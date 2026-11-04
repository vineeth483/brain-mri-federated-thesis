#!/usr/bin/env python3
"""One-round, four-client Flower connection check on the audited IID split.

This deliberately trains on two images per class per client. Its metrics are
diagnostics only, not thesis results. Run from the project root with .venv-metal.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

import numpy as np
import ray
import tensorflow as tf
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation
from federated_mobilenetv2_model import load_image, make_model


ROOT = Path(__file__).resolve().parent.parent
PARTITIONS = ROOT / "splits/clients_iid.csv"
MANIFEST = ROOT / "splits/unique_image_manifest.csv"
OUTPUT = ROOT / "outputs/flower_mobilenetv2_smoke"
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLIENT_IDS = tuple(f"client_{i}" for i in range(1, 5))
IMAGES_PER_CLASS = 2
BATCH_SIZE = 4
SEED = 42


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_partitions() -> dict[str, int]:
    """Fail before training if a client row leaves the unique training split."""
    expected = {
        row["path"]: row
        for row in read_csv(MANIFEST)
        if row["assigned_split"] == "train" and row["use_for_main_experiment"] == "yes"
    }
    rows = read_csv(PARTITIONS)
    if len(rows) != len(expected) or {row["path"] for row in rows} != set(expected):
        raise ValueError("IID partitions do not cover each unique training image exactly once")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("An exact duplicate was assigned to the IID clients")
    counts = Counter(row["client_id"] for row in rows)
    if set(counts) != set(CLIENT_IDS):
        raise ValueError("Expected exactly four IID clients")
    for row in rows:
        original = expected[row["path"]]
        if row["scenario"] != "iid" or any(
            row[field] != original[field] for field in ("class_name", "sha256")
        ):
            raise ValueError(f"Incorrect IID assignment: {row['path']}")
        if row["class_name"] not in CLASS_NAMES or not (ROOT / row["path"]).is_file():
            raise ValueError(f"Missing or invalid training image: {row['path']}")
    return {client_id: counts[client_id] for client_id in CLIENT_IDS}


def sample_client(client_id: str) -> list[dict[str, str]]:
    """Each simulated client reads and samples only its own CSV rows."""
    rows = [row for row in read_csv(PARTITIONS) if row["client_id"] == client_id]
    if not rows or any(row["scenario"] != "iid" for row in rows):
        raise ValueError(f"No valid IID rows for {client_id}")
    selected = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        pool = sorted(
            (row for row in rows if row["class_name"] == class_name),
            key=lambda row: row["path"],
        )
        if len(pool) < IMAGES_PER_CLASS:
            raise ValueError(f"Too few {class_name} images for {client_id}")
        rng = random.Random(SEED + 100 * int(client_id[-1]) + class_index)
        selected.extend(rng.sample(pool, IMAGES_PER_CLASS))
    return selected


def weights_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def make_dataset(rows: list[dict[str, str]]) -> tf.data.Dataset:
    paths = [str(ROOT / row["path"]) for row in rows]
    labels = [CLASS_NAMES.index(row["class_name"]) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    return dataset.map(load_image, num_parallel_calls=1).batch(BATCH_SIZE).prefetch(1)


client_app = ClientApp()


@client_app.train()
def train(message, context):
    partition_id = int(context.node_config["partition-id"])
    if partition_id not in range(len(CLIENT_IDS)):
        raise ValueError(f"Unexpected Flower partition ID: {partition_id}")
    client_id = CLIENT_IDS[partition_id]
    rows = sample_client(client_id)
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(f"Apple GPU is not visible inside Flower worker {client_id}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + partition_id)
    model = make_model(pretrained=False)
    incoming = message.content["arrays"].to_numpy_ndarrays()
    model.set_weights(incoming)
    initial_digest = weights_digest(model.get_weights())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    history = model.fit(make_dataset(rows), epochs=1, shuffle=False, verbose=0)
    updated = model.get_weights()
    if weights_digest(updated) == initial_digest:
        raise RuntimeError(f"No model weights changed on {client_id}")
    report = {
        "client_id": client_id,
        "assigned_images": sum(1 for row in read_csv(PARTITIONS) if row["client_id"] == client_id),
        "trained_images": len(rows),
        "sample_class_counts": dict(Counter(row["class_name"] for row in rows)),
        "sample_paths": [row["path"] for row in rows],
        "initial_weights_sha256": initial_digest,
        "updated_weights_sha256": weights_digest(updated),
        "worker_gpu_visible": True,
        "training_loss": float(history.history["loss"][-1]),
    }
    (OUTPUT / f"{client_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    reply = Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(numpy_ndarrays=updated),
                "metrics": MetricRecord(
                    {"num-examples": len(rows), "loss": report["training_loss"]}
                ),
            }
        ),
        reply_to=message,
    )
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    return reply


server_app = ServerApp()


@server_app.main()
def main(grid, context):
    tf.keras.utils.set_random_seed(SEED)
    with tf.device("/CPU:0"):
        model = make_model(pretrained=True)
        initial = model.get_weights()
    del model
    gc.collect()
    tf.keras.backend.clear_session()

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=4,
        min_available_nodes=4,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(numpy_ndarrays=initial),
        num_rounds=1,
    )
    if result.arrays is None or 1 not in result.train_metrics_clientapp:
        raise RuntimeError("Flower did not return one aggregated training round")
    final = result.arrays.to_numpy_ndarrays()
    if len(final) != len(initial) or weights_digest(final) == weights_digest(initial):
        raise RuntimeError("FedAvg did not return changed model weights")

    reports = [json.loads((OUTPUT / f"{cid}.json").read_text()) for cid in CLIENT_IDS]
    if {r["initial_weights_sha256"] for r in reports} != {weights_digest(initial)}:
        raise RuntimeError("Clients did not start from identical server weights")
    if sum(r["trained_images"] for r in reports) != 32:
        raise RuntimeError("Expected exactly 32 training images across the four clients")

    validation_rows = [
        next(
            row for row in read_csv(MANIFEST)
            if row["assigned_split"] == "validation"
            and row["use_for_main_experiment"] == "yes"
            and row["class_name"] == class_name
        )
        for class_name in CLASS_NAMES
    ]
    with tf.device("/CPU:0"):
        probe_images = tf.stack([
            load_image(tf.constant(str(ROOT / row["path"])), tf.constant(0))[0]
            for row in validation_rows
        ])
        probe_model = make_model(pretrained=False)
        probe_model.set_weights(final)
        predictions = probe_model(probe_images, training=False).numpy()
    if predictions.shape != (4, 4) or not np.isfinite(predictions).all():
        raise RuntimeError("Aggregated model did not return finite four-class predictions")
    np.testing.assert_allclose(predictions.sum(axis=1), np.ones(4), atol=1e-4)
    del probe_model
    tf.keras.backend.clear_session()

    summary = {
        "purpose": "one-round connectivity test; not a thesis accuracy result",
        "scenario": "iid",
        "model": "MobileNetV2 with ImageNet backbone and new four-class head",
        "strategy": "FedAvg",
        "rounds": 1,
        "clients_completed": len(reports),
        "images_per_client": IMAGES_PER_CLASS * len(CLASS_NAMES),
        "images_trained_total": sum(r["trained_images"] for r in reports),
        "all_clients_same_initial_weights": True,
        "initial_weights_sha256": weights_digest(initial),
        "aggregated_weights_sha256": weights_digest(final),
        "all_workers_saw_gpu": all(r["worker_gpu_visible"] for r in reports),
        "validation_probe_images": len(validation_rows),
        "aggregated_model_predictions_valid": True,
        "weighted_client_training_loss": float(result.train_metrics_clientapp[1]["loss"]),
        "clients": {r["client_id"]: r for r in reports},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("FLOWER_SMOKE_SUCCESS " + json.dumps({
        "clients": summary["clients_completed"],
        "rounds": summary["rounds"],
        "images_trained": summary["images_trained_total"],
        "all_workers_saw_gpu": summary["all_workers_saw_gpu"],
        "aggregated_model_predictions_valid": summary["aggregated_model_predictions_valid"],
    }), flush=True)


if __name__ == "__main__":
    assigned_counts = check_partitions()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("client_*.json"):
        stale.unlink()
    (OUTPUT / "summary.json").unlink(missing_ok=True)
    print("Verified IID assignments:", assigned_counts, flush=True)
    # Flower's Ray runtime_env launcher mishandles this project's path with spaces
    # on this Mac. Starting Ray first avoids that launcher and preserves GPU workers.
    ray.init(num_cpus=2, num_gpus=1, include_dashboard=False)
    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=4,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1}},
        )
    finally:
        ray.shutdown()
