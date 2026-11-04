#!/usr/bin/env python3
"""Train MobileNetV2 with four Flower clients on one audited partition.

Run one scenario at a time. Select checkpoints using only the unique validation
split; the held-out test split is not read by this script.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
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

from federated_mobilenetv2_model import (
    CLASS_NAMES,
    configure_training,
    load_image,
    make_model,
)


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "splits/unique_image_manifest.csv"
CLIENT_IDS = tuple(f"client_{i}" for i in range(1, 5))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def weights_digest(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def audit_partition(scenario: str) -> dict[str, int]:
    """Check full coverage of unique training images and split isolation."""
    selected = {
        row["path"]: row for row in read_csv(MANIFEST)
        if row["assigned_split"] == "train" and row["use_for_main_experiment"] == "yes"
    }
    rows = read_csv(ROOT / f"splits/clients_{scenario}.csv")
    if len(rows) != len(selected) or {row["path"] for row in rows} != set(selected):
        raise ValueError("Clients must contain each unique training image exactly once")
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("The client assignment contains duplicate image content")
    counts = Counter(row["client_id"] for row in rows)
    if set(counts) != set(CLIENT_IDS):
        raise ValueError("Expected four clients")
    for row in rows:
        original = selected[row["path"]]
        if row["scenario"] != scenario or any(
            row[key] != original[key] for key in ("sha256", "class_name")
        ):
            raise ValueError(f"Client assignment disagrees with manifest: {row['path']}")
        if not (ROOT / row["path"]).is_file():
            raise FileNotFoundError(ROOT / row["path"])
    return {client_id: counts[client_id] for client_id in CLIENT_IDS}


def client_rows(scenario: str, client_id: str) -> list[dict[str, str]]:
    rows = [
        row for row in read_csv(ROOT / f"splits/clients_{scenario}.csv")
        if row["client_id"] == client_id
    ]
    if not rows or any(row["scenario"] != scenario for row in rows):
        raise ValueError(f"No valid {scenario} rows for {client_id}")
    return sorted(rows, key=lambda row: row["path"])


def dataset(rows: list[dict[str, str]], batch_size: int, *, shuffle_seed: int | None):
    paths = [str(ROOT / row["path"]) for row in rows]
    labels = [CLASS_NAMES.index(row["class_name"]) for row in rows]
    data = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle_seed is not None:
        data = data.shuffle(len(rows), seed=shuffle_seed, reshuffle_each_iteration=False)
    return data.map(load_image, num_parallel_calls=2).batch(batch_size).prefetch(1)


def initial_weights(path: Path | None) -> list[np.ndarray]:
    if path is None:
        model = make_model(pretrained=True)
        arrays = model.get_weights()
        del model
        tf.keras.backend.clear_session()
        return arrays
    with np.load(path, allow_pickle=False) as data:
        arrays = [data[f"arr_{i}"] for i in range(len(data.files))]
    model = make_model(pretrained=False)
    model.set_weights(arrays)
    del model
    tf.keras.backend.clear_session()
    return arrays


def make_apps(
    scenario: str,
    output: Path,
    counts: dict[str, int],
    head_rounds: int,
    fine_rounds: int,
    batch_size: int,
    seed: int,
    starting_weights: list[np.ndarray],
):
    client_app = ClientApp()

    @client_app.train()
    def train(message, context):
        partition_id = int(context.node_config["partition-id"])
        if partition_id not in range(4):
            raise ValueError(f"Unexpected Flower partition ID: {partition_id}")
        client_id = CLIENT_IDS[partition_id]
        round_number = int(message.content["config"]["server-round"])
        phase = "head" if round_number <= head_rounds else "fine_tune"
        rows = client_rows(scenario, client_id)
        if len(rows) != counts[client_id]:
            raise ValueError(f"Assigned image count changed for {client_id}")
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError(f"Apple GPU not visible to {client_id}")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + 100 * round_number + partition_id)
        model = make_model(pretrained=False, seed=seed)
        model.set_weights(message.content["arrays"].to_numpy_ndarrays())
        before = weights_digest(model.get_weights())
        configure_training(model, phase)
        history = model.fit(
            dataset(
                rows,
                batch_size,
                shuffle_seed=seed + 100 * round_number + partition_id,
            ),
            epochs=1,
            shuffle=False,
            verbose=0,
        )
        updated = model.get_weights()
        after = weights_digest(updated)
        if before == after:
            raise RuntimeError(f"No model weights changed for {client_id} round {round_number}")
        report = {
            "scenario": scenario,
            "round": round_number,
            "phase": phase,
            "client_id": client_id,
            "images_trained": len(rows),
            "class_counts": dict(Counter(row["class_name"] for row in rows)),
            "starting_weights_sha256": before,
            "updated_weights_sha256": after,
            "gpu_visible": True,
            "training_loss": float(history.history["loss"][-1]),
            "training_accuracy": float(history.history["accuracy"][-1]),
        }
        (output / "client_runs" / f"round_{round_number:02d}_{client_id}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        reply = Message(
            content=RecordDict({
                "arrays": ArrayRecord(numpy_ndarrays=updated),
                "metrics": MetricRecord({
                    "num-examples": len(rows),
                    "loss": report["training_loss"],
                    "accuracy": report["training_accuracy"],
                }),
            }),
            reply_to=message,
        )
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return reply

    server_app = ServerApp()

    @server_app.main()
    def server_main(grid, context):
        tf.keras.utils.set_random_seed(seed)
        validation_rows = sorted(
            (
                row for row in read_csv(MANIFEST)
                if row["assigned_split"] == "validation"
                and row["use_for_main_experiment"] == "yes"
            ),
            key=lambda row: row["path"],
        )
        if len(validation_rows) != 1078:
            raise ValueError("Unexpected number of unique validation images")
        validation_data = dataset(validation_rows, batch_size, shuffle_seed=None)
        evaluation_model = make_model(pretrained=False, seed=seed)
        evaluation_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        history = []
        best_loss = float("inf")
        best_round = -1

        def evaluate(round_number, arrays):
            nonlocal best_loss, best_round
            evaluation_model.set_weights(arrays.to_numpy_ndarrays())
            loss, accuracy = evaluation_model.evaluate(validation_data, verbose=0)
            entry = {
                "round": round_number,
                "phase": "initial" if round_number == 0 else (
                    "head" if round_number <= head_rounds else "fine_tune"
                ),
                "validation_images": len(validation_rows),
                "validation_loss": float(loss),
                "validation_accuracy": float(accuracy),
            }
            history.append(entry)
            (output / "validation_history.json").write_text(json.dumps(history, indent=2) + "\n")
            if loss < best_loss:
                best_loss = float(loss)
                best_round = round_number
                evaluation_model.save_weights(output / "best.weights.h5")
            print("VALIDATION " + json.dumps(entry), flush=True)
            return MetricRecord({"loss": float(loss), "accuracy": float(accuracy)})

        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=4,
            min_available_nodes=4,
        )
        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(numpy_ndarrays=starting_weights),
            num_rounds=head_rounds + fine_rounds,
            evaluate_fn=evaluate,
        )
        if result.arrays is None or len(history) != head_rounds + fine_rounds + 1:
            raise RuntimeError("Federated training did not complete all planned rounds")
        client_reports = {}
        for round_number in range(1, head_rounds + fine_rounds + 1):
            reports = [
                json.loads((output / "client_runs" / f"round_{round_number:02d}_{cid}.json").read_text())
                for cid in CLIENT_IDS
            ]
            if sum(report["images_trained"] for report in reports) != sum(counts.values()):
                raise RuntimeError(f"Round {round_number} did not train on all assigned images")
            if len({report["starting_weights_sha256"] for report in reports}) != 1:
                raise RuntimeError(f"Clients started round {round_number} from different weights")
            client_reports[str(round_number)] = {
                report["client_id"]: report["images_trained"] for report in reports
            }
        summary = {
            "scenario": scenario,
            "architecture": "MobileNetV2 ImageNet backbone with four-class head",
            "strategy": "FedAvg",
            "seed": seed,
            "batch_size": batch_size,
            "local_epochs_per_round": 1,
            "head_rounds": head_rounds,
            "fine_tune_rounds": fine_rounds,
            "rounds_completed": head_rounds + fine_rounds,
            "clients_per_round": 4,
            "assigned_images_per_client": counts,
            "training_images_per_round": sum(counts.values()),
            "validation_images": len(validation_rows),
            "test_images_used": False,
            "initial_weights_sha256": weights_digest(starting_weights),
            "final_weights_sha256": weights_digest(result.arrays.to_numpy_ndarrays()),
            "best_validation_round": best_round,
            "best_validation_loss": best_loss,
            "best_validation_accuracy": history[best_round]["validation_accuracy"],
            "checkpoint": str(output / "best.weights.h5"),
            "client_images_by_round": client_reports,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("FEDERATED_RUN_SUCCESS " + json.dumps({
            "scenario": scenario,
            "rounds": summary["rounds_completed"],
            "best_round": best_round,
            "best_validation_accuracy": summary["best_validation_accuracy"],
        }), flush=True)

    return server_app, client_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("iid", "noniid"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--head-rounds", type=int, default=8)
    parser.add_argument("--fine-rounds", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.head_rounds < 1 or args.fine_rounds < 1 or args.batch_size < 1:
        parser.error("Round counts and batch size must be positive")
    output = (args.output or ROOT / f"outputs/flower_{args.scenario}_full").resolve()
    counts = audit_partition(args.scenario)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new --output path: {output}")
    print(f"Verified {args.scenario} assignment: {counts}", flush=True)
    tf.keras.utils.set_random_seed(args.seed)
    starting_weights = initial_weights(args.initial_weights)
    output.mkdir(parents=True)
    (output / "client_runs").mkdir()
    np.savez_compressed(output / "initial_weights.npz", *starting_weights)
    (output / "run_config.json").write_text(json.dumps({
        "scenario": args.scenario,
        "head_rounds": args.head_rounds,
        "fine_rounds": args.fine_rounds,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "starting_weights_sha256": weights_digest(starting_weights),
        "starting_weights_source": str(args.initial_weights) if args.initial_weights else "ImageNet backbone plus new seeded head",
        "assigned_images_per_client": counts,
    }, indent=2) + "\n")
    server_app, client_app = make_apps(
        args.scenario, output, counts, args.head_rounds, args.fine_rounds,
        args.batch_size, args.seed, starting_weights,
    )
    # Pre-initializing Ray avoids Flower's path-with-spaces runtime_env issue here.
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


if __name__ == "__main__":
    main()
