#!/usr/bin/env python3
"""Select between saved training phases using only the Training validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Training"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trained_efficientnet_b0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        labels="inferred",
        label_mode="int",
        image_size=(224, 224),
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="validation",
        seed=args.seed,
        # Match train_cnn.py exactly: shuffle before applying the seeded split.
        shuffle=True,
    ).prefetch(tf.data.AUTOTUNE)
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]

    results = []
    for name in ("head_best.keras", "fine_tune_best.keras"):
        path = args.output / name
        model = tf.keras.models.load_model(path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        loss, accuracy = model.evaluate(val_ds, verbose=0)
        results.append({"phase": name, "checkpoint": str(path), "validation_loss": float(loss), "validation_accuracy": float(accuracy)})

    selected = min(results, key=lambda result: result["validation_loss"])
    chosen = tf.keras.models.load_model(selected["checkpoint"], compile=False)
    chosen.save(args.output / "final.keras")
    (args.output / "class_names.json").write_text(json.dumps(class_names, indent=2) + "\n")
    summary = {
        "architecture": "EfficientNetB0 with ImageNet weights, trained on data/Training",
        "validation_split": 0.2,
        "seed": args.seed,
        "candidates": results,
        "selected_checkpoint": selected["checkpoint"],
        "selected_validation_loss": selected["validation_loss"],
        "selected_validation_accuracy": selected["validation_accuracy"],
        "final_model": str(args.output / "final.keras"),
        "test_data_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
