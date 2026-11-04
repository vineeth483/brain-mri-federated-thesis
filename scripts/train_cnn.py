#!/usr/bin/env python3
"""Train a reproducible ImageNet-transfer EfficientNet-B0 on the Training split.

Expected data layout: DATA_DIR/{glioma,meningioma,notumor,pituitary}/*
The separate Test directory is intentionally never read by this script.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Training"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trained_efficientnet_b0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        labels="inferred",
        label_mode="int",
        image_size=(224, 224),
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="training",
        seed=args.seed,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data,
        labels="inferred",
        label_mode="int",
        image_size=(224, 224),
        batch_size=args.batch_size,
        validation_split=0.2,
        subset="validation",
        seed=args.seed,
        shuffle=True,
    )
    class_names = train_ds.class_names
    expected = ["glioma", "meningioma", "notumor", "pituitary"]
    if class_names != expected:
        raise ValueError(f"Expected class folders {expected}; found {class_names}")
    image_extensions = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    train_image_count = sum(
        1 for path in args.data.rglob("*") if path.is_file() and path.suffix.lower() in image_extensions
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=args.seed),
            tf.keras.layers.RandomRotation(0.04, seed=args.seed + 1),
            tf.keras.layers.RandomZoom(0.08, seed=args.seed + 2),
            tf.keras.layers.RandomContrast(0.1, seed=args.seed + 3),
        ],
        name="mri_augmentation",
    )
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = augmentation(inputs)
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnetb0_brain_mri")

    def callbacks(checkpoint: Path) -> list[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
            ),
        ]
    history: dict[str, list[float]] = {}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    head_checkpoint = args.output / "head_best.keras"
    first = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks(head_checkpoint),
        verbose=2,
    )
    history.update({f"head_{key}": [float(v) for v in values] for key, values in first.history.items()})

    # Unfreeze the final 20% of the backbone while keeping BatchNorm stable.
    backbone.trainable = True
    unfreeze_from = int(len(backbone.layers) * 0.8)
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    fine_tune_checkpoint = args.output / "fine_tune_best.keras"
    second = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks(fine_tune_checkpoint),
        verbose=2,
    )
    history.update({f"fine_tune_{key}": [float(v) for v in values] for key, values in second.history.items()})

    candidates = [head_checkpoint, fine_tune_checkpoint]
    scored = []
    for candidate in candidates:
        candidate_model = tf.keras.models.load_model(candidate, compile=False)
        candidate_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        candidate_loss, candidate_accuracy = candidate_model.evaluate(val_ds, verbose=0)
        scored.append((candidate_loss, candidate_accuracy, candidate))
    val_loss, val_accuracy, selected_checkpoint = min(scored, key=lambda item: item[0])
    best_model = tf.keras.models.load_model(selected_checkpoint, compile=False)
    best_model.save(args.output / "final.keras")
    (args.output / "class_names.json").write_text(json.dumps(class_names, indent=2) + "\n")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    summary = {
        "architecture": "EfficientNetB0 with ImageNet weights, trained on the Training folder",
        "seed": args.seed,
        "train_images": train_image_count,
        "validation_split": 0.2,
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "class_names": class_names,
        "checkpoint": str(args.output / "final.keras"),
        "selected_phase_checkpoint": str(selected_checkpoint),
        "test_data_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
