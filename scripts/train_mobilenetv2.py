#!/usr/bin/env python3
"""Train a MobileNetV2 transfer-learning model on the brain MRI Training split.

Expected layout: DATA_DIR/{glioma,meningioma,notumor,pituitary}/*
The separate Test directory is not used for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def make_callbacks(checkpoint: Path) -> list[tf.keras.callbacks.Callback]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Training"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trained_mobilenetv2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No GPU is visible to TensorFlow. Run in an environment with the "
            "Apple Metal plugin enabled; CPU fallback is disabled for this run."
        )
    print(f"Training with TensorFlow {tf.__version__} on {gpus}", flush=True)

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

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=args.seed),
            tf.keras.layers.RandomRotation(0.04, seed=args.seed + 1),
            tf.keras.layers.RandomZoom(0.08, seed=args.seed + 2),
            tf.keras.layers.RandomContrast(0.1, seed=args.seed + 3),
        ],
        name="mri_augmentation",
    )
    with tf.device("/GPU:0"):
        inputs = tf.keras.Input(shape=(224, 224, 3), name="image_0_to_255")
        x = augment(inputs)
        # MobileNetV2 expects each RGB channel scaled from [0, 255] to [-1, 1].
        x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocessing")(x)
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        backbone.trainable = False
        x = backbone(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = tf.keras.layers.Dropout(0.3, name="classifier_dropout")(x)
        outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="class_probabilities")(x)
        model = tf.keras.Model(inputs, outputs, name="mobilenetv2_brain_mri")

    history: dict[str, list[float]] = {}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    head_checkpoint = args.output / "head_best.keras"
    with tf.device("/GPU:0"):
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs,
            callbacks=make_callbacks(head_checkpoint),
            verbose=2,
        )
    history.update({f"head_{k}": [float(v) for v in vals] for k, vals in head_history.history.items()})

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
    fine_checkpoint = args.output / "fine_tune_best.keras"
    with tf.device("/GPU:0"):
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=make_callbacks(fine_checkpoint),
            verbose=2,
        )
    history.update({f"fine_tune_{k}": [float(v) for v in vals] for k, vals in fine_history.history.items()})

    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    candidates = []
    for phase, path in (("head", head_checkpoint), ("fine_tune", fine_checkpoint)):
        candidate = tf.keras.models.load_model(path, compile=False)
        candidate.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        loss, accuracy = candidate.evaluate(val_ds, verbose=0)
        candidates.append({"phase": phase, "checkpoint": str(path), "validation_loss": float(loss), "validation_accuracy": float(accuracy)})

    selected = min(candidates, key=lambda result: result["validation_loss"])
    best = tf.keras.models.load_model(selected["checkpoint"], compile=False)
    best.save(args.output / "final.keras")
    (args.output / "class_names.json").write_text(json.dumps(class_names, indent=2) + "\n")
    image_exts = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    train_images = sum(1 for p in args.data.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)
    summary = {
        "architecture": "MobileNetV2 with ImageNet weights",
        "training_device": str(gpus[0]),
        "training_images": train_images,
        "validation_split": 0.2,
        "seed": args.seed,
        "candidates": candidates,
        "selected_phase": selected["phase"],
        "selected_validation_loss": selected["validation_loss"],
        "selected_validation_accuracy": selected["validation_accuracy"],
        "final_model": str(args.output / "final.keras"),
        "test_data_used": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
