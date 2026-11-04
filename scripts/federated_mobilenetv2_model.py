"""Shared MobileNetV2 architecture and image decoding for Flower runs."""

from __future__ import annotations

import tensorflow as tf


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")


def make_model(pretrained: bool, seed: int = 42) -> tf.keras.Model:
    """Match the centralized MobileNetV2 input pipeline and classifier."""
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.04, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.08, seed=seed + 2),
            tf.keras.layers.RandomContrast(0.1, seed=seed + 3),
        ],
        name="mri_augmentation",
    )
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_0_to_255")
    x = augment(inputs)
    x = tf.keras.layers.Rescaling(
        1.0 / 127.5, offset=-1.0, name="mobilenet_preprocessing"
    )(x)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(224, 224, 3),
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.3, name="classifier_dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="class_probabilities"
    )(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_brain_mri")


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    return tf.image.resize(image, (224, 224)), label


def configure_training(model: tf.keras.Model, phase: str) -> None:
    """Use the centralized training script's head and fine-tuning settings."""
    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2")
    )
    if phase == "head":
        backbone.trainable = False
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif phase == "fine_tune":
        backbone.trainable = True
        unfreeze_from = int(len(backbone.layers) * 0.8)
        for layer in backbone.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
    else:
        raise ValueError(f"Unknown training phase: {phase}")
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
