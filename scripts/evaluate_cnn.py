#!/usr/bin/env python3
"""Evaluate the downloaded brain MRI CNN on a labeled class-folder dataset.

Expected layout: DATA_DIR/{glioma,meningioma,notumor,pituitary}/*.(jpg|png|...)
The directory should contain the held-out test set only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def load_images(data_dir: Path, class_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        files = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            raise ValueError(f"No supported images found in {class_dir}")
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            images.append(image.astype(np.float32))  # Keep 0–255 input range per model card.
            labels.append(label)
            paths.append(str(path))
    return np.asarray(images, dtype=np.float32), np.asarray(labels), paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Held-out test folder with one subfolder per class")
    parser.add_argument("--model", type=Path, default=Path("models/centralized/xpathology_efficientnet_b0/xpathology_brain_specialist_b0.keras"))
    parser.add_argument("--classes", type=Path, default=Path("models/centralized/xpathology_efficientnet_b0/brain_class_names.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/xpathology_efficientnet_b0"))
    args = parser.parse_args()

    class_names = json.loads(args.classes.read_text())
    model = tf.keras.models.load_model(args.model, compile=False)
    images, y_true, paths = load_images(args.data, class_names)
    probabilities = np.asarray(model.predict(images, batch_size=32, verbose=1))
    y_pred = probabilities.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=range(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": str(args.model),
        "test_data": str(args.data),
        "num_images": int(len(y_true)),
        "class_names": class_names,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist(),
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "predictions.csv").open("w") as f:
        f.write("path,true_label,predicted_label,confidence\n")
        for path, truth, pred, scores in zip(paths, y_true, y_pred, probabilities):
            f.write(f'"{path.replace(chr(34), chr(34) * 2)}",{class_names[truth]},{class_names[pred]},{float(scores[pred]):.8f}\n')
    print(json.dumps({"num_images": len(y_true), "accuracy": summary["accuracy"], "metrics": str(args.output / "metrics.json"), "predictions": str(args.output / "predictions.csv")}, indent=2))


if __name__ == "__main__":
    main()
