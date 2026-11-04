#!/usr/bin/env python3
"""Audit brain MRI images and create a fixed, duplicate-aware train/validation split.

Run from the project root. Original images are only read, never moved or changed.
The separate Test folder is retained as a held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
SOURCE_SPLITS = ("Training", "Test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FIELDNAMES = (
    "path",
    "source_split",
    "assigned_split",
    "class_name",
    "sha256",
    "duplicate_group_size",
    "use_for_main_experiment",
    "width",
    "height",
    "image_format",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.load()  # Decode the entire file, not only its header.
        width, height = image.size
        image_format = image.format or "unknown"
    if width < 1 or height < 1:
        raise ValueError("image has an empty dimension")
    return width, height, image_format


def collect_images(data_root: Path, project_root: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    ignored: list[str] = []
    for source_split in SOURCE_SPLITS:
        split_dir = data_root / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing data folder: {split_dir}")
        present_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if present_classes != set(CLASS_NAMES):
            raise ValueError(
                f"Expected class folders {CLASS_NAMES} in {split_dir}; found {sorted(present_classes)}"
            )
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            paths = sorted(p for p in class_dir.rglob("*") if p.is_file())
            image_paths = [p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS]
            ignored.extend(str(p.relative_to(project_root)) for p in paths if p not in image_paths)
            if not image_paths:
                raise ValueError(f"No supported images in {class_dir}")
            for path in image_paths:
                relative_path = path.relative_to(project_root).as_posix()
                try:
                    width, height, image_format = inspect_image(path)
                    digest = sha256_file(path)
                except Exception as exc:
                    raise ValueError(f"Unreadable image {relative_path}: {exc}") from exc
                rows.append(
                    {
                        "path": relative_path,
                        "source_split": source_split,
                        "assigned_split": "test" if source_split == "Test" else "",
                        "class_name": class_name,
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "image_format": image_format,
                    }
                )
    return rows, ignored


def assign_training_validation(rows: list[dict], seed: int, val_fraction: float) -> None:
    by_class_and_hash: dict[str, dict[str, list[dict]]] = {
        class_name: defaultdict(list) for class_name in CLASS_NAMES
    }
    for row in rows:
        if row["source_split"] == "Training":
            by_class_and_hash[row["class_name"]][row["sha256"]].append(row)

    for class_index, class_name in enumerate(CLASS_NAMES):
        groups = list(by_class_and_hash[class_name].values())
        groups.sort(key=lambda group: group[0]["sha256"])
        random.Random(seed + class_index).shuffle(groups)
        total = sum(len(group) for group in groups)
        target = round(total * val_fraction)
        validation_count = 0
        for group in groups:
            assigned = "validation" if validation_count + len(group) <= target else "train"
            for row in group:
                row["assigned_split"] = assigned
            if assigned == "validation":
                validation_count += len(group)
        if validation_count != target:
            raise ValueError(
                f"Could not assign {target} validation images in {class_name} "
                "without separating duplicates"
            )


def make_summary(rows: list[dict], ignored: list[str], seed: int, val_fraction: float) -> dict:
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    for group in by_hash.values():
        for row in group:
            row["duplicate_group_size"] = len(group)
            row["use_for_main_experiment"] = "yes" if row is min(group, key=lambda item: item["path"]) else "no"

    conflicts = [
        {"sha256": digest, "paths": [row["path"] for row in group]}
        for digest, group in by_hash.items()
        if len({row["class_name"] for row in group}) > 1
    ]
    cross_source = [
        {"sha256": digest, "paths": [row["path"] for row in group]}
        for digest, group in by_hash.items()
        if len({row["source_split"] for row in group}) > 1
    ]
    split_crossings = [
        {"sha256": digest, "paths": [row["path"] for row in group]}
        for digest, group in by_hash.items()
        if len({row["assigned_split"] for row in group}) > 1
    ]
    duplicate_stats = {}
    for source_split in SOURCE_SPLITS:
        source_groups = [
            [row for row in group if row["source_split"] == source_split]
            for group in by_hash.values()
        ]
        repeated = [group for group in source_groups if len(group) > 1]
        duplicate_stats[source_split] = {
            "duplicate_groups": len(repeated),
            "redundant_copies": sum(len(group) - 1 for group in repeated),
            "unique_images": len({row["sha256"] for row in rows if row["source_split"] == source_split}),
        }

    fingerprint = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        fingerprint.update(
            f"{row['path']}\t{row['class_name']}\t{row['sha256']}\n".encode("utf-8")
        )
    return {
        "seed": seed,
        "validation_fraction": val_fraction,
        "dataset_sha256": fingerprint.hexdigest(),
        "total_images": len(rows),
        "source_counts": {
            source: dict(sorted(Counter(row["class_name"] for row in rows if row["source_split"] == source).items()))
            for source in SOURCE_SPLITS
        },
        "assigned_counts": {
            split: dict(sorted(Counter(row["class_name"] for row in rows if row["assigned_split"] == split).items()))
            for split in ("train", "validation", "test")
        },
        "unique_content_counts": {
            split: {
                class_name: len(
                    {
                        row["sha256"]
                        for row in rows
                        if row["assigned_split"] == split and row["class_name"] == class_name
                    }
                )
                for class_name in CLASS_NAMES
            }
            for split in ("train", "validation", "test")
        },
        "excluded_exact_copies": {
            split: sum(
                row["use_for_main_experiment"] == "no"
                for row in rows
                if row["assigned_split"] == split
            )
            for split in ("train", "validation", "test")
        },
        "duplicate_stats": duplicate_stats,
        "cross_source_duplicate_groups": cross_source,
        "conflicting_label_groups": conflicts,
        "cross_assigned_split_duplicate_groups": split_crossings,
        "ignored_non_image_files": ignored,
        "image_dimensions": {
            "min_width": min(row["width"] for row in rows),
            "max_width": max(row["width"] for row in rows),
            "min_height": min(row["height"] for row in rows),
            "max_height": max(row["height"] for row in rows),
        },
    }


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Brain MRI data split audit",
        "",
        "The original image files were read but not moved or edited. Exact duplicates were identified by SHA-256 of file bytes.",
        "",
        f"- Total readable images: **{summary['total_images']:,}**",
        f"- Split seed: **{summary['seed']}**",
        f"- Validation fraction of the original Training folder: **{summary['validation_fraction']:.0%}**",
        f"- Dataset fingerprint: `{summary['dataset_sha256']}`",
        "",
        "| Assigned split | Glioma | Meningioma | No tumor | Pituitary | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        counts = summary["assigned_counts"][split]
        values = [counts.get(class_name, 0) for class_name in CLASS_NAMES]
        lines.append(f"| {split} | " + " | ".join(str(value) for value in values) + f" | {sum(values):,} |")
    lines.extend(
        [
            "",
            "The file counts are class-balanced. Repeated files mean the number of distinct images is different:",
            "",
            "| Assigned split | Unique glioma | Unique meningioma | Unique no tumor | Unique pituitary |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        counts = summary["unique_content_counts"][split]
        lines.append(f"| {split} | " + " | ".join(str(counts[class_name]) for class_name in CLASS_NAMES) + " |")
    lines.extend(
        [
            "",
            "The main experiments will use `splits/unique_image_manifest.csv`: one deterministic representative per exact-duplicate group, selected by path name. All distinct images remain in use; no class downsampling or balancing was applied.",
            "The complete 7,200-file inventory remains in `splits/image_manifest.csv`, with a `use_for_main_experiment` column. Original image files were not deleted or moved.",
        ]
    )
    lines.extend(["", "## Duplicate audit", ""])
    for source in SOURCE_SPLITS:
        stats = summary["duplicate_stats"][source]
        lines.append(
            f"- {source}: {stats['duplicate_groups']} exact-duplicate groups, "
            f"{stats['redundant_copies']} redundant copies, {stats['unique_images']:,} unique files by content."
        )
    lines.extend(
        [
            f"- Duplicate groups crossing Training and Test: {len(summary['cross_source_duplicate_groups'])}.",
            f"- Duplicate groups with conflicting class labels: {len(summary['conflicting_label_groups'])}.",
            f"- Duplicate groups crossing assigned train/validation/test splits: {len(summary['cross_assigned_split_duplicate_groups'])}.",
            "",
            "## Scope and limits",
            "",
            "The validation set comes only from the original Training folder. The original Test folder remains a held-out test set; its labels and model predictions were not used to choose this split.",
            "The dataset does not provide patient identifiers, so separation of different slices from the same patient cannot be verified.",
            "This audit detects byte-identical copies, not visually similar images saved with different bytes.",
            "Class-balanced file counts do not imply unbiased examples: exact copies occur more often in the no-tumor class, and image dimensions also differ by class. The unique-image manifest removes known duplicate weighting, but source and near-duplicate effects remain unverified.",
            "The Test folder contains repeated files. The main evaluation should use its 1,584 unique images, with the full 1,600-file result also reported for comparison with the earlier baseline.",
            "The earlier centralized MobileNetV2 run used its own random validation split; a matched centralized comparison will use this manifest.",
            "",
        ]
    )
    if summary["ignored_non_image_files"]:
        lines.append("Ignored non-image files: " + ", ".join(f"`{item}`" for item in summary["ignored_non_image_files"]) + ".")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--report", type=Path, default=Path("reports/data_split_audit.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")

    data_root = args.data.resolve()
    project_root = data_root.parent
    rows, ignored = collect_images(data_root, project_root)
    # Label conflicts and Training/Test overlap must stop creation of a trusted split.
    preliminary = make_summary(rows, ignored, args.seed, args.validation_fraction)
    if preliminary["conflicting_label_groups"] or preliminary["cross_source_duplicate_groups"]:
        raise ValueError(
            "Found conflicting labels or exact Training/Test duplicates; "
            "review these groups before creating the split:\n"
            + json.dumps(
                {
                    "conflicting_labels": preliminary["conflicting_label_groups"],
                    "cross_source": preliminary["cross_source_duplicate_groups"],
                },
                indent=2,
            )
        )
    assign_training_validation(rows, args.seed, args.validation_fraction)
    summary = make_summary(rows, ignored, args.seed, args.validation_fraction)
    if summary["cross_assigned_split_duplicate_groups"]:
        raise AssertionError("An exact duplicate crossed an assigned split")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "image_manifest.csv"
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in FIELDNAMES}
            for row in sorted(rows, key=lambda item: item["path"])
        )
    unique_manifest_path = args.output / "unique_image_manifest.csv"
    unique_rows = sorted(
        (row for row in rows if row["use_for_main_experiment"] == "yes"),
        key=lambda item: item["path"],
    )
    if len(unique_rows) != len({row["sha256"] for row in rows}):
        raise AssertionError("Unique-image manifest does not have one row per content hash")
    with unique_manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({field: row[field] for field in FIELDNAMES} for row in unique_rows)
    summary_path = args.output / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.report, summary)
    print(json.dumps({"full_manifest": str(manifest_path), "unique_manifest": str(unique_manifest_path), "summary": str(summary_path), "report": str(args.report), "unique_counts": summary["unique_content_counts"], "excluded_exact_copies": summary["excluded_exact_copies"]}, indent=2))


if __name__ == "__main__":
    main()
