#!/usr/bin/env python3
"""Create reproducible four-client IID and label-skewed non-IID partitions.

Only rows marked ``train`` in the unique-image manifest are assigned to clients.
Validation and test images remain outside the client partitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLIENT_IDS = ("client_1", "client_2", "client_3", "client_4")
OUTPUT_FIELDS = ("scenario", "client_id", "path", "class_name", "sha256", "width", "height")


def load_training_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    if any(row["use_for_main_experiment"] != "yes" for row in rows):
        raise ValueError("Expected a unique-image manifest with only selected images")
    train_rows = [row for row in rows if row["assigned_split"] == "train"]
    if len({row["path"] for row in train_rows}) != len(train_rows):
        raise ValueError("Repeated path in training manifest")
    if len({row["sha256"] for row in train_rows}) != len(train_rows):
        raise ValueError("Exact duplicate in training manifest")
    if {row["class_name"] for row in train_rows} != set(CLASS_NAMES):
        raise ValueError("Training manifest does not contain the expected four classes")
    return train_rows


def class_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in CLASS_NAMES}
    for row in rows:
        grouped[row["class_name"]].append(row)
    for index, class_name in enumerate(CLASS_NAMES):
        grouped[class_name].sort(key=lambda row: row["path"])
        random.Random(seed + index).shuffle(grouped[class_name])
    return grouped


def assign_iid(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    assigned = []
    for class_name in CLASS_NAMES:
        for index, row in enumerate(grouped[class_name]):
            assigned.append(
                {"scenario": "iid", "client_id": CLIENT_IDS[index % len(CLIENT_IDS)], **row}
            )
    return assigned


def assign_noniid(grouped: dict[str, list[dict[str, str]]], dominant_fraction: float) -> list[dict[str, str]]:
    assigned = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        images = grouped[class_name]
        dominant_count = round(len(images) * dominant_fraction)
        other_clients = [i for i in range(len(CLIENT_IDS)) if i != class_index]
        allocations = [0] * len(CLIENT_IDS)
        allocations[class_index] = dominant_count
        remaining = len(images) - dominant_count
        for index in range(remaining):
            allocations[other_clients[index % len(other_clients)]] += 1
        assert sum(allocations) == len(images)
        offset = 0
        for client_index, count in enumerate(allocations):
            for row in images[offset : offset + count]:
                assigned.append(
                    {"scenario": "noniid", "client_id": CLIENT_IDS[client_index], **row}
                )
            offset += count
    return assigned


def summarize(assigned: list[dict[str, str]], train_rows: list[dict[str, str]], scenario: str) -> dict:
    paths = [row["path"] for row in assigned]
    hashes = [row["sha256"] for row in assigned]
    expected_paths = {row["path"] for row in train_rows}
    if len(assigned) != len(train_rows) or set(paths) != expected_paths or len(set(paths)) != len(paths):
        raise AssertionError(f"{scenario}: training image coverage is not exactly once")
    if len(set(hashes)) != len(hashes):
        raise AssertionError(f"{scenario}: duplicate image content was assigned")
    if {row["scenario"] for row in assigned} != {scenario}:
        raise AssertionError(f"{scenario}: scenario label mismatch")
    by_client: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assigned:
        by_client[row["client_id"]].append(row)
    if set(by_client) != set(CLIENT_IDS):
        raise AssertionError(f"{scenario}: missing client")

    global_counts = Counter(row["class_name"] for row in train_rows)
    global_size = len(train_rows)
    summary = {"scenario": scenario, "total_images": global_size, "global_class_counts": dict(global_counts), "clients": {}}
    for client_index, client_id in enumerate(CLIENT_IDS):
        rows = by_client[client_id]
        counts = Counter(row["class_name"] for row in rows)
        fractions = {name: counts[name] / len(rows) for name in CLASS_NAMES}
        global_fractions = {name: global_counts[name] / global_size for name in CLASS_NAMES}
        total_variation = sum(abs(fractions[name] - global_fractions[name]) for name in CLASS_NAMES) / 2
        square_512_fraction = sum(
            int(row["width"]) == 512 and int(row["height"]) == 512 for row in rows
        ) / len(rows)
        summary["clients"][client_id] = {
            "total_images": len(rows),
            "class_counts": {name: counts[name] for name in CLASS_NAMES},
            "class_fractions": {name: round(fractions[name], 6) for name in CLASS_NAMES},
            "class_total_variation_from_global": round(total_variation, 6),
            "square_512_fraction": round(square_512_fraction, 6),
        }
        if scenario == "iid" and max(
            abs(fractions[name] - global_fractions[name]) for name in CLASS_NAMES
        ) > 0.01:
            raise AssertionError(f"{client_id}: IID class proportions differ from global by over 1 point")
        if scenario == "noniid" and fractions[CLASS_NAMES[client_index]] < 0.5:
            raise AssertionError(f"{client_id}: dominant class is under 50%")
    return summary


def write_csv(path: Path, assigned: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in OUTPUT_FIELDS}
            for row in sorted(assigned, key=lambda row: (row["client_id"], row["path"]))
        )


def write_report(path: Path, iid: dict, noniid: dict, seed: int, dominant_fraction: float) -> None:
    lines = [
        "# Four-client data partition audit",
        "",
        "Both scenarios use exactly the same unique training images. Each image is assigned once per scenario; validation and test images are not assigned to clients.",
        f"Random seed: **{seed}**. Non-IID dominant-class allocation: **{dominant_fraction:.0%}** of each class to its designated client, with the remainder divided among the other three.",
        "",
    ]
    for summary in (iid, noniid):
        lines.extend(
            [
                f"## {summary['scenario'].upper()}",
                "",
                "| Client | Glioma | Meningioma | No tumor | Pituitary | Total | Designated class share | Class distance from global | 512x512 share |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for client_index, client_id in enumerate(CLIENT_IDS):
            client = summary["clients"][client_id]
            counts = client["class_counts"]
            dominant_share = client["class_fractions"][CLASS_NAMES[client_index]]
            lines.append(
                f"| {client_id} | "
                + " | ".join(str(counts[name]) for name in CLASS_NAMES)
                + f" | {client['total_images']} | {dominant_share:.1%} | "
                + f"{client['class_total_variation_from_global']:.1%} | {client['square_512_fraction']:.1%} |"
            )
        lines.append("")
    lines.extend(
        [
            "The designated class is glioma for client 1, meningioma for client 2, no tumor for client 3, and pituitary for client 4. Class distance from global is total variation distance: 0% means the same class proportions as the complete training pool. The 512x512 column checks one visible image-size characteristic; similar label counts alone do not establish full statistical IID.",
            "Non-IID here means deliberate **label skew**. Because image dimensions also correlate with class in this dataset, the non-IID allocation changes the image-size mix too; those two effects cannot be fully separated with the available metadata.",
            "These clients do not simulate verified differences between real hospitals, scanners, or patients, because those identifiers are unavailable.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("splits/unique_image_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--report", type=Path, default=Path("reports/client_partition_audit.md"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dominant-fraction", type=float, default=0.55)
    args = parser.parse_args()
    if not 0.5 <= args.dominant_fraction < 1:
        parser.error("--dominant-fraction must be at least 0.5 and less than 1")

    train_rows = load_training_rows(args.manifest)
    grouped = class_rows(train_rows, args.seed)
    iid_rows = assign_iid(grouped)
    noniid_rows = assign_noniid(grouped, args.dominant_fraction)
    iid_summary = summarize(iid_rows, train_rows, "iid")
    noniid_summary = summarize(noniid_rows, train_rows, "noniid")
    client_sizes = [client["total_images"] for client in noniid_summary["clients"].values()]
    mean_size = sum(client_sizes) / len(client_sizes)
    if max(abs(size - mean_size) / mean_size for size in client_sizes) > 0.1:
        raise AssertionError("Non-IID client sizes differ by more than 10% from the mean")

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "clients_iid.csv", iid_rows)
    write_csv(args.output / "clients_noniid.csv", noniid_rows)
    summary = {
        "seed": args.seed,
        "dominant_fraction": args.dominant_fraction,
        "iid": iid_summary,
        "noniid": noniid_summary,
    }
    (args.output / "client_partition_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(args.report, iid_summary, noniid_summary, args.seed, args.dominant_fraction)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
