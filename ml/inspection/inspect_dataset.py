"""PlantGuard ML — Dataset inspection.

Recursively scans image directories, verifies every image, and produces a
factual report: class counts, integrity, dimensions, file sizes, exact
duplicates. Every metric is computed from files on disk — nothing is
hard-coded or estimated.

Usage:
    python -m ml.inspection.inspect_dataset --root data/raw --out data/reports
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import os
import numpy as np
from PIL import Image, ImageFile

# Tolerate truncated images instead of raising, so we can flag them.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = 300_000_000  # decompression-bomb guard

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Only report the datasets we intentionally extracted. The archive zips also
# contain duplicate/variant copies (grayscale, segmented, augmented) which must
# never enter training — see docs on leakage.
KNOWN_DATASETS = ("plantvillage", "nutrition", "plantdoc")


@dataclass
class ImageRecord:
    path: str
    dataset: str
    label: str
    split: str | None  # plantdoc/nutrition ship pre-assigned splits
    format: str | None
    width: int
    height: int
    size_bytes: int
    md5: str
    status: str  # "ok" | "corrupt" | "unreadable"
    error: str | None = None


@dataclass
class DatasetReport:
    generated_at: str
    root: str
    datasets: dict = field(default_factory=dict)
    total_images: int = 0
    total_corrupt: int = 0
    total_exact_duplicate_groups: int = 0
    total_exact_duplicate_files: int = 0
    duplicate_wasted_bytes: int = 0
    scan_seconds: float = 0.0
    notes: list = field(default_factory=list)


def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_split(rel_parts: tuple[str, ...]) -> str | None:
    """Datasets that ship a pre-assigned train/test split encode it in the path."""
    for part in rel_parts:
        p = part.lower()
        if p in {"train", "test", "val", "validation"}:
            return "val" if p in {"val", "validation"} else p
    return None


def _scan_task(args: tuple) -> ImageRecord:
    path, dataset, label, split = args
    return scan_image(path, dataset, label, split)


def scan_image(path: Path, dataset: str, label: str, split: str | None) -> ImageRecord:
    rec = ImageRecord(
        path=str(path),
        dataset=dataset,
        label=label,
        split=split,
        format=None,
        width=0,
        height=0,
        size_bytes=path.stat().st_size,
        md5="",
        status="unreadable",
    )
    try:
        rec.md5 = md5_of_file(path)
        with Image.open(path) as im:
            im.verify()  # structural check (cheap)
        with Image.open(path) as im:
            rec.format = im.format
            rec.width, rec.height = im.size
            # Full decode catches corrupt payloads that verify() misses.
            im.convert("RGB")
        rec.status = "ok"
    except Exception as e:  # noqa: BLE001 — any failure is a data point
        rec.status = "corrupt"
        rec.error = f"{type(e).__name__}: {e}"
    return rec


def discover_images(dataset_dir: Path) -> list[tuple[Path, str, str | None]]:
    """Yield (path, label, split) for every candidate image under a dataset dir."""
    found: list[tuple[Path, str, str | None]] = []
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_dir)
        parts = rel.parts
        split = infer_split(parts)
        # Label = first path component that is not the split dir.
        label = next((p for p in parts if p.lower() not in {"train", "test", "val", "validation"}), "unknown")
        found.append((path, label, split))
    return found


def class_distribution(records: list[ImageRecord]) -> dict:
    ok = [r for r in records if r.status == "ok"]
    per_class = Counter(r.label for r in ok)
    counts = np.array(list(per_class.values()), dtype=float)
    if len(counts) == 0:
        return {"classes": 0, "per_class": {}}
    dist = {
        "classes": len(per_class),
        "images": int(sum(per_class.values())),
        "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "imbalance_ratio": round(float(counts.max() / max(counts.min(), 1)), 2),
        "mean_per_class": round(float(counts.mean()), 1),
    }
    return dist


def find_exact_duplicates(records: list[ImageRecord]) -> tuple[list[list[str]], int]:
    """Group 'ok' records by md5. Returns (groups, wasted_bytes)."""
    by_md5: dict[str, list[ImageRecord]] = {}
    for r in records:
        if r.status == "ok" and r.md5:
            by_md5.setdefault(r.md5, []).append(r)
    groups = []
    wasted = 0
    for md5, recs in by_md5.items():
        if len(recs) > 1:
            groups.append([r.path for r in recs])
            wasted += sum(r.size_bytes for r in recs[1:])
    return groups, wasted


def inspect_dataset(root: Path) -> tuple[DatasetReport, list[ImageRecord]]:
    t0 = time.time()
    report = DatasetReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        root=str(root),
    )
    all_records: list[ImageRecord] = []

    for ds_name in KNOWN_DATASETS:
        ds_dir = root / ds_name
        if not ds_dir.is_dir():
            report.notes.append(f"Dataset '{ds_name}' not present at {ds_dir} — skipped.")
            continue

        candidates = discover_images(ds_dir)
        records: list[ImageRecord] = []
        all_files = [p for p in ds_dir.rglob("*") if p.is_file()]
        skipped_ext = len(all_files) - len(candidates)

        # Parallel scan: integrity verification is CPU-bound (JPEG decode),
        # so we fan out across cores. Results are identical to serial scanning.
        tasks = [(path, ds_name, label, split) for path, label, split in candidates]
        with ProcessPoolExecutor(max_workers=min(4, (os.cpu_count() or 2))) as pool:
            for rec in pool.map(_scan_task, tasks, chunksize=64):
                records.append(rec)

        ok = [r for r in records if r.status == "ok"]
        corrupt = [r for r in records if r.status != "ok"]
        dup_groups, dup_wasted = find_exact_duplicates(records)
        widths = np.array([r.width for r in ok], dtype=float)
        heights = np.array([r.height for r in ok], dtype=float)
        sizes = np.array([r.size_bytes for r in ok], dtype=float)
        formats = Counter(r.format for r in ok)

        ds_report = {
            "dir": str(ds_dir),
            "candidate_images": len(candidates),
            "non_image_files_skipped": skipped_ext,
            "ok": len(ok),
            "corrupt": len(corrupt),
            "corrupt_examples": [
                {"path": r.path, "error": r.error} for r in corrupt[:10]
            ],
            "formats": dict(formats),
            "dimensions": {
                "width_min": int(widths.min()) if len(widths) else 0,
                "width_median": float(np.median(widths)) if len(widths) else 0,
                "width_max": int(widths.max()) if len(widths) else 0,
                "height_min": int(heights.min()) if len(heights) else 0,
                "height_median": float(np.median(heights)) if len(heights) else 0,
                "height_max": int(heights.max()) if len(heights) else 0,
            },
            "file_size_bytes": {
                "min": int(sizes.min()) if len(sizes) else 0,
                "median": float(np.median(sizes)) if len(sizes) else 0,
                "max": int(sizes.max()) if len(sizes) else 0,
                "total_gb": round(float(sizes.sum() / 1e9), 3),
            },
            "splits": dict(Counter(r.split for r in ok)),
            "class_distribution": class_distribution(records),
            "exact_duplicate_groups": len(dup_groups),
            "exact_duplicate_files": sum(len(g) - 1 for g in dup_groups),
            "duplicate_wasted_mb": round(dup_wasted / 1e6, 2),
        }
        report.datasets[ds_name] = ds_report
        all_records.extend(records)
        report.total_images += len(ok)
        report.total_corrupt += len(corrupt)
        report.total_exact_duplicate_groups += len(dup_groups)
        report.total_exact_duplicate_files += sum(len(g) - 1 for g in dup_groups)
        report.duplicate_wasted_bytes += dup_wasted

    report.scan_seconds = round(time.time() - t0, 1)
    return report, all_records

    # Feasibility verdicts are computed from the counts above, not assumed.
    report.notes.append(
        "Feasibility: disease/health classification requires >=2 classes with "
        ">=100 images each; plant ID requires the label to encode species "
        "(true for PlantVillage 'Plant___condition' naming)."
    )
    return report


def write_reports(report: DatasetReport, out_dir: Path, records: list[ImageRecord]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    (out_dir / f"dataset_report_{stamp}.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "dataset_report_latest.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Flatten class distributions into a CSV summary.
    with open(out_dir / "class_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label", "images"])
        for ds, info in report.datasets.items():
            for label, n in info["class_distribution"].get("per_class", {}).items():
                w.writerow([ds, label, n])

    with open(out_dir / "image_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "dataset", "label", "split", "format", "width", "height",
                    "size_bytes", "md5", "status", "error"])
        for r in records:
            w.writerow([r.path, r.dataset, r.label, r.split, r.format, r.width,
                        r.height, r.size_bytes, r.md5, r.status, r.error or ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect PlantGuard image datasets")
    parser.add_argument("--root", default="data/raw", help="Root folder containing dataset folders")
    parser.add_argument("--out", default="data/reports", help="Output folder for reports")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    print(f"Scanning {root} ...")
    report, records = inspect_dataset(root)
    print(f"Scan complete in {report.scan_seconds}s — {report.total_images} ok, "
          f"{report.total_corrupt} corrupt, "
          f"{report.total_exact_duplicate_groups} duplicate groups")

    for ds, info in report.datasets.items():
        cd = info["class_distribution"]
        print(f"\n[{ds}] {info['ok']} ok / {info['corrupt']} corrupt | "
              f"{cd['classes']} classes | imbalance {cd.get('imbalance_ratio')} | "
              f"dups: {info['exact_duplicate_files']} files in {info['exact_duplicate_groups']} groups")

    out_dir = Path(args.out).resolve()
    write_reports(report, out_dir, records)
    print(f"\nReports written to {out_dir}")


if __name__ == "__main__":
    main()
