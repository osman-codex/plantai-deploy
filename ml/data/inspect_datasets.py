"""Dataset inspection and report generation.

Scans extracted raw datasets, detects structure / classes / corruption /
exact duplicates / near duplicates (sampled estimate), measures class
imbalance and image statistics, and writes:

  data/reports/dataset_report.json    - full machine-readable report
  data/reports/dataset_classes.csv    - per-class counts across datasets
  data/reports/dataset_report.html    - self-contained human-readable report
  data/reports/class_distribution.png - bar charts (all datasets)

Nothing here invents numbers: every figure is derived from the actual files
on disk in `data/raw`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

from ml.config import RAW_DIR, REPORTS_DIR

log = logging.getLogger("inspect")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# PlantVillage encodes plant + disease as  plant___disease
HEALTHY_TOKENS = {"healthy"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _image_info(data: bytes) -> dict:
    """Return format/width/height/mode + integrity from a byte buffer."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # forces full decode -> catches truncated/corrupt
        fmt = (img.format or "").lower()
        return {
            "valid": True,
            "format": fmt,
            "width": img.width,
            "height": img.height,
            "mode": img.mode or "",
            "corrupt": False,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "format": None, "width": None, "height": None,
                "mode": None, "corrupt": True, "error": str(exc)[:200]}


def _ahash16(data: bytes) -> int:
    """16x16 grayscale average hash -> 256-bit int. Fast pixel-decode hash."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            g = im.convert("L").resize((16, 16), Image.BILINEAR)
            px = list(g.get_flattened_data())
            avg = sum(px) / len(px)
            bits = 0
            for v in px:
                bits = (bits << 1) | (1 if v >= avg else 0)
            return bits
    except Exception:  # noqa: BLE001 - corrupt images count as their own hash
        return 0


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


SPLIT_FOLDERS = {"train", "test", "color", "grayscale", "segmented"}


def _class_from_rel(rel: Path) -> str:
    """First path component that is a real class (skip split/colour folders)."""
    for part in rel.parts:
        if part.lower() not in SPLIT_FOLDERS:
            return part
    return rel.parts[0]


def _tasks_supported(dataset: str, classes: list[str]) -> dict:
    """Honest support matrix for the tasks the spec asks for."""
    healthy = sum(1 for c in classes if any(t in c.lower() for t in HEALTHY_TOKENS))
    has_disease = len(classes) - healthy
    has_deficiency = dataset == "nutrition"
    return {
        "plant_identification": dataset in ("plantvillage", "plantdoc"),
        "health_classification": healthy > 0 and has_disease > 0,
        "disease_classification": has_disease > 0,
        "deficiency_classification": has_deficiency,  # label semantics reviewed separately
        "severity_estimation": False,  # no severity annotations exist here
        "notes": [
            "No severity annotations exist in any dataset; severity cannot be trained.",
            "Deficiency labels use availability wording ('KAB','NAB',...); semantics require inspection.",
        ],
    }


# ---------------------------------------------------------------------------
# per-dataset scan
# ---------------------------------------------------------------------------
def scan_dataset(ds_dir: Path, sample_hash: int = 200) -> dict:
    files = sorted(p for p in ds_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)
    log.info("Scanning %s: %d candidate images", ds_dir.name, len(files))

    n_valid = n_corrupt = n_unsupported = 0
    formats: Counter = Counter()
    class_counts: Counter = Counter()
    md5_to_path: dict[str, list[str]] = defaultdict(list)
    widths: list[int] = []
    heights: list[int] = []
    errors: Counter = Counter()
    ext_count: Counter = Counter()

    # sample for perceptual hash estimate
    rng = random.Random(2026)
    sample = rng.sample(files, min(sample_hash, len(files)))
    hash_tracker: dict[int, list[tuple[str, str]]] = defaultdict(list)
    near_dup_pairs_sample = 0

    for f in files:
        ext_count[f.suffix.lower()] += 1
        cls = _class_from_rel(f.relative_to(ds_dir))
        class_counts[cls] += 1
        try:
            data = f.read_bytes()
        except OSError as exc:
            n_corrupt += 1
            errors[f"read:{type(exc).__name__}"] += 1
            continue

        # exact duplicate detection (global)
        d = _md5(data)
        md5_to_path[d].append(str(f.relative_to(ds_dir)))

        info = _image_info(data)
        if not info["valid"]:
            n_corrupt += 1
            errors["decode:" + (info["error"] or "unknown")] += 1
            continue
        n_valid += 1
        if info["format"]:
            formats[info["format"]] += 1
        if info["width"] and info["height"]:
            widths.append(info["width"])
            heights.append(info["height"])
            if info["width"] < 64 or info["height"] < 64:
                errors["tiny_image"] += 1

        if f in sample:
            h = _ahash16(data)
            hash_tracker[h].append((str(f.relative_to(ds_dir)), cls))

    # near-dup estimate from the sample (single class-single hash buckets)
    for h, entries in hash_tracker.items():
        if len(entries) > 1:
            near_dup_pairs_sample += len(entries) - 1

    duplicate_sets = [v for v in md5_to_path.values() if len(v) > 1]
    n_dup_files = sum(len(v) - 1 for v in duplicate_sets)

    sizes = [f.stat().st_size for f in files]
    total_mb = sum(sizes) / (1024 * 1024)

    dist = {k: int(v) for k, v in sorted(class_counts.items())}
    if dist:
        max_c = max(dist.values())
        min_c = min(dist.values())
        imbalance_ratio = max_c / max(min_c, 1)
    else:
        imbalance_ratio = 1.0

    return {
        "dataset": ds_dir.name,
        "root": str(ds_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files),
        "valid_images": n_valid,
        "corrupt": n_corrupt,
        "unsupported_format": n_unsupported,
        "format_counts": dict(formats),
        "extension_counts": dict(ext_count),
        "byte_count_sampled": {
            "n": max(len(sizes), 0),
            "total_mb": round(total_mb, 1),
            "mean_kb": round(sum(sizes) / max(len(sizes), 1) / 1024, 1),
        },
        "image_sizes_sampled": {
            "n": len(widths),
            "min_w_h": [min(widths), min(heights)] if widths else None,
            "max_w_h": [max(widths), max(heights)] if widths else None,
        },
        "classes": dist,
        "n_classes": len(dist),
        "class_imbalance_ratio": round(imbalance_ratio, 2),
        "exact_duplicate_groups": len(duplicate_sets),
        "exact_duplicate_files": n_dup_files,
        "near_duplicate_sample": {
            "sampled": len(sample),
            "near_dup_files_estimated": near_dup_pairs_sample,
            "note": "estimated from a uniform sample; full detection runs in preprocessing",
        },
        "decode_errors": dict(errors),
        "tasks_supported": _tasks_supported(ds_dir.name, list(dist.keys())),
    }


# ---------------------------------------------------------------------------
# charts / report writers
# ---------------------------------------------------------------------------
def _plot_class_distribution(results: list[dict], out: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * max(len(results), 1), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        dist = res["classes"]
        labels = list(dist.keys())
        vals = list(dist.values())
        ax.bar(range(len(labels)), vals, color="#2d5016")
        ax.set_title(f'{res["dataset"]} ({res["n_classes"]} classes)')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("images")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def write_csv(all_rows: list[dict], out: Path):
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "class", "count"])
        for r in all_rows:
            for cls, cnt in r["classes"].items():
                w.writerow([r["dataset"], cls, cnt])


def write_html(results: list[dict], chart_path: Path, out: Path):
    def _esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    blocks = []
    for res in results:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{v}</td></tr>"
            for k, v in sorted(res["classes"].items())
        )
        support = "".join(
            f"<tr><td>{_esc(k)}</td><td>{'yes' if v else 'no'}</td></tr>"
            for k, v in res["tasks_supported"].items()
        )
        notes = "".join(f"<li>{_esc(n)}</li>" for n in res["tasks_supported"]["notes"])
        blocks.append(
            f"""
            <section>
              <h2>{_esc(res['dataset'])}</h2>
              <table class="kv">
                <tr><td>Images (valid)</td><td>{res['valid_images']} / {res['total_files']}</td></tr>
                <tr><td>Classes</td><td>{res['n_classes']}</td></tr>
                <tr><td>Formats</td><td>{_esc(res['format_counts'])}</td></tr>
                <tr><td>Corrupt</td><td>{res['corrupt']}</td></tr>
                <tr><td>Exact duplicates (files)</td><td>{res['exact_duplicate_files']}</td></tr>
                <tr><td>Class imbalance ratio</td><td>{res['class_imbalance_ratio']}</td></tr>
                <tr><td>Total size</td><td>{res['byte_count_sampled']['total_mb']} MB</td></tr>
              </table>
              <h3>Tasks this dataset can support</h3>
              <table class="kv">{support}</table>
              <ul>{notes}</ul>
              <details><summary>Class counts</summary>
                <table>{rows}</table>
              </details>
            </section>"""
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>PlantGuard AI - Dataset Report</title>
<style>
 body {{ font-family: system-ui, Segoe UI, Arial; margin: 2rem auto; max-width: 1100px; color: #1c1c1c; }}
 h1, h2 {{ color: #2d5016; }} table.kv td:first-child {{ font-weight: 600; }}
 table {{ border-collapse: collapse; margin: .5rem 0 1rem; }}
 td, th {{ border: 1px solid #ccc; padding: .25rem .5rem; font-size: .85rem; }}
 img {{ max-width: 100%; }} .warn {{ background:#fdf3d1;border:1px solid #e0c76a;padding:.5rem .8rem;border-radius:6px; }}
</style></head><body>
<h1>PlantGuard AI — Dataset Report</h1>
<p>Generated {datetime.now(timezone.utc).isoformat()} from real files in <code>data/raw</code>.</p>
<p class="warn"><b>Integrity note:</b> no severity annotations exist in any dataset, so a trained
severity model is impossible; deficiency labels use availability wording and require inspection.</p>
<img src="{chart_path.name}" alt="class distribution">
{''.join(blocks)}
</body></html>"""
    out.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect raw datasets and write the dataset report.")
    ap.add_argument("--root", type=Path, default=RAW_DIR)
    ap.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--hash-sample", type=int, default=200)
    args = ap.parse_args()

    ds_dirs = sorted(d for d in args.root.iterdir() if d.is_dir())
    if not ds_dirs:
        log.error("No dataset folders under %s", args.root)
        return 1

    results = [scan_dataset(d, sample_hash=args.hash_sample) for d in ds_dirs]
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # combined json
    combined = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": results,
        "summary": {
            "total_valid_images": sum(r["valid_images"] for r in results),
            "total_corrupt": sum(r["corrupt"] for r in results),
        },
    }
    (args.report_dir / "dataset_report.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8")

    # csv + html + chart
    write_csv(results, args.report_dir / "dataset_classes.csv")
    chart = args.report_dir / "class_distribution.png"
    _plot_class_distribution(results, chart)
    write_html(results, chart, args.report_dir / "dataset_report.html")

    for r in results:
        log.info(
            "%s: valid=%d classes=%d corrupt=%d dup_files=%d skew=%.2fx",
            r["dataset"], r["valid_images"], r["n_classes"], r["corrupt"],
            r["exact_duplicate_files"], r["class_imbalance_ratio"],
        )
    log.info("Reports written to %s", args.report_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())