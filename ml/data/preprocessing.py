"""Preprocessing: validation, exact/near dedup, leak-safe train/val/test split.

Outputs (no image copies are made - we reference the raw files):
  data/processed/{dataset}_manifest.csv   - every valid image + md5 + ahash
  data/processed/{dataset}_split.csv      - split assignment (train/val/test)
  data/processed/{dataset}_grouped.csv    - near-duplicate cluster assignments
  data/quarantine/{corrupt|duplicates}/   - moved-out invalid / duplicate files

Split guarantees:
  * deterministic (RANDOM_SEED)
  * stratified by class (70/15/15)
  * near-duplicate clusters stay entirely inside one split (leakage guard)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from ml.config import PROCESSED_DIR, QUARANTINE_DIR, RAW_DIR, RANDOM_SEED

log = logging.getLogger("preprocess")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
HASH_SIZE = 16
NEAR_DUP_HAMMING = 4

# datasets: name -> (raw prefix relative to RAW_DIR, label mode)
#   "flat":  <prefix>/<class>/<file>          label = parts[0]
#   "split": <prefix>/{train,test}/<class>/<file>  label = first part after split seed
DATASETS = {
    "plantvillage": ("plantvillage/color", "flat"),
    "plantdoc": ("plantdoc", "split"),
    "nutrition": ("nutrition", "split"),
}

SPLIT_SEED_FOLDERS = {"train", "test", "color", "grayscale", "segmented"}


def _hash_file(path: Path) -> tuple[str, int, int, int, str]:
    """Return (md5, size, ahash16, width, height) for one image file."""
    try:
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            w, h = im.width, im.height
            g = im.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.BILINEAR)
            px = list(g.getdata())
            avg = sum(px) / len(px)
            bits = 0
            for v in px:
                bits = (bits << 1) | (1 if v >= avg else 0)
        return md5, len(data), bits, w, h
    except Exception as exc:  # noqa: BLE001
        return "", -1, 0, -1, f"{type(exc).__name__}:{exc}"


def _quarantine(path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.rename(dest_dir / f"{uuid.uuid4().hex[:10]}_{path.name}")


def _label_for(rel_parts: tuple[str, ...], mode: str) -> str:
    if mode == "flat":
        return rel_parts[0] if rel_parts else "unknown"
    for part in rel_parts:
        if part not in SPLIT_SEED_FOLDERS:
            return part
    return "unknown"


def collect_manifest(root: Path, prefix: str, mode: str, dataset: str,
                     quarantine: bool = True) -> pd.DataFrame:
    source = root / prefix
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS)
    log.info("%s: hashing %d files...", dataset, len(files))

    rows = []
    corrupt = []

    with ProcessPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(_hash_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futs)):
            f = futs[fut]
            md5, size, ahash, w, h = fut.result()
            rel = f.relative_to(source)
            parts = rel.parts
            if md5 == "":
                corrupt.append(f)
                continue
            label = _label_for(parts, mode)
            rows.append({
                "dataset": dataset, "label": label, "path": str(f),
                "md5": md5, "size_bytes": size, "ahash": ahash,
                "width": w, "height": h,
            })
            if (i + 1) % 8000 == 0:
                log.info("  %s: %d/%d", dataset, i + 1, len(files))

    if corrupt:
        log.warning("%s: %d corrupt files -> quarantine", dataset, len(corrupt))
        if quarantine:
            qdir = QUARANTINE_DIR / "corrupt"
            for f in corrupt:
                _quarantine(f, qdir)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # exact duplicates: keep one representative per md5 (same label)
    df = df.sort_values(["label", "path"])
    dup_mask = df.duplicated(subset=["md5"], keep="first")
    dups = df[dup_mask]
    if not dups.empty:
        log.warning("%s: %d exact duplicate files -> quarantine", dataset, len(dups))
        if quarantine:
            qdir = QUARANTINE_DIR / "duplicates"
            for path in dups["path"]:
                _quarantine(Path(path), qdir)
    df = df[~dup_mask].reset_index(drop=True)

    manifest = PROCESSED_DIR / f"{dataset}_manifest.csv"
    df.to_csv(manifest, index=False)
    log.info("%s manifest: %d unique images -> %s", dataset, len(df), manifest)
    return df


def cluster_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Union-find clusters of images that are near-identical (hamming <= N).

    Clustering is only done within the same label + within a size band to keep
    it fast and avoid accidental linkage of genuinely different photos.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), x)
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    g = df.groupby(["label"]).groups
    hashes = df["ahash"].to_numpy()
    sizes = df["size_bytes"].to_numpy()

    for label, idx in g.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        by_hash: dict[int, list[int]] = defaultdict(list)
        for i in idx:
            by_hash[hashes[i] // 256].append(i)  # bucket on top byte -> faster
        for bucket in by_hash.values():
            for a in bucket:
                for b in bucket:
                    if a >= b:
                        continue
                    ha, hb = hashes[a], hashes[b]
                    if (ha ^ hb).bit_count() <= NEAR_DUP_HAMMING and abs(sizes[a] - sizes[b]) <= 5 * 1024:
                        union(a, b)

    cluster_id: dict[int, int] = {}
    out = []
    for i, row in df.iterrows():
        root = find(i)
        if root not in cluster_id:
            cluster_id[root] = len(cluster_id)
        out.append(cluster_id[root])
    df = df.copy()
    df["cluster"] = out
    return df


def split_dataframe(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Stratified 70/15/15 split at CLUSTER level (leakage safe)."""
    g = df.groupby("cluster").agg(
        label=("label", "first"), size=("size_bytes", "size")).reset_index()
    g = g.sample(frac=1.0, random_state=seed)
    g = g.sort_values(["label", "size"]).reset_index(drop=True)

    assign: dict[int, str] = {}
    for label, cls in g.groupby("label", sort=True):
        n = len(cls)
        cls = cls.reset_index()
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        for i, row in cls.iterrows():
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assign[row["cluster"]] = split

    df = df.copy()
    df["split"] = df["cluster"].map(assign)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate, dedup and split datasets.")
    ap.add_argument("datasets", nargs="*", default=list(DATASETS),
                    help="which datasets to run (default: all)")
    ap.add_argument("--no-quarantine", action="store_true", help="report only, don't move files")
    ap.add_argument("--no-cluster", action="store_true", help="skip near-duplicate clustering")
    args = ap.parse_args()

    summary = {}
    for ds in args.datasets:
        if ds not in DATASETS:
            log.error("Unknown dataset %s", ds)
            return 1
        prefix, mode = DATASETS[ds]
        df = collect_manifest(RAW_DIR, prefix, mode, ds, quarantine=not args.no_quarantine)
        if df.empty:
            log.warning("%s: nothing usable", ds)
            continue
        df = add_hash_if_missing(df)
        if not args.no_cluster:
            df = cluster_near_duplicates(df)
        else:
            df["cluster"] = list(range(len(df)))
        df = split_dataframe(df)
        out = PROCESSED_DIR / f"{ds}_split.csv"
        df.to_csv(out, index=False)
        counts = df["split"].value_counts().to_dict()
        summary[ds] = {k: int(counts.get(k, 0)) for k in ("train", "val", "test")}
        log.info("%s split -> %s", ds, summary[ds])
    log.info("Split summary: %s", summary)
    return 0


def add_hash_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    # placeholder kept for parity; hashing is already part of collect_manifest
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())


def add_hash_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    # placeholder kept for parity; hashing is already part of collect_manifest
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())