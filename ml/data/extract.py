"""Extract the local datasets from archive zips into data/raw.

Only the subsets needed for training are pulled out (we never explode the
6 GB of archives on disk). After installation of this disk, the original
archives become the permanent source of truth referenced by manifests.

Mapping (zip entry prefix -> data/raw sub-folder):
  - plantvillage dataset/color/      -> data/raw/plantvillage/color
  - PlantDoc-Dataset/{train,test}/   -> data/raw/plantdoc/{train,test}
  - Nutrition_dataset/{train,test}/  -> data/raw/nutrition/{train,test}
"""
from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

from ml.config import RAW_DIR, ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")

ARCHIVES_SOURCE = ROOT_DIR.parent  # the PLANTGUARD AI folder that holds the .zip files
PLANTGUARD_AI_DIR = Path(ARCHIVES_SOURCE)

# zip-name -> (prefix filter, target relative dir)
PLAN = [
    ("archive (2).zip",            "plantvillage dataset/color", "plantvillage/color"),
    ("archive (1).zip",            "PlantDoc-Dataset",          "plantdoc"),
    ("archive.zip",                "Nutrition_dataset",         "nutrition"),
]

ZIP_NAME_TO_PREFIX = {name: prefix for name, prefix, _ in PLAN}


def _entry_ok(entry_name: str, prefix: str) -> bool:
    norm = entry_name.replace("\\", "/")
    if not norm.startswith(prefix):
        return False
    rest = norm[len(prefix):].strip("/")
    if not rest:
        return False  # directory entry
    return "/" in rest  # must be <class>/<file>


def extract_archive(zip_path: Path, prefix: str, target: Path) -> tuple[int, int]:
    target.mkdir(parents=True, exist_ok=True)
    extracted = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or not _entry_ok(name, prefix):
                skipped += 1
                continue
            rel = name[len(prefix):].strip("/")
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as out:
                out.write(src.read())
            extracted += 1
            if extracted % 5000 == 0:
                log.info("  ... %s: %d files", target.name, extracted)
    return extracted, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="*", help="specific zip basenames (default: all)")
    args = ap.parse_args()

    if not PLANTGUARD_AI_DIR.exists():
        log.error("Archive source dir not found: %s", PLANTGUARD_AI_DIR)
        return 1

    for zip_name, prefix, target_rel in PLAN:
        if args.zips and zip_name not in args.zips:
            continue
        zip_path = PLANTGUARD_AI_DIR / zip_name
        if not zip_path.exists():
            log.warning("Missing archive, skipping: %s", zip_path)
            continue
        target = RAW_DIR / target_rel
        log.info("Extracting %s -> %s", zip_name, target)
        try:
            extracted, skipped = extract_archive(zip_path, prefix, target)
            log.info("  done: %d files, %d ignored", extracted, skipped)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed extracting %s: %s", zip_name, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())