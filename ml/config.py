"""Core configuration and path handling for PlantGuard AI.

All user-configurable values read from environment (a `.env` file at the
project root is auto-loaded). No hard-coded IPs/secrets anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover - dotenv optional
    pass

# --- Project root -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # plantguard/

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))
# The dataset lives OUTSIDE the OneDrive-synced project tree on this machine.
# A junction into a synced folder makes OneDrive Files-On-Demand dehydrate the
# images (they were deleted once already); the external root is stable storage.
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "C:\\Users\\iddri\\plantguard_data")
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", DATA_DIR / "processed"))
QUARANTINE_DIR = Path(os.getenv("QUARANTINE_DIR", DATA_DIR / "quarantine"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", DATA_DIR / "reports"))

exported_model_dir = MODELS_DIR / "exported"
for _d in (MODELS_DIR, RAW_DIR, PROCESSED_DIR, QUARANTINE_DIR, REPORTS_DIR, exported_model_dir):
    _d.mkdir(parents=True, exist_ok=True)

# --- ML settings ------------------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
DEVICE = os.getenv("DEVICE", "auto")  # auto | cpu | cuda

MODEL_ARCH = os.getenv("MODEL_ARCH", "efficientnet_b0")
MODEL_VERSION = os.getenv("MODEL_VERSION", "plantguard_disease_v1")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
TOP_K = int(os.getenv("TOP_K", "3"))

# --- Databases --------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{ROOT_DIR / 'plantguard.db'}",
)

def get_device() -> str:
    if DEVICE == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return DEVICE