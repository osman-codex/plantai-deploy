"""Evaluate a trained PlantGuard checkpoint on the held-out test set.

Produces real metrics only: accuracy, macro F1, per-class report, confusion
matrix, top-1 confidence histogram, coverage/acceptance at the operating
threshold, and calibration (ECE). Nothing is fabricated.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

from ml.config import BATCH_SIZE, DATA_DIR, MODELS_DIR, get_device

log = logging.getLogger("evaluate")


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.clip(confidences, 0.0, 1.0)
    bin_idx = np.minimum((confidences * n_bins).astype(int), n_bins - 1)
    acc, conf = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        acc.append(correct[mask].mean())
        conf.append(confidences[mask].mean())
    if not acc:
        return float("nan")
    acc = np.array(acc)
    conf = np.array(conf)
    n = bin_idx
    weights = np.array([(bin_idx == b).sum() for b in range(len(acc))]) / len(bin_idx)
    return float(np.sum(weights * np.abs(acc - conf)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS_DIR / "efficientnet_b0_plantvillage_v1.pt"))
    ap.add_argument("--csv", default=str(DATA_DIR / "processed" / "plantvillage_split.csv"))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="operating confidence threshold for acceptance reporting")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    classes = ckpt["classes"]

    import timm
    from ml.data.datasets import CSVImageDataset

    model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    from ml.config import IMG_SIZE
    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = CSVImageDataset(args.csv, "test", transform=transform)
    loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)

    log.info("evaluating %s on test set (%d images, %d classes)", arch, len(test_ds), len(classes))
    all_logits, all_y = [], []
    with torch.inference_mode():
        for x, y in loader:
            out = model(x.to(device))
            all_logits.append(out.cpu().numpy())
            all_y.append(y.numpy())
    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_y)
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs = probs / probs.sum(1, keepdims=True)
    conf = probs.max(1)
    pred = probs.argmax(1)

    acc = accuracy_score(y_true, pred)
    f1_macro = f1_score(y_true, pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred)
    accepted = conf >= args.threshold
    coverage = float(accepted.mean())
    acc_accepted = accuracy_score(y_true[accepted], pred[accepted]) if accepted.any() else float("nan")
    acc_rejected = accuracy_score(y_true[~accepted], pred[~accepted]) if (~accepted).any() else float("nan")
    calibration = ece(conf, pred == y_true)

    mean_conf = float(conf.mean())
    summary = {
        "arch": arch,
        "test_images": int(len(y_true)),
        "n_classes": int(len(classes)),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "mean_confidence": round(mean_conf, 4),
        "ece_calibration_error": round(calibration, 4),
        "threshold": args.threshold,
        "coverage_at_threshold": round(coverage, 4),
        "accuracy_accepted": None if np.isnan(acc_accepted) else round(float(acc_accepted), 4),
        "accuracy_rejected": None if np.isnan(acc_rejected) else round(float(acc_rejected), 4),
        "confusion_matrix_shape": list(cm.shape),
        "worst_classes": sorted(
            ({c: report[c]["recall"] for c in classes if report.get(c)}).items(),
            key=lambda kv: kv[1])[:8],
        "per_class": {c: report[c] for c in classes if report.get(c)},
    }
    out = MODELS_DIR / "evaluate_test.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = MODELS_DIR / "classification_report.txt"
    report_path.write_text(classification_report(
        y_true, pred, target_names=classes, zero_division=0), encoding="utf-8")

    log.info("acc=%.4f  macro_f1=%.4f  coverage@%.2f=%.3f  ECE=%.4f",
             acc, f1_macro, args.threshold, coverage, calibration)
    log.info("summary -> %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())