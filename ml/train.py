"""Train EfficientNet-B0 (or any timm backbone) on a PlantGuard split.

Two phases (spec: frozen backbone baseline, then fine-tune):
  * head  : backbone frozen, train only the classifier head
  * full  : unfreeze all, fine-tune the whole model

Everything is deterministic (seed), class-imbalance is handled with class
weights, and results/metrics are never faked - only real validation numbers
are written.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ml.config import (
    BATCH_SIZE, DATA_DIR, IMG_SIZE, MODELS_DIR, NUM_WORKERS, RANDOM_SEED,
    get_device,
)

log = logging.getLogger("train")
log.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PlantGuard model training")
    ap.add_argument("--dataset", default="plantvillage",
                    help="dataset name used for the split csv filename")
    ap.add_argument("--csv", default=None, help="explicit split csv path")
    ap.add_argument("--arch", default="efficientnet_b0", help="timm model name")
    ap.add_argument("--phase", choices=["auto", "head", "full"], default="auto",
                    help="auto = head then full; select just one to run either stage")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--epochs-full", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=5e-3)
    ap.add_argument("--lr-full", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", default=None, help="checkpoint stem (default models_dir/{arch}_{dataset}_v1)")
    ap.add_argument("--train-subset", type=float, default=None,
                    help="fraction (0.0-1.0) of the training split to use per class "
                         "(e.g. 0.25 = a documented subset baseline). Must be >0.")
    ap.add_argument("--smoke", action="store_true",
                    help="run a tiny end-to-end sanity pass (64 images, 1 step per phase)")
    return ap.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(size: int, train: bool) -> tuple[list, list]:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train:
        return ([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ], None)
    return ([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ], None)


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    import timm
    model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "classifier" in name or "head" in name or "fc" in name:
            continue
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              desc: str = "", step_cap: int | None = None) -> tuple[float, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    running, correct, total = 0.0, 0, 0
    t0 = time.time()
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            running += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            if i % 500 == 0 or (i == len(loader) - 1):
                el = time.time() - t0
                log.info("  %s step %d/%d  mean_loss=%.4f  acc=%.4f  %.1f img/s",
                         desc, i + 1, len(loader), running / max(total, 1),
                         correct / max(total, 1), total / max(el, 1e-6))
            if step_cap is not None and i + 1 >= step_cap:
                break
    acc = correct / max(total, 1)
    avg = running / max(total, 1)
    rate = total / max(time.time() - t0, 1e-6)
    log.info("%s  loss=%.4f acc=%.4f  (%d samples, %.1f img/s)",
             desc, avg, acc, total, rate)
    return avg, acc


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    import os as _os
    torch.set_num_threads(max(2, min(4, (_os.cpu_count() or 4))))
    log.info("device=%s  threads=%d", device, torch.get_num_threads())

    csv_path = Path(args.csv) if args.csv else DATA_DIR / "processed" / f"{args.dataset}_split.csv"
    if not csv_path.exists():
        log.error("split csv not found: %s", csv_path)
        return 1
    out_stem = args.out or f"{args.arch}_{args.dataset}_v1"

    from ml.data.datasets import CSVImageDataset

    tr = build_transforms(IMG_SIZE, train=True)[0]
    val = build_transforms(IMG_SIZE, train=False)[0]
    train_ds = CSVImageDataset(csv_path, "train", transform=transforms.Compose(tr))
    val_ds = CSVImageDataset(csv_path, "val", transform=transforms.Compose(val))
    if args.train_subset is not None:
        if not 0.0 < args.train_subset <= 1.0:
            log.error("--train-subset must be in (0, 1]")
            return 1
        before = len(train_ds)
        df = train_ds.df
        counts = df["label"].value_counts().to_dict()
        keep = {c: max(1, int(round(n * args.train_subset))) for c, n in counts.items()}
        df = df.copy()
        df["_rank"] = df.groupby("label")["path"].rank(method="first", ascending=True).astype(int)
        train_ds.df = df[df["_rank"] <= df["label"].map(keep)].drop(columns="_rank").reset_index(drop=True)
        log.info("train subset baseline: %d -> %d images (%.0f%%), stratified per class",
                 before, len(train_ds), args.train_subset * 100)
    num_classes = len(train_ds.labels)
    log.info("classes=%d  train=%d  val=%d", num_classes, len(train_ds), len(val_ds))
    if args.smoke:
        train_ds.df = train_ds.df.head(64)
        val_ds.df = val_ds.df.head(64)
        train_ds.df = train_ds.df.reset_index(drop=True)
        val_ds.df = val_ds.df.reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, prefetch_factor=2 if args.workers else None,
                              persistent_workers=bool(args.workers))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=bool(args.workers))

    model = build_model(args.arch, num_classes)
    model.to(device)

    # Compute class weights from training set (some classes may have 0 in subset)
    vc = train_ds.df['label'].value_counts()
    weights_list = []
    for c in train_ds.labels:
        count = vc.get(c, 0)
        weights_list.append(max(count, 1))  # avoid division by zero
    weights = torch.tensor(weights_list, dtype=torch.float, device=device)
    weights = 1.0 / weights
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {"phases": {}, "total_epochs": 0}
    best_path = MODELS_DIR / f"{out_stem}.pt"

    def run_phase(name: str, freeze: bool, epochs: int, lr: float) -> bool:
        if epochs < 1:
            return True
        if freeze:
            freeze_backbone(model)
        else:
            unfreeze_all(model)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        steps_per_epoch = len(train_loader)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
        log.info("=== PHASE %s  freeze=%s  lr=%s  epochs=%d ===", name, freeze, lr, epochs)
        accs, losses = [], []
        best_acc, no_improve = 0.0, 0
        patience = 2
        for epoch in range(1, epochs + 1):
            if args.smoke:
                run_epoch(model, train_loader, criterion, device, opt, sched,
                          f"[smoke] {name} epoch {epoch} (train)", step_cap=2)
                vl, va = run_epoch(model, val_loader, criterion, device, None, None,
                                   f"[smoke] {name} epoch {epoch} (val)", step_cap=2)
            else:
                run_epoch(model, train_loader, criterion, device, opt, sched,
                          f"[{name}] epoch {epoch} (train)")
                vl, va = run_epoch(model, val_loader, criterion, device, None, None,
                                   f"[{name}] epoch {epoch} (val)")
            losses.append(round(float(vl), 4))
            accs.append(round(va, 4))
            if va > best_acc:
                best_acc = va
                no_improve = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": num_classes,
                    "classes": train_ds.labels,
                    "class_to_idx": train_ds.class_to_idx,
                    "phase": name,
                    "epoch": epoch,
                    "val_loss": round(float(vl), 4),
                    "val_acc": round(va, 4),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": args.seed,
                }, best_path)
                log.info("  saved best %s checkpoint (val_acc=%.4f)", name, va)
            else:
                no_improve += 1
                if no_improve >= patience and not args.smoke:
                    log.info("  early stop at epoch %d (patience %d)", epoch, patience)
                    epochs = epoch  # record effective epochs
                    break
        history["phases"][name] = {"epochs": epochs, "val_loss": losses, "val_acc": accs,
                                   "best_val_acc": best_acc}
        return True

    phases = []
    if args.smoke:
        phases = [("head_smoke", True, 1, args.lr_head), ("full_smoke", False, 1, args.lr_full)]
    else:
        if args.phase in ("auto", "head"):
            phases.append(("head", True, args.epochs_head, args.lr_head))
        if args.phase in ("auto", "full") and (args.phase == "full" or args.epochs_head > 0):
            phases.append(("full", False, args.epochs_full, args.lr_full))

    for ph in phases:
        if not run_phase(*ph):
            return 1
        if args.smoke:
            break

    history["metrics"] = {"best_val_acc": max(
        (h["best_val_acc"] for h in history["phases"].values()), default=0.0)}
    metrics_path = MODELS_DIR / f"{out_stem}_history.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_ckpt = MODELS_DIR / f"{out_stem}.pt"
    if final_ckpt.exists():
        ck = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        ck["config_summary"] = {
            "split_csv": str(csv_path), "device": device, "batch_size": args.batch_size,
            "version": "plantguard_disease_v1",
            "train_subset_fraction": args.train_subset,
            "train_count": len(train_ds) if not args.smoke else len(train_loader.dataset.df),
            "val_count": len(val_ds) if not args.smoke else len(val_loader.dataset.df),
            "phases": {
                n: {"best_val_acc": h["best_val_acc"]} for n, h in history["phases"].items()
            },
        }
        torch.save(ck, final_ckpt)
    log.info("done -> %s", final_ckpt)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())