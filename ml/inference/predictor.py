"""PlantGuard inference: real model predictions with quality checks, confidence
gating and honest Grad-CAM explainability. No numbers are fabricated here -
anything not supported is explicitly marked as unsupported.
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image
import torch
import torch.nn as nn

from ml.config import CONFIDENCE_THRESHOLD, DEVICE, IMG_SIZE, MODELS_DIR, TOP_K, get_device
from ml.kb import kb_for


class PredictionError(Exception):
    """Raised when an image cannot be analyzed at all."""


@dataclass
class Candidate:
    class_name: str
    display_name: str
    confidence: float

    def to_dict(self) -> dict:
        info = kb_for(self.class_name)
        return {
            "class": self.class_name,
            "display_name": self.display_name,
            "common_name": info["common_name"],
            "confidence": round(self.confidence, 6),
        }


@dataclass
class PredictionResult:
    accepted: bool
    reason: Optional[str]
    top_k: list[Candidate]
    grad_cam: Optional[dict]          # {"url"/"overlay": base64 png ...}
    quality_warnings: list[str]
    latency_ms: float
    model_version: str
    auth: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        primary = self.top_k[0] if self.top_k else None
        diagnosis = {
            "disease": primary.display_name if primary else None,
            "confidence": primary.confidence if primary else None,
            "matched_class": primary.class_name if primary else None,
            "notes": self.meta.get("notes", []),
        }
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "diagnosis": diagnosis,
            "matches": [c.to_dict() for c in self.top_k],
            "confidence": primary.confidence if primary else None,
            "severe_status": self.meta.get("severe_status"),
            "deficiency": self.meta.get("deficiency"),
            "severity": self.meta.get("severity"),
            "confidence_threshold": self.meta.get("confidence_threshold"),
            "disease_health_status": self.meta.get("disease_health_status"),
            "treatment_recommendations": self.meta.get("treatment_recommendations", []),
            "explainability": {"grad_cam": self.grad_cam},
            "quality_warnings": self.quality_warnings,
            "latency_ms": round(self.latency_ms, 2),
            "model_version": self.model_version,
        }


def _display_name(class_name: str) -> str:
    plant, _, disease = class_name.partition("___")
    plant = plant.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    return f"{plant} - {disease}" if disease else plant


class PlantGuardPredictor:
    def __init__(self, ckpt: Optional[Path] = None, threshold: Optional[float] = None):
        self.device = get_device()
        self.threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold
        self.top_k = TOP_K
        ckpt = ckpt or Path(MODELS_DIR) / "efficientnet_b0_plantvillage_v1.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"No trained checkpoint at {ckpt}")
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model_version = data.get("config_summary", {}).get("version", "plantguard_disease_v1")
        import timm
        self.model = timm.create_model(
            data["arch"], pretrained=False, num_classes=data["num_classes"])
        self.model.load_state_dict(data["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.classes: list[str] = data["classes"]
        self.class_to_idx: dict[str, int] = data["class_to_idx"]
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.14)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._cam_layer: Optional[str] = self._find_last_conv()

    def _find_last_conv(self) -> Optional[str]:
        last = None
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                last = name
        return last

    def predict_from_bytes(self, image_bytes: bytes, filename: str = "upload") -> PredictionResult:
        t0 = time.time()
        try:
            pil = PIL.Image.open(io.BytesIO(image_bytes))
            pil.load()
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Could not decode image ({exc}).") from exc
        warnings = self._quality_check(pil)

        rgb = pil.convert("RGB")
        if min(rgb.size) < 32:
            raise PredictionError("Image is too small to analyze.")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        topv, topi = probs.topk(self.top_k)
        topv, topi = topv.tolist(), topi.tolist()

        candidates = [Candidate(self.classes[i], _display_name(self.classes[i]), float(v))
                      for v, i in zip(topv, topi)]
        primary = candidates[0]
        accepted = primary.confidence >= self.threshold
        reason = None if accepted else (
            f"Confidence {primary.confidence:.2%} is below the {self.threshold:.0%} threshold; "
            "the model cannot make a reliable prediction for this image.")

        cam = None
        if accepted:
            try:
                cam = self._grad_cam(pil, logits, topi[0])
            except Exception:  # pragma: no cover - best-effort explainability
                cam = None

        meta = self._build_meta(primary, accepted)
        latency = (time.time() - t0) * 1000.0
        return PredictionResult(
            accepted=accepted, reason=reason, top_k=candidates, grad_cam=cam,
            quality_warnings=warnings, latency_ms=latency, model_version=self.model_version,
            meta=meta)

    # --- helpers -------------------------------------------------------------
    def _quality_check(self, pil) -> list[str]:
        warnings = []
        rgb = pil.convert("RGB")
        w, h = rgb.size
        if w == 0 or h == 0 or (w / max(h, 1)) > 4 or (h / max(w, 1)) > 4:
            warnings.append("Image has an unusual aspect ratio; results may be less reliable.")
        small = rgb.resize((64, 64))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        mean = float(arr.mean())
        if mean < 0.05:
            warnings.append("Image appears almost entirely black.")
        elif mean > 0.95:
            warnings.append("Image appears almost entirely white.")
        std = float(arr.std())
        if std < 0.03:
            warnings.append("Image appears nearly uniform (low detail).")
        return warnings

    def _build_meta(self, primary: Candidate, accepted: bool) -> dict:
        info = kb_for(primary.class_name)
        healthy = "healthy" in primary.class_name
        return {
            "severe_status": {
                "severe": None,
                "certainty": None,
                "reason": "Severity estimation is not supported: no severity labels exist in the training data.",
            },
            "severity": {
                "rating": None,
                "certainty": None,
                "reason": "Severity estimation is not supported: no severity labels exist in the training data.",
            },
            "deficiency": {
                "configuration": None,
                "certainty": None,
                "reason": "Deficiency is only reported by the dedicated nutrient model; a deficiency diagnosis requires a different trained model.",
            },
            "confidence_threshold": {"value": self.threshold, "met": accepted},
            "disease_health_status": "healthy" if healthy else ("sick" if accepted else "unknown"),
            "treatment_recommendations": [{
                "disease": info["common_name"],
                "description": info["description"],
                "guidance": info["treatment"],
                "scope": "Static reference guidance; verify with a specialist before acting.",
            }] if not healthy else [],
            "notes": [
                "Model predictions are probabilities from a real trained classifier.",
                "Grad-CAM shows the regions the model focused on; it is an explanation aid, not a segmentation.",
            ],
        }

    # --- Grad-CAM ------------------------------------------------------------
    def _grad_cam(self, pil, logits, pred_idx: int) -> dict:
        import torchvision.transforms.functional as F

        self.model.zero_grad(set_to_none=True)
        acts: dict[str, torch.Tensor] = {}
        grads: dict[str, torch.Tensor] = {}

        def fwd_hook(m, _in, out):
            acts[m._pg_cam_name] = out.detach()

        def bwd_hook(m, _in, out):
            grads[m._pg_cam_name] = out[0].detach()

        target = dict(self.model.named_modules())[self._cam_layer]
        target._pg_cam_name = self._cam_layer
        hf = target.register_forward_hook(fwd_hook)
        hb = target.register_full_backward_hook(bwd_hook)
        try:
            tensor = self.transform(pil.convert("RGB")).unsqueeze(0).to(self.device)
            tensor.requires_grad_(True)
            out = self.model(tensor)
            score = out[0, pred_idx]
            score.backward()
            A = acts[self._cam_layer]          # (1, C, H, W)
            G = grads[self._cam_layer]         # (1, C, H, W)
            alpha = G.mean(dim=(2, 3), keepdim=True)          # weights (1,C,1,1)
            cam = torch.relu((alpha * A).sum(dim=1, keepdim=True))[0, 0]  # (H,W)
            cam = F.resize(cam.unsqueeze(0).unsqueeze(0), (pil.height, pil.width),
                           antialias=True)[0][0]
            cam = cam - cam.min()
            if float(cam.max()) > 0:
                cam = cam / cam.max()
            heat = cam.cpu().numpy()           # 0..1 float mask
        finally:
            hf.remove()
            hb.remove()

        overlay = self._overlay(pil.convert("RGB"), heat)
        thresh = float((heat > 0.4).mean())
        return {
            "available": True,
            "layers": {"grad_cam_source": self._cam_layer},
            "high_attention_fraction_0_4": round(thresh, 4),
            "overlay": "data:image/png;base64," + self._png_b64(overlay),
        }

    @staticmethod
    def _png_b64(img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _overlay(pil_rgb, heat: np.ndarray) -> PIL.Image.Image:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        heat_rgb = (cm.jet(heat)[:, :, :3] * 255).astype(np.uint8)
        base = np.asarray(pil_rgb).copy()
        mask = heat > 0.25
        if mask.any():
            blend = 0.45
            base[mask] = (base[mask] * (1 - blend) + heat_rgb[mask] * blend).astype(np.uint8)
        return PIL.Image.fromarray(base)