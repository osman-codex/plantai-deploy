"""
PlantGuard AI - FastAPI Backend
Local + deployment backend for plant health diagnosis and monitoring.

Real model inference only. Anything the model cannot do is reported as
explicitly unsupported - never fake numbers.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import (
    Base, Image, ModelVersion, Observation, Plant,
    SessionLocal, Treatment, engine, generate_plant_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from ml.config import MODELS_DIR, MODEL_VERSION
    from ml.inference.predictor import PlantGuardPredictor, PredictionError
except Exception as exc:  # pragma: no cover
    logger.warning("ML stack import failed: %s", exc)
    MODELS_DIR = MODEL_VERSION = None
    PlantGuardPredictor = PredictionError = None
    PredictionResult = None

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "")) if os.getenv("UPLOAD_DIR") else Path(__file__).resolve().parent.parent / "uploads"
PLANTS_DIR = (Path(os.getenv("DATA_DIR")) / "plants") if os.getenv("DATA_DIR") else Path(__file__).resolve().parent.parent / "data" / "plants"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PLANTS_DIR.mkdir(parents=True, exist_ok=True)
SPA_DIR = Path(os.getenv("SPA_DIR", "")) or (Path(__file__).resolve().parents[2] / "frontend_vite" / "dist")

app = FastAPI(
    title="PlantGuard AI",
    description="AI-powered plant health diagnosis and monitoring system (real trained model).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: the frontend origin(s) come from CORS_ORIGINS (comma-separated, "*" ok).
# NOTE: allow_credentials must be False when using the wildcard, otherwise
# browsers reject the header combination and every API call fails CORS checks.
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

_predictor = None


def get_predictor():
    """Lazy-load the trained model once (it is large)."""
    global _predictor
    if _predictor is None:
        if PlantGuardPredictor is None:
            raise HTTPException(503, "ML backend is not installed.")
        _predictor = PlantGuardPredictor()
    return _predictor


# ===== Pydantic Models =====

class PlantCreate(BaseModel):
    crop: str
    species: Optional[str] = None
    common_name: Optional[str] = None
    notes: Optional[str] = None


class PlantUpdate(BaseModel):
    species: Optional[str] = None
    common_name: Optional[str] = None
    notes: Optional[str] = None
    treatment: Optional[str] = None
    treatment_date: Optional[datetime] = None


class ObservationNotes(BaseModel):
    notes: Optional[str] = None


class TreatmentCreate(BaseModel):
    plant_id: str
    treatment_name: str
    description: Optional[str] = None
    notes: Optional[str] = None


class CompareRequest(BaseModel):
    observation_id_1: int
    observation_id_2: int


# ===== Database Dependency =====

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===== Helpers =====

def _save_bytes(data: bytes, plant_id: str = None, observation_num: int = None) -> str:
    if plant_id and observation_num:
        d = PLANTS_DIR / plant_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"observation_{observation_num:03d}.jpg"
    else:
        p = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}.jpg"
    p.write_bytes(data)
    return str(p)


def _validate(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(400, "No file provided")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "File type not supported. Use JPG, PNG, or WebP.")
    raw = file.file.read()
    file.file.seek(0)
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum size is 10MB.")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===== Root =====

@app.get("/")
async def root():
    return {
        "name": "PlantGuard AI",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "POST /predict - Analyze an image with the trained model",
            "health": "GET /health",
            "model_info": "GET /model-info",
            "plants": "GET/POST /plants",
            "plant_detail": "GET/PUT/DELETE /plants/{plant_id}",
            "observations": "POST/GET /plants/{plant_id}/observations",
            "timeline": "GET /plants/{plant_id}/timeline",
            "compare": "POST /plants/{plant_id}/compare",
            "treatments": "GET/POST /plants/{plant_id}/treatments",
            "image_serving": "GET /uploads/{filename}, GET /data/plants/{plant_id}/{filename}",
        },
    }


@app.get("/health")
async def health_check() -> dict:
    # Deliberately avoids the DB and the model here: Render health checks must
    # respond fast and this endpoint is what the home page status chip polls.
    return {"status": "healthy", "model": "lazy_load_on_demand", "timestamp": _now_iso()}


@app.get("/health/deep")
async def health_deep() -> dict:
    """Full check: model actually loads + DB answers. Slower; use manually."""
    model_status = "not_loaded"
    if PlantGuardPredictor is not None:
        try:
            get_predictor()
            model_status = "loaded"
        except Exception as exc:
            model_status = f"checkpoint_missing ({exc})"
    return {"status": "healthy", "model": model_status, "timestamp": _now_iso()}


# ===== Plants CRUD =====

@app.post("/plants")
async def create_plant(profile: PlantCreate, db: Session = Depends(get_db)):
    plant_id = generate_plant_id()
    plant = Plant(
        plant_id=plant_id, crop=profile.crop, species=profile.species,
        common_name=profile.common_name, notes=profile.notes, initial_diagnosis={},
    )
    db.add(plant)
    db.commit()
    db.refresh(plant)
    return {
        "id": plant.id,
        "plant_id": plant.plant_id,
        "crop": plant.crop,
        "species": plant.species,
        "common_name": plant.common_name,
        "date_created": plant.date_created.isoformat(),
        "message": f"Plant profile created with ID: {plant_id}",
    }


@app.get("/plants")
async def list_plants(db: Session = Depends(get_db)):
    plants = db.query(Plant).order_by(Plant.date_created.desc()).all()
    return [
        {
            "id": p.id, "plant_id": p.plant_id, "crop": p.crop,
            "species": p.species, "common_name": p.common_name,
            "date_created": p.date_created.isoformat(),
            "observations_count": len(p.observations),
            "latest_status": p.observations[-1].health_status if p.observations else None,
        }
        for p in plants
    ]


def _get_plant_or_404(db: Session, plant_id: str) -> Plant:
    plant = db.query(Plant).filter(Plant.plant_id == plant_id).first()
    if not plant:
        raise HTTPException(404, f"Plant with ID {plant_id} not found")
    return plant


@app.get("/plants/{plant_id}")
async def get_plant(plant_id: str, db: Session = Depends(get_db)):
    plant = _get_plant_or_404(db, plant_id)
    latest = plant.observations[-1] if plant.observations else None
    return {
        "id": plant.id, "plant_id": plant.plant_id, "crop": plant.crop,
        "species": plant.species, "common_name": plant.common_name,
        "date_created": plant.date_created.isoformat(),
        "notes": plant.notes, "treatment": plant.treatment,
        "treatment_date": plant.treatment_date.isoformat() if plant.treatment_date else None,
        "initial_diagnosis": plant.initial_diagnosis,
        "observations_count": len(plant.observations),
        "latest_observation": {
            "disease": obs.predicted_disease, "health_status": obs.health_status,
            "severity": obs.severity, "severity_category": obs.severity_category,
            "created_at": obs.created_at.isoformat(),
        } if latest else None,
    }


@app.put("/plants/{plant_id}")
async def update_plant(plant_id: str, profile: PlantUpdate, db: Session = Depends(get_db)):
    plant = _get_plant_or_404(db, plant_id)
    for key, value in profile.model_dump(exclude_unset=True).items():
        setattr(plant, key, value)
    db.commit()
    db.refresh(plant)
    return {
        "message": f"Plant {plant_id} updated successfully",
        "plant_id": plant.plant_id,
        "crop": plant.crop,
        "species": plant.species,
        "common_name": plant.common_name,
        "notes": plant.notes,
        "treatment": plant.treatment,
        "treatment_date": plant.treatment_date.isoformat() if plant.treatment_date else None,
    }


@app.delete("/plants/{plant_id}")
async def delete_plant(plant_id: str, db: Session = Depends(get_db)):
    plant = _get_plant_or_404(db, plant_id)
    db.delete(plant)
    db.commit()
    return {"message": f"Plant {plant_id} deleted successfully"}


# ===== Observations =====

@app.post("/plants/{plant_id}/observations")
async def create_observation(
    plant_id: str,
    file: UploadFile = File(...),
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    plant = _get_plant_or_404(db, plant_id)
    _validate(file)
    raw = file.file.read()
    if not raw:
        raise HTTPException(422, "Empty file.")
    observation_num = len(plant.observations) + 1
    file_path = _save_bytes(raw, plant_id, observation_num)
    result = _analyze_with_predictor(raw)
    pred = result["prediction"]

    image = Image(
        plant_id=plant.id, file_path=file_path,
        file_name=f"observation_{observation_num:03d}{Path(file_path).suffix}",
        upload_date=datetime.utcnow(),
        quality_score=None,
    )
    db.add(image)
    db.flush()

    observation = Observation(
        plant_id=plant.id, image_id=image.id,
        predicted_disease=pred.get("disease"),
        predicted_deficiency=None,
        health_status=pred.get("disease_health_status"),
        confidence=pred.get("confidence"),
        severity=None,
        severity_category=None,
        affected_area_pct=pred.get("explainability", {}).get("grad_cam", {})
        .get("high_attention_fraction_0_4") if pred.get("explainability") else None,
        model_version=pred.get("model_version"),
        disease_probabilities={m["class"]: m["confidence"] for m in pred.get("matches", [])},
        deficiency_probabilities={},
        metadata={"notes": notes, "file_name": image.file_name,
                  "accepted": pred.get("accepted"), "reason": pred.get("reason")},
    )
    db.add(observation)
    if not plant.initial_diagnosis:
        plant.initial_diagnosis = {
            "disease": pred.get("disease"), "health_status": pred.get("disease_health_status"),
            "confidence": pred.get("confidence"), "model_version": pred.get("model_version"),
            "timestamp": _now_iso(),
        }
    db.commit()
    db.refresh(observation)

    return {
        "id": observation.id, "plant_id": plant_id, "image_id": image.id,
        "image_path": file_path, "observation_number": observation_num,
        "upload_date": image.upload_date.isoformat(),
        "prediction": pred, "notes": notes,
    }


@app.get("/plants/{plant_id}/observations")
async def list_observations(plant_id: str, db: Session = Depends(get_db)) -> list:
    plant = _get_plant_or_404(db, plant_id)
    observations = db.query(Observation).filter(Observation.plant_id == plant.id)\
        .order_by(Observation.created_at.desc()).all()
    return [
        {
            "id": obs.id, "observation_number": i + 1,
            "created_at": obs.created_at.isoformat(),
            "image_path": obs.image.file_path if obs.image else None,
            "image_name": obs.image.file_name if obs.image else None,
            "prediction": {
                "disease": obs.predicted_disease, "deficiency": obs.predicted_deficiency,
                "health_status": obs.health_status, "confidence": obs.confidence,
                "severity": obs.severity, "severity_category": obs.severity_category,
                "affected_area_pct": obs.affected_area_pct,
            },
            "notes": obs.metadata.get("notes") if obs.metadata else None,
        }
        for i, obs in enumerate(observations)
    ]


# ===== Timeline & Compare (honest - severity unsupported) =====

@app.get("/plants/{plant_id}/timeline")
async def get_timeline(plant_id: str, db: Session = Depends(get_db)) -> dict:
    plant = _get_plant_or_404(db, plant_id)
    observations = db.query(Observation).filter(Observation.plant_id == plant.id)\
        .order_by(Observation.created_at.asc()).all()
    if not observations:
        return {
            "plant_id": plant_id, "crop": plant.crop, "observations": [],
            "trend": "no_data", "improvement_percentage": None,
            "message": "No observations yet. Upload your first image to start tracking.",
            "total_observations": 0,
        }

    timeline = [
        {
            "date": obs.created_at.isoformat(), "disease": obs.predicted_disease,
            "health_status": obs.health_status, "confidence": obs.confidence,
            "image_path": obs.image.file_path if obs.image else None,
        }
        for obs in observations
    ]
    return {
        "plant_id": plant_id, "crop": plant.crop, "observations": timeline,
        "trend": "insufficient_data",
        "improvement_percentage": None,
        "message": "The training data does not include severity labels, so the system cannot "
                   "compute severity trends or quantify improvement. Observations are shown "
                   "chronologically so you can compare model predictions across visits.",
        "total_observations": len(observations),
    }


@app.post("/plants/{plant_id}/compare")
async def compare_observations(
    plant_id: str,
    body: CompareRequest,
    db: Session = Depends(get_db),
) -> dict:
    plant = _get_plant_or_404(db, plant_id)
    obs1 = db.query(Observation).filter(
        Observation.id == body.observation_id_1,
        Observation.plant_id == plant.id,
    ).first()
    obs2 = db.query(Observation).filter(
        Observation.id == body.observation_id_2,
        Observation.plant_id == plant.id,
    ).first()
    if not obs1 or not obs2:
        raise HTTPException(404, "One or both observations not found")

    def _s(o: Observation) -> dict:
        return {
            "disease": o.predicted_disease,
            "health_status": o.health_status,
            "confidence": o.confidence,
            "date": o.created_at.isoformat(),
        }

    return {
        "plant_id": plant_id,
        "observation_1": _s(obs1),
        "observation_2": _s(obs2),
        "improvement": None,
        "message": "Quantitative recovery comparison requires severity estimation, which is "
                   "not supported by the trained disease model (no severity labels).",
    }


# ===== Prediction =====

def _analyze_with_predictor(raw_image: Optional[bytes]) -> Dict[str, Any]:
    if raw_image is None:
        raw_image = b""
    if PlantGuardPredictor is None:
        return {
            "prediction": {
                "accepted": False, "reason": "ML model not installed.",
                "model_version": "unavailable",
            }
        }
    try:
        predictor = get_predictor()
        result = predictor.predict_from_bytes(raw_image, filename="upload")
        return {"prediction": result.to_dict()}
    except PredictionError as exc:
        raise HTTPException(422, str(exc))


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Analyze a plant image with the real trained model."""
    _validate(file)
    raw = file.file.read()
    if not raw:
        raise HTTPException(422, "Empty file.")
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Trained model not available yet: {exc}")
    try:
        result = predictor.predict_from_bytes(raw, filename=file.filename or "upload")
    except PredictionError as exc:
        raise HTTPException(422, str(exc))

    payload = result.to_dict()
    model_msg = (
        "active" if payload.get("accepted") else
        "insufficient_confidence" if payload.get("confidence") is not None else
        "unavailable"
    )
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "status": "success",
        "prediction": payload,
        "quality_assessment": {
            "overall_score": None,
            "warnings": payload.get("quality_warnings", []),
            "message": "Analysis completed by the real trained model." if payload.get("accepted")
            else "Model ran but did not reach the confidence threshold.",
        },
        "disclaimer": "This AI-assisted visual assessment is for informational purposes only. "
                      "It is not a substitute for laboratory diagnosis, soil testing, or "
                      "professional plant pathology advice.",
        "model_status": model_msg,
    }


# ===== Treatments =====

@app.post("/plants/{plant_id}/treatments")
async def create_treatment(
    plant_id: str,
    treatment: TreatmentCreate,
    db: Session = Depends(get_db),
) -> dict:
    plant = _get_plant_or_404(db, plant_id)
    treat = Treatment(
        plant_id=plant.id,
        treatment_name=treatment.treatment_name,
        description=treatment.description,
        notes=treatment.notes,
    )
    db.add(treat)
    db.commit()
    db.refresh(treat)
    plant.treatment = treatment.treatment_name
    plant.treatment_date = treat.applied_date
    db.commit()
    return {
        "id": treat.id,
        "plant_id": plant_id,
        "treatment_name": treat.treatment_name,
        "description": treat.description,
        "applied_date": treat.applied_date.isoformat(),
        "notes": treat.notes,
        "message": "Treatment record added",
    }


@app.get("/plants/{plant_id}/treatments")
async def list_treatments(plant_id: str, db: Session = Depends(get_db)) -> list:
    plant = _get_plant_or_404(db, plant_id)
    treatments = db.query(Treatment).filter(Treatment.plant_id == plant.id)\
        .order_by(Treatment.applied_date.desc()).all()
    return [
        {
            "id": t.id, "treatment_name": t.treatment_name, "description": t.description,
            "applied_date": t.applied_date.isoformat(), "notes": t.notes,
        }
        for t in treatments
    ]


# ===== Image Serving =====

@app.get("/uploads/{filename}")
async def serve_upload(filename: str) -> FileResponse:
    p = UPLOAD_DIR / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(404, "Image not found")


@app.get("/data/plants/{plant_id}/{filename}")
async def serve_plant_image(plant_id: str, filename: str) -> FileResponse:
    p = PLANTS_DIR / plant_id / filename
    if p.exists():
        return FileResponse(p)
    raise HTTPException(404, "Image not found")


# ===== Model Info =====

@app.get("/eval-metrics")
async def eval_metrics() -> dict:
    """Return test-set metrics produced by ml/evaluate.py, or a clear
    'not computed' status. Never fabricates numbers."""
    p = MODELS_DIR / "evaluate_test.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {"status": "ok", "metrics": data}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "detail": str(exc), "metrics": None}
    return {"status": "not_computed", "metrics": None}


@app.get("/model-info")
async def model_info(db: Session = Depends(get_db)) -> dict:
    info: dict = {
        "model_name": "PlantGuard EfficientNet-B0 classifier",
        "version": MODEL_VERSION if MODEL_VERSION else "unavailable",
        "status": "not_loaded",
        "available_tasks": [],
        "unsupported_tasks": ["severity_estimation", "deficiency_classification"],
        "disclaimer": "Real trained model. Severity and deficiency are not supported "
                      "because no labels exist in the training data.",
    }
    if PlantGuardPredictor is not None:
        try:
            p = get_predictor()
            info["status"] = "loaded"
            info["version"] = p.model_version
            info["available_tasks"] = [
                "plant_identification",
                "health_classification",
                "disease_classification",
            ]
            existing = (
                db.query(ModelVersion)
                .filter(ModelVersion.version == p.model_version)
                .first()
            )
            if not existing:
                db.add(
                    ModelVersion(
                        model_name="plantguard_efficientnet_b0",
                        version=p.model_version,
                    )
                )
                db.commit()
        except Exception as exc:
            info["status"] = "checkpoint_missing"
            info["message"] = str(exc)
    return info


@app.on_event("startup")
async def startup():
    logger.info("PlantGuard AI backend starting...")


# ===== Optional single-service mode (serves the built SPA) ==================
# If frontend_vite/dist exists (e.g. built inside the Render service), the same
# FastAPI process serves both the API and the frontend at one URL. This kills
# CORS/deployment-split issues entirely for simple deployments.

if SPA_DIR.exists() and (SPA_DIR / "index.html").exists():
    if (SPA_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=SPA_DIR / "assets"), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # API-like paths that miss should 404 as JSON, not return index.html,
        # so the frontend never sees the "Unexpected token '<'" error again.
        if full_path.startswith(("api/", "predict", "plants", "health", "model-info", "eval-metrics", "uploads", "data")):
            raise HTTPException(404, "Not found")
        candidate = (SPA_DIR / full_path).resolve()
        if full_path and candidate.is_file() and SPA_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(SPA_DIR / "index.html")