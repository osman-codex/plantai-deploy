"""
SQLite Database for PlantGuard AI
Local prototype database using SQLite with SQLAlchemy ORM.
Designed to be easily migrated to PostgreSQL later.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Use SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./plantguard.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ============== Database Models ==============

class Plant(Base):
    """Plant profile for monitoring"""
    __tablename__ = "plants"
    
    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(String(20), unique=True, nullable=False, index=True)  # PG-000001 format
    user_id = Column(Integer, nullable=True)
    species = Column(String(255), nullable=True)
    crop = Column(String(255), nullable=False)
    common_name = Column(String(255), nullable=True)
    date_created = Column(DateTime, default=datetime.utcnow)
    initial_diagnosis = Column(JSON, default=dict)
    treatment = Column(Text, nullable=True)
    treatment_date = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    
    observations = relationship("Observation", back_populates="plant", cascade="all, delete-orphan")
    images = relationship("Image", back_populates="plant", cascade="all, delete-orphan")

class Image(Base):
    """Stored image with metadata"""
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    quality_score = Column(Float, nullable=True)
    embedding = Column(JSON, nullable=True)
    
    plant = relationship("Plant", back_populates="images")
    observation = relationship("Observation", back_populates="image", uselist=False)

class Observation(Base):
    """Model prediction results for an image"""
    __tablename__ = "observations"
    
    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), unique=True)
    predicted_disease = Column(String(255), nullable=True)
    predicted_deficiency = Column(String(255), nullable=True)
    health_status = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    disease_probabilities = Column(JSON, default=dict)
    deficiency_probabilities = Column(JSON, default=dict)
    severity = Column(Float, nullable=True)
    severity_category = Column(String(50), nullable=True)
    affected_area_pct = Column(Float, nullable=True)
    plant_id_prediction = Column(String(255), nullable=True)
    plant_id_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(100), default="0.1.0")
    extra_data = Column("metadata", JSON, default=dict)
    
    plant = relationship("Plant", back_populates="observations")
    image = relationship("Image", back_populates="observation")

class Prediction(Base):
    """Simple prediction history for demos"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=True)
    image_path = Column(String(500), nullable=True)
    predicted_disease = Column(String(255), nullable=True)
    predicted_deficiency = Column(String(255), nullable=True)
    health_status = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    severity = Column(Float, nullable=True)
    severity_category = Column(String(50), nullable=True)
    model_version = Column(String(100), default="0.1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

class Treatment(Base):
    """Treatment records for plants"""
    __tablename__ = "treatments"
    
    id = Column(Integer, primary_key=True, index=True)
    plant_id = Column(Integer, ForeignKey("plants.id", ondelete="CASCADE"), nullable=False)
    treatment_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    applied_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    
    plant = relationship("Plant", back_populates="treatments")

# Add relationship to Plant
from sqlalchemy.orm import relationship as orm_relationship
Plant.treatments = orm_relationship("Treatment", back_populates="plant", cascade="all, delete-orphan")

class ModelVersion(Base):
    """Model version tracking"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(255), nullable=False)
    version = Column(String(100), nullable=False)
    training_datasets = Column(JSON, default=list)
    classes = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)
    model_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# ============== Database Functions ==============

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    
    # Create default plant IDs if none exist
    from sqlalchemy.orm import Session
    db = SessionLocal()
    try:
        max_id = db.query(Plant).order_by(Plant.id.desc()).first()
        if not max_id:
            # Initialize with starting counter
            pass
    finally:
        db.close()

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_plant_id():
    """Generate next Plant ID in format PG-XXXXXX"""
    from sqlalchemy.orm import Session
    db = SessionLocal()
    try:
        last_plant = db.query(Plant).order_by(Plant.id.desc()).first()
        if last_plant and last_plant.plant_id:
            last_num = int(last_plant.plant_id.split('-')[1])
            new_num = last_num + 1
        else:
            new_num = 1
        return f"PG-{new_num:06d}"
    finally:
        db.close()

# Create tables on import
init_db()
