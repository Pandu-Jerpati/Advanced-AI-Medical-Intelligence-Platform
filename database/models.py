from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime

from .database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    image_filename = Column(String(255), nullable=False)

    prediction = Column(String(100), nullable=False)

    confidence = Column(Float, nullable=False)

    model_version = Column(String(100), nullable=False)

    gradcam_path = Column(String(500), nullable=True)

    report = Column(Text, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )