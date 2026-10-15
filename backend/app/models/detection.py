import enum
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, ForeignKey, Enum as SQLEnum, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry
from app.db.base_class import Base


class DetectionType(str, enum.Enum):
    PEST = "pest"
    DISEASE = "disease"


class SeverityLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    field_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False, index=True
    )
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("crops.id", ondelete="SET NULL"), nullable=True, index=True
    )
    pest_disease_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pest_diseases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    detection_type: Mapped[DetectionType] = mapped_column(
        SQLEnum(DetectionType, name="detection_type", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox = mapped_column(JSONB, nullable=True)
    severity: Mapped[SeverityLevel] = mapped_column(
        SQLEnum(SeverityLevel, name="severity_level", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    density_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_unknown: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    location = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True), nullable=True
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    field = relationship("Field", back_populates="detections")
    pest_disease = relationship("PestDisease", back_populates="detections")
    alerts = relationship("Alert", back_populates="detection")
