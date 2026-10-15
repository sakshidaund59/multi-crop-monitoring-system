import enum
import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, Enum as SQLEnum, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base


class PestDiseaseCategory(str, enum.Enum):
    PEST = "pest"
    DISEASE = "disease"


class PestDisease(Base):
    __tablename__ = "pest_diseases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    crop_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("crops.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[PestDiseaseCategory] = mapped_column(
        SQLEnum(PestDiseaseCategory, name="pest_disease_category", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    crop = relationship("Crop", back_populates="pest_diseases")
    detections = relationship("Detection", back_populates="pest_disease")
