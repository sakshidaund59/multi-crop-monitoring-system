import uuid
from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    farm_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"), nullable=True, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_registry.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False, default="disease")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="completed")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    farm = relationship("Farm", back_populates="analyses")
    user = relationship("User")
    model = relationship("ModelRegistry", back_populates="analyses")
    pest_detections = relationship("PestDetection", back_populates="analysis", cascade="all, delete-orphan")
    disease_results = relationship("DiseaseResult", back_populates="analysis", cascade="all, delete-orphan")
    severity_results = relationship("SeverityResult", back_populates="analysis", cascade="all, delete-orphan")
