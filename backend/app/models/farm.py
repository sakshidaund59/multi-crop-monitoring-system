import uuid
from datetime import datetime
from sqlalchemy import String, Float, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from geoalchemy2 import Geometry
from app.db.base_class import Base


class Farm(Base):
    __tablename__ = "farms"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    location = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True), nullable=True
    )
    boundary = mapped_column(
        Geometry("POLYGON", srid=4326, spatial_index=True), nullable=True
    )
    area_ha: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    owner = relationship("User", back_populates="farms")
    fields = relationship("Field", back_populates="farm", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="farm", cascade="all, delete-orphan")
    weather_snapshots = relationship("WeatherSnapshot", back_populates="farm", cascade="all, delete-orphan")
    ndvi_snapshots = relationship("NdviSnapshot", back_populates="farm", cascade="all, delete-orphan")
    risk_results = relationship("RiskResult", back_populates="farm", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="farm", cascade="all, delete-orphan")
