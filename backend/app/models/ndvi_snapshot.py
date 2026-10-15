import uuid
from datetime import datetime, date
from sqlalchemy import Float, Date, String, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base


class NdviSnapshot(Base):
    __tablename__ = "ndvi_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    farm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("farms.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mean_ndvi: Mapped[float] = mapped_column(Float, nullable=False)
    min_ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_ndvi: Mapped[float | None] = mapped_column(Float, nullable=True)
    cloud_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    tile_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    farm = relationship("Farm", back_populates="ndvi_snapshots")
