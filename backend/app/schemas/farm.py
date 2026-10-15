import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FarmCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    boundary: Optional[Dict[str, Any]] = None  # GeoJSON Polygon dict {"type": "Polygon", "coordinates": [...]}
    area_ha: Optional[float] = Field(default=None, ge=0.0)


class FarmResponse(BaseModel):
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    boundary: Optional[Dict[str, Any]] = None
    area_ha: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
