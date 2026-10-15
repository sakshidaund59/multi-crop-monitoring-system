import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ModelRegistryResponse(BaseModel):
    id: uuid.UUID
    name: str
    version: str
    weights_hash: str
    dataset_version: str
    metrics_path: Optional[str] = None
    trained_at: datetime
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True
