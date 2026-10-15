from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.models.model_registry import ModelRegistry
from app.schemas.model import ModelRegistryResponse

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get(
    "",
    response_model=List[ModelRegistryResponse],
    summary="List registered ML models",
)
async def list_models(
    db: AsyncSession = Depends(get_db),
):
    """Retrieve all models currently registered in the model registry."""
    stmt = select(ModelRegistry).where(ModelRegistry.is_active == True).order_by(ModelRegistry.created_at.desc())
    res = await db.execute(stmt)
    return res.scalars().all()


@router.get(
    "/active",
    response_model=ModelRegistryResponse,
    summary="Get currently active ML model",
)
async def get_active_model(
    db: AsyncSession = Depends(get_db),
):
    """Retrieve the primary active model for inference."""
    stmt = select(ModelRegistry).where(ModelRegistry.is_active == True).order_by(ModelRegistry.created_at.desc())
    res = await db.execute(stmt)
    model = res.scalars().first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active machine learning model registered",
        )
    return model
