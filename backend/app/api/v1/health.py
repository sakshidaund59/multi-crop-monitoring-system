from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db.session import get_db
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Basic Application Health Check")
async def health_check():
    """Minimal application health check endpoint."""
    return {
        "status": "ok",
        "app": settings.PROJECT_NAME,
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
    }


@router.get("/health/db", summary="Database & PostGIS Health Check")
async def db_health_check(db: AsyncSession = Depends(get_db)):
    """Database connection and PostGIS spatial extension health check endpoint."""
    try:
        # Check basic postgres connectivity
        result = await db.execute(text("SELECT 1;"))
        result.scalar()

        # Check PostGIS extension
        postgis_version = None
        try:
            pg_res = await db.execute(text("SELECT PostGIS_Full_Version();"))
            postgis_version = pg_res.scalar()
        except Exception:
            postgis_version = "PostGIS extension not active yet or error querying version"

        return {
            "status": "ok",
            "database": "connected",
            "postgis_version": postgis_version,
            "database_name": settings.POSTGRES_DB,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database health check failed: {str(e)}",
        )
