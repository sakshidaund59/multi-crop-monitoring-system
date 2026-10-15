import uuid
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from geoalchemy2.elements import WKTElement
from geoalchemy2.shape import to_shape
from shapely.geometry import shape, Point, Polygon
from app.db.session import get_db
from app.models.user import User
from app.models.farm import Farm
from app.schemas.farm import FarmCreateRequest, FarmResponse
from app.api.deps import get_current_active_user

router = APIRouter(prefix="/farms", tags=["Farms"])


def serialize_farm(farm: Farm) -> FarmResponse:
    """Helper to convert GeoAlchemy2 geometries into GeoJSON / lat-lon for API response."""
    lat, lon = None, None
    if farm.location is not None:
        try:
            pt = to_shape(farm.location)
            lon = float(pt.x)
            lat = float(pt.y)
        except Exception:
            pass

    boundary_geojson = None
    if farm.boundary is not None:
        try:
            poly = to_shape(farm.boundary)
            boundary_geojson = poly.__geo_interface__
        except Exception:
            pass

    return FarmResponse(
        id=farm.id,
        owner_id=farm.owner_id,
        name=farm.name,
        latitude=lat,
        longitude=lon,
        boundary=boundary_geojson,
        area_ha=farm.area_ha,
        created_at=farm.created_at,
        updated_at=farm.updated_at,
    )


@router.post(
    "",
    response_model=FarmResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new farm with GPS point and optional boundary polygon",
)
async def create_farm(
    payload: FarmCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Register a new farm plot with GPS center point and optional boundary geometry in SRID 4326."""
    point_wkt = WKTElement(f"POINT({payload.longitude} {payload.latitude})", srid=4326)

    boundary_wkt = None
    calculated_area = payload.area_ha

    if payload.boundary:
        try:
            poly = shape(payload.boundary)
            if not isinstance(poly, Polygon):
                raise ValueError("Boundary must be a valid GeoJSON Polygon geometry")
            boundary_wkt = WKTElement(poly.wkt, srid=4326)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid boundary GeoJSON polygon: {str(e)}",
            )

    farm = Farm(
        owner_id=current_user.id,
        name=payload.name.strip(),
        location=point_wkt,
        boundary=boundary_wkt,
        area_ha=calculated_area,
    )
    db.add(farm)
    await db.commit()
    await db.refresh(farm)

    return serialize_farm(farm)


@router.get(
    "",
    response_model=List[FarmResponse],
    summary="List all farms belonging to current user",
)
async def list_farms(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve all farms registered by the authenticated user."""
    stmt = select(Farm).where(Farm.owner_id == current_user.id).order_by(Farm.created_at.desc())
    res = await db.execute(stmt)
    farms = res.scalars().all()
    return [serialize_farm(f) for f in farms]


@router.get(
    "/{farm_id}",
    response_model=FarmResponse,
    summary="Get farm details by ID",
)
async def get_farm(
    farm_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve details of a specific farm by its UUID."""
    stmt = select(Farm).where(Farm.id == farm_id)
    res = await db.execute(stmt)
    farm = res.scalar_one_or_none()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found",
        )

    if farm.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this farm",
        )

    return serialize_farm(farm)


@router.delete(
    "/{farm_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a farm plot",
)
async def delete_farm(
    farm_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a farm plot by its UUID."""
    stmt = select(Farm).where(Farm.id == farm_id)
    res = await db.execute(stmt)
    farm = res.scalar_one_or_none()

    if not farm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Farm not found",
        )

    if farm.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this farm",
        )

    await db.delete(farm)
    await db.commit()
    return None
