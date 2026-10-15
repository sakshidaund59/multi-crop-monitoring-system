import asyncio
import os
import sys

# Ensure backend root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.models.crop import Crop

CROPS_DATA = [
    {
        "name": "Rice",
        "scientific_name": "Oryza sativa",
        "description": "Cereal grain and staple food for a large part of the world population.",
    },
    {
        "name": "Wheat",
        "scientific_name": "Triticum aestivum",
        "description": "Cereal grain cultivated worldwide as a major food staple.",
    },
    {
        "name": "Maize",
        "scientific_name": "Zea mays",
        "description": "Cereal grain widely grown for food, fodder, and industrial products.",
    },
    {
        "name": "Cotton",
        "scientific_name": "Gossypium hirsutum",
        "description": "Soft, fluffy staple fiber crop used primarily for textile production.",
    },
    {
        "name": "Soybean",
        "scientific_name": "Glycine max",
        "description": "Legume species widely cultivated for its edible bean and oil production.",
    },
    {
        "name": "Sugarcane",
        "scientific_name": "Saccharum officinarum",
        "description": "Tall perennial grass cultivated for sugar production and biofuel.",
    },
]


async def seed_crops():
    """Idempotently seed the 6 initial crops."""
    print("Starting crop seeding...")
    async with AsyncSessionLocal() as session:
        async with session.begin():
            seeded_count = 0
            for crop_info in CROPS_DATA:
                stmt = select(Crop).where(Crop.name == crop_info["name"])
                res = await session.execute(stmt)
                existing = res.scalar_one_or_none()

                if not existing:
                    crop = Crop(
                        name=crop_info["name"],
                        scientific_name=crop_info["scientific_name"],
                        description=crop_info["description"],
                    )
                    session.add(crop)
                    seeded_count += 1
                    print(f"Added crop: {crop_info['name']}")
                else:
                    print(f"Crop already exists: {crop_info['name']}")

        await session.commit()
        print(f"Seeding completed. New crops added: {seeded_count}")


if __name__ == "__main__":
    asyncio.run(seed_crops())
