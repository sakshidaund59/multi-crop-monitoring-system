# Architecture Overview - Multi-Crop Monitoring System

## Overview
The Multi-Crop Monitoring System is designed as a modular monorepo architecture. It integrates web client visualization, FastAPI backend micro-services (monolith modular structure), PostgreSQL/PostGIS spatial data persistence, Redis task queues/caching, and offline/batch ML pipelines.

---

## High-Level Data Flow

```
[ Frontend (React + Vite) ] <---> [ FastAPI Backend ] <---> [ PostgreSQL + PostGIS ]
                                          |
                                          +---> [ Redis Queue ]
                                          |
                                          +---> [ ML Inference Engine ]
```

---

## Component Responsibilities

1. **Frontend (`frontend/`)**:
   - Modern React application built with Vite and styled using Tailwind CSS.
   - Responsible for user interaction, map rendering (GIS field polygons), dashboard displays, and satellite image inspection.

2. **Backend (`backend/`)**:
   - High-performance FastAPI application.
   - Database ORM using SQLAlchemy (Async) + GeoAlchemy2 for PostGIS spatial queries.
   - Database migrations managed via Alembic.
   - Exposes RESTful API endpoints for client applications and internal background tasks.

3. **Database & Cache (`db` & `redis`)**:
   - **PostgreSQL + PostGIS**: Stores relational data, crop types, field geometries (POLYGON), historical observations, and spatial indices.
   - **Redis**: In-memory cache store and job queue broker for background processing (satellite data processing, ML model predictions).

4. **Machine Learning (`ml/`)**:
   - Isolated workspace for data science notebooks, dataset preprocessing scripts, model training routines, and model artifact storage.

---

## Key Design Principles
- **Separation of Concerns**: UI, API, database access, and ML workflows remain strictly decoupled.
- **Geospatial-First**: Built around PostGIS standards (EPSG:4326 WGS84 coordinates).
- **Scalability**: Async Python execution path for non-blocking I/O operations.
