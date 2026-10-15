# Multi-Crop Monitoring System - End-to-End System Architecture Blueprint

## Executive Summary
The **Multi-Crop Monitoring System** is an enterprise-grade, geospatial-enabled agricultural decision support platform. It integrates remote sensing (Sentinel-2 NDVI satellite imagery), computer vision AI (crop disease & pest identification), meteorological risk analytics, and real-time field management into a scalable monorepo architecture.

---

## 1. System High-Level Topology & Data Flow

```
+-----------------------------------------------------------------------------------+
|                                  USER CLIENTS                                     |
|  React 18 + Vite + Tailwind CSS + Leaflet/MapLibre GIS + Chart.js                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         | REST APIs (HTTPS / JSON)
                                         v
+-----------------------------------------------------------------------------------+
|                                 FASTAPI BACKEND                                   |
|  - API Gateway / Auth Router (JWT, RBAC)                                          |
|  - Field & Spatial Services (GeoAlchemy2 / Shapely)                               |
|  - Satellite Data Ingestion & Index Calculator (NDVI / EVI)                       |
|  - Weather & Agronomic Risk Engine (GDD, Disease Outbreak Scoring)                |
|  - ML Inference Gateway (PyTorch / ONNX Runtime)                                  |
+-----------------------------------------------------------------------------------+
         |                                |                               |
         | Async DB Queries               | Background Jobs               | Inference
         v                                v                               v
+-----------------------+     +-----------------------+     +-----------------------+
|  POSTGRESQL + POSTGIS |     |   REDIS + CELERY      |     |  OFFLINE ML PIPELINE  |
|  - Relational Data    |     |  - Task Queue Broker  |     |  - Model Weights      |
|  - Polygon Geometries |     |  - NDVI Fetch Jobs    |     |  - Vision Detectors   |
|  - Spatial Indices    |     |  - Alert Dispatcher   |     |  - PyTorch / ONNX     |
+-----------------------+     +-----------------------+     +-----------------------+
```

---

## 2. Monorepo Architecture & Directory Specifications

```
multi-crop-monitoring-system/
│
├── frontend/                     # React + Vite + Tailwind CSS Client
│   ├── src/
│   │   ├── assets/               # Branding, icons, static assets
│   │   ├── components/           # Reusable UI widgets (Navbar, Cards, Modals, Tables)
│   │   ├── components/map/       # GIS Map components (Leaflet / MapLibre field polygons)
│   │   ├── components/charts/    # NDVI time-series & weather graphs
│   │   ├── pages/                # Page views (Auth, Dashboard, Fields, Scans, Alerts)
│   │   ├── services/             # Axios API client modules
│   │   ├── store/                # State management (Zustand / Context API)
│   │   └── utils/                # Coordinate formatters, GIS helper routines
│   ├── Dockerfile
│   └── package.json
│
├── backend/                      # Python FastAPI Application
│   ├── alembic/                  # Alembic DB migration revisions
│   ├── app/
│   │   ├── api/v1/               # API Router Endpoints
│   │   │   ├── auth.py           # Authentication & User Management
│   │   │   ├── fields.py         # GIS Field Polygon CRUD & Spatial Queries
│   │   │   ├── crops.py          # Crop Registration & Growth Tracking
│   │   │   ├── satellite.py      # Sentinel NDVI/EVI Processing APIs
│   │   │   ├── disease.py        # Plant Disease & Pest Image Scan APIs
│   │   │   ├── weather.py        # Weather Logs & Agronomic Risk Engine
│   │   │   ├── alerts.py         # Outbreak Warnings & Advisory Notifications
│   │   │   └── health.py         # System Health Check Endpoints
│   │   ├── core/
│   │   │   ├── config.py         # Pydantic Settings (.env handling)
│   │   │   ├── security.py       # Password Hashing & JWT Token Generation
│   │   │   └── logging.py        # Structured Logging Config
│   │   ├── db/
│   │   │   ├── base.py           # SQLAlchemy Declarative Base
│   │   │   ├── session.py        # Async Session Manager
│   │   │   └── models/           # DB Entity Models (User, Field, Crop, Scan, Alert)
│   │   ├── schemas/              # Pydantic Request/Response DTO Validation
│   │   ├── services/             # Business Logic & External Integrations
│   │   │   ├── gis_service.py    # PostGIS Polygon Computations
│   │   │   ├── ndvi_service.py   # Satellite Imagery Index Processing
│   │   │   ├── risk_service.py   # Disease Outbreak Probability Calculator
│   │   │   └── inference_service.py # ONNX ML Model Invocation
│   │   └── workers/              # Celery / Redis Background Job Workers
│   ├── alembic.ini
│   ├── Dockerfile
│   └── requirements.txt
│
├── ml/                           # Data Science & Machine Learning Workspace
│   ├── datasets/                 # Raw & Annotated Leaf/Pest Image Datasets
│   ├── notebooks/                # Exploratory Data Analysis & Model Prototyping
│   ├── models/                   # Trained Weights (.pt, .onnx) & Metadata
│   ├── training/                 # PyTorch Training Loops & Data Loaders
│   ├── inference/                # Model Preprocessing & ONNX Exporter Scripts
│   ├── README.md
│   └── requirements.txt
│
├── docs/                         # Architecture & Developer Documentation
│   ├── architecture.md           # System Blueprint (this document)
│   └── development.md            # Setup & Migration Guide
│
├── docker-compose.yml            # Multi-container Orchestrator (PostGIS, Redis, Backend, Frontend)
├── .env.example                  # Environment Configuration Template
├── .gitignore                    # Monorepo Version Control Exclusions
└── README.md                     # Project Quickstart Guide
```

---

## 3. Data Architecture & Relational Schema (PostgreSQL + PostGIS)

### Core Entities & Attributes

1. **`users` Table**:
   - `id`: UUID (Primary Key)
   - `email`: VARCHAR(255) (Unique, Indexed)
   - `hashed_password`: VARCHAR(255)
   - `full_name`: VARCHAR(100)
   - `role`: ENUM ('FARMER', 'AGRONOMIST', 'ADMIN')
   - `created_at`: TIMESTAMP WITH TIME ZONE

2. **`fields` Table** *(Geospatial)*:
   - `id`: UUID (Primary Key)
   - `user_id`: UUID (Foreign Key -> `users.id`)
   - `name`: VARCHAR(100)
   - `boundary`: GEOMETRY(Polygon, 4326) *(Spatial Index - GiST)*
   - `area_hectares`: FLOAT
   - `soil_type`: VARCHAR(50)
   - `created_at`: TIMESTAMP WITH TIME ZONE

3. **`crop_cycles` Table**:
   - `id`: UUID (Primary Key)
   - `field_id`: UUID (Foreign Key -> `fields.id`)
   - `crop_type`: ENUM ('COTTON', 'WHEAT', 'RICE', 'SUGARCANE', 'MAIZE', 'SOYBEAN')
   - `variety`: VARCHAR(100)
   - `sowing_date`: DATE
   - `expected_harvest_date`: DATE
   - `status`: ENUM ('PLANTED', 'GROWING', 'HARVESTED', 'FAILED')

4. **`satellite_scans` Table**:
   - `id`: UUID (Primary Key)
   - `field_id`: UUID (Foreign Key -> `fields.id`)
   - `acquisition_date`: TIMESTAMP WITH TIME ZONE
   - `mean_ndvi`: FLOAT
   - `mean_evi`: FLOAT
   - `cloud_coverage`: FLOAT
   - `ndvi_map_path`: VARCHAR(500) *(Path to GeoTIFF/PNG overlay)*

5. **`disease_scans` Table**:
   - `id`: UUID (Primary Key)
   - `field_id`: UUID (Foreign Key -> `fields.id`)
   - `user_id`: UUID (Foreign Key -> `users.id`)
   - `image_url`: VARCHAR(500)
   - `detected_disease`: VARCHAR(100)
   - `confidence_score`: FLOAT
   - `severity_level`: ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
   - `scanned_at`: TIMESTAMP WITH TIME ZONE

6. **`risk_alerts` Table**:
   - `id`: UUID (Primary Key)
   - `field_id`: UUID (Foreign Key -> `fields.id`)
   - `risk_type`: ENUM ('DISEASE_OUTBREAK', 'PEST_INFESTATION', 'DROUGHT_STRESS', 'FROST_WARNING')
   - `probability_score`: FLOAT
   - `advisory_message`: TEXT
   - `is_resolved`: BOOLEAN (Default: False)
   - `created_at`: TIMESTAMP WITH TIME ZONE

---

## 4. Subsystem Modules & Phased Implementation Roadmap

### **Phase 1: Foundation & Authentication**
- User registration, login, JWT token issuance, password hashing (Bcrypt).
- Role-based access control (RBAC) protecting endpoints.

### **Phase 2: Geospatial & Field Polygon Management**
- Interactive map field boundary plotting (Leaflet/MapLibre).
- PostGIS spatial validation (GeoJSON parsing, polygon intersection, hectare calculations).
- Crop cycle registration linked to field boundaries.

### **Phase 3: Satellite & Remote Sensing Integration**
- Integration with Copernicus / Open-access Sentinel-2 API.
- Automated calculation of vegetation indices:
  $$\text{NDVI} = \frac{\text{NIR} - \text{RED}}{\text{NIR} + \text{RED}}$$
- Heatmap visualization overlays for field vigor tracking.

### **Phase 4: AI & Machine Learning Detection Engine**
- Computer vision model (PyTorch ResNet/EfficientNet exported to ONNX Runtime) for crop leaf disease classification.
- Image upload, pre-processing (resizing, normalization), inference scoring, and advisory generation.

### **Phase 5: Weather & Risk Analytics Engine**
- Real-time weather data fetching (Temperature, Humidity, Rainfall).
- Agronomic Growth Degree Days (GDD) computation:
  $$\text{GDD} = \max\left(\frac{T_{\max} + T_{\min}}{2} - T_{\text{base}}, 0\right)$$
- Automated background risk engine calculating disease outbreak likelihood using weather + crop stage + satellite vigor.

### **Phase 6: Dashboards, Advisory & Notification Hub**
- Farmer dashboard summarizing field health, active alerts, NDVI trends, and recent disease scans.
- PDF advisory report export and multi-channel notification dispatcher (Redis queue worker).

---

## 5. Non-Functional Requirements & Security Standards
- **API Response Time**: $< 200\text{ms}$ for standard CRUD; $< 1.5\text{s}$ for ML inference.
- **Geospatial Precision**: WGS84 (EPSG:4326) coordinate standard with PostGIS GiST spatial indices.
- **Security**: JWT authentication with bearer tokens, CORS origin restrictions, input validation via Pydantic v2.
- **Database Migrations**: Alembic version control for schema reproducibility.
