# Multi-Crop Monitoring System

A scalable monorepo platform designed for multi-crop health monitoring, geospatial field analytics, disease/pest risk engine, and agricultural decision support.

---

## Repository Structure

```
multi-crop-monitoring-system/
│
├── frontend/             # React + Vite + Tailwind CSS web interface placeholder
├── backend/              # Python FastAPI REST API with PostGIS & Alembic
├── ml/                   # Machine learning datasets, models & training pipelines placeholder
├── docs/                 # System architecture and development documentation
├── docker-compose.yml    # Development orchestrator (PostGIS, Redis, Backend, Frontend)
├── .gitignore            # Git exclusion rules
├── .env.example          # Environment variable template
└── README.md             # Project quickstart and reference guide
```

---

## Quickstart Guide

### Prerequisites
- Docker & Docker Compose
- Node.js (v18+ or v20+) *(optional for local frontend dev)*
- Python 3.11+ *(optional for local backend dev)*

---

### Running with Docker Compose (Recommended)

> [!NOTE]
> Modern Docker Desktop uses `docker compose` (space-separated, Docker Compose v2). If you use legacy Docker Compose v1, install the plugin or use `docker compose`.

1. **Clone the repository & copy environment configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Start all services:**
   ```bash
   docker compose up -d --build
   ```

3. **Verify running containers:**
   ```bash
   docker compose ps
   ```

4. **Verify Backend Health Endpoints:**
   - App Health: [http://localhost:8000/health](http://localhost:8000/health)
   - Database Health: [http://localhost:8000/health/db](http://localhost:8000/health/db)
   - OpenAPI Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

5. **Verify Frontend Application:**
   - Web UI: [http://localhost:5173](http://localhost:5173)

---

### Stopping Services

```bash
docker compose down
```

---

## Phase 1: Foundation & Authentication

### 1. Database Migrations
Make sure the database container is running (`docker compose up -d db redis`), then apply the database migrations:
```bash
cd backend
py -m alembic upgrade head
```

### 2. Running Pytest Suite
Run the automated test suite for authentication, JWT guards, and Farm GIS endpoints:
```bash
cd backend
py -m pytest tests/ -v
```

### 3. Local Development

**Start Backend (Port 8000):**
```bash
cd backend
py -m uvicorn app.main:app --reload --port 8000
```

**Start Frontend (Port 3000):**
```bash
cd frontend
npm install
npm run dev
```

### 4. API Verification with cURL

#### Health Checks
```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/health/db
```

#### User Registration
```bash
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"farmer@example.com","password":"SecretPassword123!","full_name":"Ramesh Patil","role":"farmer"}'
```

#### User Login
```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"farmer@example.com","password":"SecretPassword123!"}'
```

#### Get Current User (Bearer Token)
```bash
curl -X GET http://127.0.0.1:8000/api/auth/me \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

#### Create Farm Plot (GPS Point + Polygon Boundary)
```bash
curl -X POST http://127.0.0.1:8000/api/farms \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"name":"Nashik Vineyard Plot 1","latitude":19.9975,"longitude":73.7898,"area_ha":8.5,"boundary":{"type":"Polygon","coordinates":[[[73.78,19.99],[73.79,19.99],[73.79,20.00],[73.78,20.00],[73.78,19.99]]]}}'
```

#### List User Farms
```bash
curl -X GET http://127.0.0.1:8000/api/farms \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

#### Delete Farm
```bash
curl -X DELETE http://127.0.0.1:8000/api/farms/<FARM_ID> \
  -H "Authorization: Bearer <ACCESS_TOKEN>"
```

---

## Development Milestones

- [x] **Step 0: Project Setup** - Monorepo structure, Docker Compose (PostGIS & Redis), FastAPI minimal skeleton with Alembic setup & `/health` endpoints.
- [x] **Phase 1: Foundation & Authentication** - Core database models (users, farms, model_registry, analyses, pest_detections, disease_results, severity_results, weather_snapshots, ndvi_snapshots, risk_results, alerts, chat_messages), JWT Auth, Farm CRUD, error handling, React Router UI with sidebar, and automated pytest suite.
- [ ] **Phase 2: Field & Geospatial Management** *(Future Phase)*
- [ ] **Phase 3: Crop Monitoring & Satellite Integration** *(Future Phase)*
- [ ] **Phase 4: AI & ML Engine Integration** *(Future Phase)*
- [ ] **Phase 5: Alert System & Dashboards** *(Future Phase)*

---

## License
Proprietary - Multi-Crop Monitoring System Team.
