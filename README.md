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

## Development Milestones

- [x] **Step 0: Project Setup** - Monorepo structure, Docker Compose (PostGIS & Redis), FastAPI minimal skeleton with Alembic setup & `/health` endpoints, React+Vite placeholder.
- [ ] **Step 1: Core Foundation & Auth** *(Future Phase)*
- [ ] **Step 2: Field & Geospatial Management** *(Future Phase)*
- [ ] **Step 3: Crop Monitoring & Satellite Integration** *(Future Phase)*
- [ ] **Step 4: AI & ML Engine Integration** *(Future Phase)*
- [ ] **Step 5: Alert System & Dashboards** *(Future Phase)*

---

## License
Proprietary - Multi-Crop Monitoring System Team.
