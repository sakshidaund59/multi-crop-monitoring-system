# Development Guide - Multi-Crop Monitoring System

## Environment Setup

### 1. Requirements
- Python 3.11+
- Node.js 18+ & npm
- Docker Desktop (includes `docker compose` v2)
- PostgreSQL client tools (optional)

---

### 2. Docker Development (Recommended)

```powershell
# Build and start all services
docker compose up -d --build

# View container status
docker compose ps

# Stop containers
docker compose down
```

---

### 2. Local Backend Development (without Docker)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run FastAPI dev server
uvicorn app.main:app --reload --port 8000
```

---

### 3. Database Migrations (Alembic)

Alembic is configured to manage schema migrations.

```bash
cd backend

# Create a new migration revision
alembic revision -m "Describe migration"

# Upgrade database to head revision
alembic upgrade head

# Downgrade one revision
alembic downgrade -1
```

---

### 4. Local Frontend Development (without Docker)

```bash
cd frontend
npm install
npm run dev
```

---

### 5. Health Verification Endpoints
- Base Health: `GET http://localhost:8000/health`
- Database & PostGIS Health: `GET http://localhost:8000/health/db`
