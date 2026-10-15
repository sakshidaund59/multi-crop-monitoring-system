from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.api.v1.health import router as health_router
from app.api.v1.auth import router as auth_router
from app.api.v1.farms import router as farms_router
from app.api.v1.models import router as models_router

# Initialize structured logging system
setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/api/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global Error Handlers returning {"error": ..., "reason": ...}
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"HTTP_{exc.status_code}", "reason": str(exc.detail)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_reasons = []
    for err in exc.errors():
        loc = ".".join(str(l) for l in err["loc"] if l != "body")
        error_reasons.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "reason": "; ".join(error_reasons)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    reason = str(exc)
    if "relation \"users\" does not exist" in reason:
        reason = "Database table 'users' does not exist. Please run 'py -m alembic upgrade head' in the backend directory."
    elif "Connect call failed" in reason or "connection refused" in reason.lower() or "connection to server at" in reason.lower():
        reason = "Cannot connect to PostgreSQL database on 127.0.0.1:15432. Please ensure docker container 'crop_monitoring_db' is running (run 'docker compose up -d db')."
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "reason": reason},
    )


# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount health endpoints at root (/health and /health/db) and under /api/v1
app.include_router(health_router)
app.include_router(health_router, prefix="/api/v1")

# Mount API modules under /api and /api/v1
app.include_router(auth_router, prefix="/api")
app.include_router(auth_router, prefix="/api/v1")

app.include_router(farms_router, prefix="/api")
app.include_router(farms_router, prefix="/api/v1")

app.include_router(models_router, prefix="/api")
app.include_router(models_router, prefix="/api/v1")


@app.get("/", summary="Root Endpoint")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": "/health",
        "db_health": "/health/db",
    }
