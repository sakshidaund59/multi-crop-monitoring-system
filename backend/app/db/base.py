from app.db.base_class import Base

# Model registry for Alembic: import all models so that Base.metadata is fully populated
from app.models.user import User  # noqa
from app.models.crop import Crop  # noqa
from app.models.farm import Farm  # noqa
from app.models.field import Field  # noqa
from app.models.pest_disease import PestDisease  # noqa
from app.models.detection import Detection  # noqa
from app.models.alert import Alert  # noqa
from app.models.model_registry import ModelRegistry  # noqa
from app.models.analysis import Analysis  # noqa
from app.models.pest_detection import PestDetection  # noqa
from app.models.disease_result import DiseaseResult  # noqa
from app.models.severity_result import SeverityResult  # noqa
from app.models.weather_snapshot import WeatherSnapshot  # noqa
from app.models.ndvi_snapshot import NdviSnapshot  # noqa
from app.models.risk_result import RiskResult  # noqa
from app.models.chat_message import ChatMessage  # noqa

__all__ = [
    "Base",
    "User",
    "Crop",
    "Farm",
    "Field",
    "PestDisease",
    "Detection",
    "Alert",
    "ModelRegistry",
    "Analysis",
    "PestDetection",
    "DiseaseResult",
    "SeverityResult",
    "WeatherSnapshot",
    "NdviSnapshot",
    "RiskResult",
    "ChatMessage",
]
