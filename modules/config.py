from pydantic import BaseSettings
from typing import Dict, List
import secrets

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Stock Analysis"
    
    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000"
    ]
    
    # Market Data APIs
    MARKET_DATA_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    
    # Email Configuration
    EMAIL_USERNAME: str = ""
    EMAIL_PASSWORD: str = ""
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    
    # Quality Alert Thresholds
    QUALITY_ALERT_THRESHOLDS: Dict[str, float] = {
        'overall_score': 0.8,
        'completeness': 0.9,
        'accuracy': 0.85,
        'consistency': 0.9,
        'timeliness': 0.95,
        'uniqueness': 0.99
    }
    
    # Alert Settings
    ALERT_NOTIFICATION_CHANNELS: List[str] = ['email', 'slack']
    
    # Environment
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # Trading Settings
    DEFAULT_RISK_TOLERANCE: float = 0.5
    MAX_POSITION_SIZE: float = 0.1
    STOP_LOSS_PERCENTAGE: float = 0.02
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 