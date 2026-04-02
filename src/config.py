from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Centralized configuration management for MarketPulse AI.
    Loads variables from the .env file and environment.
    """
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database
    mongo_uri: str = Field(default="mongodb://localhost:27017/", description="MongoDB connection URI")
    db_name: str = Field(default="marketpulse", description="Main database name")

    # Paths
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts", description="Directory to store ML artifacts")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs", description="Directory to store application logs")
    
    # API Settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )

    # ML Parameters
    pca_components: int = Field(default=2, description="Number of PCA components")
    max_clusters: int = Field(default=8, description="Maximum number of clusters for K-Means tuning")

settings = Settings()

# Ensure directories exist
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
settings.logs_dir.mkdir(parents=True, exist_ok=True)
