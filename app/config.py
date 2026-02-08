"""
Application Configuration

Loads environment variables using pydantic-settings.
All settings can be overridden via a .env file or environment variables.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-2.0-flash"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    PDF_PATH: str = "data/ng12.pdf"
    PATIENTS_PATH: str = "data/patients.json"

    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), extra="ignore")


settings = Settings()
