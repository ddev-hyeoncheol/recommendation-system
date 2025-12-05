from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------
# Application Settings
# ---------------------------------------------------------
class Settings(BaseSettings):
    """
    Application Settings managed by Pydantic.
    Reads configuration from environment variables and .env file.
    """

    # ---------------------------------------------------------
    # Vespa Configuration
    # ---------------------------------------------------------
    vespa_host: str = Field(
        default="vespa",
        validation_alias="VESPA_HOST",
        description="Hostname of the Vespa container or service",
    )
    vespa_port: int = Field(
        default=8080,
        validation_alias="VESPA_PORT",
        description="Port number for Vespa Query/Container API",
    )

    # ---------------------------------------------------------
    # FastAPI Metadata
    # ---------------------------------------------------------
    api_title: str = Field(default="Recommendation Service API", validation_alias="API_TITLE")
    api_version: str = Field(default="0.1.0", validation_alias="API_VERSION")
    api_description: str = Field(
        default="API for User & Product Recommendation backed by Vespa",
        validation_alias="API_DESCRIPTION",
    )

    # ---------------------------------------------------------
    # Recommendation Tuning Parameters
    # ---------------------------------------------------------
    recommend_hits: int = Field(
        default=5,
        validation_alias="RECOMMEND_HITS",
        description="Number of final recommendations to return",
    )
    recommend_target_hits: int = Field(
        default=10,
        validation_alias="RECOMMEND_TARGET_HITS",
        description="Number of candidates to search in HNSW graph (Approximate Search Accuracy)",
    )

    # ---------------------------------------------------------
    # Pydantic Configuration
    # ---------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ---------------------------------------------------------
# Settings Provider
# ---------------------------------------------------------
@lru_cache()
def get_settings() -> Settings:
    """
    Creates and returns a cached Settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return Settings()


# Global Settings Instance
settings = get_settings()
