from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SonoLesion-AI API"
    model_path: str | None = None
    results_dir: str = "results"
    allow_demo_mode: bool = True
    model_input_size: int = 256
    segmentation_threshold: float = 0.5
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SONOLESION_",
        extra="ignore",
    )

    @property
    def results_path(self) -> Path:
        path = Path(self.results_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
