"""Environment-driven config."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FEATUREFLOW_",
        extra="ignore",
    )

    # Online store (Redis)
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50

    # Offline store
    offline_store_path: str = "./data/offline"  # parquet files live here

    # Registry (sqlite via duckdb for simplicity in OSS; production uses postgres)
    registry_db_path: str = "./data/registry.duckdb"

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Streaming
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "featureflow"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9100

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
