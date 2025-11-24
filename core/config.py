"""Central configuration declarations."""
import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings shared across services."""

    vectorstore_root: Path = Field(default=Path("storage/vectorstore"))
    vectorstore_tmp_suffix: str = Field(default=".tmp")
    vectorstore_default_namespace: str = Field(default="kb_default")
    vectorstore_api_url: str = Field(default=os.getenv("QA_AGENT_VECTORSTORE_API_URL", "http://vectorstore:8082"))
    milvus_host: str = Field(default="milvus")
    milvus_port: int = Field(default=19530)
    milvus_user: str = Field(default="")
    milvus_password: str = Field(default="")
    milvus_secure: bool = Field(default=False)
    llm_proxy_url: str = Field(default="http://model:8001/v1/completions")
    weather_api_key: str = Field(default=os.getenv("QA_AGENT_WEATHER_API_KEY",""))
    

    class Config:
        env_prefix = "QA_AGENT_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
