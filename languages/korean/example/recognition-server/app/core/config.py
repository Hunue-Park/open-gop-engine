import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_KEY: str = os.getenv("APP_KEY", "your_default_app_key")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your_default_secret_key")
    API_VERSION: str = "v1"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
