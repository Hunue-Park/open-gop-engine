import os
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings:
    # 프로젝트 기본 경로
    BASE_DIR = Path("/app/open-gop-engine")  # 볼륨 마운트된 경로로 변경

    # 모델 관련 설정
    MODEL_DIR = BASE_DIR / "languages" / "korean" / "models"
    ONNX_MODEL_PATH = str(MODEL_DIR / "wav2vec2_ctc_combined.onnx")
    TOKENIZER_PATH = str(MODEL_DIR / "tokenizer.json")
    
    # 엔진 설정
    DEVICE = "CPU"
    CONFIDENCE_THRESHOLD = 20
    
    # API 설정
    API_V1_STR = "/api/v1"
    PROJECT_NAME = "음성인식 API"


@lru_cache()
def get_settings():
    return Settings()

# settings 인스턴스 생성 및 export
settings = get_settings()
