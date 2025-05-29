from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.speech_recognition import router as speech_router

app = FastAPI(
    title="음성인식 API",
    description="멀티파트 폼 데이터를 이용한 음성인식 API",
    version="0.1.0",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(speech_router)


@app.get("/")
def read_root():
    return {"status": "online", "service": "Speech Recognition API"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
