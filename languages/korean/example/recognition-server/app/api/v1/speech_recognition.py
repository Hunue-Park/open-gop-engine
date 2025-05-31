from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
from app.services.recognition_service import RecognitionService
from app.schemas.recognition import (
    SessionCreateRequest,
    SessionResponse,
    EvaluationResponse
)

router = APIRouter(prefix="/api/v1/speech", tags=["speech"])
recognition_service = RecognitionService()


@router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """새로운 음성 인식 세션 생성"""
    try:
        result = recognition_service.create_session(request.sentence, request.options)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate/{session_id}", response_model=EvaluationResponse)
async def evaluate_audio(session_id: str, audio_file: UploadFile = File(...)):
    """오디오 평가 수행"""
    try:
        audio_data = await audio_file.read()
        result = recognition_service.evaluate_audio(session_id, audio_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """세션 상태 조회"""
    try:
        return recognition_service.get_session_status(session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/session/{session_id}")
async def close_session(session_id: str):
    """세션 종료"""
    try:
        return recognition_service.close_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
