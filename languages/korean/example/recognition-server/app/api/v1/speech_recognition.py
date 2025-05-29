from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional

from app.services.speech_recognition import SpeechRecognitionService
from app.services.auth import AuthService
from app.core.config import get_settings
from app.schemas.speech_recognition import (
    SpeechRecognitionResponse,
    SpeechRecognitionCommand,
)

router = APIRouter(prefix="/api/v1/speech", tags=["speech"])


# 의존성 주입
def get_speech_service():
    return SpeechRecognitionService()


def get_auth_service():
    settings = get_settings()
    return AuthService(settings.SECRET_KEY, settings.APP_KEY)


@router.post("/recognize", response_model=SpeechRecognitionResponse)
async def recognize_speech(
    text: str = Form(...),
    audio: UploadFile = File(...),
    speech_service: SpeechRecognitionService = Depends(get_speech_service),
    auth_service: AuthService = Depends(get_auth_service),
):
    """
    음성 인식 API 엔드포인트
    - text: JSON 형식의 명령 매개변수
    - audio: 오디오 파일 데이터
    """
    try:
        # 오디오 데이터 읽기
        audio_data = await audio.read()

        # 인증 검증 - 실제 구현에서는 이 부분을 활성화
        # command = json.loads(text)
        # if "connect" in command:
        #     connect_params = command["connect"]["param"]["app"]
        #     if not auth_service.verify_connect_signature(
        #         connect_params["timestamp"],
        #         connect_params["sig"]
        #     ):
        #         raise HTTPException(status_code=401, detail="Invalid signature")
        # elif "start" in command:
        #     start_params = command["start"]["param"]["app"]
        #     if not auth_service.verify_start_signature(
        #         start_params["timestamp"],
        #         start_params["userId"],
        #         start_params["sig"]
        #     ):
        #         raise HTTPException(status_code=401, detail="Invalid signature")

        # 명령 처리 및 음성 인식
        result = await speech_service.process_command(text, audio_data)
        return result

    except Exception as e:
        return SpeechRecognitionResponse(
            code=500, message=f"처리 중 오류 발생: {str(e)}", result=None
        )
