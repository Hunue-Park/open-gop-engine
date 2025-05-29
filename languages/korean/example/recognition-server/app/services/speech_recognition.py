import os
import uuid
import json
import tempfile
from typing import Dict, Any, Optional, BinaryIO
import speech_recognition as sr

from app.schemas.speech_recognition import (
    SpeechRecognitionCommand,
    SpeechRecognitionResponse,
    ConnectRequest,
    StartRequest,
)


class SpeechRecognitionService:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # 세션 저장소
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def process_command(
        self, command_data: str, audio_data: bytes
    ) -> SpeechRecognitionResponse:
        """명령 및 오디오 데이터 처리"""
        try:
            # JSON 파싱
            command_json = json.loads(command_data)
            command = SpeechRecognitionCommand(**command_json)

            # Connect 명령 처리
            if command.connect:
                return await self._handle_connect(command.connect)

            # Start 명령 처리
            elif command.start:
                return await self._handle_start(command.start, audio_data)

            # 알 수 없는 명령
            else:
                return SpeechRecognitionResponse(
                    code=400, message="Unknown command", result=None
                )

        except json.JSONDecodeError:
            return SpeechRecognitionResponse(
                code=400, message="Invalid JSON format", result=None
            )
        except Exception as e:
            return SpeechRecognitionResponse(
                code=500, message=f"처리 중 오류 발생: {str(e)}", result=None
            )

    async def _handle_connect(
        self, connect: ConnectRequest
    ) -> SpeechRecognitionResponse:
        """Connect 명령 처리"""
        # 여기서는 인증 로직은 생략하고 응답만 반환
        session_id = str(uuid.uuid4())

        # 세션 생성
        self.sessions[session_id] = {
            "connected": True,
            "sdk_version": connect.param.sdk.version,
            "app_id": connect.param.app.applicationId,
        }

        return SpeechRecognitionResponse(
            code=0, message="Connected successfully", result={"sessionId": session_id}
        )

    async def _handle_start(
        self, start: StartRequest, audio_data: bytes
    ) -> SpeechRecognitionResponse:
        """Start 명령 처리 및 오디오 인식"""
        try:
            # 임시 파일로 오디오 저장
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{start.param.audio.audioType}"
            ) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # 음성 인식 수행
            recognition_result = await self._recognize_audio(
                temp_audio_path,
                start.param.audio.audioType,
                start.param.audio.sampleRate,
            )

            # 임시 파일 삭제
            os.unlink(temp_audio_path)

            # 음성 인식 결과 반환
            return SpeechRecognitionResponse(
                code=0,
                message="Recognition completed",
                result={
                    "tokenId": start.param.request.tokenId,
                    "text": recognition_result.get("text", ""),
                    "confidence": recognition_result.get("confidence", 0),
                    "status": "done",
                },
            )

        except Exception as e:
            return SpeechRecognitionResponse(
                code=500, message=f"Speech recognition failed: {str(e)}", result=None
            )

    async def _recognize_audio(
        self, audio_path: str, audio_type: str, sample_rate: int
    ) -> Dict[str, Any]:
        """오디오 파일 인식"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)

                # Google의 음성인식 API 사용 (다른 엔진으로 변경 가능)
                text = self.recognizer.recognize_google(audio, language="ko-KR")

                return {
                    "text": text,
                    "confidence": 0.9,  # 실제 신뢰도는 엔진에 따라 다를 수 있음
                }
        except sr.UnknownValueError:
            return {"text": "", "confidence": 0}
        except sr.RequestError as e:
            raise Exception(f"API 요청 실패: {e}")
