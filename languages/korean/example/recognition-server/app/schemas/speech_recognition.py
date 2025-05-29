from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# SDK 정보
class SdkInfo(BaseModel):
    version: int
    source: int
    protocol: int


# 앱 인증 정보 (Connect 용)
class ConnectAppInfo(BaseModel):
    applicationId: str
    timestamp: str
    sig: str


# Connect 요청 파라미터
class ConnectParams(BaseModel):
    sdk: SdkInfo
    app: ConnectAppInfo


# Connect 요청
class ConnectRequest(BaseModel):
    cmd: str = "connect"
    param: ConnectParams


# 앱 인증 정보 (Start 용)
class StartAppInfo(BaseModel):
    userId: str
    applicationId: str
    timestamp: str
    sig: str


# 오디오 정보
class AudioInfo(BaseModel):
    audioType: str
    channel: int
    sampleBytes: int
    sampleRate: int


# 요청 정보
class RequestInfo(BaseModel):
    tokenId: str
    coreType: str
    # 추가 파라미터를 동적으로 처리하기 위한 필드
    extraParams: Optional[Dict[str, Any]] = None


# Start 요청 파라미터
class StartParams(BaseModel):
    app: StartAppInfo
    audio: AudioInfo
    request: RequestInfo


# Start 요청
class StartRequest(BaseModel):
    cmd: str = "start"
    param: StartParams


# 요청에 포함될 수 있는 모든 명령
class SpeechRecognitionCommand(BaseModel):
    connect: Optional[ConnectRequest] = None
    start: Optional[StartRequest] = None


# 응답 모델
class SpeechRecognitionResponse(BaseModel):
    code: int
    message: str
    result: Optional[Dict[str, Any]] = None
