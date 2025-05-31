from typing import Dict, Any, Optional
from realtime_engine_ko.recognition_engine import EngineCoordinator
from app.core.config import settings

class RecognitionService:
    def __init__(self):
        self.engine = EngineCoordinator(
            onnx_model_path=settings.ONNX_MODEL_PATH,
            tokenizer_path=settings.TOKENIZER_PATH,
            device=settings.DEVICE,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD
        )
    
    def create_session(self, sentence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """새로운 인식 세션 생성"""
        return self.engine.create_session(sentence, engine_options=options or {})
    
    def evaluate_audio(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """오디오 데이터 평가"""
        return self.engine.evaluate_audio(session_id, audio_data)
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        return self.engine.get_session_status(session_id)
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """세션 종료"""
        return self.engine.close_session(session_id)
