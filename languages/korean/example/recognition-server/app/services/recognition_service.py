import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from realtime_engine_ko.recognition_engine import EngineCoordinator
from app.core.config import settings
from pathlib import Path

# 로거 설정
logger = logging.getLogger(__name__)

class RecognitionService:
    def __init__(self):
        self.engine = EngineCoordinator(
            onnx_model_path=settings.ONNX_MODEL_PATH,
            tokenizer_path=settings.TOKENIZER_PATH,
            device=settings.DEVICE,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD
        )
        # 오디오 파일 저장 디렉토리 설정
        # self.audio_dir = "/app/open-gop-engine/languages/korean/example/recognition-server/recordings/debug"
        # logger.info(f"Audio directory set to: {self.audio_dir}")
        
        # try:
        #     os.makedirs(self.audio_dir, exist_ok=True)
        #     logger.info(f"Successfully created audio directory")
        # except Exception as e:
        #     logger.error(f"Failed to create audio directory: {e}")
        
        # # 세션별 오디오 카운터
        # self.session_counters: Dict[str, int] = {}
    
    def create_session(self, sentence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """새로운 인식 세션 생성"""
        result = self.engine.create_session(sentence, engine_options=options or {})
        return result
    
    def evaluate_audio(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """오디오 데이터 평가"""
            
        # 원래 평가 로직 실행
        result = self.engine.evaluate_audio(session_id, audio_data)
        logger.info("Successfully completed audio evaluation")
        return result
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        return self.engine.get_session_status(session_id)
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """세션 종료"""
        # 세션 카운터 정리
        if session_id in self.session_counters:
            del self.session_counters[session_id]
        return self.engine.close_session(session_id)
