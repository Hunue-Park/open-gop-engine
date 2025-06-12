import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from realtime_engine_ko.recognition_engine import EngineCoordinator
from app.core.config import settings
from pathlib import Path
import time
import threading

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
        
        # 단순한 세션 타임아웃만
        self._start_simple_cleanup()
    
    def _start_simple_cleanup(self):
        """10분마다 오래된 세션 정리"""
        def cleanup():
            while True:
                try:
                    self.engine.cleanup_inactive_sessions(600)  # 10분
                except:
                    pass
                time.sleep(600)  # 10분마다
        threading.Thread(target=cleanup, daemon=True).start()
    
    def create_session(self, sentence: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """새로운 인식 세션 생성"""
        result = self.engine.create_session(sentence, engine_options=options or {})
        return result
    
    def evaluate_audio(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """오디오 데이터 평가"""
            
        # 원래 평가 로직 실행
        result = self.engine.evaluate_audio(session_id, audio_data)
        return result
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        return self.engine.get_session_status(session_id)
    
    def close_session(self, session_id: str):
        try:
            return self.engine.close_session(session_id)
        except Exception as e:
            # 실패해도 성공으로 처리 (이미 정리되었을 수 있음)
            return {"status": "session_closed", "session_id": session_id}
