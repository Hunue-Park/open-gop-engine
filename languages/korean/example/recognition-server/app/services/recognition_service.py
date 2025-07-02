import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import time
import threading

# 로거 설정
logger = logging.getLogger(__name__)

def load_pyrealtime_engine():
    """pyrealtime.so 파일을 직접 로드하여 EngineCoordinator 반환"""
    try:
        # recognition-server 디렉토리의 pyrealtime.so 파일 경로
        current_file = Path(__file__)  # app/services/recognition_service.py
        recognition_server_dir = current_file.parent.parent.parent  # recognition-server/
        so_file_path = recognition_server_dir / "pyrealtime.so"
        
        if not so_file_path.exists():
            raise FileNotFoundError(f"pyrealtime.so 파일을 찾을 수 없습니다: {so_file_path}")
        
        # sys.path에 pyrealtime.so가 있는 디렉토리 추가
        so_dir = str(so_file_path.parent)
        if so_dir not in sys.path:
            sys.path.insert(0, so_dir)
        
        # pyrealtime 모듈 직접 import
        import pyrealtime
        return pyrealtime.EngineCoordinator
        
    except Exception as e:
        print(f"❌ pyrealtime.so 로드 실패: {e}")
        raise ImportError(f"pyrealtime.so 로드 실패: {e}")

# EngineCoordinator 로드
EngineCoordinator = load_pyrealtime_engine()

class RecognitionService:
    def __init__(self):
        from app.core.config import settings
        
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
        result = self.engine.create_session(sentence, options=options or {})
        return result
    
    def evaluate_audio(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """오디오 데이터 평가"""
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
