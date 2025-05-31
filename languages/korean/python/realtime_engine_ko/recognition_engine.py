import time
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np

from realtime_engine_ko.sentence_block import SentenceBlockManager, BlockStatus
from realtime_engine_ko.progress_tracker import ProgressTracker
from realtime_engine_ko.audio_processor import AudioProcessor
from realtime_engine_ko.w2v_onnx_core import Wav2VecCTCOnnxCore
from realtime_engine_ko.eval_manager import EvaluationController

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EngineCoordinator")

class RecordListener:
    """SpeechSuper OnRecordListener와 유사한 인터페이스"""
    
    def __init__(
        self, 
        on_start=None,         # 녹음 시작
        on_tick=None,          # 진행 틱
        on_start_record_fail=None,  # 녹음 시작 실패 
        on_record_end=None,    # 녹음 종료
        on_score=None          # 평가 결과
    ):
        self.on_start = on_start
        self.on_tick = on_tick
        self.on_start_record_fail = on_start_record_fail
        self.on_record_end = on_record_end
        self.on_score = on_score

class EngineCoordinator:
    """
    음성 인식 엔진의 전체 컴포넌트를 조율하는 최상위 클래스
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        tokenizer_path: str,
        device: str = "CPU",
        confidence_threshold: float = 0.7
    ):
        """
        엔진 코디네이터 초기화
        
        Args:
            onnx_model_path: ONNX 모델 파일 경로
            tokenizer_path: 토크나이저 파일 경로
            device: 추론 장치 ("CPU" 또는 "CUDA")
            confidence_threshold: 인식 신뢰도 임계값
        """
        # 인식 엔진 초기화
        self.recognition_engine = Wav2VecCTCOnnxCore(
            onnx_model_path=onnx_model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
        logger.info("RecognitionEngine 초기화 완료")
        
        # 세션 관리
        self.sessions = {}
        
        # 설정값
        self.confidence_threshold = confidence_threshold
        
        logger.info("EngineCoordinator 초기화 완료")
    
    def create_session(self, sentence: str, engine_options: Dict[str, Any] = {}) -> Dict[str, Any]:
        """
        평가 세션 생성 (문장 초기화)
        
        Args:
            sentence: 평가할 문장
            engine_options: 엔진 옵션 (예: {"confidence_threshold": 0.7})
        Returns:
            Dict[str, Any]: 세션 정보
        """
        session_id = str(uuid.uuid4())
        
        # 엔진 옵션 처리
        confidence_threshold = engine_options.get("confidence_threshold", self.confidence_threshold)
        min_time_between_evals = engine_options.get("min_time_between_evals", 0.5)
        
        # 세션별 컴포넌트 초기화
        sentence_manager = SentenceBlockManager(sentence)
        progress_tracker = ProgressTracker(
            total_blocks=len(sentence_manager.blocks),
            window_size=3,
            time_based_advance=False
        )
        audio_processor = AudioProcessor(
            sample_rate=16000
        )
        eval_controller = EvaluationController(
            recognition_engine=self.recognition_engine,
            sentence_manager=sentence_manager,
            progress_tracker=progress_tracker,
            confidence_threshold=confidence_threshold,
            min_time_between_evals=min_time_between_evals
        )
        
        # 세션 정보 저장
        self.sessions[session_id] = {
            "sentence_manager": sentence_manager,
            "progress_tracker": progress_tracker,
            "audio_processor": audio_processor,
            "eval_controller": eval_controller,
            "created_at": time.time(),
            "last_activity": time.time()
        }
        
        # 진행 추적 시작
        progress_tracker.start()
        
        return {
            "session_id": session_id,
            "status": "initialized",
            "sentence": sentence,
            "blocks": len(sentence_manager.blocks)
        }
    
    def evaluate_audio(self, session_id: str, binary_data: bytes) -> Dict[str, Any]:
        """
        오디오 바이너리 데이터 평가
        
        Args:
            session_id: 세션 ID
            binary_data: 바이너리 오디오 데이터
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        # 세션 확인
        if session_id not in self.sessions:
            return {"error": "invalid_session", "message": "세션이 존재하지 않거나 만료되었습니다"}
        
        session = self.sessions[session_id]
        session["last_activity"] = time.time()
        
        # 오디오 데이터 처리 - 바이너리 형식 감지 및 처리는 AudioProcessor에서 담당
        audio_tensor = session["audio_processor"].process_audio_binary(binary_data)
        
        if audio_tensor is None:
            return {
                "session_id": session_id,
                "status": "no_valid_audio",
                "result": self._create_empty_result()
            }
        
        # 평가 진행
        result = session["eval_controller"].process_recognition_result(
            audio_chunk=audio_tensor,
            metadata={"timestamp": time.time()}
        )
        
        # 진행 상태 정보 추가
        result["session_id"] = session_id
        result["status"] = "in_progress"
        
        # 모든 블록 평가 완료 확인
        all_blocks_evaluated = session["eval_controller"].are_all_blocks_evaluated()
        if all_blocks_evaluated:
            result["status"] = "completed"
        
        return result
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """
        세션 종료 및 정리
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Dict[str, Any]: 결과 상태
        """
        if session_id in self.sessions:
            # 세션 정리
            del self.sessions[session_id]
            return {"status": "session_closed", "session_id": session_id}
        return {"error": "invalid_session"}
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """빈 결과 객체 생성"""
        return {
            "overall": 0.0,
            "pronunciation": 0.0,
            "resource_version": "1.0.0",
            "words": [],
            "eof": False
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        세션 상태 조회
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Dict[str, Any]: 세션 상태 정보
        """
        if session_id not in self.sessions:
            return {"error": "invalid_session", "message": "세션이 존재하지 않거나 만료되었습니다"}
        
        session = self.sessions[session_id]
        
        # 각 컴포넌트에서 상태 정보 수집
        summary = session["eval_controller"].get_evaluation_summary()
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "current_progress": {
                "active_block": session["sentence_manager"].active_block_id,
                "total_blocks": len(session["sentence_manager"].blocks),
                "overall_score": summary["overall_score"]
            },
            "all_completed": session["eval_controller"].are_all_blocks_evaluated()
        }
    
    def cleanup_inactive_sessions(self, max_inactive_time: float = 3600.0) -> int:
        """
        비활성 세션 정리
        
        Args:
            max_inactive_time: 최대 비활성 시간 (초, 기본값: 1시간)
            
        Returns:
            int: 정리된 세션 수
        """
        current_time = time.time()
        inactive_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_activity"] > max_inactive_time:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            self.close_session(session_id)
        
        return len(inactive_sessions)
