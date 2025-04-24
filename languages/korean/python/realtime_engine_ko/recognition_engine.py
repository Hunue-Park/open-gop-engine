import time
import logging
import threading
import json
from typing import Dict, Any, List, Optional, Callable, Union

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
        update_interval: float = 0.3,
        confidence_threshold: float = 0.7
    ):
        """
        엔진 코디네이터 초기화
        
        Args:
            onnx_model_path: ONNX 모델 파일 경로
            tokenizer_path: 토크나이저 파일 경로
            device: 추론 장치 ("CPU" 또는 "CUDA")
            update_interval: 결과 업데이트 간격 (초)
            confidence_threshold: 인식 신뢰도 임계값
        """
        # 인식 엔진 초기화
        self.recognition_engine = Wav2VecCTCOnnxCore(
            onnx_model_path=onnx_model_path,
            tokenizer_path=tokenizer_path,
            device=device
        )
        logger.info("RecognitionEngine 초기화 완료")
        
        # 나머지 컴포넌트는 필요시 초기화
        self.sentence_manager: Optional[SentenceBlockManager] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.eval_controller: Optional[EvaluationController] = None
        
        # 상태 관리
        self.is_initialized = False
        self.is_running = False
        self.update_interval = update_interval
        self.confidence_threshold = confidence_threshold
        self.timer_thread: Optional[threading.Thread] = None
        
        # RecordListener 관련
        self.record_listener: Optional[RecordListener] = None
        
        logger.info("EngineCoordinator 초기화 완료")
    
    def set_record_listener(self, record_listener: RecordListener) -> None:
        """
        RecordListener 설정
        
        Args:
            record_listener: 녹음 이벤트 처리 리스너
        """
        self.record_listener = record_listener
    
    def initialize(self, sentence: str, audio_polling_interval: float = 0.03, min_time_between_evals: float = 0.5) -> bool:
        """
        주어진 문장으로 시스템 초기화
        
        Args:
            sentence: 평가할 문장
            
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            # 문장 블록 관리자 초기화
            self.sentence_manager = SentenceBlockManager(sentence)
            
            # 진행 추적기 초기화
            self.progress_tracker = ProgressTracker(
                total_blocks=len(self.sentence_manager.blocks),
                window_size=3,
                time_based_advance=True
            )
            
            # 오디오 프로세서 초기화
            self.audio_processor = AudioProcessor(
                sample_rate=16000,
                chunk_duration=2.0,
                polling_interval=audio_polling_interval
            )
            
            # 평가 컨트롤러 초기화
            self.eval_controller = EvaluationController(
                recognition_engine=self.recognition_engine,
                sentence_manager=self.sentence_manager,
                progress_tracker=self.progress_tracker,
                confidence_threshold=self.confidence_threshold,
                min_time_between_evals=min_time_between_evals
            )
            
            # 오디오 처리 이벤트 등록
            self.audio_processor.add_chunk_callback(self._on_new_chunk)
            
            self.is_initialized = True
            logger.info(f"시스템 초기화 완료: '{sentence}' ({len(self.sentence_manager.blocks)} 블록)")
            
            return True
            
        except Exception as e:
            logger.error(f"초기화 오류: {e}")
            return False
    
    def start_evaluation(self, audio_file_path: str) -> bool:
        """
        평가 시작
        
        Args:
            audio_file_path: 모니터링할 오디오 파일 경로
            
        Returns:
            bool: 시작 성공 여부
        """
        if not self.is_initialized:
            error_msg = "초기화되지 않은 상태에서 평가를 시작할 수 없습니다."
            logger.error(error_msg)
            if self.record_listener and self.record_listener.on_start_record_fail:
                self.record_listener.on_start_record_fail(error_msg)
            return False
            
        if self.is_running:
            logger.warning("이미 평가가 진행 중입니다.")
            return False
        
        try:
            # 오디오 파일 설정
            if not self.audio_processor.set_audio_file(audio_file_path):
                error_msg = f"오디오 파일 설정 실패: {audio_file_path}"
                logger.error(error_msg)
                if self.record_listener and self.record_listener.on_start_record_fail:
                    self.record_listener.on_start_record_fail(error_msg)
                return False
                
            # 오디오 모니터링 시작
            if not self.audio_processor.start_monitoring():
                error_msg = "오디오 모니터링 시작 실패"
                logger.error(error_msg)
                if self.record_listener and self.record_listener.on_start_record_fail:
                    self.record_listener.on_start_record_fail(error_msg)
                return False
                
            # 진행 추적 시작
            self.progress_tracker.start()
            
            # 타이머 스레드 시작 (주기적 틱 이벤트용)
            self.is_running = True
            self.timer_thread = threading.Thread(
                target=self._timer_loop,
                daemon=True
            )
            self.timer_thread.start()
            
            # 시작 이벤트 호출
            if self.record_listener and self.record_listener.on_start:
                self.record_listener.on_start()
            
            logger.info("평가 시작")
            return True
            
        except Exception as e:
            error_msg = f"평가 시작 오류: {e}"
            logger.error(error_msg)
            if self.record_listener and self.record_listener.on_start_record_fail:
                self.record_listener.on_start_record_fail(error_msg)
            return False
    
    def stop_evaluation(self) -> None:
        """평가 중지"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=1.0)
            
        if self.audio_processor:
            self.audio_processor.stop_monitoring()
            
        # 종료 이벤트 호출
        if self.record_listener and self.record_listener.on_record_end:
            self.record_listener.on_record_end()
            
        logger.info("평가 중지")
    
    def _timer_loop(self) -> None:
        """
        주기적 틱 이벤트 생성 (내부 스레드에서 실행)
        """
        while self.is_running:
            try:
                # 진행 상태 확인
                if self.sentence_manager and self.record_listener and self.record_listener.on_tick:
                    current = self.sentence_manager.active_block_id + 1
                    total = len(self.sentence_manager.blocks)
                    self.record_listener.on_tick(current, total)
                
                # 다음 틱까지 대기
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"타이머 루프 오류: {e}")
                time.sleep(self.update_interval)
    
    def _on_new_chunk(self, audio_chunk, metadata):
        """새 오디오 청크 이벤트 핸들러"""
        try:
            if not self.is_running:
                return
                
            # 시간 기반 진행 확인 -> 사용하지 않음.
            # if self.progress_tracker.should_advance():
            #     self.sentence_manager.advance_active_block()
            #     self.progress_tracker.advance()
            #     logger.info(f"시간 기반 진행: 블록 {self.sentence_manager.active_block_id}")
            
            # 인식 결과 처리
            if audio_chunk is not None:
                result = self.eval_controller.process_recognition_result(
                    audio_chunk=audio_chunk,
                    metadata=metadata
                )
                
                # 결과 스코어 이벤트 호출
                if self.record_listener and self.record_listener.on_score:
                    # JSON 문자열로 변환 (SpeechSuper와 유사하게)
                    result_json = json.dumps(result)
                    self.record_listener.on_score(result_json)
                
        except Exception as e:
            logger.error(f"청크 처리 오류: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        현재 시스템 상태 정보 반환
        
        Returns:
            Dict[str, Any]: 현재 상태 정보
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        result = {
            "status": "running" if self.is_running else "stopped",
            "progress": {
                "current": self.sentence_manager.active_block_id + 1,
                "total": len(self.sentence_manager.blocks)
            }
        }
        
        # 평가 요약 정보 추가
        if self.eval_controller:
            result.update(self.eval_controller.get_evaluation_summary())
            
        return result
    
    def reset(self) -> None:
        """시스템 상태 초기화"""
        self.stop_evaluation()
        
        if self.sentence_manager:
            self.sentence_manager.reset()
            
        if self.progress_tracker:
            self.progress_tracker.reset()
            
        if self.eval_controller:
            self.eval_controller.reset()
            
        if self.audio_processor:
            self.audio_processor.reset()
            
        logger.info("시스템 초기화됨")
    
    # --- 외부 API 메서드 ---
    
    def evaluate_speech(self, sentence: str, audio_file_path: str, record_listener: Optional[RecordListener] = None) -> Dict[str, Any]:
        """
        음성 평가 시작 (편의 메서드)
        
        Args:
            sentence: 평가할 문장
            audio_file_path: 오디오 파일 경로
            record_listener: 선택적 리스너
            
        Returns:
            Dict[str, Any]: 초기 상태 정보
        """
        if record_listener:
            self.set_record_listener(record_listener)
            
        if not self.initialize(sentence):
            if self.record_listener and self.record_listener.on_start_record_fail:
                self.record_listener.on_start_record_fail("초기화 실패")
            return {"status": "initialization_failed"}
            
        if not self.start_evaluation(audio_file_path):
            if self.record_listener and self.record_listener.on_start_record_fail:
                self.record_listener.on_start_record_fail("평가 시작 실패")
            return {"status": "start_failed"}
            
        return self.get_current_state()
    
    def get_results(self) -> Dict[str, Any]:
        """
        현재 평가 결과 반환
        
        Returns:
            Dict[str, Any]: 평가 결과
        """
        return self.get_current_state()
