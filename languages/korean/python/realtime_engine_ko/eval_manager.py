import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np

from realtime_engine_ko.sentence_block import SentenceBlockManager, BlockStatus
from realtime_engine_ko.progress_tracker import ProgressTracker
from realtime_engine_ko.w2v_onnx_core import Wav2VecCTCOnnxCore

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EvaluationController")

class EvaluationController:
    """
    인식 결과와 블록 매핑 및 평가 제어를 담당하는 클래스
    """
    
    def __init__(
        self,
        recognition_engine: Wav2VecCTCOnnxCore,
        sentence_manager: SentenceBlockManager,
        progress_tracker: ProgressTracker,
        confidence_threshold: float = 10,
        min_time_between_evals: float = 0.1
    ):
        """
        평가 컨트롤러 초기화
        
        Args:
            recognition_engine: 음성 인식 엔진
            sentence_manager: 문장 블록 관리자
            progress_tracker: 진행 상황 추적기
            confidence_threshold: 인식 신뢰도 임계값
            min_time_between_evals: 블록 간 최소 평가 간격 (초)
        """
        self.recognition_engine = recognition_engine
        self.sentence_manager = sentence_manager
        self.progress_tracker = progress_tracker
        self.confidence_threshold = confidence_threshold
        self.min_time_between_evals = min_time_between_evals
        
        # 평가 상태 추적
        self.last_eval_time: Optional[float] = None
        self.pending_evaluations: Dict[int, Dict[str, Any]] = {}
        self.cached_results: Dict[int, Dict[str, Any]] = {}
        
    def process_recognition_result(
        self, 
        audio_chunk: torch.Tensor, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        음성 인식 결과를 처리하고 평가 진행
        
        Args:
            audio_chunk: 전처리된 오디오 청크 (torch.Tensor)
            metadata: 청크 메타데이터
            
        Returns:
            Dict[str, Any]: 평가 결과 및 상태 정보
        """
        # 활성 윈도우 내 블록 ID 목록 가져오기
        active_window = self.progress_tracker.get_active_window()
        
        # 오디오 청크가 없으면 현재 상태만 반환
        if audio_chunk is None:
            return self._create_result_format()
        
        # 활성 윈도우 내 모든 블록에 대해 매칭 시도
        best_match_id = None
        best_match_score = -float('inf')
        
        for block_id in active_window:
            block = self.sentence_manager.get_block(block_id)
            if not block:
                continue
                
            # 이미 평가된 블록은 건너뛰기
            if block.status == BlockStatus.EVALUATED:
                continue
            
            # 현재 블록의 컨텍스트 수집
            context_before = ""
            context_after = ""
            
            # 이전 블록들을 context_before로 수집 (최대 2개)
            prev_blocks = []
            for i in range(max(0, block_id-2), block_id):
                prev_block = self.sentence_manager.get_block(i)
                if prev_block:
                    prev_blocks.append(prev_block.text)
            context_before = " ".join(prev_blocks)
            
            # 다음 블록들을 context_after로 수집 (최대 2개)
            next_blocks = []
            for i in range(block_id+1, min(block_id+3, len(self.sentence_manager.blocks))):
                next_block = self.sentence_manager.get_block(i)
                if next_block:
                    next_blocks.append(next_block.text)
            context_after = " ".join(next_blocks)
            
            # 블록 텍스트로 GOP 계산 (컨텍스트 포함)
            try:
                gop_result = self.recognition_engine.calculate_gop_with_context(
                    audio_chunk, 
                    block.text,
                    context_before,
                    context_after,
                    # 컨텍스트 내 위치는 항상 0 (단독 블록 평가 시)
                    target_index=0 if not context_before else None
                )
                
                # 전체 발음 점수 추출
                overall_score = gop_result.get("overall", 0.0)
                
                # 현재 블록이 최적 매치인지 확인
                if overall_score > best_match_score:
                    best_match_score = overall_score
                    best_match_id = block_id
                    
                # 결과 캐싱
                self.cached_results[block_id] = {
                    "gop_score": overall_score,
                    "details": gop_result,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"블록 {block_id} GOP 계산 중 오류: {e}")
        # 최적 매치 블록을 찾았으면 해당 블록 평가 진행
        if best_match_id is not None and best_match_score >= self.confidence_threshold:
            # 평가 가능한 시점인지 확인
            current_time = time.time()
            if (self.last_eval_time is None or 
                current_time - self.last_eval_time >= self.min_time_between_evals):
                
                # 어떤 블록이든 매치된 블록 평가
                self._evaluate_block(best_match_id, self.cached_results[best_match_id])
                self.last_eval_time = current_time
                
                # 활성 블록 업데이트 (케이스별 처리)
                if best_match_id == self.sentence_manager.active_block_id:
                    # 1. 현재 활성 블록이 인식된 경우 - 다음 블록으로 진행
                    self.sentence_manager.advance_active_block()
                    
                    # ProgressTracker 업데이트
                    self.progress_tracker.set_current_index(self.sentence_manager.active_block_id)
                    
                elif best_match_id < self.sentence_manager.active_block_id:
                    # 2. 이전 블록이 인식된 경우 (순서가 뒤바뀐 발화)
                    logger.info(f"이전 블록 {best_match_id}가 인식됨 (현재 활성 블록: {self.sentence_manager.active_block_id})")
                    
                    # 활성 블록 상태 업데이트
                    self.sentence_manager.set_active_block(best_match_id)
                    
                    # 평가 후에는 다음 블록으로 이동
                    self.sentence_manager.advance_active_block()
                    
                    # ProgressTracker 업데이트
                    self.progress_tracker.set_current_index(self.sentence_manager.active_block_id)
                    
                elif best_match_id > self.sentence_manager.active_block_id:
                    # 3. 다음 블록이 인식된 경우 (블록을 건너뛴 경우)
                    logger.info(f"건너뛴 블록 {best_match_id}가 인식됨 (현재 활성 블록: {self.sentence_manager.active_block_id})")
                    
                    # 중간 블록들 처리 (선택적으로 추가 가능)
                    # 여기서는 중간 블록들을 넘어가고, 바로 매치된 블록으로 이동
                    
                    # 활성 블록을 인식된 블록 다음으로 설정
                    self.sentence_manager.set_active_block(best_match_id + 1)
                    if best_match_id + 1 >= len(self.sentence_manager.blocks):
                        # 마지막 블록이면 마지막 블록을 활성 상태로 유지
                        self.sentence_manager.set_active_block(best_match_id)
                    
                    # ProgressTracker 업데이트
                    self.progress_tracker.set_current_index(self.sentence_manager.active_block_id)
        
        # 새 형식으로 결과 반환
        return self._create_result_format()
    
    def _create_result_format(self) -> Dict[str, Any]:
        """
        요청된 형식에 맞게 결과 생성
        
        Returns:
            Dict[str, Any]: 형식화된 결과
        """
        evaluated_blocks = [b for b in self.sentence_manager.blocks 
                           if b.status == BlockStatus.EVALUATED]
        
        # 평가된 블록이 없으면 빈 결과 반환
        if not evaluated_blocks:
            return {
                "result": {
                    "overall": 0.0,
                    "pronunciation": 0.0,
                    "resource_version": "1.0.0",
                    "words": [],
                    "eof": False
                }
            }
        
        # 평균 점수 계산
        avg_score = sum(b.gop_score for b in evaluated_blocks if b.gop_score is not None) / len(evaluated_blocks)
        avg_score_rounded = round(avg_score, 1)
        
        # 단어별 점수 구성
        words = []
        for block in evaluated_blocks:
            if block.gop_score is not None:
                words.append({
                    "word": block.text,
                    "scores": {
                        "pronunciation": round(block.gop_score, 1)
                    }
                })
        
        # 모든 블록이 평가 완료되었는지 확인
        all_blocks_evaluated = len(evaluated_blocks) == len(self.sentence_manager.blocks)
        
        # 기본 결과 구조
        result = {
            "result": {
                "overall": avg_score_rounded,
                "pronunciation": avg_score_rounded,
                "resource_version": "1.0.0",
                "words": words,
                "eof": False
            }
        }
        
        # 모든 블록이 평가 완료된 경우 추가 정보 포함
        if all_blocks_evaluated:
            result["result"]["eof"] = True
            result["result"]["final_score"] = avg_score_rounded
            
            # 강화된 결과 데이터 추가
            result["result"]["details"] = {
                "total_blocks": len(self.sentence_manager.blocks),
                "completion_time": time.time(),
                "score_breakdown": {
                    "min_score": round(min((b.gop_score for b in evaluated_blocks if b.gop_score is not None), default=0), 1),
                    "max_score": round(max((b.gop_score for b in evaluated_blocks if b.gop_score is not None), default=0), 1)
                }
            }
        
        return result
    
    def _evaluate_block(self, block_id: int, evaluation_data: Dict[str, Any]) -> None:
        """
        블록 평가 결과 적용
        
        Args:
            block_id: 평가할 블록 ID
            evaluation_data: 평가 데이터
        """
        block = self.sentence_manager.get_block(block_id)
        if not block:
            return
            
        # 상태 업데이트
        if block.status == BlockStatus.PENDING or block.status == BlockStatus.ACTIVE:
            self.sentence_manager.update_block_status(block_id, BlockStatus.RECOGNIZED)
            
        # GOP 점수 설정
        self.sentence_manager.set_block_score(block_id, evaluation_data["gop_score"])
        
        # 확신도 설정 (있는 경우)
        # if "confidence" in evaluation_data:
        #     block.set_confidence(evaluation_data["confidence"])

        self.sentence_manager.update_block_status(block_id, BlockStatus.EVALUATED)
        
        logger.info(f"블록 {block_id} ({block.text}) 평가 완료: 점수={block.gop_score}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        현재까지의 평가 요약 정보 반환
        
        Returns:
            Dict[str, Any]: 평가 요약 정보
        """
        evaluated_blocks = [b for b in self.sentence_manager.blocks 
                           if b.status == BlockStatus.EVALUATED]
        
        # 평가된 블록이 없으면 빈 요약 반환
        if not evaluated_blocks:
            return {
                "overall_score": 0.0,
                "progress": {
                    "completed": 0,
                    "total": len(self.sentence_manager.blocks)
                },
                "blocks": []
            }
        
        # 평균 점수 계산
        avg_score = sum(b.gop_score for b in evaluated_blocks if b.gop_score is not None) / len(evaluated_blocks)
        
        return {
            "overall_score": round(avg_score, 1),
            "progress": {
                "completed": len(evaluated_blocks),
                "total": len(self.sentence_manager.blocks)
            },
            "blocks": [b.to_dict() for b in self.sentence_manager.blocks]
        }
    
    def reset(self) -> None:
        """
        평가 상태 초기화
        """
        self.last_eval_time = None
        self.pending_evaluations.clear()
        self.cached_results.clear()
