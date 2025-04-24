from enum import Enum
from typing import List, Optional, Dict, Any
import time

class BlockStatus(Enum):
    """블록의 상태를 나타내는 열거형"""
    PENDING = "pending"       # 아직 처리되지 않음
    ACTIVE = "active"         # 현재 활성화됨
    RECOGNIZED = "recognized" # 인식됨
    EVALUATED = "evaluated"   # 평가 완료됨


class SentenceBlock:
    """문장의 개별 블록(단어 또는 구)을 나타내는 클래스"""
    
    def __init__(self, text: str, block_id: int):
        self.text = text
        self.block_id = block_id
        self.status = BlockStatus.PENDING
        self.gop_score: Optional[float] = None
        self.confidence: Optional[float] = None
        self.recognized_at: Optional[float] = None  # 인식된 시간
        self.evaluated_at: Optional[float] = None   # 평가된 시간
    
    def set_status(self, status: BlockStatus) -> None:
        """블록 상태 변경"""
        self.status = status
    
    def set_score(self, score: float) -> None:
        """GOP 점수 설정"""
        self.gop_score = score
    
    def set_confidence(self, confidence: float) -> None:
        """인식 신뢰도 점수 설정"""
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """블록 정보를 사전 형태로 변환"""
        return {
            "text": self.text,
            "block_id": self.block_id,
            "status": self.status.value,
            "gop_score": self.gop_score,
            "confidence": self.confidence,
            "recognized_at": self.recognized_at,
            "evaluated_at": self.evaluated_at
        }


class SentenceBlockManager:
    """문장 블록을 관리하는 클래스"""
    
    def __init__(self, sentence: str, delimiter: str = " "):
        """
        문장을 받아 블록으로 분할하여 초기화
        
        Args:
            sentence: 분할할 전체 문장
            delimiter: 블록 분할 기준 (기본값: 공백)
        """
        self.blocks: List[SentenceBlock] = []
        self.active_block_id: int = 0
        
        # 문장을 블록으로 분할
        blocks_text = sentence.split(delimiter)
        for i, block_text in enumerate(blocks_text):
            if block_text.strip():  # 빈 블록 제외
                self.blocks.append(SentenceBlock(block_text.strip(), i))
        
        # 첫 번째 블록은 기본적으로 ACTIVE 상태로 설정
        if self.blocks:
            self.blocks[0].set_status(BlockStatus.ACTIVE)
    
    def get_block(self, block_id: int) -> Optional[SentenceBlock]:
        """특정 ID의 블록 반환"""
        if 0 <= block_id < len(self.blocks):
            return self.blocks[block_id]
        return None
    
    def get_active_block(self) -> Optional[SentenceBlock]:
        """현재 활성 블록 반환"""
        return self.get_block(self.active_block_id)
    
    def set_active_block(self, block_id: int) -> bool:
        """
        활성 블록 변경
        
        Args:
            block_id: 활성화할 블록 ID
            
        Returns:
            bool: 성공 여부
        """
        if 0 <= block_id < len(self.blocks):
            # 이전 활성 블록 상태 변경
            current_active = self.get_active_block()
            if current_active and current_active.status == BlockStatus.ACTIVE:
                current_active.set_status(BlockStatus.PENDING)
            
            # 새 활성 블록 설정
            self.active_block_id = block_id
            self.blocks[block_id].set_status(BlockStatus.ACTIVE)
            return True
        return False
    
    def advance_active_block(self) -> bool:
        """다음 블록으로 활성 블록 이동"""
        next_id = self.active_block_id + 1
        return self.set_active_block(next_id)
    
    def get_window(self, window_size: int = 3) -> List[SentenceBlock]:
        """
        현재 활성 블록 주변의 윈도우 반환
        
        Args:
            window_size: 윈도우 크기 (기본값: 3)
            
        Returns:
            List[SentenceBlock]: 활성 블록 + 이전 블록들
        """
        start = max(0, self.active_block_id - window_size + 1)
        end = self.active_block_id + 1
        return self.blocks[start:end]
    
    def update_block_status(self, block_id: int, status: BlockStatus) -> bool:
        """
        특정 블록의 상태 업데이트
        
        Args:
            block_id: 업데이트할 블록 ID
            status: 새 상태
            
        Returns:
            bool: 성공 여부
        """
        block = self.get_block(block_id)
        if block:
            block.set_status(status)
            if status == BlockStatus.RECOGNIZED:
                block.recognized_at = time.time()
            if status == BlockStatus.EVALUATED:
                block.evaluated_at = time.time()
            return True
        return False
    
    def set_block_score(self, block_id: int, score: float) -> bool:
        """
        블록의 GOP 점수 설정
        
        Args:
            block_id: 점수를 설정할 블록 ID
            score: GOP 점수
            
        Returns:
            bool: 성공 여부
        """
        block = self.get_block(block_id)
        if block:
            block.set_score(score)
            return True
        return False
    
    def get_all_blocks_status(self) -> List[Dict[str, Any]]:
        """모든 블록의 상태 정보 반환"""
        return [block.to_dict() for block in self.blocks]
    
    def reset(self) -> None:
        """모든 블록 상태 초기화"""
        for block in self.blocks:
            block.set_status(BlockStatus.PENDING)
            block.gop_score = None
            block.confidence = None
            block.recognized_at = None
            block.evaluated_at = None
        
        # 첫 번째 블록을 활성화
        if self.blocks:
            self.active_block_id = 0
            self.blocks[0].set_status(BlockStatus.ACTIVE)
