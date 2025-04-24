import time
from typing import List, Tuple, Optional


class ProgressTracker:
    """
    음성 인식 진행 상황과 허용 인식 범위를 관리하는 클래스
    """
    
    def __init__(self, total_blocks: int, window_size: int = 3, time_based_advance: bool = True):
        """
        진행 추적기 초기화
        
        Args:
            total_blocks: 전체 블록 수
            window_size: 인식 윈도우 크기 (현재 블록 + 이전 N개 블록)
            time_based_advance: 시간 기반 자동 진행 활성화 여부
        """
        self.total_blocks = total_blocks
        self.window_size = window_size
        self.time_based_advance = time_based_advance
        
        self.current_index = 0
        self.start_time: Optional[float] = None
        self.last_advance_time: Optional[float] = None
        
        # 블록당 평균 소요 시간 (초) - 기본값, 실제 사용자 데이터로 조정 가능
        self.avg_time_per_block = 2.0  
        
        # 블록 자동 진행 최소 시간 (초)
        self.min_time_for_advance = 1.5
    
    def start(self) -> None:
        """진행 추적 시작"""
        self.start_time = time.time()
        self.last_advance_time = self.start_time
    
    def is_started(self) -> bool:
        """진행 추적 시작 여부 확인"""
        return self.start_time is not None
    
    def get_elapsed_time(self) -> float:
        """시작 이후 경과 시간 반환 (초)"""
        if not self.is_started():
            return 0.0
        return time.time() - self.start_time
    
    def get_time_since_last_advance(self) -> float:
        """마지막 진행 이후 경과 시간 반환 (초)"""
        if not self.last_advance_time:
            return 0.0
        return time.time() - self.last_advance_time
    
    def get_active_window(self) -> List[int]:
        """
        현재 활성 인식 윈도우 범위 반환
        현재 블록 + 이전 N개 블록의 인덱스 목록
        
        Returns:
            List[int]: 활성 윈도우 내 블록 인덱스 목록
        """
        start = max(0, self.current_index - self.window_size + 1)
        end = self.current_index + 1
        return list(range(start, end))
    
    def get_expected_block_index(self) -> int:
        """
        경과 시간 기준으로 예상되는 블록 인덱스 계산
        시간 기반 진행에 사용
        
        Returns:
            int: 예상 블록 인덱스
        """
        if not self.is_started():
            return 0
            
        elapsed = self.get_elapsed_time()
        expected_index = min(
            int(elapsed / self.avg_time_per_block),
            self.total_blocks - 1
        )
        return expected_index
    
    def should_advance(self) -> bool:
        """
        현재 블록에서 다음 블록으로 진행해야 하는지 확인
        시간 기반 진행 로직
        
        Returns:
            bool: 다음 블록으로 진행해야 하면 True
        """
        if not self.time_based_advance or not self.is_started():
            return False
            
        # 이미 마지막 블록이면 진행 불가
        if self.current_index >= self.total_blocks - 1:
            return False
            
        # 경과 시간 기준 예상 블록이 현재보다 앞서 있고,
        # 마지막 진행 이후 최소 시간이 지났으면 진행
        expected_index = self.get_expected_block_index()
        time_since_last = self.get_time_since_last_advance()
        
        return (expected_index > self.current_index and 
                time_since_last >= self.min_time_for_advance)
    
    def advance(self) -> bool:
        """
        다음 블록으로 진행
        
        Returns:
            bool: 진행 성공 여부
        """
        if self.current_index >= self.total_blocks - 1:
            return False
            
        self.current_index += 1
        self.last_advance_time = time.time()
        return True
    
    def set_current_index(self, index: int) -> bool:
        """
        현재 인덱스 직접 설정
        특정 블록으로 이동하거나 재설정할 때 사용
        
        Args:
            index: 설정할 인덱스
            
        Returns:
            bool: 설정 성공 여부
        """
        if 0 <= index < self.total_blocks:
            self.current_index = index
            self.last_advance_time = time.time()
            return True
        return False
    
    def reset(self) -> None:
        """진행 상태 초기화"""
        self.current_index = 0
        self.start_time = None
        self.last_advance_time = None
    
    def adjust_time_parameters(self, avg_time_per_block: float, min_time_for_advance: float) -> None:
        """
        시간 관련 파라미터 조정
        사용자 패턴에 맞게 동적 조정 가능
        
        Args:
            avg_time_per_block: 블록당 평균 소요 시간(초)
            min_time_for_advance: 블록 자동 진행 최소 시간(초)
        """
        if avg_time_per_block > 0:
            self.avg_time_per_block = avg_time_per_block
        if min_time_for_advance > 0:
            self.min_time_for_advance = min_time_for_advance
