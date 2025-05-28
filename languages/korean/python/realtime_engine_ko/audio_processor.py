import os
import time
import numpy as np
import soundfile as sf
import torch
from typing import Optional, Tuple, Dict, Any, List
import threading
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AudioProcessor")

class AudioProcessor:
    """
    오디오 데이터 처리를 담당하는 클래스 (바이너리 입력 지원)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_buffer_seconds: float = 10.0
    ):
        """
        오디오 프로세서 초기화
        
        Args:
            sample_rate: 목표 샘플링 레이트 (Hz)
            max_buffer_seconds: 최대 버퍼 길이 (초)
        """
        self.sample_rate = sample_rate
        self.max_buffer_length = int(sample_rate * max_buffer_seconds)
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # 처리 상태 추적
        self.last_process_time: Optional[float] = None
        self.total_duration: float = 0.0
        self.latest_chunk: Optional[torch.Tensor] = None
        
    def process_audio_binary(self, binary_data) -> Optional[torch.Tensor]:
        """
        바이너리 오디오 데이터 처리
        
        Args:
            binary_data: 바이너리 오디오 데이터 (일반적으로 int16 형식)
            
        Returns:
            Optional[torch.Tensor]: 처리된 오디오 텐서 또는 None
        """
        try:
            # 먼저 int16으로 가정하고 처리 (PyAudio 기본 포맷)
            audio_data = np.frombuffer(binary_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 버퍼에 추가
            self.audio_buffer = np.append(self.audio_buffer, audio_data)
            
            # 최대 길이 제한
            if len(self.audio_buffer) > self.max_buffer_length:
                self.audio_buffer = self.audio_buffer[-self.max_buffer_length:]
            
            # 전체 버퍼에 대한 전처리 및 반환
            return self._preprocess_audio_data(self.audio_buffer)
            
        except Exception as e:
            logger.error(f"오디오 바이너리 처리 중 오류 발생: {e}")
            return None
    
    def _preprocess_audio_data(self, audio_data: np.ndarray, do_normalize: bool = True) -> Optional[torch.Tensor]:
        """
        오디오 데이터 직접 전처리
        
        Args:
            audio_data: numpy 배열 형태의 오디오 데이터
            do_normalize: 정규화 여부
            
        Returns:
            Optional[torch.Tensor]: 전처리된 오디오 텐서, 유효하지 않으면 None
        """
        # VAD 검사 - 음성이 없으면 None 반환
        if not self._detect_voice_activity(audio_data):
            return None
            
        # 1) 모노화 (이미 모노인 경우 건너뜀)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 2) 정규화
        if do_normalize:
            m = audio_data.mean()
            s = audio_data.std()
            audio_data = (audio_data - m) / (s + 1e-8)
        
        # 3) float32로 캐스팅
        audio_data = audio_data.astype(np.float32)
        
        # 4) torch.Tensor로 변환 및 배치 차원 추가
        tensor = torch.from_numpy(audio_data).unsqueeze(0)  # shape: [1, T]
        
        return tensor
    
    def _detect_voice_activity(self, audio_data: np.ndarray, 
                            energy_threshold: float = 0.0005,
                            min_speech_frames: int = 10) -> bool:
        """
        간단한 에너지 기반 VAD 구현
        
        Args:
            audio_data: 오디오 데이터 (numpy 배열)
            energy_threshold: 음성으로 판단할 에너지 임계값
            min_speech_frames: 음성으로 판단할 최소 프레임 수
            
        Returns:
            bool: 음성이 있는지 여부
        """
        # 모노 데이터로 변환
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # 프레임 단위로 분할 (10ms 프레임)
        frame_size = int(self.sample_rate * 0.01)
        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        
        # 각 프레임의 에너지 계산
        energies = [np.sum(frame**2) / len(frame) for frame in frames if len(frame) == frame_size]
        
        # 임계값을 넘는 프레임 수 계산
        speech_frames = sum(1 for energy in energies if energy > energy_threshold)
        
        # 로깅 (디버깅용)
        avg_energy = np.mean(energies) if energies else 0
        logger.debug(f"VAD: 평균 에너지={avg_energy:.6f}, 음성 프레임={speech_frames}/{len(energies)}")
        
        # 임계값을 넘는 프레임이 충분한지 확인
        return speech_frames >= min_speech_frames
    
    def reset(self) -> None:
        """상태 초기화"""
        self.total_duration = 0.0
        self.last_process_time = None
        self.latest_chunk = None
