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
    실시간 오디오 파일 모니터링 및 처리를 담당하는 클래스
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.5,  # 2.5초로 증가
        polling_interval: float = 0.1
    ):
        """
        오디오 프로세서 초기화
        
        Args:
            sample_rate: 목표 샘플링 레이트 (Hz)
            chunk_duration: 처리할 청크 단위 시간 (초)
            polling_interval: 파일 변경 확인 간격 (초)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.polling_interval = polling_interval
        
        # 파일 모니터링 상태
        self.audio_file_path: Optional[str] = None
        self.last_file_size: int = 0
        self.last_processed_pos: int = 0
        self.is_monitoring: bool = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 청크 처리를 위한 상태
        self.buffer: List[np.ndarray] = []
        self.last_chunk_time: Optional[float] = None
        self.total_duration: float = 0.0
        self.latest_chunk: Optional[torch.Tensor] = None
        
        # 이벤트 기반 메커니즘
        self.chunk_callbacks = []  # 청크 생성 시 호출할 콜백 함수 목록
        
    def set_audio_file(self, file_path: str) -> bool:
        """
        모니터링할 오디오 파일 설정
        
        Args:
            file_path: 오디오 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        if not os.path.exists(file_path):
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return False
            
        self.audio_file_path = file_path
        self.last_file_size = os.path.getsize(file_path)
        self.last_processed_pos = 0
        self.total_duration = 0.0
        self.buffer = []
        self.latest_chunk = None
        return True
        
    def start_monitoring(self) -> bool:
        """
        오디오 파일 모니터링 시작
        
        Returns:
            bool: 성공 여부
        """
        if self.is_monitoring:
            logger.warning("이미 모니터링 중입니다.")
            return False
            
        if not self.audio_file_path:
            logger.error("모니터링할 오디오 파일이 설정되지 않았습니다.")
            return False
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"오디오 파일 모니터링 시작: {self.audio_file_path}")
        return True
        
    def stop_monitoring(self) -> None:
        """모니터링 중지"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("오디오 파일 모니터링 중지")
        
    def _monitoring_loop(self) -> None:
        """
        오디오 파일 변경 모니터링 루프
        (내부 스레드에서 실행)
        """
        if not self.audio_file_path:
            return
            
        while self.is_monitoring:
            try:
                # 파일 크기 확인
                current_size = os.path.getsize(self.audio_file_path)
                
                # 파일 크기가 증가했으면 새 데이터 처리
                if current_size > self.last_file_size:
                    self._process_new_audio_data()
                    self.last_file_size = current_size
                    
                # 다음 확인까지 대기
                time.sleep(self.polling_interval)
                
            except Exception as e:
                logger.error(f"파일 모니터링 중 오류 발생: {e}")
                time.sleep(self.polling_interval)
                
    def _process_new_audio_data(self) -> None:
        """
        새로 추가된 오디오 데이터 처리
        """
        if not self.audio_file_path:
            return
            
        try:
            # 파일에서 새 데이터 읽기
            with sf.SoundFile(self.audio_file_path, 'r') as f:
                # 마지막 처리 위치로 이동
                f.seek(self.last_processed_pos)
                
                # 새 데이터 읽기
                frames = f.read()
                
                if len(frames) > 0:
                    # 새 데이터 처리
                    self._add_to_buffer(frames)
                    
                    # 처리 위치 업데이트
                    self.last_processed_pos = f.tell()
                    
        except Exception as e:
            logger.error(f"새 오디오 데이터 처리 중 오류 발생: {e}")
            
    def _add_to_buffer(self, audio_data: np.ndarray) -> None:
        """
        오디오 데이터를 버퍼에 추가
        
        Args:
            audio_data: 추가할 오디오 데이터 (numpy 배열)
        """
        # 데이터 정규화 (필요시)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
            
        # 채널 수 확인 (모노로 변환)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # 버퍼에 추가
        self.buffer.append(audio_data)
        
        # 총 녹음 시간 업데이트
        self.total_duration += len(audio_data) / self.sample_rate
        
        # 새 청크 생성 가능한지 확인
        self._check_and_process_chunks()
        
    def _check_and_process_chunks(self) -> None:
        """
        버퍼에서 청크 단위로 처리 가능한지 확인하고 처리
        """
        # 버퍼의 전체 샘플 수 계산
        # total_samples = sum(len(data) for data in self.buffer)
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        # 청크 추출
        chunk = self._extract_chunk(chunk_samples)
        
        # 청크 전처리 및 저장
        self.latest_chunk = self._preprocess_chunk(chunk)
        # 청크 타임스탬프 업데이트
        self.last_chunk_time = time.time()
        
        # # 처리된 샘플 수 업데이트
        # total_samples -= chunk_samples
        
        # 청크 생성 후 콜백 호출
        if self.latest_chunk is not None:
            metadata = {
                "timestamp": time.time(),
                "duration": self.chunk_duration,
                "total_duration": self.total_duration
            }
            for callback in self.chunk_callbacks:
                callback(self.latest_chunk, metadata)
            
        
    def _extract_chunk(self, chunk_samples: int) -> np.ndarray:
        """
        버퍼에서 지정된 크기의 청크 추출
        
        Args:
            chunk_samples: 추출할 샘플 수
            
        Returns:
            np.ndarray: 추출된 청크
        """
        result = np.array([], dtype=np.float32)
        samples_collected = 0
        
        while samples_collected < chunk_samples and self.buffer:
            buffer_data = self.buffer[0]
            samples_needed = chunk_samples - samples_collected
            
            if len(buffer_data) <= samples_needed:
                # 버퍼 데이터 전체 사용
                result = np.append(result, buffer_data)
                samples_collected += len(buffer_data)
                self.buffer.pop(0)
            else:
                # 버퍼 데이터 일부만 사용
                result = np.append(result, buffer_data[:samples_needed])
                self.buffer[0] = buffer_data[samples_needed:]
                samples_collected += samples_needed
                
        return result
        
    def _preprocess_chunk(self, chunk: np.ndarray, do_normalize: bool = True) -> torch.Tensor:
        """
        오디오 청크를 w2v_onnx_core와 호환되는 포맷으로 전처리
        
        Args:
            chunk: 처리할 오디오 청크
            do_normalize: 정규화 여부
            
        Returns:
            torch.Tensor: 처리된 오디오 텐서 [1, T]
        """
        # VAD 검사 - 음성이 없으면 None 반환
        if not self._detect_voice_activity(chunk):
            return None
        # 1) 모노화 (이미 모노인 경우 건너뜀)
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)
        
        # 2) 정규화
        if do_normalize:
            m = chunk.mean()
            s = chunk.std()
            chunk = (chunk - m) / (s + 1e-8)
        
        # 3) float32로 캐스팅
        chunk = chunk.astype(np.float32)
        
        # 4) torch.Tensor로 변환 및 배치 차원 추가
        tensor = torch.from_numpy(chunk).unsqueeze(0)  # shape: [1, T]
        
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
        
    def get_latest_chunk(self) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        가장 최근 처리된 청크와 메타데이터 반환
        w2v_onnx_core와 호환되는 형식으로 반환
        
        Returns:
            Tuple[Optional[torch.Tensor], Dict[str, Any]]: 
                (전처리된 청크 텐서[1,T], 메타데이터)
        """
        metadata = {
            "timestamp": time.time(),
            "duration": self.chunk_duration,
            "total_duration": self.total_duration
        }
        
        # 실제 청크가 없는 경우
        if self.latest_chunk is None:
            return None, metadata
            
        return self.latest_chunk, metadata
        
    def reset(self) -> None:
        """상태 초기화"""
        self.stop_monitoring()
        self.audio_file_path = None
        self.last_file_size = 0
        self.last_processed_pos = 0
        self.buffer = []
        self.total_duration = 0.0
        self.last_chunk_time = None
        self.latest_chunk = None
        
    def add_chunk_callback(self, callback):
        """청크 생성 시 호출할 콜백 함수 등록"""
        self.chunk_callbacks.append(callback)
