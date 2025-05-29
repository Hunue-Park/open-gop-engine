import os
import time
import threading
import json
import pyaudio
import wave
import pprint
import numpy as np

from realtime_engine_ko import EngineCoordinator

class RealtimeAudioProcessor:
    def __init__(self, engine, output_dir=None):
        self.engine = engine
        self.output_dir = output_dir
        self.session_id = None
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.frames_buffer = []
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.current_result = None
        self.session_text = None
        self.save_to_file = bool(output_dir)

    def start_session(self, text):
        """새 평가 세션 시작"""
        config_options = {
            "min_time_between_evals": 0.5,
            "confidence_threshold": 30.0
        }
        response = self.engine.create_session(text, config_options)
        self.session_id = response["session_id"]
        self.session_text = text
        self.current_result = None
        print(f"세션 생성됨: {self.session_id}")
        return response

    def start_recording(self):
        """녹음 시작 및 처리 스레드 실행"""
        if self.is_recording:
            return
            
        self.frames_buffer = []
        self.is_recording = True
        self.stop_event.clear()
        
        # 파일 저장 시 사용할 경로 생성 (선택적)
        self.current_file = None
        if self.save_to_file and self.output_dir:
            timestamp = int(time.time())
            self.current_file = os.path.join(
                self.output_dir, 
                f"recorded_audio_{timestamp}.wav"
            )
        
        # 오디오 스트림 시작
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        self.stream.start_stream()
        
        # 오디오 처리 스레드 시작
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("녹음 및 실시간 처리 시작...")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """오디오 데이터 콜백 - 버퍼에 추가"""
        if self.is_recording:
            self.frames_buffer.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """주기적으로 오디오 청크를 처리하고 평가 결과 획득"""
        counter = 0
        total_processed_frames = 0
        
        while not self.stop_event.is_set() and self.is_recording:
            if self.frames_buffer and self.session_id:
                counter += 1
                
                # 현재 버퍼의 프레임만 처리하기 위해 복사 후 비우기
                # 스레드 안전성을 위해 임시 변수에 복사
                current_frames = self.frames_buffer.copy()
                # 버퍼 비우기 - 중요!
                self.frames_buffer = []
                
                # 현재 처리할 오디오 정보 표시
                frame_count = len(current_frames)
                total_processed_frames += frame_count
                
                print(f"[처리 #{counter}] 현재 처리: {frame_count} 프레임, 총 처리됨: {total_processed_frames} 프레임")
                
                # 오디오 바이너리 데이터 생성
                audio_binary = b''.join(current_frames)
                binary_size = len(audio_binary)
                
                # 오디오 길이 계산 (16000Hz, 16bit)
                audio_seconds = binary_size / (16000 * 2)
                print(f"[처리 #{counter}] 오디오 길이: {audio_seconds:.2f}초, 크기: {binary_size} 바이트")
                
                # 엔진 호출 전
                start_time = time.time()
                
                # 엔진 호출
                result = self.engine.evaluate_audio(self.session_id, audio_binary)
                
                # 엔진 호출 후
                end_time = time.time()
                print(f"[엔진 호출] 소요 시간: {end_time - start_time:.3f}초")
                
                self.current_result = result
                
                # 결과 상태 확인
                print(f"[처리 #{counter}] 결과 상태: {result.get('status', 'unknown')}")
                
                # 결과 표시
                self._display_result(result)
                
                # 결과 분석
                if "result" in result:
                    word_count = len(result["result"].get("words", []))
                    print(f"[엔진 응답] 인식된 단어 수: {word_count}")
                    if "error" in result:
                        print(f"[엔진 오류] {result['error']}: {result.get('message', '')}")
            
            # 처리 주기 (0.3초)
            time.sleep(0.3)
    
    def _display_result(self, result):
        """평가 결과 표시"""
        if not result:
            return
            
        status = result.get("status", "")
        
        # 진행 상태 출력
        if "result" in result:
            words = result["result"].get("words", [])
            total_blocks = len(self.session_text.split())
            progress = f"{len(words)}/{total_blocks}"
            score = result["result"].get("overall", 0)
            pprint.pprint(result)
            
            # print(f"\r상태: {status} | 진행: {progress} | 현재 점수: {score}", end="")
            
            # 완료된 경우 최종 결과 표시
            if status == "completed":
                print("\n\n===== 최종 평가 결과 =====")
                print(f"전체 점수: {score}")
                print("\n단어별 점수:")
                for word in words:
                    print(f"- {word['word']}: {word['scores']['pronunciation']}")
                print("=============================\n")
    
    def stop_recording(self):
        """녹음 중지 및 세션 종료"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.stop_event.set()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # 오디오 파일 저장 (선택적)
        if self.save_to_file and self.current_file and self.frames_buffer:
            self._save_audio_file()
        
        # 최종 평가 결과 처리
        if self.session_id and self.current_result:
            final_result = self.engine.get_session_status(self.session_id)
            print("\n녹음 종료 - 최종 평가 상태")
            self._display_result(self.current_result)
        
        # 세션 종료
        if self.session_id:
            self.engine.close_session(self.session_id)
            print(f"세션 종료: {self.session_id}")
            self.session_id = None
        
        print("\n녹음 및 평가 종료")
    
    def _save_audio_file(self):
        """녹음된 오디오를 파일로 저장"""
        if not self.current_file or not self.frames_buffer:
            return
            
        # 저장 직전 로깅
        total_frames = len(self.frames_buffer)
        total_bytes = sum(len(frame) for frame in self.frames_buffer)
        total_seconds = total_bytes / (16000 * 2)  # 16000Hz, 16bit(2바이트)
        
        print(f"\n저장 정보: {total_frames} 프레임, {total_bytes} 바이트, 약 {total_seconds:.2f}초")
        
        try:
            wf = wave.open(self.current_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            
            audio_data = b''.join(self.frames_buffer)
            print(f"실제 저장 크기: {len(audio_data)} 바이트")
            
            wf.writeframes(audio_data)
            wf.close()
            print(f"녹음 파일 저장됨: {self.current_file}")
        except Exception as e:
            print(f"파일 저장 오류: {e}")
    
    def __del__(self):
        """소멸자: 모든 자원 정리"""
        self.stop_event.set()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        if self.session_id:
            try:
                self.engine.close_session(self.session_id)
            except:
                pass
        self.audio.terminate()


def main():
    # 프로젝트 루트 경로 계산
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # 설정
    MODEL_PATH = os.path.join(BASE_DIR, "models/wav2vec2_ctc_combined.onnx")
    TOKENIZER_PATH = os.path.join(BASE_DIR, "models/tokenizer.json")
    MATRIX_PATH = os.path.join(BASE_DIR, "models/wav2vec2_ctc_combined")
    REFERENCE_TEXT = "안녕하세요 제 이름은 박현우 입니다. 만나서 반갑습니다."
    OUTPUT_DIR = os.path.join(BASE_DIR, "recordings")
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 엔진 초기화
    print("엔진 초기화 중...")
    engine = EngineCoordinator(
        onnx_model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        confidence_threshold=30,  # 약간 낮은 임계값 설정
        # matrix_path=MATRIX_PATH   # 추가: 매트릭스 경로 전달
    )
    print("엔진 초기화 완료!")
    
    # 오디오 프로세서 생성
    processor = RealtimeAudioProcessor(engine, OUTPUT_DIR)
    
    try:
        print(f"\n정답 텍스트: {REFERENCE_TEXT}")
        print("\n명령어:")
        print("r: 녹음 시작/종료")
        print("q: 프로그램 종료")
        
        recording = False
        
        while True:
            cmd = input("\n> ")
            
            if cmd.lower() == 'r':
                if not recording:
                    # 세션 생성 및 녹음 시작
                    processor.start_session(REFERENCE_TEXT)
                    processor.start_recording()
                    recording = True
                else:
                    # 녹음 종료
                    processor.stop_recording()
                    recording = False
            elif cmd.lower() == 'q':
                if recording:
                    processor.stop_recording()
                break
            else:
                print("알 수 없는 명령어입니다. 'r'로 녹음 시작/종료, 'q'로 종료하세요.")
    
    except KeyboardInterrupt:
        print("프로그램을 종료합니다...")
    finally:
        if recording:
            processor.stop_recording()


if __name__ == "__main__":
    main()