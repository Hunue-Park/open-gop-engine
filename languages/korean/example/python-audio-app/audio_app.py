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
        response = self.engine.create_session(text)
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
        while not self.stop_event.is_set() and self.is_recording:
            if self.frames_buffer and self.session_id:
                # 버퍼에서 일정량 데이터 가져오기 (복사 후 처리)
                frames_to_process = self.frames_buffer.copy()
                
                # 바이너리 데이터로 변환
                audio_binary = b''.join(frames_to_process)
                
                # 평가 요청
                result = self.engine.evaluate_audio(self.session_id, audio_binary)
                self.current_result = result
                
                # 결과 출력 
                self._display_result(result)
                
            time.sleep(0.3)  # 300ms 간격으로 처리
    
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
            
            print(f"\r상태: {status} | 진행: {progress} | 현재 점수: {score}", end="")
            
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
            
        try:
            wf = wave.open(self.current_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames_buffer))
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
    MODEL_PATH = os.path.join(BASE_DIR, "models/wav2vec2_ctc_dynamic.onnx")
    TOKENIZER_PATH = os.path.join(BASE_DIR, "models/tokenizer.json")
    REFERENCE_TEXT = "미래는 그 누구도 알 수 없습니다. 어제를 돌아보면 비로소 내일이 보입니다. 지금껏 내가 이룬 것들이 내일과 이어진다는 믿음을 갖고 전진해야 합니다."
    OUTPUT_DIR = os.path.join(BASE_DIR, "recordings")
    
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 엔진 초기화
    print("엔진 초기화 중...")
    engine = EngineCoordinator(
        onnx_model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        confidence_threshold=30  # 약간 낮은 임계값 설정
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