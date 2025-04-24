import os
import time
import threading
import json
import pyaudio
import wave
import pprint

from pyrealtime import EngineCoordinator, RecordListener

class RealtimeAudioRecorder:
    def __init__(self, output_dir, filename_prefix="recorded_audio_"):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.current_file = None
        self.recording_thread = None
        self.temp_file_thread = None
        self.stop_event = threading.Event()
        
    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.stop_event.clear()
        
        # 녹음 타임스탬프로 파일명 생성
        timestamp = int(time.time())
        self.current_file = os.path.join(
            self.output_dir, 
            f"{self.filename_prefix}{timestamp}.wav"
        )
        
        # 빈 파일 먼저 생성 (모니터링을 위해)
        wf = wave.open(self.current_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b'')  # 빈 데이터 쓰기
        wf.close()
        
        print(f"파일 생성됨: {self.current_file}")
        
        # 오디오 스트림 시작
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._callback
        )
        self.stream.start_stream()
        print("녹음 시작...")
        
        # 중간 파일 저장 스레드 시작
        self.temp_file_thread = threading.Thread(target=self._save_temp_files)
        self.temp_file_thread.daemon = True
        self.temp_file_thread.start()
        
    def _callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _save_temp_files(self):
        """실시간 처리를 위해 주기적으로 임시 파일 저장"""
        while not self.stop_event.is_set():
            if self.is_recording and self.frames:
                # 현재까지의 프레임으로 임시 파일 저장
                self._write_current_frames()
            time.sleep(0.3)  # 300ms 간격으로 파일 업데이트
            
    def _write_current_frames(self):
        """현재까지 녹음된 프레임을 파일로 저장"""
        try:
            # 현재 프레임 복사 (스레드 안전성)
            current_frames = self.frames.copy()
            
            if not current_frames:
                return
                
            temp_file = f"{self.current_file}.temp"
            
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(current_frames))
            wf.close()
            
            # 완성된 임시 파일을 현재 파일로 이동 (원자적 연산)
            os.rename(temp_file, self.current_file)
        except Exception as e:
            print(f"임시 파일 저장 중 오류: {e}")
    
    def stop_recording(self):
        if not self.is_recording:
            return None
            
        self.is_recording = False
        self.stop_event.set()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # 최종 파일 저장
        if self.frames:
            wf = wave.open(self.current_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames))
            wf.close()
        
        print(f"녹음 완료: {self.current_file}")
        final_path = self.current_file
        self.current_file = None
        return final_path
        
    def __del__(self):
        self.stop_event.set()
        if self.temp_file_thread and self.temp_file_thread.is_alive():
            self.temp_file_thread.join(timeout=1.0)
        self.audio.terminate()


# EngineCoordinator를 위한 콜백 핸들러
class EngineCallbackHandler(RecordListener):
    def __init__(self):
        super().__init__(
            on_start=self.handle_start,
            on_tick=self.handle_tick,
            on_start_record_fail=self.handle_start_record_fail,
            on_record_end=self.handle_record_end,
            on_score=self.handle_score
        )
        self.latest_score = None
        
    def handle_start(self):
        print("\n[이벤트] 평가 시작")
        
    def handle_tick(self, current, total):
        progress_percent = (current / total) * 100
        print(f"\r[이벤트] 진행 상황: {current}/{total} ({progress_percent:.1f}%)", end="")
        
    def handle_start_record_fail(self, error_msg):
        print(f"\n[이벤트] 평가 시작 실패: {error_msg}")
        
    def handle_record_end(self):
        print("\n[이벤트] 평가 종료")
        if self.latest_score:
            self.print_final_result()
        
    def handle_score(self, result_json):
        try:
            result = json.loads(result_json)
            self.latest_score = result

            pprint.pprint(result)
            
        except json.JSONDecodeError:
            print(f"잘못된 JSON 형식: {result_json}")
        except Exception as e:
            print(f"결과 처리 중 오류: {e}")
    
    def print_final_result(self):
        if not self.latest_score:
            return
            
        print("\n\n===== 최종 평가 결과 =====")
        print(f"전체 점수: {self.latest_score.get('overall_score', 0)}")
        
        if "blocks" in self.latest_score:
            print("\n단어별 점수:")
            for block in self.latest_score.get("blocks", []):
                print(f"- {block['text']}: {block.get('gop_score', 0)}")
                
        print("=============================\n")


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
        confidence_threshold=30,  # 약간 낮은 임계값 설정
        update_interval=0.3  # 0.5초 간격으로 틱 이벤트
    )
    print("엔진 초기화 완료!")
    
    # 콜백 핸들러 생성
    callback_handler = EngineCallbackHandler()
    engine.SetRecordListener(callback_handler)
    
    # 녹음기 생성
    recorder = RealtimeAudioRecorder(OUTPUT_DIR)
    
    try:
        print(f"\n정답 텍스트: {REFERENCE_TEXT}")
        print("\n명령어:")
        print("r: 녹음 시작/종료")
        print("q: 프로그램 종료")
        
        recording = False
        evaluating = False
        
        while True:
            cmd = input("\n> ")
            
            if cmd.lower() == 'r':
                if not recording:
                    # 녹음 시작
                    recorder.start_recording()
                    recording = True
                    # 이미 평가 중이 아니면 초기화 및 평가 시작
                    if not evaluating:
                        engine.Initialize(REFERENCE_TEXT, audio_polling_interval=0.03, min_time_between_evals=0.5)
                        # 녹음이 시작되면 바로 평가도 시작
                        if recorder.current_file:
                            engine.StartEvaluation(recorder.current_file)
                            evaluating = True
                else:
                    # 녹음 종료
                    final_path = recorder.stop_recording()
                    recording = False
                    # 평가 중지
                    if evaluating:
                        engine.StopEvaluation()
                        evaluating = False
                        # 최종 결과 출력
                        callback_handler.print_final_result()
            elif cmd.lower() == 'q':
                if recording:
                    recorder.stop_recording()
                if evaluating:
                    engine.StopEvaluation()
                break
            else:
                print("알 수 없는 명령어입니다. 'r'로 녹음 시작/종료, 'q'로 종료하세요.")
    
    except KeyboardInterrupt:
        print("프로그램을 종료합니다...")
    finally:
        if recording:
            recorder.stop_recording()
        if evaluating:
            engine.StopEvaluation()


if __name__ == "__main__":
    main()