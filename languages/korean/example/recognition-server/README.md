# speech-recognition-backend

FastAPI 기반의 음성 인식 API 서버 애플리케이션입니다.

## 실행 방법

- languages/korean/models 폴더에 wav2vec2_ctc_combined.onnx 파일을 다운로드 받습니다.
- cd docker
- make local-with-logs

## 서버 실행시 c++ 모듈 사용하지 않고 파이썬 패키지를 사용하는 방법

- languages/korean/python/realtime_engine_ko/init.py 파일을 수정합니다.
- from .pyrealtime import EngineCoordinator
- 주석 처리
- from .recognition_engine import EngineCoordinator
- 주석 해제
