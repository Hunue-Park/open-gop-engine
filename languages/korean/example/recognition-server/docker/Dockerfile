# Python 람다 함수를 위한 Dockerfile(Linux/ARM64)
FROM python:3.12.3

# Poetry 설치
RUN pip install poetry

# 앱 코드가 위치할 디렉토리 생성
WORKDIR /app

# Poetry 설정 - 가상환경 생성하지 않도록 설정
RUN poetry config virtualenvs.create false

# realtime_engine_ko 의존성 먼저 설치
RUN pip install torch numpy onnxruntime

# realtime_engine_ko 패키지 복사 및 설치
COPY languages/korean/python /app/languages/korean/python
RUN pip install -e /app/languages/korean/python

# 프로젝트 의존성 파일 복사 및 설치
COPY languages/korean/example/recognition-server/pyproject.toml /app/languages/korean/example/recognition-server/
WORKDIR /app/languages/korean/example/recognition-server
RUN poetry install --no-root

# 애플리케이션 코드 복사
COPY languages/korean/example/recognition-server/app /app/languages/korean/example/recognition-server/app

# PYTHONPATH 설정
ENV PYTHONPATH=/app/languages/korean/example/recognition-server

# Poetry를 통해 실행하되, 작업 디렉토리를 명시적으로 지정
WORKDIR /app/languages/korean/example/recognition-server
CMD poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info