# Python 람다 함수를 위한 Dockerfile(Linux/ARM64 또는 x86_64)
FROM python:3.12.3

ARG DEBIAN_FRONTEND=noninteractive

# C++ 런타임 의존성 설치
RUN apt-get update && \
    apt-get install -y \
        libsndfile1 \
        libuuid1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ONNX Runtime 설치
ARG ONNXRUNTIME_VERSION=1.18.0 
ARG ORT_ARCH_FOR_SERVER=linux-aarch64 
RUN curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ORT_ARCH_FOR_SERVER}-${ONNXRUNTIME_VERSION}.tgz && \
    tar -xzf onnxruntime.tgz && \
    cp -R onnxruntime-${ORT_ARCH_FOR_SERVER}-${ONNXRUNTIME_VERSION}/lib/* /usr/lib/ && \
    rm -rf onnxruntime.tgz onnxruntime-${ORT_ARCH_FOR_SERVER}-${ONNXRUNTIME_VERSION} && \
    ldconfig

# Poetry 설치
RUN pip install poetry

# 앱 코드가 위치할 디렉토리 생성
WORKDIR /app

# Poetry 설정
RUN poetry config virtualenvs.create false

# ✅ 기존 realtime_engine_ko 패키지 설치 부분 제거
# ❌ COPY languages/korean/python /app/languages/korean/python
# ❌ RUN pip install -e /app/languages/korean/python

# ✅ pyrealtime.so 파일만 직접 복사
COPY languages/korean/example/recognition-server/pyrealtime.so /app/languages/korean/example/recognition-server/pyrealtime.so

# 프로젝트 의존성 파일 복사 및 설치 (realtime-engine-ko 의존성 제거된 버전)
COPY languages/korean/example/recognition-server/pyproject.toml /app/languages/korean/example/recognition-server/
WORKDIR /app/languages/korean/example/recognition-server
RUN poetry install --no-root

# 애플리케이션 코드 복사
COPY languages/korean/example/recognition-server/app /app/languages/korean/example/recognition-server/app

WORKDIR /app/languages/korean/example/recognition-server
CMD poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info