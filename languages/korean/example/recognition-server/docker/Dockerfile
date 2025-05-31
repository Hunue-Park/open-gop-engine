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

# (만약 cpp/Dockerfile.build와 동일하게 ONNX Runtime을 설치해야 한다면)
ARG ONNXRUNTIME_VERSION=1.18.0 
# 서버 이미지가 실행될 아키텍처 (linux_build.sh의 ORT_ARCH_ARG와 일치시켜야 함)
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

# Poetry 설정 - 가상환경 생성하지 않도록 설정
RUN poetry config virtualenvs.create false

# realtime_engine_ko 파이썬 패키지 의존성 (예: numpy)
# RUN pip install numpy # poetry가 처리할 것이므로 중복 가능성

# realtime_engine_ko 패키지 복사 (이때 내부에 Linux용 pyrealtime.so 포함됨)
COPY languages/korean/python /app/languages/korean/python
RUN pip install -e /app/languages/korean/python 
# 위 install 과정에서 realtime_engine_ko/__init__.py가 실행되면서 C++ 모듈 로드 시도

# 프로젝트 의존성 파일 복사 및 설치
COPY languages/korean/example/recognition-server/pyproject.toml /app/languages/korean/example/recognition-server/
WORKDIR /app/languages/korean/example/recognition-server
RUN poetry install --no-root # --no-dev 옵션도 고려

# 애플리케이션 코드 복사 (config.py 포함)
COPY languages/korean/example/recognition-server/app /app/languages/korean/example/recognition-server/app

# 모델 파일 복사 라인 주석 처리 또는 삭제 (볼륨 마운트 사용 시)
# COPY models /app/models 

WORKDIR /app/languages/korean/example/recognition-server
CMD poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info