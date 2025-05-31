#!/bin/bash
set -e

# 스크립트가 위치한 디렉토리 (cpp 디렉토리)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}" # cpp 디렉토리가 컨텍스트 루트
IMAGE_NAME="realtime_engine_ko_builder_linux"
OUTPUT_DIR="${PROJECT_ROOT}/linux_build_output"
PYTHON_VERSION_ARG="3.12" # 서버 애플리케이션 Python 버전에 맞춤
ORT_ARCH_ARG="linux-aarch64" # 명시적으로 aarch64로 설정 (이전 단계에서 확인된 값)

echo "Building Linux .so file using Docker..."
echo "Target Python version: ${PYTHON_VERSION_ARG}"
echo "Target ONNX Runtime architecture: ${ORT_ARCH_ARG}"

# Docker 이미지 빌드
# docker buildx build 명령어 형식에 맞게 옵션 배치
# 일반적으로 -f 옵션은 빌드 컨텍스트 경로 앞에 명시하지만, 
# buildx는 -f를 --file로 사용하는 것을 더 선호할 수 있으며,
# 컨텍스트는 마지막에 명확히 전달합니다.
docker buildx build \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION_ARG}" \
  --build-arg ORT_ARCH="${ORT_ARCH_ARG}" \
  -t "${IMAGE_NAME}" \
  --file "${PROJECT_ROOT}/Dockerfile.build" \
  "${PROJECT_ROOT}" # 빌드 컨텍스트는 마지막에 하나만

echo "Copying .so file from Docker image..."

# 출력 디렉토리 생성
mkdir -p "${OUTPUT_DIR}"

# Docker 컨테이너를 임시로 실행하여 /output 디렉토리의 내용 복사
TEMP_CONTAINER_ID=$(docker create "${IMAGE_NAME}")
docker cp "${TEMP_CONTAINER_ID}:/output/." "${OUTPUT_DIR}/"
docker rm "${TEMP_CONTAINER_ID}"

echo "Build complete. Linux .so file(s) are in ${OUTPUT_DIR}"
ls -l "${OUTPUT_DIR}"
