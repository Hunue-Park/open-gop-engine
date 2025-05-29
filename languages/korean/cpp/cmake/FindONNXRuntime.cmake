# src/cpp/cmake/FindONNXRuntime.cmake
# 이 파일은 ONNX Runtime 라이브러리를 찾는 CMake 모듈입니다.

# 다음 변수를 검색합니다:
#   ONNXRUNTIME_INCLUDE_DIRS
#   ONNXRUNTIME_LIBS
#   ONNXRUNTIME_FOUND

# 사용자가 직접 경로를 설정할 수 있도록 변수 정의
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "ONNX Runtime의 루트 디렉토리")

# 헤더 파일 검색
find_path(ONNXRUNTIME_INCLUDE_DIRS onnxruntime_cxx_api.h
  HINTS
    ${ONNXRUNTIME_ROOT_DIR}/include
    ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime
    /opt/homebrew/Cellar/onnxruntime/1.21.0/include
    /opt/homebrew/Cellar/onnxruntime/1.21.0/include/onnxruntime
    /opt/homebrew/include
    /opt/homebrew/include/onnxruntime
)

# 라이브러리 파일 검색
find_library(ONNXRUNTIME_LIBS
  NAMES onnxruntime
  HINTS
    ${ONNXRUNTIME_ROOT_DIR}/lib
    /opt/homebrew/Cellar/onnxruntime/1.21.0/lib
    /opt/homebrew/lib
)

# 표준 CMake find_package 변수 설정
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime DEFAULT_MSG ONNXRUNTIME_LIBS ONNXRUNTIME_INCLUDE_DIRS)

# 변수 설정
if(ONNXRUNTIME_FOUND)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIRS})
    set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_LIBS})
endif()

# 변수 캐시 설정
mark_as_advanced(
    ONNXRUNTIME_INCLUDE_DIRS
    ONNXRUNTIME_LIBS
)