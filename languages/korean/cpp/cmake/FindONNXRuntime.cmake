# src/cpp/cmake/FindONNXRuntime.cmake
# 이 파일은 ONNX Runtime 라이브러리를 찾는 CMake 모듈입니다.

# 다음 변수를 검색합니다:
#   ONNXRUNTIME_INCLUDE_DIRS
#   ONNXRUNTIME_LIBS
#   ONNXRUNTIME_FOUND

# 사용자가 직접 경로를 설정할 수 있도록 변수 정의
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "ONNX Runtime의 루트 디렉토리")

# 헤더 파일 검색
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT_DIR}/include
        /usr/include
        /usr/local/include
    PATH_SUFFIXES onnxruntime
    DOC "ONNX Runtime 헤더 파일 디렉토리"
)

# 라이브러리 파일 검색
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT_DIR}/lib
        /usr/lib
        /usr/local/lib
    DOC "ONNX Runtime 라이브러리 파일"
)

# 표준 CMake find_package 변수 설정
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS
        ONNXRUNTIME_INCLUDE_DIR
        ONNXRUNTIME_LIBRARY
)

# 변수 설정
if(ONNXRUNTIME_FOUND)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
    set(ONNXRUNTIME_LIBS ${ONNXRUNTIME_LIBRARY})
endif()

# 변수 캐시 설정
mark_as_advanced(
    ONNXRUNTIME_INCLUDE_DIR
    ONNXRUNTIME_LIBRARY
)