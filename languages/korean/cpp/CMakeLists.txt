# src/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(realtime_engine_ko_cpp VERSION 0.1.0 LANGUAGES CXX)

# 컴파일러 옵션 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # PIC 추가

# 빌드 타입 설정
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# tokenizers-cpp 서브모듈 추가
add_subdirectory(third_party/tokenizers-cpp)

# nlohmann_json을 자동으로 가져오기 위한 FetchContent 설정
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# pybind11 설정 추가
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.10.4
)
FetchContent_MakeAvailable(pybind11)

# 외부 패키지 찾기
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 필수 패키지
find_package(Eigen3 REQUIRED)
find_package(ONNXRuntime REQUIRED)
find_package(SndFile REQUIRED)

# C++ 라이브러리 소스 및 헤더 파일
set(SOURCES
    src/common.cpp
    src/sentence_block.cpp
    src/progress_tracker.cpp
    src/audio_processor.cpp
    src/w2v_onnx_core.cpp
    src/eval_manager.cpp
    src/recognition_engine.cpp
    dtw/dtw_algorithm.cpp
)

set(HEADERS
    include/realtime_engine_ko/common.h
    include/realtime_engine_ko/sentence_block.h
    include/realtime_engine_ko/progress_tracker.h
    include/realtime_engine_ko/audio_processor.h
    include/realtime_engine_ko/w2v_onnx_core.h
    include/realtime_engine_ko/eval_manager.h
    include/realtime_engine_ko/recognition_engine.h
    dtw/dtw_algorithm.h
)

# C++ 코어 라이브러리 생성
add_library(realtime_engine_ko_cpp STATIC ${SOURCES} ${HEADERS})

# 헤더 파일 경로 설정
target_include_directories(realtime_engine_ko_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include
    ${SNDFILE_INCLUDE_DIRS}
)

# 라이브러리 연결
target_link_libraries(realtime_engine_ko_cpp PUBLIC
    ${ONNXRUNTIME_LIBS}
    tokenizers_cpp
    ${SNDFILE_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Python 바인딩 모듈 빌드
pybind11_add_module(pyrealtime python_bindings.cpp)
target_link_libraries(pyrealtime PRIVATE realtime_engine_ko_cpp)

# Python 모듈 출력 디렉토리 설정 (선택 사항)
set_target_properties(pyrealtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

# 예제 및 테스트 빌드 설정 (선택적)
option(BUILD_EXAMPLES "Build example applications" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()