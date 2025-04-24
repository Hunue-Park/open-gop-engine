# src/cpp/cmake/FindTokenizersCpp.cmake
# 이 파일은 tokenizers-cpp 라이브러리를 찾는 CMake 모듈입니다.

# 다음 변수를 검색합니다:
#   TOKENIZERSCPP_INCLUDE_DIRS
#   TOKENIZERSCPP_LIBRARIES
#   TOKENIZERSCPP_FOUND

# 사용자가 직접 경로를 설정할 수 있도록 변수 정의
set(TOKENIZERSCPP_ROOT_DIR "" CACHE PATH "tokenizers-cpp의 루트 디렉토리")

# 헤더 파일 검색
find_path(TOKENIZERSCPP_INCLUDE_DIR
    NAMES tokenizers_cpp.h
    PATHS
        ${TOKENIZERSCPP_ROOT_DIR}/include
        /usr/include
        /usr/local/include
    DOC "tokenizers-cpp 헤더 파일 디렉토리"
)

# 라이브러리 파일 검색
find_library(TOKENIZERSCPP_LIBRARY
    NAMES tokenizers_cpp
    PATHS
        ${TOKENIZERSCPP_ROOT_DIR}/lib
        /usr/lib
        /usr/local/lib
    DOC "tokenizers-cpp 라이브러리 파일"
)

# 필요한 나머지 라이브러리들 검색
find_library(TOKENIZERSC_LIBRARY
    NAMES tokenizers_c
    PATHS
        ${TOKENIZERSCPP_ROOT_DIR}/lib
        /usr/lib
        /usr/local/lib
    DOC "tokenizers_c 라이브러리 파일"
)

find_library(SENTENCEPIECE_LIBRARY
    NAMES sentencepiece
    PATHS
        ${TOKENIZERSCPP_ROOT_DIR}/lib
        /usr/lib
        /usr/local/lib
    DOC "sentencepiece 라이브러리 파일"
)

# 표준 CMake find_package 변수 설정
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TokenizersCpp
    REQUIRED_VARS
        TOKENIZERSCPP_INCLUDE_DIR
        TOKENIZERSCPP_LIBRARY
        TOKENIZERSC_LIBRARY
        SENTENCEPIECE_LIBRARY
)

# 변수 설정
if(TOKENIZERSCPP_FOUND)
    set(TOKENIZERSCPP_INCLUDE_DIRS ${TOKENIZERSCPP_INCLUDE_DIR})
    set(TOKENIZERSCPP_LIBRARIES
        ${TOKENIZERSCPP_LIBRARY}
        ${TOKENIZERSC_LIBRARY}
        ${SENTENCEPIECE_LIBRARY}
    )
endif()

# 변수 캐시 설정
mark_as_advanced(
    TOKENIZERSCPP_INCLUDE_DIR
    TOKENIZERSCPP_LIBRARY
    TOKENIZERSC_LIBRARY
    SENTENCEPIECE_LIBRARY
)