// src/cpp/src/common.cpp
#include "realtime_engine_ko/common.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace realtime_engine_ko {

// 로깅 유틸리티 구현
void Logger(LogLevel level, const std::string& component, const std::string& message) {
    // 현재 시간 가져오기
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    // 시간 포맷팅
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    
    // 로그 레벨 문자열
    std::string level_str;
    switch (level) {
        case LogLevel::DEBUG:   level_str = "DEBUG"; break;
        case LogLevel::INFO:    level_str = "INFO"; break;
        case LogLevel::WARNING: level_str = "WARNING"; break;
        case LogLevel::ERROR:   level_str = "ERROR"; break;
    }
    
    // 로그 메시지 출력
    std::cout << ss.str() << " - " << component << " - " << level_str << " - " << message << std::endl;
}

} // namespace realtime_engine_ko