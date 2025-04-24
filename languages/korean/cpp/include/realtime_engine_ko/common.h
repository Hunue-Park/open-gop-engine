// src/cpp/include/realtime_engine_ko/common.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <any>
#include <memory>
#include <optional>
#include <functional>
#include <chrono>
#include <Eigen/Dense>

namespace realtime_engine_ko {

// 기본 타입 정의
using AudioTensor = Eigen::Matrix<float, Eigen::Dynamic, 1>;
using MetadataMap = std::map<std::string, std::any>;
using ResultMap = std::map<std::string, std::any>;

// 로깅 유틸리티 (간단한 구현)
enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

void Logger(LogLevel level, const std::string& component, const std::string& message);

// 간단한 로깅 매크로
#define LOG_DEBUG(component, message) realtime_engine_ko::Logger(realtime_engine_ko::LogLevel::DEBUG, component, message)
#define LOG_INFO(component, message) realtime_engine_ko::Logger(realtime_engine_ko::LogLevel::INFO, component, message)
#define LOG_WARNING(component, message) realtime_engine_ko::Logger(realtime_engine_ko::LogLevel::WARNING, component, message)
#define LOG_ERROR(component, message) realtime_engine_ko::Logger(realtime_engine_ko::LogLevel::ERROR, component, message)

} // namespace realtime_engine_ko