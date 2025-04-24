// src/cpp/src/progress_tracker.cpp
#include "realtime_engine_ko/progress_tracker.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <algorithm>

namespace realtime_engine_ko {

ProgressTracker::ProgressTracker(int total_blocks, int window_size, bool time_based_advance)
    : total_blocks(total_blocks), window_size(window_size), time_based_advance(time_based_advance),
      current_index(0), start_time(std::nullopt), last_advance_time(std::nullopt),
      avg_time_per_block(2.0), min_time_for_advance(1.5) {
    
    std::stringstream ss;
    ss << "ProgressTracker 초기화: " << total_blocks << " 블록, 윈도우 크기=" << window_size;
    LOG_INFO("ProgressTracker", ss.str());
}

void ProgressTracker::Start() {
    start_time = std::chrono::system_clock::now();
    last_advance_time = start_time;
    LOG_INFO("ProgressTracker", "진행 추적 시작");
}

bool ProgressTracker::IsStarted() const {
    return start_time.has_value();
}

double ProgressTracker::GetElapsedTime() const {
    if (!IsStarted()) {
        return 0.0;
    }
    
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time.value());
    return duration.count() / 1000.0;
}

double ProgressTracker::GetTimeSinceLastAdvance() const {
    if (!last_advance_time.has_value()) {
        return 0.0;
    }
    
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_advance_time.value());
    return duration.count() / 1000.0;
}

std::vector<int> ProgressTracker::GetActiveWindow() const {
    int start = std::max(0, current_index - window_size + 1);
    int end = current_index + 1;
    
    std::vector<int> result;
    for (int i = start; i < end; ++i) {
        result.push_back(i);
    }
    
    return result;
}

int ProgressTracker::GetExpectedBlockIndex() const {
    if (!IsStarted()) {
        return 0;
    }
    
    double elapsed = GetElapsedTime();
    int expected_index = std::min(
        static_cast<int>(elapsed / avg_time_per_block),
        total_blocks - 1
    );
    
    return expected_index;
}

bool ProgressTracker::ShouldAdvance() const {
    if (!time_based_advance || !IsStarted()) {
        return false;
    }
    
    // 이미 마지막 블록이면 진행 불가
    if (current_index >= total_blocks - 1) {
        return false;
    }
    
    // 경과 시간 기준 예상 블록이 현재보다 앞서 있고,
    // 마지막 진행 이후 최소 시간이 지났으면 진행
    int expected_index = GetExpectedBlockIndex();
    double time_since_last = GetTimeSinceLastAdvance();
    
    return (expected_index > current_index && time_since_last >= min_time_for_advance);
}

bool ProgressTracker::Advance() {
    if (current_index >= total_blocks - 1) {
        return false;
    }
    
    current_index++;
    last_advance_time = std::chrono::system_clock::now();
    
    std::stringstream ss;
    ss << "블록 진행: " << current_index << "/" << total_blocks;
    LOG_INFO("ProgressTracker", ss.str());
    
    return true;
}

bool ProgressTracker::SetCurrentIndex(int index) {
    if (index < 0 || index >= total_blocks) {
        return false;
    }
    
    current_index = index;
    last_advance_time = std::chrono::system_clock::now();
    
    std::stringstream ss;
    ss << "현재 인덱스 설정: " << index;
    LOG_INFO("ProgressTracker", ss.str());
    
    return true;
}

void ProgressTracker::Reset() {
    current_index = 0;
    start_time = std::nullopt;
    last_advance_time = std::nullopt;
    
    LOG_INFO("ProgressTracker", "진행 상태 초기화");
}

void ProgressTracker::AdjustTimeParameters(double avg_time_per_block, double min_time_for_advance) {
    if (avg_time_per_block > 0) {
        this->avg_time_per_block = avg_time_per_block;
    }
    
    if (min_time_for_advance > 0) {
        this->min_time_for_advance = min_time_for_advance;
    }
    
    std::stringstream ss;
    ss << "시간 파라미터 조정: 블록당 평균=" << this->avg_time_per_block 
       << "초, 최소 진행 시간=" << this->min_time_for_advance << "초";
    LOG_INFO("ProgressTracker", ss.str());
}

} // namespace realtime_engine_ko