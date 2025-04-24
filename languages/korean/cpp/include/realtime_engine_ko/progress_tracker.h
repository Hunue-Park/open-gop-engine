// progress_tracker.h
#pragma once

#include <vector>
#include <chrono>
#include <optional>

namespace realtime_engine_ko {

class ProgressTracker {
public:
    ProgressTracker(int total_blocks, int window_size = 3, bool time_based_advance = true);
    
    void Start();
    bool IsStarted() const;
    double GetElapsedTime() const;
    double GetTimeSinceLastAdvance() const;
    std::vector<int> GetActiveWindow() const;
    int GetExpectedBlockIndex() const;
    bool ShouldAdvance() const;
    bool Advance();
    bool SetCurrentIndex(int index);
    void Reset();
    void AdjustTimeParameters(double avg_time_per_block, double min_time_for_advance);
    
private:
    int total_blocks;
    int window_size;
    bool time_based_advance;
    
    int current_index;
    std::optional<std::chrono::system_clock::time_point> start_time;
    std::optional<std::chrono::system_clock::time_point> last_advance_time;
    
    // 블록당 평균 소요 시간 (초)
    double avg_time_per_block = 2.0;
    
    // 블록 자동 진행 최소 시간 (초)
    double min_time_for_advance = 1.5;
};

} // namespace realtime_engine_ko