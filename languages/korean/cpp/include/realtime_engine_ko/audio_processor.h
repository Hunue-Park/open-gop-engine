// audio_processor.h
#pragma once

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <string>

namespace realtime_engine_ko {

class AudioProcessor {
public:
    AudioProcessor(int sample_rate = 16000, float max_buffer_seconds = 10.0);
    ~AudioProcessor();
    
    // 핵심 기능만 유지
    Eigen::Matrix<float, Eigen::Dynamic, 1> ProcessAudioBinary(const std::vector<uint8_t>& binary_data);
    void Reset();
    
private:
    Eigen::Matrix<float, Eigen::Dynamic, 1> PreprocessAudioData(
        const std::vector<float>& audio_data, bool do_normalize = true);
    bool DetectVoiceActivity(const std::vector<float>& audio_data, 
                           float energy_threshold = 0.00005f,
                           int min_speech_frames = 5);
    
    // 핵심 멤버 변수만 유지
    int sample_rate;
    int max_buffer_length;
    std::vector<float> audio_buffer;
    
    // 상태 추적용 변수
    std::chrono::system_clock::time_point last_process_time;
    float total_duration;
};

} // namespace realtime_engine_ko