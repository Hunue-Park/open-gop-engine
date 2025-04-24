// audio_processor.h
#pragma once

#include <string>
#include <vector>
#include <thread>
#include <functional>
#include <mutex>
#include <atomic>
#include <memory>
#include <any>
#include <map>
#include <Eigen/Dense>
#include <sndfile.h>

namespace realtime_engine_ko {

class AudioProcessor {
public:
    using AudioTensor = Eigen::Matrix<float, Eigen::Dynamic, 1>;
    using CallbackFunc = std::function<void(const AudioTensor&, const std::map<std::string, std::any>&)>;
    
    AudioProcessor(int sample_rate = 16000, float chunk_duration = 2.5, float polling_interval = 0.1);
    ~AudioProcessor();
    
    bool SetAudioFile(const std::string& file_path);
    bool StartMonitoring();
    void StopMonitoring();
    std::pair<AudioTensor, std::map<std::string, std::any>> GetLatestChunk() const;
    void Reset();
    void AddChunkCallback(CallbackFunc callback);
    
private:
    void MonitoringLoop();
    void ProcessNewAudioData();
    void AddToBuffer(const std::vector<float>& audio_data);
    void CheckAndProcessChunks();
    AudioTensor ExtractChunk(int chunk_samples);
    AudioTensor PreprocessChunk(const std::vector<float>& chunk, bool do_normalize = true);
    bool DetectVoiceActivity(const std::vector<float>& audio_data, float energy_threshold = 0.0005, int min_speech_frames = 10);
    
    int sample_rate;
    float chunk_duration;
    float polling_interval;
    
    std::string audio_file_path;
    sf_count_t last_file_size;
    sf_count_t last_processed_pos;
    std::atomic<bool> is_monitoring;
    std::unique_ptr<std::thread> monitoring_thread;
    
    std::vector<std::vector<float>> buffer;
    std::chrono::system_clock::time_point last_chunk_time;
    float total_duration;
    AudioTensor latest_chunk;
    
    std::vector<CallbackFunc> chunk_callbacks;
    std::mutex buffer_mutex;
};

} // namespace realtime_engine_ko