// recognition_engine.h
#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>

#include "sentence_block.h"
#include "progress_tracker.h"
#include "audio_processor.h"
#include "w2v_onnx_core.h"
#include "eval_manager.h"

namespace realtime_engine_ko {

class RecordListener {
public:
    using StartCallback = std::function<void()>;
    using TickCallback = std::function<void(int, int)>;
    using FailCallback = std::function<void(const std::string&)>;
    using EndCallback = std::function<void()>;
    using ScoreCallback = std::function<void(const std::string&)>;
    
    RecordListener(
        StartCallback on_start = nullptr,
        TickCallback on_tick = nullptr,
        FailCallback on_start_record_fail = nullptr,
        EndCallback on_record_end = nullptr,
        ScoreCallback on_score = nullptr);
    
    StartCallback on_start;
    TickCallback on_tick;
    FailCallback on_start_record_fail;
    EndCallback on_record_end;
    ScoreCallback on_score;
};

class EngineCoordinator {
public:
    EngineCoordinator(
        const std::string& onnx_model_path,
        const std::string& tokenizer_path,
        const std::string& device = "CPU",
        float update_interval = 0.3f,
        float confidence_threshold = 0.7f);
    
    ~EngineCoordinator();
    
    void SetRecordListener(const RecordListener& record_listener);
    bool Initialize(const std::string& sentence, float audio_polling_interval = 0.03f, float min_time_between_evals = 0.5f);
    bool StartEvaluation(const std::string& audio_file_path);
    void StopEvaluation();
    std::map<std::string, std::any> GetCurrentState() const;
    void Reset();
    
    // 외부 API 메서드
    std::map<std::string, std::any> EvaluateSpeech(
        const std::string& sentence,
        const std::string& audio_file_path,
        const RecordListener& record_listener = RecordListener());
    
    std::map<std::string, std::any> GetResults() const;
    
private:
    void TimerLoop();
    void OnNewChunk(const AudioProcessor::AudioTensor& audio_chunk, const std::map<std::string, std::any>& metadata);
    
    std::shared_ptr<Wav2VecCTCOnnxCore> recognition_engine;
    std::shared_ptr<SentenceBlockManager> sentence_manager;
    std::shared_ptr<ProgressTracker> progress_tracker;
    std::shared_ptr<AudioProcessor> audio_processor;
    std::shared_ptr<EvaluationController> eval_controller;
    
    bool is_initialized;
    std::atomic<bool> is_running;
    float update_interval;
    float confidence_threshold;
    std::unique_ptr<std::thread> timer_thread;
    
    RecordListener record_listener;
};

} // namespace realtime_engine_ko