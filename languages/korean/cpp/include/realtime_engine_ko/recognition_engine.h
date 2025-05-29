// recognition_engine.h
#pragma once

#include <string>
#include <vector>
#include <map>
#include <any>
#include <functional>
#include <memory>
#include <chrono>
#include "realtime_engine_ko/w2v_onnx_core.h"
#include "realtime_engine_ko/sentence_block.h"
#include "realtime_engine_ko/progress_tracker.h"
#include "realtime_engine_ko/audio_processor.h"
#include "realtime_engine_ko/eval_manager.h"

namespace realtime_engine_ko {

// RecordListener 클래스 - Python 구현과 일치
class RecordListener {
public:
    RecordListener(
        std::function<void()> on_start = nullptr,
        std::function<void()> on_tick = nullptr,
        std::function<void(const std::string&)> on_start_record_fail = nullptr,
        std::function<void()> on_record_end = nullptr,
        std::function<void(const std::map<std::string, std::any>&)> on_score = nullptr
    ) : on_start(on_start),
        on_tick(on_tick),
        on_start_record_fail(on_start_record_fail),
        on_record_end(on_record_end),
        on_score(on_score) {}
        
    std::function<void()> on_start;                                  // 녹음 시작
    std::function<void()> on_tick;                                  // 진행 틱
    std::function<void(const std::string&)> on_start_record_fail;   // 녹음 시작 실패
    std::function<void()> on_record_end;                            // 녹음 종료
    std::function<void(const std::map<std::string, std::any>&)> on_score;  // 평가 결과
};

// EngineCoordinator 클래스
class EngineCoordinator {
public:
    // 세션 데이터 구조체
    struct SessionData {
        std::shared_ptr<SentenceBlockManager> sentence_manager;
        std::shared_ptr<ProgressTracker> progress_tracker;
        std::shared_ptr<AudioProcessor> audio_processor;
        std::shared_ptr<EvaluationController> eval_controller;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point last_activity;
    };
    
    // 생성자
    EngineCoordinator(
        const std::string& onnx_model_path,
        const std::string& tokenizer_path,
        const std::string& device = "CPU",
        float confidence_threshold = 0.7f,
        const std::string& matrix_path = "");
    
    // 소멸자
    ~EngineCoordinator();
    
    // 세션 관리 메서드 - Python 구현과 일치
    std::map<std::string, std::any> CreateSession(
        const std::string& sentence,
        const std::map<std::string, std::any>& engine_options = {});
    
    std::map<std::string, std::any> EvaluateAudio(
        const std::string& session_id,
        const std::vector<uint8_t>& binary_data);
    
    std::map<std::string, std::any> CloseSession(const std::string& session_id);
    
    std::map<std::string, std::any> GetSessionStatus(const std::string& session_id);
    
    int CleanupInactiveSessions(float max_inactive_time = 3600.0f);
    
private:
    // 인식 엔진
    std::shared_ptr<Wav2VecCTCOnnxCore> recognition_engine;
    
    // 세션 관리
    std::map<std::string, SessionData> sessions;
    
    // 설정값
    float confidence_threshold;
    
    // 유틸리티 메서드
    std::map<std::string, std::any> CreateEmptyResult() const;
};

} // namespace realtime_engine_ko