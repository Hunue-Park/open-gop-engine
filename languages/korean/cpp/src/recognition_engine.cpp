// src/cpp/src/recognition_engine.cpp
#include "realtime_engine_ko/recognition_engine.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>
#include <uuid/uuid.h>

namespace realtime_engine_ko {

// EngineCoordinator 구현
EngineCoordinator::EngineCoordinator(
    const std::string& onnx_model_path,
    const std::string& tokenizer_path,
    const std::string& device,
    float confidence_threshold)
    : confidence_threshold(confidence_threshold) {
    
    try {
        // 인식 엔진 초기화
        recognition_engine = std::make_shared<Wav2VecCTCOnnxCore>(
            onnx_model_path, tokenizer_path, device);
        
        LOG_INFO("EngineCoordinator", "RecognitionEngine 초기화 완료");
        LOG_INFO("EngineCoordinator", "EngineCoordinator 초기화 완료");
    } catch (const std::exception& e) {
        std::string error_msg = "EngineCoordinator 초기화 오류: " + std::string(e.what());
        LOG_ERROR("EngineCoordinator", error_msg);
        throw std::runtime_error(error_msg);
    }
}

EngineCoordinator::~EngineCoordinator() {
    // 모든 활성 세션 정리
    std::vector<std::string> session_ids_to_close;
    for (const auto& [session_id, session] : sessions) {
        session_ids_to_close.push_back(session_id);
    }
    
    for (const auto& session_id : session_ids_to_close) {
        CloseSession(session_id);
    }
}

std::map<std::string, std::any> EngineCoordinator::CreateSession(
    const std::string& sentence,
    const std::map<std::string, std::any>& engine_options) {
    
    try {
        // UUID 생성
        uuid_t uuid;
        uuid_generate(uuid);
        char uuid_str[37];
        uuid_unparse_lower(uuid, uuid_str);
        std::string session_id = uuid_str;
        
        // 엔진 옵션 처리
        float session_confidence_threshold = confidence_threshold;
        float min_time_between_evals = 0.5f;
        
        if (engine_options.count("confidence_threshold")) {
            session_confidence_threshold = std::any_cast<float>(engine_options.at("confidence_threshold"));
        }
        
        if (engine_options.count("min_time_between_evals")) {
            min_time_between_evals = std::any_cast<float>(engine_options.at("min_time_between_evals"));
        }
        
        // 세션 컴포넌트 초기화
        auto sentence_manager = std::make_shared<SentenceBlockManager>(sentence);
        auto progress_tracker = std::make_shared<ProgressTracker>(
            sentence_manager->blocks.size(), 3, false);
        auto audio_processor = std::make_shared<AudioProcessor>(16000);
        auto eval_controller = std::make_shared<EvaluationController>(
            recognition_engine,
            sentence_manager,
            progress_tracker,
            session_confidence_threshold,
            min_time_between_evals);
        
        // 세션 정보 저장
        SessionData session_data;
        session_data.sentence_manager = sentence_manager;
        session_data.progress_tracker = progress_tracker;
        session_data.audio_processor = audio_processor;
        session_data.eval_controller = eval_controller;
        session_data.created_at = std::chrono::system_clock::now();
        session_data.last_activity = std::chrono::system_clock::now();
        
        sessions[session_id] = session_data;
        
        // 진행 추적 시작
        progress_tracker->Start();
        
        // 결과 구성
        std::map<std::string, std::any> result;
        result["session_id"] = session_id;
        result["status"] = std::string("initialized");
        result["sentence"] = sentence;
        result["blocks"] = sentence_manager->blocks.size();
        
        LOG_INFO("EngineCoordinator", "세션 생성: " + session_id);
        return result;
    } catch (const std::exception& e) {
        std::string error_msg = "세션 생성 오류: " + std::string(e.what());
        LOG_ERROR("EngineCoordinator", error_msg);
        
        std::map<std::string, std::any> error_result;
        error_result["error"] = std::string("session_creation_failed");
        error_result["message"] = error_msg;
        return error_result;
    }
}

std::map<std::string, std::any> EngineCoordinator::EvaluateAudio(
    const std::string& session_id,
    const std::vector<uint8_t>& binary_data) {
    
    // 세션 확인
    if (sessions.find(session_id) == sessions.end()) {
        std::map<std::string, std::any> error_result;
        error_result["error"] = std::string("invalid_session");
        error_result["message"] = std::string("세션이 존재하지 않거나 만료되었습니다");
        return error_result;
    }
    
    auto& session = sessions[session_id];
    session.last_activity = std::chrono::system_clock::now();
    
    try {
        // 오디오 데이터 처리
        Eigen::Matrix<float, Eigen::Dynamic, 1> audio_tensor = 
            session.audio_processor->ProcessAudioBinary(binary_data);
        
        if (audio_tensor.size() == 0) {
            std::map<std::string, std::any> no_audio_result;
            no_audio_result["session_id"] = session_id;
            no_audio_result["status"] = std::string("no_valid_audio");
            no_audio_result["result"] = CreateEmptyResult();
            return no_audio_result;
        }
        
        // 평가 진행
        std::map<std::string, std::any> metadata;
        metadata["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        
        auto result = session.eval_controller->ProcessRecognitionResult(audio_tensor, metadata);
        
        // 진행 상태 정보 추가
        result["session_id"] = session_id;
        result["status"] = std::string("in_progress");
        
        // 모든 블록 평가 완료 확인
        bool all_blocks_evaluated = session.eval_controller->AreAllBlocksEvaluated();
        if (all_blocks_evaluated) {
            result["status"] = std::string("completed");
        }
        
        return result;
    } catch (const std::exception& e) {
        std::string error_msg = "오디오 평가 오류: " + std::string(e.what());
        LOG_ERROR("EngineCoordinator", error_msg);
        
        std::map<std::string, std::any> error_result;
        error_result["error"] = std::string("evaluation_failed");
        error_result["message"] = error_msg;
        return error_result;
    }
}

std::map<std::string, std::any> EngineCoordinator::CloseSession(const std::string& session_id) {
    if (sessions.find(session_id) == sessions.end()) {
        std::map<std::string, std::any> error_result;
        error_result["error"] = std::string("invalid_session");
        return error_result;
    }
    
    // 세션 정리
    sessions.erase(session_id);
    
    std::map<std::string, std::any> result;
    result["status"] = std::string("session_closed");
    result["session_id"] = session_id;
    
    LOG_INFO("EngineCoordinator", "세션 종료: " + session_id);
    return result;
}

std::map<std::string, std::any> EngineCoordinator::GetSessionStatus(const std::string& session_id) {
    if (sessions.find(session_id) == sessions.end()) {
        std::map<std::string, std::any> error_result;
        error_result["error"] = std::string("invalid_session");
        error_result["message"] = std::string("세션이 존재하지 않거나 만료되었습니다");
        return error_result;
    }
    
    auto& session = sessions[session_id];
    
    // 각 컴포넌트에서 상태 정보 수집
    auto summary = session.eval_controller->GetEvaluationSummary();
    
    std::map<std::string, std::any> result;
    result["session_id"] = session_id;
    result["created_at"] = std::chrono::system_clock::to_time_t(session.created_at);
    result["last_activity"] = std::chrono::system_clock::to_time_t(session.last_activity);
    
    std::map<std::string, std::any> current_progress;
    current_progress["active_block"] = session.sentence_manager->active_block_id;
    current_progress["total_blocks"] = session.sentence_manager->blocks.size();
    current_progress["overall_score"] = summary["overall_score"];
    
    result["current_progress"] = current_progress;
    result["all_completed"] = session.eval_controller->AreAllBlocksEvaluated();
    
    return result;
}

int EngineCoordinator::CleanupInactiveSessions(float max_inactive_time) {
    std::vector<std::string> sessions_to_remove;
    auto current_time = std::chrono::system_clock::now();
    
    for (const auto& [session_id, session] : sessions) {
        auto inactive_duration = std::chrono::duration_cast<std::chrono::seconds>(
            current_time - session.last_activity).count();
        
        if (inactive_duration > max_inactive_time) {
            sessions_to_remove.push_back(session_id);
        }
    }
    
    for (const auto& session_id : sessions_to_remove) {
        CloseSession(session_id);
    }
    
    LOG_INFO("EngineCoordinator", "비활성 세션 정리: " + std::to_string(sessions_to_remove.size()) + "개 제거됨");
    return sessions_to_remove.size();
}

std::map<std::string, std::any> EngineCoordinator::CreateEmptyResult() const {
    std::map<std::string, std::any> result;
    result["total_avg_score"] = 0.0f;
    result["pronunciation"] = 0.0f;
    result["resource_version"] = std::string("1.0.0");
    result["words"] = std::vector<std::map<std::string, std::any>>();
    result["eof"] = false;
    return result;
}

} // namespace realtime_engine_ko