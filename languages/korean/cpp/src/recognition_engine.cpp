// src/cpp/src/recognition_engine.cpp
#include "realtime_engine_ko/recognition_engine.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <chrono>
#include <thread>
#include <nlohmann/json.hpp>

namespace realtime_engine_ko {

// RecordListener 구현
RecordListener::RecordListener(
    StartCallback on_start,
    TickCallback on_tick,
    FailCallback on_start_record_fail,
    EndCallback on_record_end,
    ScoreCallback on_score)
    : on_start(on_start), on_tick(on_tick), on_start_record_fail(on_start_record_fail),
      on_record_end(on_record_end), on_score(on_score) {
}

// EngineCoordinator 구현
EngineCoordinator::EngineCoordinator(
    const std::string& onnx_model_path,
    const std::string& tokenizer_path,
    const std::string& device,
    float update_interval,
    float confidence_threshold)
    : is_initialized(false), is_running(false),
      update_interval(update_interval), confidence_threshold(confidence_threshold),
      timer_thread(nullptr) {
    
    try {
        // 인식 엔진 초기화 (unique_ptr 대신 shared_ptr 사용)
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
    StopEvaluation();
}

void EngineCoordinator::SetRecordListener(const RecordListener& record_listener) {
    this->record_listener = record_listener;
}

bool EngineCoordinator::Initialize(const std::string& sentence, float audio_polling_interval, float min_time_between_evals) {
    try {
        // 문장 블록 관리자 초기화
        sentence_manager = std::make_shared<SentenceBlockManager>(sentence);
        
        // 진행 추적기 초기화
        progress_tracker = std::make_shared<ProgressTracker>(
            sentence_manager->blocks.size(), 3, true);
        
        // 오디오 프로세서 초기화
        audio_processor = std::make_shared<AudioProcessor>(
            16000, 2.0f, audio_polling_interval);
        
        // 평가 컨트롤러 초기화
        eval_controller = std::make_shared<EvaluationController>(
            recognition_engine,
            sentence_manager,
            progress_tracker,
            confidence_threshold,
            min_time_between_evals);
        
        // 오디오 처리 이벤트 등록
        audio_processor->AddChunkCallback(
            [this](const AudioProcessor::AudioTensor& chunk, const std::map<std::string, std::any>& metadata) {
                this->OnNewChunk(chunk, metadata);
            });
        
        is_initialized = true;
        
        std::stringstream ss;
        ss << "시스템 초기화 완료: '" << sentence << "' (" 
           << sentence_manager->blocks.size() << " 블록)";
        LOG_INFO("EngineCoordinator", ss.str());
        
        return true;
    } catch (const std::exception& e) {
        std::string error_msg = "초기화 오류: " + std::string(e.what());
        LOG_ERROR("EngineCoordinator", error_msg);
        return false;
    }
}

bool EngineCoordinator::StartEvaluation(const std::string& audio_file_path) {
    if (!is_initialized) {
        std::string error_msg = "초기화되지 않은 상태에서 평가를 시작할 수 없습니다.";
        LOG_ERROR("EngineCoordinator", error_msg);
        if (record_listener.on_start_record_fail) {
            record_listener.on_start_record_fail(error_msg);
        }
        return false;
    }
    
    if (is_running) {
        LOG_WARNING("EngineCoordinator", "이미 평가가 진행 중입니다.");
        return false;
    }
    
    try {
        // 오디오 파일 설정
        if (!audio_processor->SetAudioFile(audio_file_path)) {
            std::string error_msg = "오디오 파일 설정 실패: " + audio_file_path;
            LOG_ERROR("EngineCoordinator", error_msg);
            if (record_listener.on_start_record_fail) {
                record_listener.on_start_record_fail(error_msg);
            }
            return false;
        }
        
        // 오디오 모니터링 시작
        if (!audio_processor->StartMonitoring()) {
            std::string error_msg = "오디오 모니터링 시작 실패";
            LOG_ERROR("EngineCoordinator", error_msg);
            if (record_listener.on_start_record_fail) {
                record_listener.on_start_record_fail(error_msg);
            }
            return false;
        }
        
        // 진행 추적 시작
        progress_tracker->Start();
        
        // 타이머 스레드 시작 (주기적 틱 이벤트용)
        is_running = true;
        timer_thread = std::make_unique<std::thread>(&EngineCoordinator::TimerLoop, this);
        
        // 시작 이벤트 호출
        if (record_listener.on_start) {
            record_listener.on_start();
        }
        
        LOG_INFO("EngineCoordinator", "평가 시작");
        return true;
    } catch (const std::exception& e) {
        std::string error_msg = "평가 시작 오류: " + std::string(e.what());
        LOG_ERROR("EngineCoordinator", error_msg);
        if (record_listener.on_start_record_fail) {
            record_listener.on_start_record_fail(error_msg);
        }
        return false;
    }
}

void EngineCoordinator::StopEvaluation() {
    if (!is_running) {
        return;
    }
    
    is_running = false;
    
    if (timer_thread && timer_thread->joinable()) {
        timer_thread->join();
        timer_thread.reset();
    }
    
    if (audio_processor) {
        audio_processor->StopMonitoring();
    }
    
    // 종료 이벤트 호출
    if (record_listener.on_record_end) {
        record_listener.on_record_end();
    }
    
    LOG_INFO("EngineCoordinator", "평가 중지");
}

void EngineCoordinator::TimerLoop() {
    while (is_running) {
        try {
            // 진행 상태 확인
            if (sentence_manager && record_listener.on_tick) {
                int current = sentence_manager->active_block_id + 1;
                int total = sentence_manager->blocks.size();
                record_listener.on_tick(current, total);
            }
            
            // 다음 틱까지 대기
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(update_interval * 1000)));
        } catch (const std::exception& e) {
            std::string error_msg = "타이머 루프 오류: " + std::string(e.what());
            LOG_ERROR("EngineCoordinator", error_msg);
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(update_interval * 1000)));
        }
    }
}

void EngineCoordinator::OnNewChunk(
    const AudioProcessor::AudioTensor& audio_chunk, 
    const std::map<std::string, std::any>& metadata) {
    
    try {
        if (!is_running) {
            return;
        }
        
        // 인식 결과 처리
        if (audio_chunk.size() > 0) {
            auto result = eval_controller->ProcessRecognitionResult(audio_chunk, metadata);
            
            // 결과 스코어 이벤트 호출
            if (record_listener.on_score) {
                // JSON 문자열로 변환 (SpeechSuper와 유사하게)
                nlohmann::json json_result;
                
                // ResultMap을 JSON으로 변환
                std::function<void(const std::map<std::string, std::any>&, nlohmann::json&)> convert_map;
                convert_map = [&convert_map](const std::map<std::string, std::any>& map, nlohmann::json& json) {
                    for (const auto& [key, value] : map) {
                        try {
                            if (value.type() == typeid(int)) {
                                json[key] = std::any_cast<int>(value);
                            } else if (value.type() == typeid(float)) {
                                json[key] = std::any_cast<float>(value);
                            } else if (value.type() == typeid(double)) {
                                json[key] = std::any_cast<double>(value);
                            } else if (value.type() == typeid(bool)) {
                                json[key] = std::any_cast<bool>(value);
                            } else if (value.type() == typeid(std::string)) {
                                json[key] = std::any_cast<std::string>(value);
                            } else if (value.type() == typeid(std::vector<std::map<std::string, std::any>>)) {
                                auto vec = std::any_cast<std::vector<std::map<std::string, std::any>>>(value);
                                nlohmann::json json_array = nlohmann::json::array();
                                for (const auto& item : vec) {
                                    nlohmann::json json_item;
                                    convert_map(item, json_item);
                                    json_array.push_back(json_item);
                                }
                                json[key] = json_array;
                            } else if (value.type() == typeid(std::map<std::string, std::any>)) {
                                auto nested_map = std::any_cast<std::map<std::string, std::any>>(value);
                                nlohmann::json nested_json;
                                convert_map(nested_map, nested_json);
                                json[key] = nested_json;
                            }
                        } catch (const std::exception& e) {
                            LOG_ERROR("EngineCoordinator", "JSON 변환 오류: " + std::string(e.what()));
                        }
                    }
                };
                
                convert_map(result, json_result);
                std::string result_json = json_result.dump();
                record_listener.on_score(result_json);
            }
        }
    } catch (const std::exception& e) {
        std::string error_msg = "청크 처리 오류: " + std::string(e.what());
        LOG_ERROR("EngineCoordinator", error_msg);
    }
}

std::map<std::string, std::any> EngineCoordinator::GetCurrentState() const {
    if (!is_initialized) {
        std::map<std::string, std::any> result;
        result["status"] = std::string("not_initialized");
        return result;
    }
    
    std::map<std::string, std::any> result;
    result["status"] = std::string(is_running ? "running" : "stopped");
    
    std::map<std::string, std::any> progress;
    progress["current"] = sentence_manager->active_block_id + 1;
    progress["total"] = sentence_manager->blocks.size();
    result["progress"] = progress;
    
    // 평가 요약 정보 추가
    if (eval_controller) {
        auto evaluation_summary = eval_controller->GetEvaluationSummary();
        for (const auto& [key, value] : evaluation_summary) {
            result[key] = value;
        }
    }
    
    return result;
}

void EngineCoordinator::Reset() {
    StopEvaluation();
    
    if (sentence_manager) {
        sentence_manager->Reset();
    }
    
    if (progress_tracker) {
        progress_tracker->Reset();
    }
    
    if (eval_controller) {
        eval_controller->Reset();
    }
    
    if (audio_processor) {
        audio_processor->Reset();
    }
    
    LOG_INFO("EngineCoordinator", "시스템 초기화됨");
}

std::map<std::string, std::any> EngineCoordinator::EvaluateSpeech(
    const std::string& sentence,
    const std::string& audio_file_path,
    const RecordListener& record_listener) {
    
    SetRecordListener(record_listener);
    
    if (!Initialize(sentence)) {
        if (this->record_listener.on_start_record_fail) {
            this->record_listener.on_start_record_fail("초기화 실패");
        }
        
        std::map<std::string, std::any> result;
        result["status"] = std::string("initialization_failed");
        return result;
    }
    
    if (!StartEvaluation(audio_file_path)) {
        if (this->record_listener.on_start_record_fail) {
            this->record_listener.on_start_record_fail("평가 시작 실패");
        }
        
        std::map<std::string, std::any> result;
        result["status"] = std::string("start_failed");
        return result;
    }
    
    return GetCurrentState();
}

std::map<std::string, std::any> EngineCoordinator::GetResults() const {
    return GetCurrentState();
}

} // namespace realtime_engine_ko