// realtime_engine_c.cpp
#include "realtime_engine_ko_c/realtime_engine_c.h"
#include "realtime_engine_ko/recognition_engine.h"
#include <string>
#include <cstring>
#include <nlohmann/json.hpp>

using namespace realtime_engine_ko;

// 문자열 복사 헬퍼 함수
static char* copy_string(const std::string& str) {
    char* result = new char[str.length() + 1];
    strcpy(result, str.c_str());
    return result;
}

// JSON으로 변환 헬퍼 함수
static char* convert_to_json(const std::map<std::string, std::any>& map_data) {
    nlohmann::json json_data;
    
    for (const auto& [key, value] : map_data) {
        try {
            if (value.type() == typeid(std::string)) {
                json_data[key] = std::any_cast<std::string>(value);
            } 
            else if (value.type() == typeid(int)) {
                json_data[key] = std::any_cast<int>(value);
            }
            else if (value.type() == typeid(float)) {
                json_data[key] = std::any_cast<float>(value);
            }
            else if (value.type() == typeid(double)) {
                json_data[key] = std::any_cast<double>(value);
            }
            else if (value.type() == typeid(bool)) {
                json_data[key] = std::any_cast<bool>(value);
            }
            // 다른 타입들은 필요에 따라 추가
        } 
        catch (const std::bad_any_cast&) {
            json_data[key] = nullptr;
        }
    }
    
    std::string json_str = json_data.dump();
    return copy_string(json_str);
}

extern "C" {

EngineCoordinatorHandle engine_create(
    const char* onnx_model_path,
    const char* tokenizer_path,
    const char* device,
    float update_interval,
    float confidence_threshold)
{
    try {
        return reinterpret_cast<EngineCoordinatorHandle>(
            new realtime_engine_ko::EngineCoordinator(
                onnx_model_path,
                tokenizer_path,
                device,
                update_interval,
                confidence_threshold
            )
        );
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void engine_set_listener(
    EngineCoordinatorHandle handle,
    StartCallbackFn on_start,
    TickCallbackFn on_tick,
    FailCallbackFn on_fail,
    EndCallbackFn on_end,
    ScoreCallbackFn on_score)
{
    if (!handle) return;
    
    std::function<void()> start_fn = on_start ? 
        [on_start]() { on_start(); } : std::function<void()>();
        
    std::function<void(int, int)> tick_fn = on_tick ? 
        [on_tick](int current, int total) { on_tick(current, total); } : std::function<void(int, int)>();
        
    std::function<void(const std::string&)> fail_fn = on_fail ? 
        [on_fail](const std::string& msg) { on_fail(msg.c_str()); } : std::function<void(const std::string&)>();
        
    std::function<void()> end_fn = on_end ? 
        [on_end]() { on_end(); } : std::function<void()>();
        
    std::function<void(const std::string&)> score_fn = on_score ? 
        [on_score](const std::string& json) { on_score(json.c_str()); } : std::function<void(const std::string&)>();
    
    RecordListener listener(start_fn, tick_fn, fail_fn, end_fn, score_fn);
    
    reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->SetRecordListener(listener);
}

bool engine_initialize(
    EngineCoordinatorHandle handle,
    const char* sentence,
    float audio_polling_interval,
    float min_time_between_evals)
{
    if (!handle) return false;
    return reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->Initialize(
        sentence, 
        audio_polling_interval, 
        min_time_between_evals
    );
}

bool engine_start_evaluation(
    EngineCoordinatorHandle handle, 
    const char* audio_file_path)
{
    if (!handle) return false;
    return reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->StartEvaluation(audio_file_path);
}

void engine_stop_evaluation(EngineCoordinatorHandle handle)
{
    if (!handle) return;
    reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->StopEvaluation();
}

void engine_reset(EngineCoordinatorHandle handle)
{
    if (!handle) return;
    reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->Reset();
}

const char* engine_get_results(EngineCoordinatorHandle handle)
{
    if (!handle) return nullptr;
    auto results = reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->GetResults();
    return convert_to_json(results);
}

const char* engine_evaluate_speech(
    EngineCoordinatorHandle handle,
    const char* sentence,
    const char* audio_file_path)
{
    if (!handle) return nullptr;
    
    RecordListener empty_listener;
    auto results = reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle)->EvaluateSpeech(
        sentence, 
        audio_file_path,
        empty_listener
    );
    
    return convert_to_json(results);
}

void engine_destroy(EngineCoordinatorHandle handle)
{
    if (handle) {
        delete reinterpret_cast<realtime_engine_ko::EngineCoordinator*>(handle);
    }
}

void engine_free_string(char* str)
{
    if (str) {
        delete[] str;
    }
}

} // extern "C"