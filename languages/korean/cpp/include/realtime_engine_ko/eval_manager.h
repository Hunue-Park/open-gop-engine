// eval_manager.h
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "sentence_block.h"
#include "progress_tracker.h"
#include "w2v_onnx_core.h"

namespace realtime_engine_ko {

class EvaluationController {
public:
    EvaluationController(
        std::shared_ptr<Wav2VecCTCOnnxCore> recognition_engine,
        std::shared_ptr<SentenceBlockManager> sentence_manager,
        std::shared_ptr<ProgressTracker> progress_tracker,
        float confidence_threshold = 10.0f,
        float min_time_between_evals = 0.1f);
    
    std::map<std::string, std::any> ProcessRecognitionResult(
        const Eigen::Matrix<float, Eigen::Dynamic, 1>& audio_chunk,
        const std::map<std::string, std::any>& metadata);
    
    std::map<std::string, std::any> GetEvaluationSummary() const;
    void Reset();
    
private:
    std::map<std::string, std::any> CreateResultFormat() const;
    void EvaluateBlock(int block_id, const std::map<std::string, std::any>& evaluation_data);
    
    std::shared_ptr<Wav2VecCTCOnnxCore> recognition_engine;
    std::shared_ptr<SentenceBlockManager> sentence_manager;
    std::shared_ptr<ProgressTracker> progress_tracker;
    float confidence_threshold;
    float min_time_between_evals;
    
    std::optional<std::chrono::system_clock::time_point> last_eval_time;
    std::map<int, std::map<std::string, std::any>> pending_evaluations;
    std::map<int, std::map<std::string, std::any>> cached_results;
};

} // namespace realtime_engine_ko