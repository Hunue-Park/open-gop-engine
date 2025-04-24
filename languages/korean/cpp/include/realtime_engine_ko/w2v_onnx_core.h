// w2v_onnx_core.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <any>
#include <optional>
#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

namespace realtime_engine_ko {

class Wav2VecCTCOnnxCore {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;
    
    Wav2VecCTCOnnxCore(const std::string& onnx_model_path, 
                       const std::string& tokenizer_path,
                       const std::string& device = "CPU");
    
    std::pair<std::vector<int>, std::vector<int>> DtwAlign(const MatrixXf& X, const MatrixXf& Y);
    std::string Transcribe(const std::string& audio_path, const std::vector<int>& raw_ids);
    float SigmoidWeight(float score, float mid = 35.0f, float steepness = 0.2f);
    float WeightedAvgWithSigmoid(const std::vector<std::pair<std::string, float>>& syllables, 
                                float mid = 35.0f, float steepness = 0.2f);
    std::vector<std::map<std::string, std::any>> GroupWordsSigmoid(
        const std::vector<std::pair<std::string, float>>& syllable_scores);
    
    std::map<std::string, std::any> CalculateGopFromTensor(
        const Eigen::Matrix<float, Eigen::Dynamic, 1>& audio_tensor,
        const std::string& text,
        float eps = 1e-8f);
    
    std::map<std::string, std::any> CalculateGopWithContext(
        const Eigen::Matrix<float, Eigen::Dynamic, 1>& audio_tensor,
        const std::string& target_text,
        const std::string& context_before = "",
        const std::string& context_after = "",
        std::optional<int> target_index = std::nullopt);
    
private:
    float weight_norm_mid = 50.0f;
    float weight_norm_steepness = 0.2f;
    
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    MatrixXf prototype_matrix;
    
    std::string input_name;
    std::string hidden_name;
    std::string logits_name;
};

} // namespace realtime_engine_ko