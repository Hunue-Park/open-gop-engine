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
    
    // 생성자 - device는 "CPU" 또는 "CUDA"
    Wav2VecCTCOnnxCore(const std::string& onnx_model_path, 
                       const std::string& tokenizer_path,
                       const std::string& device = "CPU",
                       const std::string& matrix_path = "");
    
    // 정렬 및 디코딩 관련 메서드
    std::pair<std::vector<int>, std::vector<int>> DtwAlign(const MatrixXf& X, const MatrixXf& Y);
    std::string Transcribe(const std::vector<int>& raw_ids);
    
    // 점수 계산 관련 메서드
    float SigmoidWeight(float score, float mid = 50.0f, float steepness = 0.2f);
    float WeightedAvgWithSigmoid(const std::vector<std::pair<std::string, float>>& syllables, 
                               float mid = 50.0f, float steepness = 0.2f);
    std::vector<std::map<std::string, std::any>> GroupWordsSigmoid(
        const std::vector<std::pair<std::string, float>>& syllable_scores);
    
    // GOP 계산 메서드
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
    // 가중치 정규화 파라미터
    float weight_norm_mid;
    float weight_norm_steepness;
    
    // ONNX 런타임 관련
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    
    // 프로토타입 매트릭스
    MatrixXf prototype_matrix;
    
    // 입출력 텐서 이름
    std::string input_name;
    std::string hidden_name;
    std::string logits_name;
    
    // 초기화 도우미 메서드
    void LoadPrototypeMatrix(int vocab_size, int hidden_dim);
};

} // namespace realtime_engine_ko