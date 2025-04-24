// src/cpp/src/w2v_onnx_core.cpp
#include "realtime_engine_ko/w2v_onnx_core.h"
#include "realtime_engine_ko/common.h"
#include "dtw/dtw_algorithm.h"
#include <sstream>
#include <fstream>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tokenizers_cpp.h>  // tokenizers-cpp 헤더 추가

namespace realtime_engine_ko {

Wav2VecCTCOnnxCore::Wav2VecCTCOnnxCore(
    const std::string& onnx_model_path,
    const std::string& tokenizer_path,
    const std::string& device)
    : weight_norm_mid(50.0f), weight_norm_steepness(0.2f) {
    
    try {
        // 1) ONNX 세션 설정
        Ort::SessionOptions session_options;
        if (device == "CPU") {
            session_options.SetIntraOpNumThreads(1);
            session_options.SetInterOpNumThreads(1);
            session_options.SetExecutionMode(ORT_SEQUENTIAL);
            
            // CPU 프로바이더 사용
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Wav2VecCTCOnnxCore");
            session = std::make_unique<Ort::Session>(env, onnx_model_path.c_str(), session_options);
        } else {
            // CUDA 프로바이더 사용
            Ort::SessionOptions cuda_options;
            cuda_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
            
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Wav2VecCTCOnnxCore");
            session = std::make_unique<Ort::Session>(env, onnx_model_path.c_str(), cuda_options);
        }
        
        // 2) 토크나이저 로드 - tokenizers-cpp 사용
        // 파일에서 바이트 로드
        std::string tokenizer_content;
        {
            std::ifstream file(tokenizer_path, std::ios::binary);
            if (!file) {
                throw std::runtime_error("토크나이저 파일을 열 수 없습니다: " + tokenizer_path);
            }
            
            // 파일 내용을 std::string으로 직접 읽어옴
            file.seekg(0, std::ios::end);
            tokenizer_content.resize(file.tellg());
            file.seekg(0, std::ios::beg);
            file.read(&tokenizer_content[0], tokenizer_content.size());
        }
        
        // tokenizers-cpp API를 사용하여 토크나이저 초기화 - 네임스페이스 수정
        tokenizer = std::unique_ptr<tokenizers::Tokenizer>(
            tokenizers::Tokenizer::FromBlobJSON(tokenizer_content)
        );
        
        LOG_INFO("Wav2VecCTCOnnxCore", "토크나이저 초기화 완료");
        
        // 3) 입출력 이름 가져오기
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 입력 정보
        size_t num_input_nodes = session->GetInputCount();
        if (num_input_nodes == 0) {
            throw std::runtime_error("모델에 입력 노드가 없습니다.");
        }
        
        // ONNX Runtime API 문서에 따라 수정
        auto input_name_ptr = session->GetInputNameAllocated(0, allocator);
        input_name = input_name_ptr.get();
        
        // 출력 정보
        size_t num_output_nodes = session->GetOutputCount();
        if (num_output_nodes < 2) {
            throw std::runtime_error("모델에 충분한 출력 노드가 없습니다. 최소 2개 필요.");
        }
        
        auto hidden_name_ptr = session->GetOutputNameAllocated(0, allocator);
        hidden_name = hidden_name_ptr.get();
        
        auto logits_name_ptr = session->GetOutputNameAllocated(1, allocator);
        logits_name = logits_name_ptr.get();
        
        LOG_INFO("Wav2VecCTCOnnxCore", "모델 I/O 이름: input=" + std::string(input_name) + 
                ", hidden=" + std::string(hidden_name) + ", logits=" + std::string(logits_name));
        
        // 4) hidden_dim & vocab_size 추론
        // 출력 텐서 정보
        Ort::TypeInfo hidden_type_info = session->GetOutputTypeInfo(0);
        Ort::TypeInfo logits_type_info = session->GetOutputTypeInfo(1);
        
        auto hidden_tensor_info = hidden_type_info.GetTensorTypeAndShapeInfo();
        auto logits_tensor_info = logits_type_info.GetTensorTypeAndShapeInfo();
        
        auto hidden_shape = hidden_tensor_info.GetShape();
        auto logits_shape = logits_tensor_info.GetShape();
        
        int hidden_dim = hidden_shape[2];
        int vocab_size = logits_shape[2];
        
        LOG_INFO("Wav2VecCTCOnnxCore", "hidden_dim=" + std::to_string(hidden_dim) + 
                ", vocab_size=" + std::to_string(vocab_size));
        
        // 5) prototype 매트릭스 로드 (C++에서는 복잡해서 간소화)
        // 실제 구현에서는 ONNX 모델에서 가중치를 직접 추출하는 로직 필요
        // 여기서는 간단히 랜덤 행렬로 초기화
        prototype_matrix = MatrixXf::Random(vocab_size, hidden_dim);
        
        LOG_INFO("Wav2VecCTCOnnxCore", "Wav2VecCTCOnnxCore 초기화 완료");
    } catch (const Ort::Exception& e) {
        std::string error_msg = "ONNX 초기화 오류: " + std::string(e.what());
        LOG_ERROR("Wav2VecCTCOnnxCore", error_msg);
        throw std::runtime_error(error_msg);
    } catch (const std::exception& e) {
        std::string error_msg = "초기화 오류: " + std::string(e.what());
        LOG_ERROR("Wav2VecCTCOnnxCore", error_msg);
        throw std::runtime_error(error_msg);
    }
}

std::pair<std::vector<int>, std::vector<int>> Wav2VecCTCOnnxCore::DtwAlign(const MatrixXf& X, const MatrixXf& Y) {
    // Eigen 행렬을 표준 벡터로 변환
    std::vector<dtw::VecD> x_vecs;
    std::vector<dtw::VecD> y_vecs;
    
    for (int i = 0; i < X.rows(); ++i) {
        dtw::VecD row;
        for (int j = 0; j < X.cols(); ++j) {
            row.push_back(static_cast<double>(X(i, j)));
        }
        x_vecs.push_back(row);
    }
    
    for (int i = 0; i < Y.rows(); ++i) {
        dtw::VecD row;
        for (int j = 0; j < Y.cols(); ++j) {
            row.push_back(static_cast<double>(Y(i, j)));
        }
        y_vecs.push_back(row);
    }
    
    // DTW 알고리즘 호출
    return dtw::dtw_align(x_vecs, y_vecs);
}

std::string Wav2VecCTCOnnxCore::Transcribe(const std::string& audio_path, const std::vector<int>& raw_ids) {
    LOG_DEBUG("Wav2VecCTCOnnxCore", "로그: " + audio_path);
    
    // special tokens
    std::string blank_token = "|";      // CTC blank
    std::string pad_token = "[PAD]";
    std::string unk_token = "[UNK]";
    
    // ID 조회 - tokenizers-cpp API 사용
    int blank_id = tokenizer->TokenToId(blank_token);
    int pad_id = tokenizer->TokenToId(pad_token);
    int unk_id = tokenizer->TokenToId(unk_token);
    
    std::unordered_set<int> special_tokens = {blank_id, pad_id, unk_id};
    
    std::vector<int> dedup_ids;
    int prev = -1;
    
    for (int idx : raw_ids) {
        // special token이면 prev도 초기화
        if (special_tokens.find(idx) != special_tokens.end()) {
            prev = -1;
            continue;
        }
        
        // non‑special인데 연속 중복이면 skip
        if (idx == prev) {
            continue;
        }
        
        dedup_ids.push_back(idx);
        prev = idx;
    }
    
    std::stringstream ss;
    ss << "CTC‑decoded IDs: ";
    for (int id : dedup_ids) {
        ss << id << " ";
    }
    LOG_DEBUG("Wav2VecCTCOnnxCore", ss.str());
    
    // 최종 디코딩 - tokenizers-cpp API 사용
    std::string text = tokenizer->Decode(dedup_ids);
    LOG_DEBUG("Wav2VecCTCOnnxCore", "디코딩된 텍스트: " + text);
    
    return text;
}

float Wav2VecCTCOnnxCore::SigmoidWeight(float score, float mid, float steepness) {
    // 0.5~1.5 범위의 가중치 생성
    return 0.5f + 1.0f / (1.0f + std::exp(-steepness * (score - mid)));
}

float Wav2VecCTCOnnxCore::WeightedAvgWithSigmoid(
    const std::vector<std::pair<std::string, float>>& syllables,
    float mid, float steepness) {
    
    float total_weighted_score = 0.0f;
    float total_weight = 0.0f;
    
    for (const auto& [syl, score] : syllables) {
        if (syl == "|") {  // 구분자 건너뛰기
            continue;
        }
        float weight = SigmoidWeight(score, mid, steepness);
        total_weighted_score += score * weight;
        total_weight += weight;
    }
    
    float raw_score = total_weight > 0.0f ? total_weighted_score / total_weight : 0.0f;
    return std::min(raw_score, 100.0f);
}

std::vector<std::map<std::string, std::any>> Wav2VecCTCOnnxCore::GroupWordsSigmoid(
    const std::vector<std::pair<std::string, float>>& syllable_scores) {
    
    std::vector<std::map<std::string, std::any>> words;
    std::vector<std::pair<std::string, float>> current_word;
    
    for (size_t i = 0; i < syllable_scores.size(); ++i) {
        const auto& [syl, score] = syllable_scores[i];
        
        if (syl == "|") {
            if (!current_word.empty()) {
                std::string word_text;
                for (const auto& [s, _] : current_word) {
                    word_text += s;
                }
                
                float word_score = std::round(
                    WeightedAvgWithSigmoid(current_word, weight_norm_mid, weight_norm_steepness));
                
                std::map<std::string, std::any> word_map;
                word_map["word"] = word_text;
                
                std::map<std::string, std::any> scores_map;
                scores_map["pronunciation"] = static_cast<int>(word_score);
                word_map["scores"] = scores_map;
                
                words.push_back(word_map);
                current_word.clear();
            }
        } else {
            current_word.push_back({syl, score});
        }
    }
    
    // 마지막 단어 처리
    if (!current_word.empty()) {
        std::string word_text;
        for (const auto& [s, _] : current_word) {
            word_text += s;
        }
        
        float word_score = std::round(
            WeightedAvgWithSigmoid(current_word, weight_norm_mid, weight_norm_steepness));
        
        std::map<std::string, std::any> word_map;
        word_map["word"] = word_text;
        
        std::map<std::string, std::any> scores_map;
        scores_map["pronunciation"] = static_cast<int>(word_score);
        word_map["scores"] = scores_map;
        
        words.push_back(word_map);
    }
    
    return words;
}

std::map<std::string, std::any> Wav2VecCTCOnnxCore::CalculateGopFromTensor(
    const Eigen::Matrix<float, Eigen::Dynamic, 1>& audio_tensor,
    const std::string& text,
    float eps) {
    
    try {
        // 입력 텐서 준비 (배치 차원 추가)
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(audio_tensor.size())};
        std::vector<float> input_data(audio_tensor.data(), audio_tensor.data() + audio_tensor.size());
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
        
        // 입출력 이름 설정
        std::vector<const char*> input_names = {input_name.c_str()};
        std::vector<const char*> output_names = {hidden_name.c_str(), logits_name.c_str()};
        
        // 모델 실행
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            output_names.size()
        );
        
        if (output_tensors.size() != 2) {
            LOG_ERROR("Wav2VecCTCOnnxCore", "ONNX 모델 실행 결과가 예상과 다릅니다.");
            throw std::runtime_error("ONNX 모델 실행 결과가 예상과 다릅니다.");
        }
        
        // 출력 텐서 정보
        auto* hidden_data = output_tensors[0].GetTensorData<float>();
        auto* logits_data = output_tensors[1].GetTensorData<float>();
        
        auto hidden_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto logits_info = output_tensors[1].GetTensorTypeAndShapeInfo();
        
        auto hidden_shape = hidden_info.GetShape();
        auto logits_shape = logits_info.GetShape();
        
        // 배치 차원 제거
        int T = hidden_shape[1];  // 시퀀스 길이
        int D = hidden_shape[2];  // 히든 차원
        int V = logits_shape[2];  // 어휘 크기
        
        // hidden 및 logits를 Eigen 행렬로 변환
        MatrixXf X(T, D);
        for (int t = 0; t < T; ++t) {
            for (int d = 0; d < D; ++d) {
                X(t, d) = hidden_data[t * D + d];
            }
        }
        
        MatrixXf logits(T, V);
        for (int t = 0; t < T; ++t) {
            for (int v = 0; v < V; ++v) {
                logits(t, v) = logits_data[t * V + v];
            }
        }
        
        // 3) temperature‐scaled softmax → probs
        MatrixXf scaled = logits;
        VectorXf max_vals = scaled.rowwise().maxCoeff();
        MatrixXf exp_logits = (scaled.colwise() - max_vals).array().exp();
        VectorXf sum_exp = exp_logits.rowwise().sum();
        MatrixXf probs = exp_logits.array().colwise() / sum_exp.array();
        
        // 4) 텍스트 토큰화 - tokenizers-cpp API 사용
        std::string processed_text = text;
        std::replace(processed_text.begin(), processed_text.end(), ' ', '|');
        std::vector<int> token_ids = tokenizer->Encode(processed_text);
        
        // special token 처리
        std::string blank_token = "|";
        int blank_id = tokenizer->TokenToId(blank_token);
        
        std::vector<int> safe_ids;
        for (int tid : token_ids) {
            if (tid >= 0 && tid < V) {
                safe_ids.push_back(tid);
            } else {
                safe_ids.push_back(blank_id);
            }
        }
        
        // 5) prototype 확장 및 DTW
        MatrixXf proto(safe_ids.size(), D);
        for (size_t i = 0; i < safe_ids.size(); ++i) {
            proto.row(i) = prototype_matrix.row(safe_ids[i]);
        }
        
        int M = safe_ids.size();
        int avg = std::max(1, T / M);
        
        // Y 확장
        MatrixXf Yexp(M * avg, D);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < avg; ++j) {
                Yexp.row(i * avg + j) = proto.row(i);
            }
        }
        
        // DTW 정렬
        auto [pX, pYexp] = DtwAlign(X, Yexp);
        
        std::vector<int> pY;
        for (int y : pYexp) {
            pY.push_back(y / avg);
        }
        
        // 6) 토큰별 프레임 수집
        std::map<int, std::vector<int>> frames;
        for (size_t i = 0; i < pX.size(); ++i) {
            frames[pY[i]].push_back(pX[i]);
        }
        
        // 7) 토큰별 로그 확률 점수
        std::vector<std::pair<std::string, float>> tok_scores;
        for (size_t idx = 0; idx < safe_ids.size(); ++idx) {
            int tid = safe_ids[idx];
            std::string tok = tokenizer->IdToToken(tid);
            
            float score;
            const auto& frs = frames[idx];
            
            if (!frs.empty()) {
                float sum_log_p = 0.0f;
                for (int fr : frs) {
                    sum_log_p += std::log(probs(fr, tid) + eps);
                }
                score = sum_log_p / frs.size();
            } else {
                score = -std::numeric_limits<float>::infinity();
            }
            
            tok_scores.push_back({tok, score});
        }
        
        // 8) [0,100] 범위로 정규화
        std::vector<float> raw;
        for (const auto& [_, s] : tok_scores) {
            if (std::isfinite(s)) {
                raw.push_back(s);
            }
        }
        
        std::vector<std::pair<std::string, float>> norm;
        if (!raw.empty()) {
            float mn = *std::min_element(raw.begin(), raw.end());
            float mx = *std::max_element(raw.begin(), raw.end());
            float span = (mx > mn) ? (mx - mn) : eps;
            
            for (const auto& [t, s] : tok_scores) {
                float normalized = std::isfinite(s) ? (s - mn) / span * 100.0f : 0.0f;
                norm.push_back({t, normalized});
            }
        } else {
            for (const auto& [t, _] : tok_scores) {
                norm.push_back({t, 0.0f});
            }
        }
        
        // 9) 단어로 그룹화
        auto words = GroupWordsSigmoid(norm);
        
        // 전체 점수 계산
        float overall = 0.0f;
        if (!words.empty()) {
            for (const auto& word : words) {
                auto scores = std::any_cast<std::map<std::string, std::any>>(word.at("scores"));
                overall += std::any_cast<int>(scores.at("pronunciation"));
            }
            overall /= words.size();
        }
        
        // 결과 맵 생성
        std::map<std::string, std::any> result;
        result["overall"] = std::round(overall * 10) / 10;  // 소수점 첫째 자리까지
        result["pronunciation"] = std::round(overall * 10) / 10;
        result["words"] = words;
        
        return result;
        
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Wav2VecCTCOnnxCore", "ONNX 실행 오류: " + std::string(e.what()));
        
        // 오류 시 빈 결과 반환
        std::map<std::string, std::any> result;
        result["overall"] = 0.0f;
        result["pronunciation"] = 0.0f;
        result["words"] = std::vector<std::map<std::string, std::any>>();
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Wav2VecCTCOnnxCore", "GOP 계산 오류: " + std::string(e.what()));
        
        // 오류 시 빈 결과 반환
        std::map<std::string, std::any> result;
        result["overall"] = 0.0f;
        result["pronunciation"] = 0.0f;
        result["words"] = std::vector<std::map<std::string, std::any>>();
        return result;
    }
}

std::map<std::string, std::any> Wav2VecCTCOnnxCore::CalculateGopWithContext(
    const Eigen::Matrix<float, Eigen::Dynamic, 1>& audio_tensor,
    const std::string& target_text,
    const std::string& context_before,
    const std::string& context_after,
    std::optional<int> target_index) {
    
    // 컨텍스트를 포함한 전체 텍스트
    std::string full_text = (context_before.empty() ? "" : context_before + " ") + 
                           target_text + 
                           (context_after.empty() ? "" : " " + context_after);
    full_text = full_text.empty() ? target_text : full_text;
    
    // 대상 텍스트의 인덱스 계산
    int actual_target_index = 0;
    if (!target_index.has_value()) {
        // context_before의 단어 수 계산
        if (!context_before.empty()) {
            int words_before = 0;
            std::istringstream iss(context_before);
            std::string word;
            while (iss >> word) {
                words_before++;
            }
            actual_target_index = words_before;
        }
    } else {
        actual_target_index = target_index.value();
    }
    
    // 대상 텍스트의 단어 수 계산
    int target_word_count = 0;
    std::istringstream iss(target_text);
    std::string word;
    while (iss >> word) {
        target_word_count++;
    }
    
    // 전체 텍스트로 GOP 계산
    auto result = CalculateGopFromTensor(audio_tensor, full_text);
    
    // 모든 단어가 있는지 확인
    auto words = std::any_cast<std::vector<std::map<std::string, std::any>>>(result["words"]);
    
    if (words.empty() || words.size() <= actual_target_index) {
        // 전체 텍스트 처리에 실패한 경우, 대상 텍스트만으로 시도
        return CalculateGopFromTensor(audio_tensor, target_text);
    }
    
    // target_index 위치의 단어들에 해당하는 결과 추출
    // 인덱스 범위 유효성 검사
    int end_index = std::min(actual_target_index + target_word_count, static_cast<int>(words.size()));
    std::vector<std::map<std::string, std::any>> target_words;
    
    for (int i = actual_target_index; i < end_index; ++i) {
        target_words.push_back(words[i]);
    }
    
    // 대상 블록에 대한 결과 생성
    float target_score = 0.0f;
    if (!target_words.empty()) {
        for (const auto& word : target_words) {
            auto scores = std::any_cast<std::map<std::string, std::any>>(word.at("scores"));
            target_score += std::any_cast<int>(scores.at("pronunciation"));
        }
        target_score /= target_words.size();
    }
    
    std::map<std::string, std::any> target_result;
    target_result["overall"] = std::round(target_score * 10) / 10;
    target_result["pronunciation"] = std::round(target_score * 10) / 10;
    target_result["words"] = target_words;
    
    return target_result;
}

} // namespace realtime_engine_ko