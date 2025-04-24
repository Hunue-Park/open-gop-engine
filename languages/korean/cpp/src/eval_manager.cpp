// src/cpp/src/eval_manager.cpp
#include "realtime_engine_ko/eval_manager.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <algorithm>
#include <chrono>

namespace realtime_engine_ko {

EvaluationController::EvaluationController(
    std::shared_ptr<Wav2VecCTCOnnxCore> recognition_engine,
    std::shared_ptr<SentenceBlockManager> sentence_manager,
    std::shared_ptr<ProgressTracker> progress_tracker,
    float confidence_threshold,
    float min_time_between_evals)
    : recognition_engine(recognition_engine),
      sentence_manager(sentence_manager),
      progress_tracker(progress_tracker),
      confidence_threshold(confidence_threshold),
      min_time_between_evals(min_time_between_evals),
      last_eval_time(std::nullopt) {
    
    LOG_INFO("EvaluationController", "EvaluationController 초기화 완료");
}

std::map<std::string, std::any> EvaluationController::ProcessRecognitionResult(
    const Eigen::Matrix<float, Eigen::Dynamic, 1>& audio_chunk,
    const std::map<std::string, std::any>& metadata) {
    
    // 활성 윈도우 내 블록 ID 목록 가져오기
    std::vector<int> active_window = progress_tracker->GetActiveWindow();
    
    // 오디오 청크가 비어 있으면 현재 상태만 반환
    if (audio_chunk.size() == 0) {
        return CreateResultFormat();
    }
    
    // 활성 윈도우 내 모든 블록에 대해 매칭 시도
    int best_match_id = -1;
    float best_match_score = -std::numeric_limits<float>::infinity();
    
    for (int block_id : active_window) {
        auto block = sentence_manager->GetBlock(block_id);
        if (!block) {
            continue;
        }
        
        // 이미 평가된 블록은 건너뛰기
        if (block->status == BlockStatus::EVALUATED) {
            continue;
        }
        
        // 현재 블록의 컨텍스트 수집
        std::string context_before = "";
        std::string context_after = "";
        
        // 이전 블록들을 context_before로 수집 (최대 2개)
        std::vector<std::string> prev_blocks;
        for (int i = std::max(0, block_id - 2); i < block_id; ++i) {
            auto prev_block = sentence_manager->GetBlock(i);
            if (prev_block) {
                prev_blocks.push_back(prev_block->text);
            }
        }
        
        if (!prev_blocks.empty()) {
            for (size_t i = 0; i < prev_blocks.size(); ++i) {
                context_before += prev_blocks[i];
                if (i < prev_blocks.size() - 1) {
                    context_before += " ";
                }
            }
        }
        
        // 다음 블록들을 context_after로 수집 (최대 2개)
        std::vector<std::string> next_blocks;
        for (int i = block_id + 1; i < std::min(block_id + 3, static_cast<int>(sentence_manager->blocks.size())); ++i) {
            auto next_block = sentence_manager->GetBlock(i);
            if (next_block) {
                next_blocks.push_back(next_block->text);
            }
        }
        
        if (!next_blocks.empty()) {
            for (size_t i = 0; i < next_blocks.size(); ++i) {
                context_after += next_blocks[i];
                if (i < next_blocks.size() - 1) {
                    context_after += " ";
                }
            }
        }
        
        // 블록 텍스트로 GOP 계산 (컨텍스트 포함)
        try {
            auto gop_result = recognition_engine->CalculateGopWithContext(
                audio_chunk,
                block->text,
                context_before,
                context_after,
                // 컨텍스트 내 위치는 항상 0 (단독 블록 평가 시)
                context_before.empty() ? std::optional<int>(0) : std::nullopt
            );
            
            // 전체 발음 점수 추출
            float overall_score = std::any_cast<float>(gop_result["overall"]);
            
            // 현재 블록이 최적 매치인지 확인
            if (overall_score > best_match_score) {
                best_match_score = overall_score;
                best_match_id = block_id;
            }
            
            // 결과 캐싱
            std::map<std::string, std::any> cache_entry;
            cache_entry["gop_score"] = overall_score;
            cache_entry["details"] = gop_result;
            cache_entry["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            
            cached_results[block_id] = cache_entry;
            
        } catch (const std::exception& e) {
            std::stringstream ss;
            ss << "블록 " << block_id << " GOP 계산 중 오류: " << e.what();
            LOG_ERROR("EvaluationController", ss.str());
        }
    }
    
    // 최적 매치 블록을 찾았으면 해당 블록 평가 진행
    if (best_match_id >= 0 && best_match_score >= confidence_threshold) {
        // 평가 가능한 시점인지 확인
        auto current_time = std::chrono::system_clock::now();
        bool can_evaluate = !last_eval_time.has_value() || 
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             current_time - last_eval_time.value()).count() / 1000.0 >= min_time_between_evals;
        
        if (can_evaluate) {
            // 어떤 블록이든 매치된 블록 평가
            EvaluateBlock(best_match_id, cached_results[best_match_id]);
            last_eval_time = current_time;
            
            // 활성 블록 업데이트 (케이스별 처리)
            if (best_match_id == sentence_manager->active_block_id) {
                // 1. 현재 활성 블록이 인식된 경우 - 다음 블록으로 진행
                sentence_manager->AdvanceActiveBlock();
                
                // ProgressTracker 업데이트
                progress_tracker->SetCurrentIndex(sentence_manager->active_block_id);
                
            } else if (best_match_id < sentence_manager->active_block_id) {
                // 2. 이전 블록이 인식된 경우 (순서가 뒤바뀐 발화)
                std::stringstream ss;
                ss << "이전 블록 " << best_match_id << "가 인식됨 (현재 활성 블록: " 
                   << sentence_manager->active_block_id << ")";
                LOG_INFO("EvaluationController", ss.str());
                
                // 활성 블록 상태 업데이트
                sentence_manager->SetActiveBlock(best_match_id);
                
                // 평가 후에는 다음 블록으로 이동
                sentence_manager->AdvanceActiveBlock();
                
                // ProgressTracker 업데이트
                progress_tracker->SetCurrentIndex(sentence_manager->active_block_id);
                
            } else if (best_match_id > sentence_manager->active_block_id) {
                // 3. 다음 블록이 인식된 경우 (블록을 건너뛴 경우)
                std::stringstream ss;
                ss << "건너뛴 블록 " << best_match_id << "가 인식됨 (현재 활성 블록: " 
                   << sentence_manager->active_block_id << ")";
                LOG_INFO("EvaluationController", ss.str());
                
                // 활성 블록을 인식된 블록 다음으로 설정
                int next_block_id = best_match_id + 1;
                if (next_block_id >= static_cast<int>(sentence_manager->blocks.size())) {
                    // 마지막 블록이면 마지막 블록을 활성 상태로 유지
                    sentence_manager->SetActiveBlock(best_match_id);
                } else {
                    sentence_manager->SetActiveBlock(next_block_id);
                }
                
                // ProgressTracker 업데이트
                progress_tracker->SetCurrentIndex(sentence_manager->active_block_id);
            }
        }
    }
    
    // 새 형식으로 결과 반환
    return CreateResultFormat();
}

std::map<std::string, std::any> EvaluationController::CreateResultFormat() const {
    // 평가된 블록 수집
    std::vector<std::shared_ptr<SentenceBlock>> evaluated_blocks;
    for (const auto& block : sentence_manager->blocks) {
        if (block->status == BlockStatus::EVALUATED) {
            evaluated_blocks.push_back(block);
        }
    }
    
    // 평가된 블록이 없으면 빈 결과 반환
    if (evaluated_blocks.empty()) {
        std::map<std::string, std::any> empty_result;
        
        std::map<std::string, std::any> inner_result;
        inner_result["overall"] = 0.0f;
        inner_result["pronunciation"] = 0.0f;
        inner_result["resource_version"] = std::string("1.0.0");
        inner_result["words"] = std::vector<std::map<std::string, std::any>>();
        inner_result["eof"] = false;
        
        empty_result["result"] = inner_result;
        
        return empty_result;
    }
    
    // 평균 점수 계산
    float avg_score = 0.0f;
    for (const auto& block : evaluated_blocks) {
        if (block->gop_score.has_value()) {
            avg_score += block->gop_score.value();
        }
    }
    avg_score /= evaluated_blocks.size();
    
    // 소수점 첫째 자리까지 반올림
    float avg_score_rounded = std::round(avg_score * 10) / 10;
    
    // 단어별 점수 구성
    std::vector<std::map<std::string, std::any>> words;
    for (const auto& block : evaluated_blocks) {
        if (block->gop_score.has_value()) {
            std::map<std::string, std::any> word_map;
            word_map["word"] = block->text;
            
            std::map<std::string, std::any> scores_map;
            scores_map["pronunciation"] = std::round(block->gop_score.value() * 10) / 10;
            word_map["scores"] = scores_map;
            
            words.push_back(word_map);
        }
    }
    
    // 모든 블록이 평가 완료되었는지 확인
    bool all_blocks_evaluated = evaluated_blocks.size() == sentence_manager->blocks.size();
    
    // 기본 결과 구조
    std::map<std::string, std::any> result;
    std::map<std::string, std::any> inner_result;
    
    inner_result["overall"] = avg_score_rounded;
    inner_result["pronunciation"] = avg_score_rounded;
    inner_result["resource_version"] = std::string("1.0.0");
    inner_result["words"] = words;
    inner_result["eof"] = false;
    
    // 모든 블록이 평가 완료된 경우 추가 정보 포함
    if (all_blocks_evaluated) {
        inner_result["eof"] = true;
        inner_result["final_score"] = avg_score_rounded;
        
        // 강화된 결과 데이터 추가
        std::map<std::string, std::any> details;
        details["total_blocks"] = sentence_manager->blocks.size();
        details["completion_time"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        
        std::map<std::string, std::any> score_breakdown;
        
        // 최소 점수 계산
        float min_score = std::numeric_limits<float>::max();
        for (const auto& block : evaluated_blocks) {
            if (block->gop_score.has_value() && block->gop_score.value() < min_score) {
                min_score = block->gop_score.value();
            }
        }
        
        // 최대 점수 계산
        float max_score = -std::numeric_limits<float>::max();
        for (const auto& block : evaluated_blocks) {
            if (block->gop_score.has_value() && block->gop_score.value() > max_score) {
                max_score = block->gop_score.value();
            }
        }
        
        score_breakdown["min_score"] = std::round(min_score * 10) / 10;
        score_breakdown["max_score"] = std::round(max_score * 10) / 10;
        
        details["score_breakdown"] = score_breakdown;
        inner_result["details"] = details;
    }
    
    result["result"] = inner_result;
    
    return result;
}

void EvaluationController::EvaluateBlock(int block_id, const std::map<std::string, std::any>& evaluation_data) {
    auto block = sentence_manager->GetBlock(block_id);
    if (!block) {
        return;
    }
    
    // 상태 업데이트
    if (block->status == BlockStatus::PENDING || block->status == BlockStatus::ACTIVE) {
        sentence_manager->UpdateBlockStatus(block_id, BlockStatus::RECOGNIZED);
    }
    
    // GOP 점수 설정
    float gop_score = std::any_cast<float>(evaluation_data.at("gop_score"));
    sentence_manager->SetBlockScore(block_id, gop_score);
    
    // 상태를 EVALUATED로 업데이트
    sentence_manager->UpdateBlockStatus(block_id, BlockStatus::EVALUATED);
    
    std::stringstream ss;
    ss << "블록 " << block_id << " (" << block->text << ") 평가 완료: 점수=" << block->gop_score.value();
    LOG_INFO("EvaluationController", ss.str());
}

std::map<std::string, std::any> EvaluationController::GetEvaluationSummary() const {
    // 평가된 블록 수집
    std::vector<std::shared_ptr<SentenceBlock>> evaluated_blocks;
    for (const auto& block : sentence_manager->blocks) {
        if (block->status == BlockStatus::EVALUATED) {
            evaluated_blocks.push_back(block);
        }
    }
    
    // 평가된 블록이 없으면 빈 요약 반환
    if (evaluated_blocks.empty()) {
        std::map<std::string, std::any> empty_summary;
        empty_summary["overall_score"] = 0.0f;
        
        std::map<std::string, std::any> progress;
        progress["completed"] = 0;
        progress["total"] = sentence_manager->blocks.size();
        empty_summary["progress"] = progress;
        
        empty_summary["blocks"] = std::vector<std::map<std::string, std::any>>();
        
        return empty_summary;
    }
    
    // 평균 점수 계산
    float avg_score = 0.0f;
    for (const auto& block : evaluated_blocks) {
        if (block->gop_score.has_value()) {
            avg_score += block->gop_score.value();
        }
    }
    avg_score /= evaluated_blocks.size();
    
    // 결과 구성
    std::map<std::string, std::any> summary;
    summary["overall_score"] = std::round(avg_score * 10) / 10;
    
    std::map<std::string, std::any> progress;
    progress["completed"] = evaluated_blocks.size();
    progress["total"] = sentence_manager->blocks.size();
    summary["progress"] = progress;
    
    std::vector<std::map<std::string, std::any>> blocks_status;
    for (const auto& block : sentence_manager->blocks) {
        blocks_status.push_back(block->ToDict());
    }
    summary["blocks"] = blocks_status;
    
    return summary;
}

void EvaluationController::Reset() {
    last_eval_time = std::nullopt;
    pending_evaluations.clear();
    cached_results.clear();
    
    LOG_INFO("EvaluationController", "평가 상태 초기화");
}

} // namespace realtime_engine_ko