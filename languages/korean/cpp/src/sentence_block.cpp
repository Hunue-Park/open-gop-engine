// src/cpp/src/sentence_block.cpp
#include "realtime_engine_ko/sentence_block.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <algorithm>

namespace realtime_engine_ko {

// SentenceBlock 구현
SentenceBlock::SentenceBlock(const std::string& text, int block_id)
    : text(text), block_id(block_id), status(BlockStatus::PENDING),
      gop_score(std::nullopt), confidence(std::nullopt),
      recognized_at(std::nullopt), evaluated_at(std::nullopt) {
}

void SentenceBlock::SetStatus(BlockStatus status) {
    this->status = status;
}

void SentenceBlock::SetScore(float score) {
    this->gop_score = score;
}

void SentenceBlock::SetConfidence(float confidence) {
    this->confidence = confidence;
}

std::map<std::string, std::any> SentenceBlock::ToDict() const {
    std::map<std::string, std::any> result;
    
    result["text"] = text;
    result["block_id"] = block_id;
    
    // 상태를 문자열로 변환
    std::string status_str;
    switch (status) {
        case BlockStatus::PENDING:    status_str = "pending"; break;
        case BlockStatus::ACTIVE:     status_str = "active"; break;
        case BlockStatus::RECOGNIZED: status_str = "recognized"; break;
        case BlockStatus::EVALUATED:  status_str = "evaluated"; break;
    }
    result["status"] = status_str;
    
    // 선택적 값들
    if (gop_score.has_value()) {
        result["gop_score"] = gop_score.value();
    }
    if (confidence.has_value()) {
        result["confidence"] = confidence.value();
    }
    
    // 시간 정보 (epoch 시간으로 변환)
    if (recognized_at.has_value()) {
        auto time_point = recognized_at.value();
        auto duration = time_point.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        result["recognized_at"] = static_cast<double>(seconds);
    }
    if (evaluated_at.has_value()) {
        auto time_point = evaluated_at.value();
        auto duration = time_point.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        result["evaluated_at"] = static_cast<double>(seconds);
    }
    
    return result;
}

// SentenceBlockManager 구현
SentenceBlockManager::SentenceBlockManager(const std::string& sentence, const std::string& delimiter)
    : active_block_id(0) {
    // 문자열 분할 함수
    auto split = [](const std::string& str, const std::string& delim) -> std::vector<std::string> {
        std::vector<std::string> tokens;
        size_t prev = 0, pos = 0;
        do {
            pos = str.find(delim, prev);
            if (pos == std::string::npos) pos = str.length();
            std::string token = str.substr(prev, pos - prev);
            if (!token.empty()) tokens.push_back(token);
            prev = pos + delim.length();
        } while (pos < str.length() && prev < str.length());
        return tokens;
    };
    
    // 문장을 블록으로 분할
    std::vector<std::string> blocks_text = split(sentence, delimiter);
    for (size_t i = 0; i < blocks_text.size(); ++i) {
        std::string& block_text = blocks_text[i];
        
        // 앞뒤 공백 제거
        block_text.erase(0, block_text.find_first_not_of(" \t\n\r\f\v"));
        block_text.erase(block_text.find_last_not_of(" \t\n\r\f\v") + 1);
        
        if (!block_text.empty()) {
            blocks.push_back(std::make_shared<SentenceBlock>(block_text, static_cast<int>(i)));
        }
    }
    
    // 첫 번째 블록은 ACTIVE 상태로 설정
    if (!blocks.empty()) {
        blocks[0]->SetStatus(BlockStatus::ACTIVE);
    }
    
    std::stringstream ss;
    ss << "SentenceBlockManager 초기화: " << blocks.size() << " 블록 생성됨";
    LOG_INFO("SentenceBlockManager", ss.str());
}

std::shared_ptr<SentenceBlock> SentenceBlockManager::GetBlock(int block_id) const {
    if (block_id >= 0 && block_id < static_cast<int>(blocks.size())) {
        return blocks[block_id];
    }
    return nullptr;
}

std::shared_ptr<SentenceBlock> SentenceBlockManager::GetActiveBlock() const {
    return GetBlock(active_block_id);
}

bool SentenceBlockManager::SetActiveBlock(int block_id) {
    if (block_id < 0 || block_id >= static_cast<int>(blocks.size())) {
        return false;
    }
    
    // 이전 활성 블록 상태 변경
    auto current_active = GetActiveBlock();
    if (current_active && current_active->status == BlockStatus::ACTIVE) {
        current_active->SetStatus(BlockStatus::PENDING);
    }
    
    // 새 활성 블록 설정
    active_block_id = block_id;
    blocks[block_id]->SetStatus(BlockStatus::ACTIVE);
    
    std::stringstream ss;
    ss << "활성 블록 변경: " << block_id;
    LOG_INFO("SentenceBlockManager", ss.str());
    
    return true;
}

bool SentenceBlockManager::AdvanceActiveBlock() {
    int next_id = active_block_id + 1;
    return SetActiveBlock(next_id);
}

std::vector<std::shared_ptr<SentenceBlock>> SentenceBlockManager::GetWindow(int window_size) const {
    int start = std::max(0, active_block_id - window_size + 1);
    int end = active_block_id + 1;
    
    std::vector<std::shared_ptr<SentenceBlock>> result;
    for (int i = start; i < end && i < static_cast<int>(blocks.size()); ++i) {
        result.push_back(blocks[i]);
    }
    
    return result;
}

bool SentenceBlockManager::UpdateBlockStatus(int block_id, BlockStatus status) {
    auto block = GetBlock(block_id);
    if (!block) {
        return false;
    }
    
    block->SetStatus(status);
    if (status == BlockStatus::RECOGNIZED) {
        block->recognized_at = std::chrono::system_clock::now();
    }
    if (status == BlockStatus::EVALUATED) {
        block->evaluated_at = std::chrono::system_clock::now();
    }
    
    return true;
}

bool SentenceBlockManager::SetBlockScore(int block_id, float score) {
    auto block = GetBlock(block_id);
    if (!block) {
        return false;
    }
    
    block->SetScore(score);
    return true;
}

std::vector<std::map<std::string, std::any>> SentenceBlockManager::GetAllBlocksStatus() const {
    std::vector<std::map<std::string, std::any>> result;
    for (const auto& block : blocks) {
        result.push_back(block->ToDict());
    }
    return result;
}

void SentenceBlockManager::Reset() {
    for (auto& block : blocks) {
        block->SetStatus(BlockStatus::PENDING);
        block->gop_score = std::nullopt;
        block->confidence = std::nullopt;
        block->recognized_at = std::nullopt;
        block->evaluated_at = std::nullopt;
    }
    
    // 첫 번째 블록을 활성화
    if (!blocks.empty()) {
        active_block_id = 0;
        blocks[0]->SetStatus(BlockStatus::ACTIVE);
    }
    
    LOG_INFO("SentenceBlockManager", "모든 블록 상태 초기화됨");
}

} // namespace realtime_engine_ko