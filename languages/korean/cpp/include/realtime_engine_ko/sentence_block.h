// sentence_block.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <map>
#include <chrono>
#include <any>

namespace realtime_engine_ko {

enum class BlockStatus {
    PENDING,     // 아직 처리되지 않음
    ACTIVE,      // 현재 활성화됨
    RECOGNIZED,  // 인식됨
    EVALUATED    // 평가 완료됨
};

class SentenceBlock {
public:
    SentenceBlock(const std::string& text, int block_id);
    
    void SetStatus(BlockStatus status);
    void SetScore(float score);
    void SetConfidence(float confidence);
    
    std::map<std::string, std::any> ToDict() const;
    
    std::string text;
    int block_id;
    BlockStatus status;
    std::optional<float> gop_score;
    std::optional<float> confidence;
    std::optional<std::chrono::system_clock::time_point> recognized_at;
    std::optional<std::chrono::system_clock::time_point> evaluated_at;
};

class SentenceBlockManager {
public:
    SentenceBlockManager(const std::string& sentence, const std::string& delimiter = " ");
    
    std::shared_ptr<SentenceBlock> GetBlock(int block_id) const;
    std::shared_ptr<SentenceBlock> GetActiveBlock() const;
    bool SetActiveBlock(int block_id);
    bool AdvanceActiveBlock();
    std::vector<std::shared_ptr<SentenceBlock>> GetWindow(int window_size = 3) const;
    bool UpdateBlockStatus(int block_id, BlockStatus status);
    bool SetBlockScore(int block_id, float score);
    std::vector<std::map<std::string, std::any>> GetAllBlocksStatus() const;
    void Reset();
    
    std::vector<std::shared_ptr<SentenceBlock>> blocks;
    int active_block_id;
};

} // namespace realtime_engine_ko