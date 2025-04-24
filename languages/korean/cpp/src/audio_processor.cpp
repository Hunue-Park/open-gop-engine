// src/cpp/src/audio_processor.cpp
#include "realtime_engine_ko/audio_processor.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

namespace realtime_engine_ko {

AudioProcessor::AudioProcessor(int sample_rate, float chunk_duration, float polling_interval)
    : sample_rate(sample_rate), chunk_duration(chunk_duration), polling_interval(polling_interval),
      audio_file_path(""), last_file_size(0), last_processed_pos(0), is_monitoring(false),
      monitoring_thread(nullptr), total_duration(0.0) {
    
    std::stringstream ss;
    ss << "AudioProcessor 초기화: 샘플 레이트=" << sample_rate 
       << "Hz, 청크 길이=" << chunk_duration << "초";
    LOG_INFO("AudioProcessor", ss.str());
}

AudioProcessor::~AudioProcessor() {
    StopMonitoring();
}

bool AudioProcessor::SetAudioFile(const std::string& file_path) {
    // 파일 존재 확인
    SF_INFO sf_info;
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sf_info);
    if (!file) {
        std::stringstream ss;
        ss << "파일이 존재하지 않거나 열 수 없습니다: " << file_path;
        LOG_ERROR("AudioProcessor", ss.str());
        return false;
    }
    sf_close(file);
    
    audio_file_path = file_path;
    last_file_size = sf_info.frames;
    last_processed_pos = 0;
    total_duration = 0.0;
    
    // 버퍼 초기화
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        buffer.clear();
    }
    
    // 최신 청크 초기화
    latest_chunk.resize(0);
    
    std::stringstream ss;
    ss << "오디오 파일 설정: " << file_path << " (샘플 레이트=" << sf_info.samplerate 
       << "Hz, 채널=" << sf_info.channels << ", 프레임=" << sf_info.frames << ")";
    LOG_INFO("AudioProcessor", ss.str());
    
    return true;
}

bool AudioProcessor::StartMonitoring() {
    if (is_monitoring) {
        LOG_WARNING("AudioProcessor", "이미 모니터링 중입니다.");
        return false;
    }
    
    if (audio_file_path.empty()) {
        LOG_ERROR("AudioProcessor", "모니터링할 오디오 파일이 설정되지 않았습니다.");
        return false;
    }
    
    is_monitoring = true;
    monitoring_thread = std::make_unique<std::thread>(&AudioProcessor::MonitoringLoop, this);
    
    std::stringstream ss;
    ss << "오디오 파일 모니터링 시작: " << audio_file_path;
    LOG_INFO("AudioProcessor", ss.str());
    
    return true;
}

void AudioProcessor::StopMonitoring() {
    is_monitoring = false;
    
    if (monitoring_thread && monitoring_thread->joinable()) {
        monitoring_thread->join();
    }
    
    LOG_INFO("AudioProcessor", "오디오 파일 모니터링 중지");
}

void AudioProcessor::MonitoringLoop() {
    if (audio_file_path.empty()) {
        return;
    }
    
    while (is_monitoring) {
        try {
            // 파일 크기 확인
            SF_INFO sf_info;
            SNDFILE* file = sf_open(audio_file_path.c_str(), SFM_READ, &sf_info);
            
            if (!file) {
                LOG_ERROR("AudioProcessor", "파일 크기 확인 중 오류 발생");
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    static_cast<int>(polling_interval * 1000)));
                continue;
            }
            
            sf_count_t current_size = sf_info.frames;
            sf_close(file);
            
            // 파일 크기가 증가했으면 새 데이터 처리
            if (current_size > last_file_size) {
                ProcessNewAudioData();
                last_file_size = current_size;
            }
            
            // 다음 확인까지 대기
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(polling_interval * 1000)));
            
        } catch (const std::exception& e) {
            std::stringstream ss;
            ss << "파일 모니터링 중 오류 발생: " << e.what();
            LOG_ERROR("AudioProcessor", ss.str());
            
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(polling_interval * 1000)));
        }
    }
}

void AudioProcessor::ProcessNewAudioData() {
    if (audio_file_path.empty()) {
        return;
    }
    
    try {
        // 파일 열기
        SF_INFO sf_info;
        SNDFILE* file = sf_open(audio_file_path.c_str(), SFM_READ, &sf_info);
        
        if (!file) {
            LOG_ERROR("AudioProcessor", "새 오디오 데이터 처리 중 파일 열기 실패");
            return;
        }
        
        // 마지막 처리 위치로 이동
        sf_seek(file, last_processed_pos, SEEK_SET);
        
        // 새 데이터 읽기
        sf_count_t frames_to_read = sf_info.frames - last_processed_pos;
        std::vector<float> frames(frames_to_read * sf_info.channels);
        
        sf_count_t frames_read = sf_readf_float(file, frames.data(), frames_to_read);
        
        if (frames_read > 0) {
            // 스테레오를 모노로 변환
            std::vector<float> mono_frames;
            if (sf_info.channels > 1) {
                mono_frames.resize(frames_read);
                for (sf_count_t i = 0; i < frames_read; ++i) {
                    float sum = 0.0f;
                    for (int ch = 0; ch < sf_info.channels; ++ch) {
                        sum += frames[i * sf_info.channels + ch];
                    }
                    mono_frames[i] = sum / static_cast<float>(sf_info.channels);
                }
            } else {
                mono_frames = frames;
            }
            
            // 새 데이터 처리
            AddToBuffer(mono_frames);
            
            // 처리 위치 업데이트
            last_processed_pos += frames_read;
        }
        
        sf_close(file);
        
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "새 오디오 데이터 처리 중 오류 발생: " << e.what();
        LOG_ERROR("AudioProcessor", ss.str());
    }
}

void AudioProcessor::AddToBuffer(const std::vector<float>& audio_data) {
    if (audio_data.empty()) {
        return;
    }
    
    // 데이터 정규화 (필요시)
    std::vector<float> normalized_data = audio_data;
    float max_abs = 0.0f;
    for (float sample : audio_data) {
        max_abs = std::max(max_abs, std::abs(sample));
    }
    
    if (max_abs > 1.0f) {
        for (float& sample : normalized_data) {
            sample /= max_abs;
        }
    }
    
    // 버퍼에 추가
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        buffer.push_back(std::move(normalized_data));
    }
    
    // 총 녹음 시간 업데이트
    total_duration += static_cast<float>(audio_data.size()) / static_cast<float>(sample_rate);
    
    // 새 청크 생성 가능한지 확인
    CheckAndProcessChunks();
}

void AudioProcessor::CheckAndProcessChunks() {
    // 청크 단위 샘플 수 계산
    int chunk_samples = static_cast<int>(chunk_duration * sample_rate);
    
    // 청크 추출
    AudioTensor chunk = ExtractChunk(chunk_samples);
    
    // 청크가 비어있으면 종료
    if (chunk.size() == 0) {
        return;
    }
    
    // 청크 전처리 및 저장
    latest_chunk = chunk;
    
    // 청크 타임스탬프 업데이트
    last_chunk_time = std::chrono::system_clock::now();
    
    // 청크 생성 후 콜백 호출
    if (latest_chunk.size() > 0) {
        MetadataMap metadata;
        metadata["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        metadata["duration"] = chunk_duration;
        metadata["total_duration"] = total_duration;
        
        for (const auto& callback : chunk_callbacks) {
            if (callback) {
                callback(latest_chunk, metadata);
            }
        }
    }
}

AudioProcessor::AudioTensor AudioProcessor::ExtractChunk(int chunk_samples) {
    std::lock_guard<std::mutex> lock(buffer_mutex);
    
    std::vector<float> result;
    result.reserve(chunk_samples);
    
    int samples_collected = 0;
    
    while (samples_collected < chunk_samples && !buffer.empty()) {
        auto& buffer_data = buffer.front();
        int samples_needed = chunk_samples - samples_collected;
        
        if (static_cast<int>(buffer_data.size()) <= samples_needed) {
            // 버퍼 데이터 전체 사용
            result.insert(result.end(), buffer_data.begin(), buffer_data.end());
            samples_collected += buffer_data.size();
            buffer.erase(buffer.begin());
        } else {
            // 버퍼 데이터 일부만 사용
            result.insert(result.end(), buffer_data.begin(), buffer_data.begin() + samples_needed);
            buffer_data.erase(buffer_data.begin(), buffer_data.begin() + samples_needed);
            samples_collected += samples_needed;
        }
    }
    
    // 빈 청크인 경우 그대로 반환
    if (result.empty()) {
        return AudioTensor();
    }
    
    // 결과를 AudioTensor로 변환
    AudioTensor tensor(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        tensor(i) = result[i];
    }
    
    return tensor;
}

AudioProcessor::AudioTensor AudioProcessor::PreprocessChunk(const std::vector<float>& chunk, bool do_normalize) {
    // VAD 검사 - 음성이 없으면 빈 텐서 반환
    if (!DetectVoiceActivity(chunk)) {
        return AudioTensor();
    }
    
    // 1) 정규화
    std::vector<float> processed = chunk;
    if (do_normalize) {
        // 평균 및 표준편차 계산
        float mean = 0.0f;
        for (float sample : chunk) {
            mean += sample;
        }
        mean /= chunk.size();
        
        float stddev = 0.0f;
        for (float sample : chunk) {
            stddev += (sample - mean) * (sample - mean);
        }
        stddev = std::sqrt(stddev / chunk.size());
        
        // 정규화 적용
        for (size_t i = 0; i < processed.size(); ++i) {
            processed[i] = (processed[i] - mean) / (stddev + 1e-8f);
        }
    }
    
    // 2) AudioTensor로 변환
    AudioTensor tensor(processed.size());
    for (size_t i = 0; i < processed.size(); ++i) {
        tensor(i) = processed[i];
    }
    
    return tensor;
}

bool AudioProcessor::DetectVoiceActivity(const std::vector<float>& audio_data, 
                                        float energy_threshold, int min_speech_frames) {
    // 프레임 단위로 분할 (10ms 프레임)
    int frame_size = static_cast<int>(sample_rate * 0.01);
    std::vector<std::vector<float>> frames;
    
    for (size_t i = 0; i < audio_data.size(); i += frame_size) {
        size_t end = std::min(i + frame_size, audio_data.size());
        if (end - i == frame_size) {  // 완전한 프레임만 고려
            frames.push_back(std::vector<float>(audio_data.begin() + i, audio_data.begin() + end));
        }
    }
    
    // 각 프레임의 에너지 계산
    std::vector<float> energies;
    for (const auto& frame : frames) {
        float energy = 0.0f;
        for (float sample : frame) {
            energy += sample * sample;
        }
        energy /= frame.size();
        energies.push_back(energy);
    }
    
    // 임계값을 넘는 프레임 수 계산
    int speech_frames = 0;
    for (float energy : energies) {
        if (energy > energy_threshold) {
            speech_frames++;
        }
    }
    
    // 로깅 (디버깅용)
    float avg_energy = 0.0f;
    if (!energies.empty()) {
        for (float energy : energies) {
            avg_energy += energy;
        }
        avg_energy /= energies.size();
    }
    
    std::stringstream ss;
    ss << "VAD: 평균 에너지=" << avg_energy << ", 음성 프레임=" 
       << speech_frames << "/" << energies.size();
    LOG_DEBUG("AudioProcessor", ss.str());
    
    // 임계값을 넘는 프레임이 충분한지 확인
    return speech_frames >= min_speech_frames;
}

std::pair<AudioProcessor::AudioTensor, std::map<std::string, std::any>> AudioProcessor::GetLatestChunk() const {
    MetadataMap metadata;
    metadata["timestamp"] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    metadata["duration"] = chunk_duration;
    metadata["total_duration"] = total_duration;
    
    // 실제 청크가 없는 경우
    if (latest_chunk.size() == 0) {
        return {AudioTensor(), metadata};
    }
    
    return {latest_chunk, metadata};
}

void AudioProcessor::Reset() {
    StopMonitoring();
    
    audio_file_path = "";
    last_file_size = 0;
    last_processed_pos = 0;
    
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        buffer.clear();
    }
    
    total_duration = 0.0;
    latest_chunk.resize(0);
    
    LOG_INFO("AudioProcessor", "상태 초기화 완료");
}

void AudioProcessor::AddChunkCallback(CallbackFunc callback) {
    if (callback) {
        chunk_callbacks.push_back(std::move(callback));
        LOG_INFO("AudioProcessor", "청크 콜백 함수 등록됨");
    }
}

} // namespace realtime_engine_ko