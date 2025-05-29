// src/cpp/src/audio_processor.cpp
#include "realtime_engine_ko/audio_processor.h"
#include "realtime_engine_ko/common.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <Eigen/Dense>
#include <numeric>

namespace realtime_engine_ko {

AudioProcessor::AudioProcessor(int sample_rate, float max_buffer_seconds)
    : sample_rate(sample_rate),
      max_buffer_length(static_cast<int>(sample_rate * max_buffer_seconds)),
      total_duration(0.0f) {
    
    std::stringstream ss;
    ss << "AudioProcessor 초기화: 샘플 레이트=" << sample_rate 
       << "Hz, 최대 버퍼 길이=" << max_buffer_length << " 샘플";
    LOG_INFO("AudioProcessor", ss.str());
}

AudioProcessor::~AudioProcessor() {
    // 정리 작업 (필요시)
}

Eigen::Matrix<float, Eigen::Dynamic, 1> AudioProcessor::ProcessAudioBinary(
    const std::vector<uint8_t>& binary_data) {
    
    try {
        // 바이너리 데이터 정보 출력
        LOG_INFO("AudioProcessor", "원본 바이너리 크기: " + std::to_string(binary_data.size()) + " 바이트");
        
        // 바이너리 데이터를 int16_t로 변환
        if (binary_data.size() % 2 != 0) {
            LOG_ERROR("AudioProcessor", "바이너리 데이터 크기가 올바르지 않습니다");
            return Eigen::Matrix<float, Eigen::Dynamic, 1>();
        }
        
        // int16을 float로 변환
        std::vector<float> audio_float(binary_data.size() / 2);
        const int16_t* int16_data = reinterpret_cast<const int16_t*>(binary_data.data());
        
        for (size_t i = 0; i < audio_float.size(); ++i) {
            audio_float[i] = static_cast<float>(int16_data[i]) / 32768.0f;
        }
        
        float expected_seconds = static_cast<float>(audio_float.size()) / sample_rate;
        LOG_INFO("AudioProcessor", "샘플 수: " + std::to_string(audio_float.size()) + 
                            ", 예상 시간: " + std::to_string(expected_seconds) + "초");
        
        // 버퍼 상태 출력
        LOG_INFO("AudioProcessor", "버퍼 추가 전: " + std::to_string(audio_buffer.size()) + 
                            " 샘플, 버퍼 시간: " + std::to_string(static_cast<float>(audio_buffer.size()) / sample_rate) + "초");
        
        // 오디오 버퍼에 추가
        audio_buffer.insert(audio_buffer.end(), audio_float.begin(), audio_float.end());
        
        // 최대 길이 제한
        if (audio_buffer.size() > max_buffer_length) {
            LOG_INFO("AudioProcessor", "오디오 버퍼 최대 길이에 도달했습니다. 길이: " + 
                std::to_string(audio_buffer.size()));
                
            // 버퍼 뒷부분(최신 데이터)만 유지
            audio_buffer.erase(audio_buffer.begin(), 
                audio_buffer.begin() + (audio_buffer.size() - max_buffer_length));
        }
        
        // 버퍼 통계
        float max_abs = 0.0f;
        for (const auto& sample : audio_buffer) {
            max_abs = std::max(max_abs, std::abs(sample));
        }
        
        LOG_INFO("AudioProcessor", "[CPP][AudioProcessor] 버퍼 통계: 최대값=" + std::to_string(max_abs) + 
                               ", 길이=" + std::to_string(audio_buffer.size()) + 
                               ", 시간=" + std::to_string(audio_buffer.size() / sample_rate) + "초");
        
        // VAD 직접 파라미터 지정 및 로깅
        const float vad_threshold = 0.00005f;
        const int min_frames = 10;  // Python과 일치시킴
        
        LOG_INFO("AudioProcessor", "[CPP][VAD] 파라미터: 임계값=" + std::to_string(vad_threshold) + 
                               ", 최소프레임=" + std::to_string(min_frames));
        
        bool has_voice = DetectVoiceActivity(audio_buffer, vad_threshold, min_frames);
        if (!has_voice) {
            LOG_INFO("AudioProcessor", "[CPP][AudioProcessor] VAD 실패: 음성 감지되지 않음");
            return Eigen::Matrix<float, Eigen::Dynamic, 1>();  // 빈 텐서 반환
        }
        
        // 전체 버퍼에 대한 전처리 및 반환
        auto result = PreprocessAudioData(audio_buffer);
        
        // 출력 텐서 통계
        if (result.size() > 0) {
            float max_val = 0.0f;
            for (int i = 0; i < result.size(); i++) {
                max_val = std::max(max_val, std::abs(result(i)));
            }
            LOG_INFO("AudioProcessor", "[CPP][AudioProcessor] 출력 텐서: 크기=" + std::to_string(result.size()) +
                                  ", 최대값=" + std::to_string(max_val));
        }
        
        return result;
    } catch (const std::exception& e) {
        std::stringstream ss;
        ss << "오디오 바이너리 처리 중 오류 발생: " << e.what();
        LOG_ERROR("AudioProcessor", ss.str());
        return Eigen::Matrix<float, Eigen::Dynamic, 1>();
    }
}

Eigen::Matrix<float, Eigen::Dynamic, 1> AudioProcessor::PreprocessAudioData(
    const std::vector<float>& audio_data, bool do_normalize) {
    
    // VAD 검사 - 음성이 없으면 빈 텐서 반환
    if (!DetectVoiceActivity(audio_data)) {
        return Eigen::Matrix<float, Eigen::Dynamic, 1>();
    }
    
    // 정규화
    std::vector<float> processed = audio_data;
    if (do_normalize) {
        // 평균 계산
        float mean = std::accumulate(audio_data.begin(), audio_data.end(), 0.0f) / audio_data.size();
        
        // 표준편차 계산
        float var = 0.0f;
        for (const auto& sample : audio_data) {
            var += (sample - mean) * (sample - mean);
        }
        float stddev = std::sqrt(var / audio_data.size()) + 1e-8f;
        
        // 정규화 적용
        for (size_t i = 0; i < processed.size(); ++i) {
            processed[i] = (processed[i] - mean) / stddev;
        }
    }
    
    // Eigen::Matrix로 변환
    Eigen::Matrix<float, Eigen::Dynamic, 1> result(processed.size());
    for (size_t i = 0; i < processed.size(); ++i) {
        result(i) = processed[i];
    }
    
    // 처리 통계 업데이트
    last_process_time = std::chrono::system_clock::now();
    total_duration = static_cast<float>(audio_buffer.size()) / static_cast<float>(sample_rate);
    
    return result;
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
    
    // 평균 에너지 계산
    float avg_energy = 0.0f;
    if (!energies.empty()) {
        avg_energy = std::accumulate(energies.begin(), energies.end(), 0.0f) / energies.size();
    }
    
    // 최대 에너지 계산 추가
    float max_energy = 0.0f;
    if (!energies.empty()) {
        max_energy = *std::max_element(energies.begin(), energies.end());
    }
    
    LOG_INFO("AudioProcessor", "[CPP][VAD] 에너지 통계: 최대=" + std::to_string(max_energy) +
                          ", 평균=" + std::to_string(avg_energy) + 
                          ", 음성 프레임=" + std::to_string(speech_frames) + 
                          "/" + std::to_string(energies.size()));
    
    // 임계값을 넘는 프레임이 충분한지 확인
    return speech_frames >= min_speech_frames;
}

void AudioProcessor::Reset() {
    audio_buffer.clear();
    total_duration = 0.0f;
    last_process_time = std::chrono::system_clock::time_point();
    
    LOG_INFO("AudioProcessor", "상태 초기화 완료");
}

} // namespace realtime_engine_ko