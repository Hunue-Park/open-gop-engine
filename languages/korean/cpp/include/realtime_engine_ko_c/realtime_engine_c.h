// realtime_engine_c.h
#ifndef REALTIME_ENGINE_KO_C_H
#define REALTIME_ENGINE_KO_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// 불투명 포인터로 C++의 EngineCoordinator 클래스 인스턴스를 참조
typedef struct EngineCoordinator* EngineCoordinatorHandle;

// 콜백 함수 타입 정의
typedef void (*StartCallbackFn)(void);
typedef void (*TickCallbackFn)(int current, int total);
typedef void (*FailCallbackFn)(const char* message);
typedef void (*EndCallbackFn)(void);
typedef void (*ScoreCallbackFn)(const char* score_json);

/**
 * 엔진 인스턴스 생성
 * @param onnx_model_path ONNX 모델 파일 경로
 * @param tokenizer_path 토크나이저 파일 경로
 * @param device 실행 디바이스 (예: "CPU")
 * @param update_interval 업데이트 간격 (초)
 * @param confidence_threshold 신뢰도 임계값
 * @return 성공 시 엔진 핸들, 실패 시 NULL
 */
EngineCoordinatorHandle engine_create(
    const char* onnx_model_path,
    const char* tokenizer_path,
    const char* device,
    float update_interval,
    float confidence_threshold);

/**
 * 리스너 콜백 설정
 * @param handle 엔진 핸들
 * @param on_start 시작 콜백
 * @param on_tick 진행 상황 콜백
 * @param on_fail 실패 콜백
 * @param on_end 종료 콜백
 * @param on_score 점수 콜백
 */
void engine_set_listener(
    EngineCoordinatorHandle handle,
    StartCallbackFn on_start,
    TickCallbackFn on_tick,
    FailCallbackFn on_fail,
    EndCallbackFn on_end,
    ScoreCallbackFn on_score);

/**
 * 엔진 초기화
 * @param handle 엔진 핸들
 * @param sentence 평가할 문장
 * @param audio_polling_interval 오디오 폴링 간격 (초)
 * @param min_time_between_evals 평가 사이 최소 시간 (초)
 * @return 성공 여부
 */
bool engine_initialize(
    EngineCoordinatorHandle handle,
    const char* sentence,
    float audio_polling_interval,
    float min_time_between_evals);

/**
 * 평가 시작
 * @param handle 엔진 핸들
 * @param audio_file_path 오디오 파일 경로
 * @return 성공 여부
 */
bool engine_start_evaluation(
    EngineCoordinatorHandle handle, 
    const char* audio_file_path);

/**
 * 평가 중지
 * @param handle 엔진 핸들
 */
void engine_stop_evaluation(EngineCoordinatorHandle handle);

/**
 * 엔진 초기화
 * @param handle 엔진 핸들
 */
void engine_reset(EngineCoordinatorHandle handle);

/**
 * 결과 가져오기
 * @param handle 엔진 핸들
 * @return JSON 형식의 결과 문자열 (메모리 해제 필요)
 */
const char* engine_get_results(EngineCoordinatorHandle handle);

/**
 * 음성 평가 (한번에 모든 과정 처리)
 * @param handle 엔진 핸들
 * @param sentence 평가할 문장
 * @param audio_file_path 오디오 파일 경로
 * @return JSON 형식의 결과 문자열 (메모리 해제 필요)
 */
const char* engine_evaluate_speech(
    EngineCoordinatorHandle handle,
    const char* sentence,
    const char* audio_file_path);

/**
 * 엔진 인스턴스 제거
 * @param handle 엔진 핸들
 */
void engine_destroy(EngineCoordinatorHandle handle);

/**
 * API에서 반환된 문자열 메모리 해제
 * @param str 해제할 문자열 포인터
 */
void engine_free_string(char* str);

#ifdef __cplusplus
}
#endif

#endif // REALTIME_ENGINE_KO_C_H