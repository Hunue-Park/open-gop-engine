from realtime_engine_ko.recognition_engine import EngineCoordinator, RecordListener

# 클래스 메서드 별칭 추가 (내장 함수 setattr 사용)
setattr(EngineCoordinator, 'SetRecordListener', EngineCoordinator.set_record_listener)
setattr(EngineCoordinator, 'Initialize', EngineCoordinator.initialize)
setattr(EngineCoordinator, 'StartEvaluation', EngineCoordinator.start_evaluation)
setattr(EngineCoordinator, 'StopEvaluation', EngineCoordinator.stop_evaluation)

# 또는 다음과 같이 직접 클래스 변수에 할당할 수도 있습니다:
# EngineCoordinator.SetRecordListener = EngineCoordinator.set_record_listener
# EngineCoordinator.Initialize = EngineCoordinator.initialize
# EngineCoordinator.StartEvaluation = EngineCoordinator.start_evaluation
# EngineCoordinator.StopEvaluation = EngineCoordinator.stop_evaluation

__all__ = ['EngineCoordinator', 'RecordListener']