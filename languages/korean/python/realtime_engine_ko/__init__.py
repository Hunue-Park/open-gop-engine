from realtime_engine_ko.recognition_engine import EngineCoordinator

# 클래스 메서드 별칭 추가 (필요시)
setattr(EngineCoordinator, 'CreateSession', EngineCoordinator.create_session)
setattr(EngineCoordinator, 'EvaluateAudio', EngineCoordinator.evaluate_audio)
setattr(EngineCoordinator, 'CloseSession', EngineCoordinator.close_session)
setattr(EngineCoordinator, 'GetSessionStatus', EngineCoordinator.get_session_status)
setattr(EngineCoordinator, 'CleanupInactiveSessions', EngineCoordinator.cleanup_inactive_sessions)

# 또는 직접 클래스 변수에 할당:
# EngineCoordinator.CreateSession = EngineCoordinator.create_session
# EngineCoordinator.EvaluateAudio = EngineCoordinator.evaluate_audio
# EngineCoordinator.CloseSession = EngineCoordinator.close_session
# 등...

__all__ = ['EngineCoordinator']