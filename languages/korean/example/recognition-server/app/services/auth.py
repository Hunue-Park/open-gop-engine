import time
import hashlib
from typing import Tuple, Optional


class AuthService:
    def __init__(self, secret_key: str, app_key: str):
        self.secret_key = secret_key
        self.app_key = app_key
        # 허용되는 타임스탬프 차이 (초)
        self.timestamp_threshold = 300  # 5분

    def verify_connect_signature(self, timestamp: str, signature: str) -> bool:
        """Connect 요청의 서명 검증"""
        expected_sig = self._generate_connect_signature(timestamp)
        return self._safe_compare(signature, expected_sig)

    def verify_start_signature(
        self, timestamp: str, user_id: str, signature: str
    ) -> bool:
        """Start 요청의 서명 검증"""
        expected_sig = self._generate_start_signature(timestamp, user_id)
        return self._safe_compare(signature, expected_sig)

    def verify_timestamp(self, timestamp: str) -> bool:
        """타임스탬프 유효성 검증"""
        try:
            ts = int(timestamp)
            current_time = int(time.time())
            return abs(current_time - ts) < self.timestamp_threshold
        except (ValueError, TypeError):
            return False

    def _generate_connect_signature(self, timestamp: str) -> str:
        """Connect 요청의 서명 생성: appKey + timestamp + secretKey"""
        message = f"{self.app_key}{timestamp}{self.secret_key}"
        return hashlib.sha1(message.encode()).hexdigest()

    def _generate_start_signature(self, timestamp: str, user_id: str) -> str:
        """Start 요청의 서명 생성: appKey + timestamp + userId + secretKey"""
        message = f"{self.app_key}{timestamp}{user_id}{self.secret_key}"
        return hashlib.sha1(message.encode()).hexdigest()

    def _safe_compare(self, a: str, b: str) -> bool:
        """일정 시간 문자열 비교 (타이밍 공격 방지)"""
        if len(a) != len(b):
            return False

        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)

        return result == 0
