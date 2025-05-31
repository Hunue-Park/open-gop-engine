from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class SessionCreateRequest(BaseModel):
    sentence: str
    options: Optional[Dict[str, Any]] = None

class SessionResponse(BaseModel):
    session_id: str
    status: str
    sentence: str
    blocks: int

class WordScores(BaseModel):
    pronunciation: float

class WordResult(BaseModel):
    word: str
    scores: WordScores

class EvaluationResult(BaseModel):
    overall: float
    pronunciation: float
    words: List[WordResult]
    eof: bool
    resource_version: str = "1.0.0"  # 엔진에서 제공하는 기본값

class EvaluationResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[EvaluationResult]  # result가 없을 수 있음 (에러 상황 등)
