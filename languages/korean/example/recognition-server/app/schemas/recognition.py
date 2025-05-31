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

class WordResult(BaseModel):
    word: str
    score: float
    start_time: float
    end_time: float

class EvaluationResponse(BaseModel):
    session_id: str
    status: str
    overall: float
    pronunciation: float
    words: List[WordResult]
    eof: bool
