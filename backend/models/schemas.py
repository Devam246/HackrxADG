from typing import List, Optional
from pydantic import BaseModel, HttpUrl


class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]


class HackathonResponse(BaseModel):
    answers: List[str]
    processing_time: Optional[float] = None
    document_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    environment: str


class PingResponse(BaseModel):
    message: str
    timestamp: float
