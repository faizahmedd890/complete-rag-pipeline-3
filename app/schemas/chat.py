from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    user_id: str
    chat_id: str


class ChatResponse(BaseModel):
    request_id: str
    chat_id: str
    answer: str
    cache_hit: bool
    latency_ms: float