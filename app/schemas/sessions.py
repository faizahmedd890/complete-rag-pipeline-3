from datetime import datetime

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    user_id: str


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DeleteSessionResponse(BaseModel):
    deleted: bool