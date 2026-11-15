from fastapi import APIRouter, HTTPException

from app.core.config import SUPABASE_URL, SUPABASE_KEY
from app.schemas.sessions import (
    CreateSessionRequest,
    SessionResponse,
    DeleteSessionResponse
)
from app.services.session_service import SessionService

from supabase import create_client


router = APIRouter(
    prefix="/api/sessions",
    tags=["Sessions"]
)


def get_supabase():
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )


@router.post("", response_model=SessionResponse)
def create_session(
    request: CreateSessionRequest
):
    supabase = get_supabase()

    service = SessionService(supabase)

    session = service.create(
        user_id=request.user_id
    )

    return session


@router.get(
    "/{user_id}",
    response_model=list[SessionResponse]
)
def list_sessions(user_id: str):
    supabase = get_supabase()

    service = SessionService(supabase)

    return service.list_sessions(user_id)


@router.delete(
    "/{chat_id}/{user_id}",
    response_model=DeleteSessionResponse
)
def delete_session(
    chat_id: str,
    user_id: str
):
    supabase = get_supabase()

    service = SessionService(supabase)

    deleted = service.delete(
        chat_id,
        user_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return {
        "deleted": True
    }