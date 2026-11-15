import time
import uuid

from fastapi import APIRouter, HTTPException

from app.cache.llm_cache import LLMCache
from app.rag.pipeline import RAGPipeline
from app.schemas.chat import (
    ChatRequest,
    ChatResponse
)
from app.services.memory_service import MemoryService
from app.services.session_service import SessionService

from app.core.config import (
    SUPABASE_URL,
    SUPABASE_KEY
)

from supabase import create_client


router = APIRouter(
    prefix="/api",
    tags=["Chat"]
)


pipeline = RAGPipeline()
cache = LLMCache()


def get_supabase():
    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY
    )


def save_message(
    supabase,
    chat_id,
    role,
    content
):
    content = str(content).strip()

    if not content:
        return

    supabase.table("messages").insert(
        {
            "chat_id": chat_id,
            "role": role,
            "content": content
        }
    ).execute()


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    supabase = get_supabase()

    session_service = SessionService(
        supabase
    )

    memory_service = MemoryService(
        supabase
    )

    try:

        # ==================================================
        # 1. VERIFY SESSION
        # ==================================================

        session = session_service.get_session(
            request.chat_id,
            request.user_id
        )

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found."
            )

        # ==================================================
        # 2. LOAD MEMORY
        # ==================================================

        memory = memory_service.load_memory(
            request.chat_id
        )

        # ==================================================
        # 3. SAVE ORIGINAL USER QUESTION
        # ==================================================

        save_message(
            supabase,
            request.chat_id,
            "user",
            question
        )

        # ==================================================
        # 4. FIRST QUESTION → SESSION TITLE
        # ==================================================

        messages = memory_service.load_messages(
            request.chat_id
        )

        user_messages = [
            message
            for message in messages
            if message.get("role") == "user"
        ]

        if len(user_messages) == 1:

            session_service.update_title(
                request.chat_id,
                request.user_id,
                question
            )

        # ==================================================
        # 5. RESOLVE FOLLOW-UP
        # ==================================================

        contextualized_question = (
            memory_service.contextualize(
                question,
                memory
            )
        )

        # ==================================================
        # 6. CACHE
        # ==================================================

        cached = None

        # Do not use the simple cache for
        # context-dependent follow-ups.
        if contextualized_question == question:

            cached = cache.get(
                user_id=request.user_id,
                question=question
            )

        if cached:

            answer = cached["answer"]

            save_message(
                supabase,
                request.chat_id,
                "assistant",
                answer
            )

            memory_service.update_memory(
                request.chat_id
            )

            session_service.touch(
                request.chat_id,
                request.user_id
            )

            latency_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            return ChatResponse(
                request_id=request_id,
                chat_id=request.chat_id,
                answer=answer,
                cache_hit=True,
                latency_ms=latency_ms
            )

        # ==================================================
        # 7. RAG
        # ==================================================

        answer = pipeline.ask(
            contextualized_question
        )

        # ==================================================
        # 8. SAVE CACHE
        # ==================================================

        if contextualized_question == question:

            cache.set(
                user_id=request.user_id,
                question=question,
                answer=answer
            )

        # ==================================================
        # 9. SAVE ASSISTANT MESSAGE
        # ==================================================

        save_message(
            supabase,
            request.chat_id,
            "assistant",
            answer
        )

        # ==================================================
        # 10. UPDATE MEMORY
        # ==================================================

        memory_service.update_memory(
            request.chat_id
        )

        # ==================================================
        # 11. UPDATE SESSION ACTIVITY
        # ==================================================

        session_service.touch(
            request.chat_id,
            request.user_id
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        return ChatResponse(
            request_id=request_id,
            chat_id=request.chat_id,
            answer=answer,
            cache_hit=False,
            latency_ms=latency_ms
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"Chat error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )