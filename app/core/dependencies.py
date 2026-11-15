from functools import lru_cache

from app.rag.pipeline import RAGPipeline


@lru_cache
def get_rag_pipeline():
    return RAGPipeline()


def get_current_user(
    user_id: str | None = None
):
    if not user_id:
        return "anonymous"

    return user_id