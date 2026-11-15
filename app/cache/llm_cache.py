from datetime import datetime, timezone

from supabase import create_client
from sentence_transformers import SentenceTransformer

from app.core.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    EMBEDDING_MODEL
)


CACHE_SIMILARITY_THRESHOLD = 0.85
CACHE_MATCH_COUNT = 1


class LLMCache:
    def __init__(self):
        if not SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL is missing")

        if not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_KEY is missing")

        self.supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

        self.embeddings = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def _embed(self, question: str):
        embedding = self.embeddings.encode(
            question,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def get(self, user_id: str, question: str):
        try:
            query_embedding = self._embed(question)

            response = self.supabase.rpc(
                "match_answer_cache",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": CACHE_SIMILARITY_THRESHOLD,
                    "match_count": CACHE_MATCH_COUNT,
                    "p_user_id": user_id
                }
            ).execute()

            results = response.data or []

            if not results:
                return None

            best_match = results[0]

            return {
                "answer": best_match["answer"],
                "question": best_match["question"],
                "similarity": float(
                    best_match["similarity"]
                )
            }

        except Exception as exc:
            print(f"Cache GET error: {exc}")
            return None

    def set(
        self,
        user_id: str,
        question: str,
        answer: str
    ):
        try:
            question_embedding = self._embed(question)

            self.supabase.table(
                "answer_cache"
            ).insert({
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "question_embedding": question_embedding,
                "created_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat()
            }).execute()

        except Exception as exc:
            print(f"Cache SET error: {exc}")

    def delete(
        self,
        user_id: str,
        question: str
    ):
        try:
            self.supabase.table(
                "answer_cache"
            ).delete().eq(
                "user_id",
                user_id
            ).eq(
                "question",
                question
            ).execute()

        except Exception as exc:
            print(f"Cache DELETE error: {exc}")

    def clear_user_cache(
        self,
        user_id: str
    ):
        try:
            response = self.supabase.table(
                "answer_cache"
            ).delete().eq(
                "user_id",
                user_id
            ).execute()

            return len(response.data or [])

        except Exception as exc:
            print(f"Cache CLEAR error: {exc}")
            return 0