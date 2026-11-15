from sentence_transformers import CrossEncoder

from app.core.config import (
    RERANKER_MODEL,
    RERANK_TOP_K
)

from app.rag.utils import clean_text


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512
        )

    def rerank(
        self,
        query,
        documents
    ):
        if not documents:
            return []

        pairs = [
            [
                query,
                clean_text(
                    doc.page_content
                )
            ]
            for doc in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(
                documents,
                scores
            ),
            key=lambda item:
            float(item[1]),
            reverse=True
        )

        return [
            doc
            for doc, score in ranked[
                :RERANK_TOP_K
            ]
        ]