
from app.llm.service import LLMService

from app.rag.query_expansion import expand_query
from app.rag.retrieval import (
    multi_query_hybrid_search,
    build_bm25
)
from app.rag.reranking import Reranker
from app.rag.prompts import (
    build_prompt,
    secure_output
)
from app.rag.vector_store import (
    load_vectorstore,
    load_documents
)


class RAGPipeline:

    def __init__(
        self,
        vectorstore=None,
        documents=None,
        bm25_retriever=None,
        llm_service=None,
        reranker=None
    ):

        if vectorstore is None:
            vectorstore = load_vectorstore()

        self.vectorstore = vectorstore

        if documents is None:
            documents = load_documents(vectorstore)

        if not documents:
            raise RuntimeError(
                "No documents found in Supabase."
            )

        self.documents = documents

        if bm25_retriever is None:
            bm25_retriever = build_bm25(documents)

        self.bm25_retriever = bm25_retriever

        self.llm = llm_service or LLMService()

        self.reranker = reranker or Reranker()

    def ask(self, query):

        query = query.strip()

        if not query:
            raise ValueError(
                "Question cannot be empty"
            )

        from app.core.security import security_check

        security_result = security_check(query)

        if not security_result["allowed"]:
            return (
                "I can't provide private or "
                "confidential employee information."
            )

        expanded_queries = expand_query(
            query,
            self.llm
        )

        documents = multi_query_hybrid_search(
            query,
            expanded_queries,
            self.vectorstore,
            self.bm25_retriever,
            self.documents
        )

        reranked_documents = self.reranker.rerank(
            query,
            documents
        )

        prompt = build_prompt(
            query,
            reranked_documents
        )

        answer = self.llm.generate([
            {
                "role": "user",
                "content": prompt
            }
        ])

        return secure_output(answer)
