
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from supabase import create_client
import uuid

from app.core.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    EMBEDDING_MODEL,
    VECTOR_TABLE_NAME
)


class SupabaseVectorStore:

    def __init__(self):

        if not SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL is missing")

        if not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_KEY is missing")

        self.supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ):

        query_embedding = self.embeddings.embed_query(
            query
        )

        response = self.supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter": {}
            }
        ).execute()

        results = response.data or []

        documents = []

        for result in results:

            documents.append(
                Document(
                    page_content=result["content"],
                    metadata=result.get("metadata") or {}
                )
            )

        return documents

    def add_documents(
        self,
        documents,
        batch_size=100
    ):

        total_inserted = 0

        for start in range(
            0,
            len(documents),
            batch_size
        ):

            batch_docs = documents[
                start:start + batch_size
            ]

            batch_texts = [
                doc.page_content
                for doc in batch_docs
            ]

            batch_embeddings = (
                self.embeddings.embed_documents(
                    batch_texts
                )
            )

            records = []

            for doc, embedding in zip(
                batch_docs,
                batch_embeddings
            ):

                records.append({
                    "document_id": str(uuid.uuid4()),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embedding
                })

            self.supabase.table(
                VECTOR_TABLE_NAME
            ).insert(records).execute()

            total_inserted += len(records)

        return total_inserted

    def count_documents(self):

        response = (
            self.supabase
            .table(VECTOR_TABLE_NAME)
            .select("id", count="exact")
            .limit(1)
            .execute()
        )

        return response.count or 0

    def delete_all_documents(self):

        print("Deleting old Supabase documents...")

        response = (
            self.supabase
            .table(VECTOR_TABLE_NAME)
            .delete()
            .gt("id", 0)
            .execute()
        )

        print("Old documents deleted successfully.")

        return response

    def replace_documents(self, documents):

        self.delete_all_documents()

        print("Generating new embeddings...")

        inserted_count = self.add_documents(
            documents
        )

        print(
            f"New documents inserted: {inserted_count}"
        )

        return inserted_count


def load_vectorstore():

    return SupabaseVectorStore()


def load_documents(vectorstore):

    documents = []

    page_size = 1000
    start = 0

    while True:

        response = (
            vectorstore.supabase
            .table(VECTOR_TABLE_NAME)
            .select("content, metadata")
            .range(
                start,
                start + page_size - 1
            )
            .execute()
        )

        rows = response.data or []

        for row in rows:

            content = str(
                row.get("content") or ""
            ).strip()

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata=row.get("metadata") or {}
                )
            )

        if len(rows) < page_size:
            break

        start += page_size

    return documents
