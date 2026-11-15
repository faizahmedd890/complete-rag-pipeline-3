from langchain_community.retrievers import (
    BM25Retriever
)

from app.core.config import TOP_K

from app.rag.employee import (
    exact_employee_search,
    normalize_employee_query
)

from app.rag.utils import clean_text


def build_bm25(documents):
    retriever = BM25Retriever.from_documents(
        documents
    )

    retriever.k = TOP_K

    return retriever


def hybrid_search(
    query,
    vectorstore,
    bm25_retriever
):
    vector_docs = (
        vectorstore.similarity_search(
            query,
            k=TOP_K
        )
    )

    bm25_docs = bm25_retriever.invoke(
        query
    )

    combined = []

    seen = set()

    for doc in vector_docs + bm25_docs:

        content = clean_text(
            doc.page_content
        )

        if not content:
            continue

        if content in seen:
            continue

        seen.add(content)

        combined.append(doc)

    return combined


def multi_query_hybrid_search(
    query,
    expanded_queries,
    vectorstore,
    bm25_retriever,
    all_documents
):
    exact_documents, normalized_query = (
        exact_employee_search(
            query,
            all_documents
        )
    )

    all_retrieved = []
    seen = set()

    for doc in exact_documents:

        content = clean_text(
            doc.page_content
        )

        if content and content not in seen:
            seen.add(content)
            all_retrieved.append(doc)

    queries = [
        query,
        normalized_query
    ]

    queries.extend(
        expanded_queries
    )

    _, employee_variants = (
        normalize_employee_query(query)
    )

    queries.extend(
        employee_variants
    )

    queries = list(
        dict.fromkeys(
            clean_text(q)
            for q in queries
            if clean_text(q)
        )
    )

    for search_query in queries:

        documents = hybrid_search(
            search_query,
            vectorstore,
            bm25_retriever
        )

        for doc in documents:

            content = clean_text(
                doc.page_content
            )

            if not content:
                continue

            if content in seen:
                continue

            seen.add(content)
            all_retrieved.append(doc)

    return all_retrieved