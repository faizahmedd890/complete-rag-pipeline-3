from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL
)

from app.rag.utils import clean_text


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )

    return vectorstore


def load_documents(vectorstore):
    data = vectorstore.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = []

    texts = data.get(
        "documents",
        []
    )

    metadatas = data.get(
        "metadatas",
        []
    )

    for index, text in enumerate(texts):

        metadata = {}

        if index < len(metadatas):
            metadata = metadatas[index] or {}

        cleaned_text = clean_text(text)

        if not cleaned_text:
            continue

        documents.append(
            Document(
                page_content=cleaned_text,
                metadata=metadata
            )
        )

    return documents