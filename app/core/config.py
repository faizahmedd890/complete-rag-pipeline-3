import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

CHROMA_DIR = BASE_DIR / os.getenv(
    "CHROMA_DIR",
    "chroma_db"
)

DATA_DIR = BASE_DIR / os.getenv(
    "DATA_DIR",
    "data"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag_documents"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-base-en-v1.5"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-base"
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

RERANK_TOP_K = int(
    os.getenv("RERANK_TOP_K", "3")
)

LLM_TIMEOUT = int(
    os.getenv("LLM_TIMEOUT", "30")
)

LLM_MAX_RETRIES = int(
    os.getenv("LLM_MAX_RETRIES", "3")
)