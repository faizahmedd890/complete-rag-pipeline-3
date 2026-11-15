# Complete RAG Pipeline - Refactored

This version keeps the main functionality of the uploaded RAG pipeline but separates responsibilities into modules.

## Architecture

FastAPI
  -> rate limit (Redis)
  -> answer cache (Redis)
  -> RAG
      -> exact employee lookup
      -> query expansion
      -> Chroma semantic retrieval
      -> BM25 retrieval
      -> hybrid retrieval
      -> BGE reranking
      -> LLM
      -> output security
  -> Redis cache

## Run Redis

```powershell
docker compose up -d
```

## Environment

Copy `.env.example` to `.env` and add your Groq key.

## Run API

```powershell
uv run uvicorn app.main:app --reload
```

Open:

http://127.0.0.1:8000/docs

## Test

```powershell
uv run pytest
```

## Example request

```powershell
curl.exe -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"Who is the manager of employee 8?\",\"user_id\":\"user-1\"}"
```

The first request should be a cache miss. Repeating the same question for the same user should produce a cache hit.
