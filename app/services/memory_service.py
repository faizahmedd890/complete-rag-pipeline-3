from datetime import datetime, timezone

from app.llm.client import LLMClient


SESSION_HISTORY_MESSAGES = 20
SESSION_MEMORY_MAX_CHARS = 12000


class MemoryService:
    def __init__(self, supabase):
        self.supabase = supabase
        self.llm = LLMClient()

    def load_messages(self, chat_id):
        response = (
            self.supabase
            .table("messages")
            .select("role, content, created_at")
            .eq("chat_id", chat_id)
            .order("created_at", desc=True)
            .limit(SESSION_HISTORY_MESSAGES)
            .execute()
        )

        messages = response.data or []
        messages.reverse()

        return messages

    def load_memory(self, chat_id):
        response = (
            self.supabase
            .table("session_memory")
            .select("memory")
            .eq("chat_id", chat_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            return ""

        return response.data[0].get("memory", "")

    def build_memory(self, messages):
        if not messages:
            return ""

        parts = []

        for message in messages:
            role = message.get("role", "")
            content = str(message.get("content", "")).strip()

            if not content:
                continue

            prefix = "User" if role == "user" else "Assistant"

            parts.append(
                f"{prefix}: {content}"
            )

        memory = "\n".join(parts)

        if len(memory) > SESSION_MEMORY_MAX_CHARS:
            memory = memory[-SESSION_MEMORY_MAX_CHARS:]

            first_newline = memory.find("\n")

            if first_newline != -1:
                memory = memory[first_newline + 1:]

        return memory.strip()

    def update_memory(self, chat_id):
        messages = self.load_messages(chat_id)

        memory = self.build_memory(messages)

        if not memory:
            return

        now = datetime.now(timezone.utc).isoformat()

        self.supabase.table("session_memory").upsert(
            {
                "chat_id": chat_id,
                "memory": memory,
                "updated_at": now
            },
            on_conflict="chat_id"
        ).execute()

    def contextualize(self, question, memory):
        question = question.strip()

        if not question or not memory:
            return question

        prompt = f"""
Rewrite the user's latest question into one standalone question.

Use the conversation context only to resolve references such as:

- he
- she
- they
- it
- this
- that
- the above
- previous employee
- previous person
- his role
- her department

Rules:

1. Do not answer the question.
2. Do not add facts that are not present in the conversation.
3. Resolve references using the conversation.
4. If the question is already standalone, keep it unchanged.
5. Return only the rewritten question.

Conversation context:

{memory}

Latest user question:

{question}
"""

        response = self.llm.chat(
            [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = str(response).strip()

        return result or question