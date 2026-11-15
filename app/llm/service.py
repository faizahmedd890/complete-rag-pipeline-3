import time

from app.core.config import (
    LLM_MAX_RETRIES
)

from app.llm.client import LLMClient


class LLMService:
    def __init__(self):
        self.client = LLMClient()

    def generate(self, messages):
        last_error = None

        for attempt in range(LLM_MAX_RETRIES):
            try:
                return self.client.chat(messages)

            except Exception as exc:
                last_error = exc

                if attempt == LLM_MAX_RETRIES - 1:
                    break

                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"LLM request failed after retries: {last_error}"
        )