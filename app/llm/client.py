from groq import Groq

from app.core.config import GROQ_API_KEY


class LLMClient:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is missing"
            )

        self.client = Groq(
            api_key=GROQ_API_KEY,
            timeout=30.0
        )

    def chat(self, messages):
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages
        )

        return response.choices[0].message.content