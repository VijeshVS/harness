import os

from openai import OpenAI

from config import settings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=settings.llm.request_timeout,
    max_retries=settings.llm.client_max_retries,
)