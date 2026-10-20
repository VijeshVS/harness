from typing import List
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)
from services.client import client
from services.hooks import get_session_id

MODEL = "gpt-5.6-luna"


def create_completion(
    messages: List[ChatCompletionMessageParam],
    tools: List[ChatCompletionFunctionToolParam],
) -> ChatCompletion:
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        prompt_cache_key=get_session_id(),
        prompt_cache_options={"mode": "implicit"},
        reasoning_effort="none",
    )
