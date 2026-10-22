import time
from typing import List
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)
from services.client import client
from services.hooks import get_session_id
from openai import BadRequestError, InternalServerError, RateLimitError, APIConnectionError

MODEL = "gpt-5.6-luna"

def create_completion(
    messages: List[ChatCompletionMessageParam],
    tools: List[ChatCompletionFunctionToolParam],
    max_tokens: int = 1024,
) -> ChatCompletion:
    max_retries = 5
    exponential_backoff = 2
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                prompt_cache_key=get_session_id(),
                prompt_cache_options={"mode": "implicit"},
                reasoning_effort="none",
                max_completion_tokens=max_tokens,
            )
            return res
        except BadRequestError as e:
            print(f"BadRequestError: {e.message}")
            print(f"Request details: {e.code}, {e.param}, {e.type}")
            if e.code == "context_length_exceeded":
                print("[llm] Context length exceeded. Trimming messages and retrying...")
                compact_messages = messages[-10:]
                for i in range(len(compact_messages)):
                    content = compact_messages[i].get("content")
                    if isinstance(content, str) and len(content) > 1000:
                        compact_messages[i]["content"] = content[:1000] + "..."
                messages = compact_messages
            elif e.code == "content_filter":
                print("[llm] Content filter triggered.")
                pass
        except InternalServerError as e:
            print(f"[llm] Internal server error: {e.message}. Retrying...")
        except RateLimitError as e:
            print(f"[llm] Rate limit exceeded: {e.message}. Retrying...")
        except APIConnectionError as e:
            print(f"[llm] API connection error: {e.message}. Retrying...")
        print(f"Attempt {attempt + 1} failed. Retrying in {exponential_backoff ** attempt} seconds...")
        time.sleep(exponential_backoff ** attempt)
    raise Exception("Failed to create completion after multiple attempts.")