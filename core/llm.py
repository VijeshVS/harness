import random
import time
from typing import List

from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)

from config import settings
from core.compact import compact_messages, needs_compaction
from services.client import client
from services.hooks import get_session_id

RETRYABLE_ERRORS = (InternalServerError, RateLimitError, APIConnectionError)


class CompletionFailedError(RuntimeError):
    pass


def _retry_reason(error: Exception) -> str:
    if isinstance(error, InternalServerError):
        return "Internal server error"
    if isinstance(error, RateLimitError):
        return "Rate limit exceeded"
    return "API connection error"


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return retry_after
    ceiling = min(
        settings.llm.max_delay,
        settings.llm.base_delay * (2 ** attempt),
    )
    return random.uniform(0, ceiling)


def _retry_after(error: Exception) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _request_completion(
    messages: List[ChatCompletionMessageParam],
    tools: List[ChatCompletionFunctionToolParam],
    max_tokens: int,
) -> ChatCompletion:
    return client.chat.completions.create(
        model=settings.llm.model,
        messages=messages,
        tools=tools,
        prompt_cache_key=get_session_id(),
        prompt_cache_options={"mode": "implicit"},
        reasoning_effort="none",
        max_completion_tokens=max_tokens,
    )


def _handle_bad_request(
    error: BadRequestError,
    messages: List[ChatCompletionMessageParam],
) -> bool:
    print(f"BadRequestError: {error.message}")
    print(f"Request details: {error.code}, {error.param}, {error.type}")

    if error.code == "context_length_exceeded":
        print("[llm] Context length exceeded. Compacting messages and retrying...")
        messages[:] = compact_messages(messages)
        return True

    if error.code == "content_filter":
        print("[llm] Content filter triggered.")

    return False


def _wait_before_retry(attempt: int, error: Exception) -> None:
    delay = backoff_delay(attempt, _retry_after(error))
    print(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f} seconds...")
    time.sleep(delay)


def create_completion(
    messages: List[ChatCompletionMessageParam],
    tools: List[ChatCompletionFunctionToolParam],
    max_tokens: int = settings.llm.default_max_tokens,
) -> ChatCompletion:
    last_error: Exception | None = None

    for attempt in range(settings.llm.max_retries):
        if needs_compaction(messages):
            messages[:] = compact_messages(messages)
        try:
            return _request_completion(messages, tools, max_tokens)
        except BadRequestError as error:
            if not _handle_bad_request(error, messages):
                raise
            last_error = error
        except RETRYABLE_ERRORS as error:
            reason = _retry_reason(error)
            print(f"[llm] {reason}: {error.message}. Retrying...")
            last_error = error

        if attempt < settings.llm.max_retries - 1 and last_error is not None:
            _wait_before_retry(attempt, last_error)

    raise CompletionFailedError(
        "Failed to create completion after multiple attempts."
    ) from last_error
