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
            )
            return res
        except BadRequestError as e:
            print(f"BadRequestError: {e.message}")
            # identify different 400 possible then handle accordingly
            # have to find a way to parse the error message to identify the specific issue
            pass
        except InternalServerError as e:
            print(f"InternalServerError: {e.message}")
        except RateLimitError as e:
            pass
        except APIConnectionError as e:
            pass
        print(f"Attempt {attempt + 1} failed. Retrying in {exponential_backoff ** attempt} seconds...")
        time.sleep(exponential_backoff ** attempt)
    raise Exception("Failed to create completion after multiple attempts.")