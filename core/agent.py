from typing import List, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionMessageToolCallUnionParam,
)

from config import settings
from core.llm import create_completion
from core.output import emit
from core.tool import execute_tool_calls
from services.tools import tools

MAX_COMPLETION_TOKENS = settings.llm.max_completion_tokens


def run_turn(messages: List[ChatCompletionMessageParam]) -> None:
    token_index = settings.llm.starting_token_index

    while True:
        response = create_completion(
            messages, tools, max_tokens=MAX_COMPLETION_TOKENS[token_index]
        )
        choice = response.choices[0]
        reason = choice.finish_reason
        tool_calls: List[ChatCompletionMessageToolCallUnion] | None = (
            choice.message.tool_calls
        )

        if reason != "tool_calls" or tool_calls is None:
            if reason == "length" and token_index < len(MAX_COMPLETION_TOKENS) - 1:
                token_index += 1
                emit(
                    "status",
                    "Completion length exceeded. "
                    f"Increasing max tokens to {MAX_COMPLETION_TOKENS[token_index]} and retrying...",
                    prefix="[llm] ",
                )
                continue
            if reason == "length":
                emit(
                    "status",
                    "Completion length exceeded. "
                    "Maximum token limit reached. Cannot increase further.",
                    prefix="[llm] ",
                )

            content = choice.message.content
            if content:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=content
                    )
                )
            emit("debug", f"[stop-reason] {reason}")
            if content:
                emit("assistant", content, prefix="[llm] ")
            return

        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=choice.message.content,
                tool_calls=cast(List[ChatCompletionMessageToolCallUnionParam], tool_calls),
            )
        )

        execute_tool_calls(tool_calls, messages)
