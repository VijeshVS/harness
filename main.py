import builtins
from typing import List, cast
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionMessageToolCallUnionParam,
)
from config import settings
from core.compact import compact_messages
from core.llm import create_completion
from core.tool import execute_tool_calls
from core.user_input import get_user_query
from services.tools import tools
from services.hooks import trigger_hooks
from services.args import parse_args
from services.prompt import assemble_system_prompt

MAX_COMPLETION_TOKENS = settings.llm.max_completion_tokens

args = parse_args()
agent = args.agent

if agent:
    def print(*args, **kwargs):
        builtins.print(f"[{agent}]", *args, **kwargs)

trigger_hooks("Start")

SYSTEM_PROMPT = assemble_system_prompt()


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
                print(
                    "[llm] Completion length exceeded. "
                    f"Increasing max tokens to {MAX_COMPLETION_TOKENS[token_index]} and retrying..."
                )
                continue
            if reason == "length":
                print(
                    "[llm] Completion length exceeded. "
                    "Maximum token limit reached. Cannot increase further."
                )

            content = choice.message.content
            if content:
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=content
                    )
                )
            print(f"[stop-reason] {reason}")
            print(f"[llm] {content}")
            return

        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=choice.message.content,
                tool_calls=cast(List[ChatCompletionMessageToolCallUnionParam], tool_calls),
            )
        )

        execute_tool_calls(tool_calls, messages)


def main() -> None:
    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT)
    ]

    try:
        if agent:
            if not args.query:
                raise ValueError("Subagent mode requires a non-empty --query")
            messages.append(
                ChatCompletionUserMessageParam(role="user", content=args.query)
            )
            run_turn(messages)
        else:
            first_query = args.query
            while True:
                query = get_user_query(first_query, allow_exit=True)
                first_query = None
                if query is None:
                    break

                command = query.strip().lower()
                if command.startswith("/"):
                    if command == "/compact":
                        messages[:] = compact_messages(messages)
                    else:
                        print(f"[command] invalid operation: {query.strip()}")
                    continue

                snapshot = list(messages)
                messages.append(
                    ChatCompletionUserMessageParam(role="user", content=query)
                )
                try:
                    run_turn(messages)
                except Exception as error:
                    print(f"[llm] turn failed: {error}")
                    messages[:] = snapshot
    except KeyboardInterrupt:
        print("\n[system] Interrupted. Exiting...")
        return

    try:
        trigger_hooks("Stop", messages=list(messages))
    except KeyboardInterrupt:
        print("\n[system] Interrupted. Exiting...")


if __name__ == "__main__":
    main()
