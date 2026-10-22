import builtins
from typing import List

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from core.agent import run_turn
from core.compact import compact_messages
from core.output import set_output_prefix
from core.user_input import get_user_query
from services.args import parse_args
from services.hooks import trigger_hooks
from services.prompt import assemble_system_prompt


def _run_agent(args) -> None:
    if not args.query:
        raise ValueError("Subagent mode requires a non-empty --query")

    agent = args.agent
    set_output_prefix(f"[{agent}] ")
    trigger_hooks("Start")

    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=assemble_system_prompt()
        ),
        ChatCompletionUserMessageParam(role="user", content=args.query),
    ]

    try:
        run_turn(messages)
    except KeyboardInterrupt:
        builtins.print("\n[system] Interrupted. Exiting...")
        return

    try:
        trigger_hooks("Stop", messages=messages)
    except KeyboardInterrupt:
        builtins.print("\n[system] Interrupted. Exiting...")


def _run_cli(args) -> None:
    trigger_hooks("Start")

    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=assemble_system_prompt()
        )
    ]

    try:
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
        trigger_hooks("Stop", messages=messages)
    except KeyboardInterrupt:
        print("\n[system] Interrupted. Exiting...")


def main() -> None:
    args = parse_args()

    if args.agent:
        _run_agent(args)
    elif args.cli:
        _run_cli(args)
    else:
        from tui.app import run_tui

        run_tui(args)


if __name__ == "__main__":
    main()
