import builtins
import queue
import select
import sys
from typing import List

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from core.agent import run_turn
from core.background import (
    BackgroundEvent,
    add_background_listener,
    format_background_event,
    remove_background_listener,
    terminate_background_tasks,
)
from core.compact import compact_messages
from core.output import set_output_prefix
from core.user_input import EXIT_COMMANDS
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


def _drain_background_events(
    events: queue.Queue[BackgroundEvent],
) -> list[BackgroundEvent]:
    drained: list[BackgroundEvent] = []
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            return drained
        if event.kind in {"completed", "failed"}:
            drained.append(event)


def _next_cli_turn(
    first_query: str | None,
    background_events: queue.Queue[BackgroundEvent],
) -> tuple[str, str | list[BackgroundEvent]] | None:
    if first_query is not None:
        return "user", first_query

    print("User Query: ", end="", flush=True)
    while True:
        pending = _drain_background_events(background_events)
        if pending:
            print()
            return "background", pending

        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        except (OSError, ValueError):
            ready = [sys.stdin]

        if not ready:
            continue

        line = sys.stdin.readline()
        if line == "":
            print("\n[system] End of input. Exiting...")
            return "exit", ""

        text = line.strip()
        if not text:
            print("User Query: ", end="", flush=True)
            continue
        if text.lower() in EXIT_COMMANDS:
            return "exit", ""
        return "user", text


def _run_cli(args) -> None:
    trigger_hooks("Start")

    messages: List[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=assemble_system_prompt()
        )
    ]

    background_events: queue.Queue[BackgroundEvent] = queue.Queue()

    def on_background_event(event: BackgroundEvent) -> None:
        if event.kind in {"completed", "failed"}:
            background_events.put(event)

    add_background_listener(on_background_event)
    background_retry_used = False

    try:
        first_query = args.query
        while True:
            item = _next_cli_turn(first_query, background_events)
            first_query = None
            if item is None:
                break

            kind, payload = item
            if kind == "exit":
                break

            events: list[BackgroundEvent] = []
            if kind == "background":
                events = payload
                print(
                    "\n"
                    + "\n".join(
                        f"[background] {event.task_id} {event.kind} "
                        f"in {event.duration:.1f}s"
                        for event in events
                    )
                )
                query = "\n\n".join(
                    format_background_event(event) for event in events
                )
            else:
                query = payload
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
                background_retry_used = False
            except Exception as error:
                print(f"[llm] turn failed: {error}")
                messages[:] = snapshot
                if kind == "background" and not background_retry_used:
                    background_retry_used = True
                    for event in events:
                        background_events.put(event)
    except KeyboardInterrupt:
        print("\n[system] Interrupted. Exiting...")
        return
    finally:
        remove_background_listener(on_background_event)
        terminate_background_tasks()

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
