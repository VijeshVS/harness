import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from config import settings
from services.hooks import trigger_hooks
from services.permission import check_rules, ask_user
from services.tools import TOOL_MAPPING

PARALLEL_TOOLS = {"spawn_subagent"}


def _run_handler(
    fn_name: str,
    handler: Callable[..., str],
    arguments: dict,
) -> str:
    try:
        return handler(**arguments)
    except Exception as e:
        return f"Tool '{fn_name}' execution failed with error: {str(e)}"


def execute_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCallUnion],
    messages: List[ChatCompletionMessageParam],
) -> None:
    results: dict[str, str] = {}
    arguments_by_id: dict[str, dict] = {}
    parallel_calls: list[
        tuple[ChatCompletionMessageFunctionToolCall, Callable[..., str], dict]
    ] = []

    for tool in tool_calls:
        trigger_hooks("PreToolCall", messages=messages, tool=tool)

        if not isinstance(tool, ChatCompletionMessageFunctionToolCall):
            continue

        fn_name = tool.function.name
        arguments = json.loads(tool.function.arguments)
        tool_call_id = tool.id
        arguments_by_id[tool_call_id] = arguments

        check_msg = check_rules(fn_name, arguments)

        if check_msg:
            user_res = ask_user(fn_name, arguments, check_msg)
            if not user_res:
                results[tool_call_id] = (
                    f"Tool '{fn_name}' execution denied by user."
                )
                continue

        handler = TOOL_MAPPING.get(fn_name)

        if not handler:
            results[tool_call_id] = f"Tool '{fn_name}' not found."
            continue

        if fn_name in PARALLEL_TOOLS:
            parallel_calls.append((tool, handler, arguments))
            continue

        result = _run_handler(fn_name, handler, arguments)
        results[tool_call_id] = result
        trigger_hooks("PostToolCall", messages=messages, tool=tool, result=result)

    if parallel_calls:
        max_workers = min(
            settings.subagents.max_parallel, len(parallel_calls)
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_handler, tool.function.name, handler, arguments
                ): tool
                for tool, handler, arguments in parallel_calls
            }
            for future in as_completed(futures):
                tool = futures[future]
                result = future.result()
                results[tool.id] = result
                trigger_hooks(
                    "PostToolCall",
                    messages=messages,
                    tool=tool,
                    result=result,
                )

    for tool in tool_calls:
        if not isinstance(tool, ChatCompletionMessageFunctionToolCall):
            continue

        result = results.get(tool.id)
        if result is None:
            continue

        messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                content=result,
                tool_call_id=tool.id,
            )
        )

        print(
            f"[tool] {tool.function.name}({arguments_by_id[tool.id]}) -> {result}"
        )
