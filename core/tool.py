import json
from typing import List
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from services.hooks import trigger_hooks
from services.permission import check_rules, ask_user
from services.tools import TOOL_MAPPING


def execute_tool_calls(
    tool_calls: List[ChatCompletionMessageToolCallUnion],
    messages: List[ChatCompletionMessageParam],
) -> None:
    for tool in tool_calls:
        trigger_hooks("PreToolCall", messages=messages, tool=tool)

        if not isinstance(tool, ChatCompletionMessageFunctionToolCall):
            continue

        fn_name = tool.function.name
        arguments = json.loads(tool.function.arguments)
        tool_call_id = tool.id

        check_msg = check_rules(fn_name, arguments)

        if check_msg:
            user_res = ask_user(fn_name, arguments, check_msg)
            if not user_res:
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=f"Tool '{fn_name}' execution denied by user.",
                        tool_call_id=tool_call_id,
                    )
                )
                print(f"[tool] {fn_name}({arguments}) -> Execution denied by user.")
                continue

        handler = TOOL_MAPPING.get(fn_name)

        if not handler:
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    content=f"Tool '{fn_name}' not found.",
                    tool_call_id=tool_call_id,
                )
            )
            print(f"[tool] {fn_name}({arguments}) -> Tool not found.")
            continue

        try:
            result = handler(**arguments)
        except Exception as e:
            result = f"Tool '{fn_name}' execution failed with error: {str(e)}"
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call_id,
                )
            )
            print(f"[tool] {fn_name}({arguments}) -> Execution failed with error: {str(e)}")
            continue

        messages.append(
            ChatCompletionToolMessageParam(
                role="tool",
                content=result,
                tool_call_id=tool_call_id,
            )
        )

        trigger_hooks("PostToolCall", messages=messages, tool=tool, result=result)
        print(f"[tool] {fn_name}({arguments}) -> {result}")
