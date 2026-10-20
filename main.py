from pathlib import Path
import json
from openai.types.chat import ChatCompletionUserMessageParam,ChatCompletionMessageFunctionToolCall, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam,ChatCompletionToolMessageParam, ChatCompletionSystemMessageParam,ChatCompletionMessageToolCallUnion, ChatCompletionMessageFunctionToolCall, ChatCompletionMessageToolCallUnionParam
from typing import List, cast
from services.client import client
from services.tools import TOOL_MAPPING, tools
from services.permission import check_rules, ask_user
from services.hooks import trigger_hooks, get_session_id
from services.args import parse_args
from services.logger import enable_trace
from services.prompt import assemble_system_prompt
import builtins

args = parse_args()
query = args.query
agent = args.agent

if args.trace.lower() == "true":
    enable_trace()

if agent:
    def print(*args, **kwargs):
        builtins.print(f"[{agent}]", *args, **kwargs)

trigger_hooks("Start")
session_id = get_session_id()

if not query:
    query = input("User Query: ")

SYSTEM_PROMPT = assemble_system_prompt()
messages: List[ChatCompletionMessageParam]  = [ChatCompletionSystemMessageParam(role="system",content=SYSTEM_PROMPT),
                                               ChatCompletionUserMessageParam(role="user",content=query)]

def main():
    while True:
        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=messages,
            tools=tools,
            prompt_cache_key=session_id,
            prompt_cache_options={"mode": "implicit"},
            reasoning_effort="none",
        )
        
        reason = response.choices[0].finish_reason

        if reason != "tool_calls" or response.choices[0].message.tool_calls == None:
            print(f"[llm] {response.choices[0].message.content}")
            print("[SYSTEM] llm execution finished successfully !!")
            trigger_hooks("Stop", messages=messages)
            return

        tool_calls: List[ChatCompletionMessageToolCallUnion] = response.choices[0].message.tool_calls

        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=response.choices[0].message.content,
                tool_calls=cast(List[ChatCompletionMessageToolCallUnionParam], tool_calls)
            )
        )

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
                    messages.append(ChatCompletionToolMessageParam(role="tool",content=f"Tool '{fn_name}' execution denied by user.",tool_call_id=tool_call_id))
                    print(f"[tool] {fn_name}({arguments}) -> Execution denied by user.")
                    continue
            
            handler = TOOL_MAPPING.get(fn_name)

            if not handler:
                messages.append(ChatCompletionToolMessageParam(role="tool",content=f"Tool '{fn_name}' not found.",tool_call_id=tool_call_id))
                print(f"[tool] {fn_name}({arguments}) -> Tool not found.")
                continue

            try:
                result = handler(**arguments)
            except Exception as e:
                result = f"Tool '{fn_name}' execution failed with error: {str(e)}"
                messages.append(ChatCompletionToolMessageParam(role="tool",content=result,tool_call_id=tool_call_id))
                print(f"[tool] {fn_name}({arguments}) -> Execution failed with error: {str(e)}")
                continue
            
            messages.append(ChatCompletionToolMessageParam(role="tool",content=result,tool_call_id=tool_call_id))

            trigger_hooks("PostToolCall", messages=messages, tool=tool, result=result)
            print(f"[tool] {fn_name}({arguments}) -> {result}")

if __name__ == "__main__":
    main()
