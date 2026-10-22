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
from core.llm import create_completion
from core.tool import execute_tool_calls
from core.user_input import get_user_query
from services.tools import tools
from services.hooks import trigger_hooks, get_session_id
from services.args import parse_args
from services.logger import enable_trace
from services.prompt import assemble_system_prompt

MAX_COMPLETION_TOKENS = [64, 1024, 2048, 4096, 8192, 16384, 32768]
TOKEN_INDEX = 0

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
query = get_user_query(query)

SYSTEM_PROMPT = assemble_system_prompt()

messages: List[ChatCompletionMessageParam] = [
    ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
    ChatCompletionUserMessageParam(role="user", content=query),
]


def main():
    global TOKEN_INDEX
    global MAX_COMPLETION_TOKENS
    
    while True:
        response = create_completion(messages, tools,max_tokens=MAX_COMPLETION_TOKENS[TOKEN_INDEX])
        reason = response.choices[0].finish_reason

        if reason != "tool_calls" or response.choices[0].message.tool_calls == None:
            if reason == "length":
                TOKEN_INDEX = min(TOKEN_INDEX + 1, len(MAX_COMPLETION_TOKENS) - 1)
                if TOKEN_INDEX < len(MAX_COMPLETION_TOKENS):
                    print(f"[llm] Completion length exceeded. Increasing max tokens to {MAX_COMPLETION_TOKENS[TOKEN_INDEX]} and retrying...")
                    continue
                else:
                    print("[llm] Completion length exceeded. Maximum token limit reached. Cannot increase further.")
            print(f"[stop-reason] {reason}")
            print(f"[llm] {response.choices[0].message.content}")
            print("[system] llm execution finished successfully !!")
            trigger_hooks("Stop", messages=messages)
            return

        tool_calls: List[ChatCompletionMessageToolCallUnion] = response.choices[0].message.tool_calls

        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=response.choices[0].message.content,
                tool_calls=cast(List[ChatCompletionMessageToolCallUnionParam], tool_calls),
            )
        )

        execute_tool_calls(tool_calls, messages)


if __name__ == "__main__":
    main()
