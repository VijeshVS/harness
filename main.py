from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import json 
import sys
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionFunctionToolParam, ChatCompletionMessageFunctionToolCall, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam,ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionMessageToolCallUnion, ChatCompletionMessageFunctionToolCall, ChatCompletionMessageToolCallUnionParam, ChatCompletionMessageCustomToolCall
from openai.types.shared_params.function_definition import FunctionDefinition
from typing import List, cast

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def run_bash_command(cmd):
    """
    Executes the bash command and returns stdout, stderr, return code
    """
    result = subprocess.run(cmd,shell=True,capture_output=True,text=True)
    parsed_result = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.returncode
    }
    return json.dumps(parsed_result)

tools = [
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="run_bash_command",
            description="Executes a bash command and returns stdout, stderr, and the return code.",
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["cmd"],
            },
        ),
    )
]

TOOL_MAPPING = {
    "run_bash_command": run_bash_command
}

query = input("user query: ")
messages: List[ChatCompletionMessageParam]  = [ChatCompletionUserMessageParam(role="user",content=query)]

def main():
    while True:
        response = client.chat.completions.create(
            model="gemini-3.8-flash",
            messages=messages,
            tools=tools
        )
        
        reason = response.choices[0].finish_reason

        if reason != "tool_calls" or response.choices[0].message.tool_calls == None:
            print(f"[llm] {response.choices[0].message.content}")
            print("[SYSTEM] llm execution finished successfully !!")
            return

        tool_calls: List[ChatCompletionMessageToolCallUnion] = response.choices[0].message.tool_calls

        if not tool_calls:
            print(f"[llm] {response.choices[0].message.content}")
            print("[SYSTEM] llm execution finished successfully !!")
            return

        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=response.choices[0].message.content,
                tool_calls=cast(List[ChatCompletionMessageToolCallUnionParam], tool_calls)
            )
        )

        for tool in tool_calls:
            if not isinstance(tool, ChatCompletionMessageFunctionToolCall):
                continue

            fn_name = tool.function.name
            arguments = json.loads(tool.function.arguments)
            tool_call_id = tool.id

            result = TOOL_MAPPING[fn_name](**arguments)
            messages.append(ChatCompletionToolMessageParam(role="tool",content=result,tool_call_id=tool_call_id))
            print(f"[tool] {fn_name}({arguments}) -> {result}")

if __name__ == "__main__":
    main()
