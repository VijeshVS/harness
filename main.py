from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import json
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionFunctionToolParam, ChatCompletionMessageFunctionToolCall, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam,ChatCompletionToolMessageParam
from openai.types.chat import ChatCompletionMessageToolCallUnion, ChatCompletionMessageFunctionToolCall, ChatCompletionMessageToolCallUnionParam, ChatCompletionMessageCustomToolCall
from openai.types.shared_params.function_definition import FunctionDefinition
from typing import List, cast, Dict
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60.0,
    max_retries=1,
)

WORKDIR = Path(subprocess.run("pwd",shell=True,capture_output=True,text=True).stdout.strip())

PERMISSION_RULES = [
    {
        "tools": ["run_bash_command", "read_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["run_bash_command"],
        "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    },
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

def ask_user(tool_name: str, args: dict) -> bool:
    user_res = input(f"Tool '{tool_name}' is requesting to run with arguments {args}. Do you allow this? (y/n): ")
    return user_res.lower() == "y"

class ToolModel(BaseModel):
    name: str
    description: str
    parameters: Dict[str,object]

def run_bash_command(cmd: str) -> str:
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
    
def read_file(path: str) -> str:
    """
    Reads the file and returns the content of the file
    """
    result = subprocess.run(f"cat {path}",shell=True,capture_output=True,text=True)
    parsed_result = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
    }
    return json.dumps(parsed_result)

TOOL_DEFNS: List[ToolModel] = [
    ToolModel(name="run_bash_command",description="Executes a bash command and returns stdout, stderr, and the return code.",parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["cmd"],
    }),
    ToolModel(name="read_file",description="Reads a file and returns its contents.",parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the file to read.",
                    }
                },
                "required": ["path"],
    })]

TOOL_MAPPING = {
    "run_bash_command": run_bash_command,
    "read_file": read_file
}

tools: List[ChatCompletionFunctionToolParam] = [
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name=tool_defn.name,
            description=tool_defn.description,
            parameters=tool_defn.parameters,
        ),
    )
    for tool_defn in TOOL_DEFNS
]

query = input("user query: ")
messages: List[ChatCompletionMessageParam]  = [ChatCompletionUserMessageParam(role="user",content=query)]

def main():
    while True:
        response = client.chat.completions.create(
            model="gemini-3-flash-live",
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

            check_msg = check_rules(fn_name, arguments)

            if check_msg:
                user_res = ask_user(fn_name, arguments)
                if not user_res:
                    messages.append(ChatCompletionToolMessageParam(role="tool",content=f"Tool '{fn_name}' execution denied by user.",tool_call_id=tool_call_id))
                    print(f"[tool] {fn_name}({arguments}) -> Execution denied by user.")
                    continue
            
            handler = TOOL_MAPPING.get(fn_name)

            if not handler:
                messages.append(ChatCompletionToolMessageParam(role="tool",content=f"Tool '{fn_name}' not found.",tool_call_id=tool_call_id))
                print(f"[tool] {fn_name}({arguments}) -> Tool not found.")
                continue

            result = handler(**arguments)
            messages.append(ChatCompletionToolMessageParam(role="tool",content=result,tool_call_id=tool_call_id))
            print(f"[tool] {fn_name}({arguments}) -> {result}")

if __name__ == "__main__":
    main()
