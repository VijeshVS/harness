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

def is_safe_path(path: str):
    # Read must happen in current directory only
    result = subprocess.run("pwd",shell=True,capture_output=True,text=True)
    current_dir = result.stdout

    if current_dir in path:
        return True
    else:
        return False

    # Add other conditions subsequently
    
def read_file(path: str) -> str:
    """
    Reads the file and returns the content of the file
    """
    if not is_safe_path(path):
        return json.dumps({
            "err": "Not a safe path. The intented operation was not allowed."
        })
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
            model="gemini-3.5-flash",
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
