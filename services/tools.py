import subprocess
import json
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from schema.client import ToolModel
from typing import List

CURRENT_TODOS: list[dict] = []

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

def run_todo_write(todos: list) -> str:
    """
    Writes the todos to the CURRENT_TODOS list
    """
    global CURRENT_TODOS
    CURRENT_TODOS = todos

    lines = ["[todos] Current Todos:\n"]
    for t in CURRENT_TODOS:
        icon = {"pending": "⧗", "in_progress": "▸", "completed": "✓"}[t["status"]]
        lines.append(f"{icon} {t['title']}\n")
    print("".join(lines))

    return f"Updated {len(CURRENT_TODOS)} todos."

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
    }),
    ToolModel(name="run_todo_write",description="Writes the todos to the CURRENT_TODOS list.",parameters={
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the task.",
                                },
                                "status": {
                                    "type": "string",
                                    "description": "The status of the task.",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["title", "status"],
                        },
                    }
                },
                "required": ["todos"],
    })]

TOOL_MAPPING = {
    "run_bash_command": run_bash_command,
    "read_file": read_file,
    "run_todo_write": run_todo_write
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
