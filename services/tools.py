import subprocess
import json
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition
from schema.client import ToolModel
from services.skills import SKILL_REGISTRY
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

    lines = ["[todos] Current Todos:"]
    for t in CURRENT_TODOS:
        icon = {"pending": "⧗", "in_progress": "▸", "completed": "✓"}[t["status"]]
        lines.append(f"{icon} {t['title']}")
    print("\n".join(lines))

    return f"Updated {len(CURRENT_TODOS)} todos."

def spawn_subagent(agent_name: str, query: str) -> str:
    """Spawns a sub-agent with the given name and query."""

    proc = subprocess.Popen(
        ["uv", "run", "main.py", "--query", query, "--agent", agent_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    prefix = f"[{agent_name}] "
    last_llm = None
    llm_lines = None

    for line in proc.stdout:
        print(line, end="", flush=True)

        clean = line.removeprefix(prefix)
        if clean.startswith("[llm]"):
            llm_lines = [clean.removeprefix("[llm]")]
        elif llm_lines is not None:
            if clean.startswith("[SYSTEM]"):
                last_llm = "".join(llm_lines).strip()
                llm_lines = None
            else:
                llm_lines.append(line)

    if llm_lines is not None:
        last_llm = "".join(llm_lines).strip()

    proc.wait()

    if last_llm:
        return f"Subagent {agent_name} result: {last_llm}"

    return f"Subagent {agent_name} executed {query}"

def load_skill(name: str) -> str:
    skill = SKILL_REGISTRY.get(name)
    if not skill:
        available = ", ".join(SKILL_REGISTRY) or "none"
        return f"Skill '{name}' not found. Available skills: {available}"
    return skill["content"]

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
    }),
    ToolModel(name="spawn_subagent",description="Spawns a sub-agent with the given name and query.",parameters={
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "The name of the sub-agent to spawn.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The query to pass to the sub-agent.",
                    }
                },
                "required": ["agent_name", "query"],
    }),
    ToolModel(name="load_skill",description="Loads the full instructions for a skill by name.",parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load.",
                    }
                },
                "required": ["name"],
    })
    ]


TOOL_MAPPING = {
    "run_bash_command": run_bash_command,
    "read_file": read_file,
    "spawn_subagent": spawn_subagent,
    "run_todo_write": run_todo_write,
    "load_skill": load_skill
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
