import subprocess
import json
import os
import re
from pathlib import Path
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
    
def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    """
    Reads the file and returns the requested lines
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        selected = lines[offset:] if limit is None else lines[offset:offset + limit]
        parsed_result = {
                "stdout": "".join(selected),
                "stderr": "",
                "return_code": 0
        }
    except OSError as e:
        parsed_result = {
                "stdout": "",
                "stderr": str(e),
                "return_code": 1
        }
    return json.dumps(parsed_result)

def write_file(path: str, content: str, offset: int = 0, limit: int | None = None) -> str:
    """
    Writes content, replacing `limit` lines from `offset` (or to EOF when limit is None)
    """
    p = Path(path)
    existing = p.read_text(encoding="utf-8").splitlines(keepends=True) if p.exists() else []
    new_lines = content.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    end = len(existing) if limit is None else offset + limit
    updated = existing[:offset] + new_lines + existing[end:]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(updated), encoding="utf-8")
    return f"Wrote {len(new_lines)} lines to {path} at line {offset}."

def search_files(pattern: str, path: str = ".") -> str:
    """
    Searches for file paths matching the given regex pattern
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {e}", "matches": []})

    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "logs"}
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, path)
            if regex.search(rel_path):
                matches.append(rel_path)

    matches.sort()
    return json.dumps({"matches": matches, "count": len(matches)})

def search_code(pattern: str, path: str = ".") -> str:
    """
    Searches file contents for the given regex and returns matching files with line offsets
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex: {e}", "matches": []})

    skip_dirs = {".git", ".venv", "__pycache__", "node_modules", "logs"}
    matches = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            full_path = os.path.join(root, name)
            rel_path = os.path.relpath(full_path, path)
            try:
                with open(full_path, encoding="utf-8") as f:
                    for offset, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append({
                                "file": rel_path,
                                "offset": offset,
                                "line": line.rstrip("\n"),
                            })
            except (OSError, UnicodeDecodeError):
                continue

    return json.dumps({"matches": matches, "count": len(matches)})

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
    ToolModel(name="read_file",description="Reads a file and returns its contents, optionally a slice of lines.",parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based line offset to start reading from. Defaults to 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Defaults to all remaining lines.",
                    }
                },
                "required": ["path"],
    }),
    ToolModel(name="write_file",description="Writes content to a file, optionally replacing a range of lines.",parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based line offset to start writing at. Defaults to 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of existing lines to replace. Defaults to all lines from offset to EOF.",
                    }
                },
                "required": ["path", "content"],
    }),
    ToolModel(name="search_files",description="Recursively searches for files whose relative path matches the given regex pattern.",parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to match against file paths, e.g. 'tools\\.py' or '\\.md$'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to the current directory.",
                    }
                },
                "required": ["pattern"],
    }),
    ToolModel(name="search_code",description="Searches file contents for a regex pattern and returns matching files with the line offset and line text.",parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for in file contents, e.g. 'def read_file' or 'TOOL_MAPPING'.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to the current directory.",
                    }
                },
                "required": ["pattern"],
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
    "write_file": write_file,
    "search_files": search_files,
    "search_code": search_code,
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
