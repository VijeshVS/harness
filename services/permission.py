import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

from core.output import emit

WORKDIR = Path(subprocess.run("pwd",shell=True,capture_output=True,text=True).stdout.strip())

PromptHandler = Callable[[str, dict, str], bool]

_prompt_handler: PromptHandler | None = None
_prompt_lock = threading.Lock()


def set_prompt_handler(handler: PromptHandler) -> None:
    global _prompt_handler
    _prompt_handler = handler


def clear_prompt_handler() -> None:
    global _prompt_handler
    _prompt_handler = None

PERMISSION_RULES = [
    {
        "tools": ["run_bash_command", "read_file", "write_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["run_bash_command"],
        "check": lambda args: any(kw in args.get("cmd", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    },
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

def ask_user(tool_name: str, args: dict, reason: str) -> bool:
    with _prompt_lock:
        if _prompt_handler is not None:
            return bool(_prompt_handler(tool_name, args, reason))

        if os.environ.get("HARNESS_SUBAGENT_ID"):
            return _ask_parent(tool_name, args, reason)

        user_res = input(f"Tool '{tool_name}' is requesting to run with arguments {args}. Reason: {reason}. Do you allow this? (y/n): ")
        return user_res.lower() == "y"


def _ask_parent(tool_name: str, args: dict, reason: str) -> bool:
    request = json.dumps(
        {"tool": tool_name, "args": args, "reason": reason},
        ensure_ascii=False,
        default=str,
    )
    emit("permission", request, prefix="[permission] ")
    sys.stdout.flush()

    response = sys.stdin.readline()
    if not response:
        return False
    return response.strip().lower().startswith("y")
