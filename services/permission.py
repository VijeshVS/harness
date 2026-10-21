import subprocess
from pathlib import Path

WORKDIR = Path(subprocess.run("pwd",shell=True,capture_output=True,text=True).stdout.strip())

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
    user_res = input(f"Tool '{tool_name}' is requesting to run with arguments {args}. Reason: {reason}. Do you allow this? (y/n): ")
    return user_res.lower() == "y"