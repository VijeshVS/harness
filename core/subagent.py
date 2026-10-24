from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from services.permission import ask_user

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_TAG_RE = re.compile(
    r"^\[(llm|tool|system|session|summary|compact|todos|error|command|stop-reason|skill|permission)\]\s?(.*)$"
)

_TAG_KINDS = {
    "llm": "assistant",
    "tool": "tool",
    "system": "status",
    "session": "status",
    "summary": "status",
    "compact": "status",
    "todos": "status",
    "error": "error",
    "command": "error",
    "skill": "status",
}


@dataclass(frozen=True)
class SubagentEvent:
    subagent_id: str
    name: str
    kind: str
    text: str = ""
    query: str = ""
    elapsed: float = 0.0
    status: str = ""


SubagentListener = Callable[[SubagentEvent], None]

_listeners: list[SubagentListener] = []
_listeners_lock = threading.Lock()

_procs: dict[str, subprocess.Popen[str]] = {}
_procs_lock = threading.Lock()
_print_lock = threading.Lock()


def add_subagent_listener(listener: SubagentListener) -> None:
    with _listeners_lock:
        if listener not in _listeners:
            _listeners.append(listener)


def remove_subagent_listener(listener: SubagentListener) -> None:
    with _listeners_lock:
        try:
            _listeners.remove(listener)
        except ValueError:
            pass


def terminate_subagents() -> None:
    with _procs_lock:
        procs = list(_procs.values())
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass


def run_subagent(agent_name: str, query: str) -> str:
    subagent_id = f"{agent_name}-{uuid.uuid4().hex[:8]}"
    started = time.monotonic()
    publish_event(
        SubagentEvent(
            subagent_id=subagent_id,
            name=agent_name,
            kind="spawned",
            query=query,
        )
    )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["HARNESS_SUBAGENT_ID"] = subagent_id
    env["HARNESS_SUBAGENT_NAME"] = agent_name

    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "main.py"),
        "--query",
        query,
        "--agent",
        agent_name,
    ]

    proc: subprocess.Popen[str] | None = None
    last_llm: str | None = None
    return_code = 1
    error_message: str | None = None

    try:
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with _procs_lock:
            _procs[subagent_id] = proc

        last_llm = _consume_output(subagent_id, agent_name, proc)
        return_code = proc.wait()
    except OSError as error:
        error_message = f"Subagent {agent_name} failed to start: {error}"
        publish_event(
            SubagentEvent(
                subagent_id=subagent_id,
                name=agent_name,
                kind="error",
                text=error_message,
            )
        )
    except Exception as error:
        error_message = f"Subagent {agent_name} failed: {error}"
        publish_event(
            SubagentEvent(
                subagent_id=subagent_id,
                name=agent_name,
                kind="error",
                text=error_message,
            )
        )
    finally:
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.poll() is None:
                proc.terminate()
        with _procs_lock:
            _procs.pop(subagent_id, None)

    elapsed = time.monotonic() - started

    if error_message is not None:
        result = error_message
        status = "error"
    elif return_code != 0:
        result = (
            f"Subagent {agent_name} result: {last_llm}"
            if last_llm
            else f"Subagent {agent_name} exited with code {return_code}."
        )
        status = "error"
    else:
        result = (
            f"Subagent {agent_name} result: {last_llm}"
            if last_llm
            else f"Subagent {agent_name} executed {query}"
        )
        status = "completed"

    publish_event(
        SubagentEvent(
            subagent_id=subagent_id,
            name=agent_name,
            kind="finished",
            text=result,
            elapsed=elapsed,
            status=status,
        )
    )
    return result


def publish_event(event: SubagentEvent) -> None:
    with _listeners_lock:
        listeners = tuple(_listeners)

    if not listeners:
        _print_event(event)
        return

    for listener in listeners:
        try:
            listener(event)
        except Exception:
            pass


def _print_event(event: SubagentEvent) -> None:
    with _print_lock:
        if event.kind == "spawned":
            print(f"[subagent] {event.name} spawned: {event.query}")
        elif event.kind == "finished":
            print(
                f"[subagent] {event.name} {event.status} in {event.elapsed:.1f}s"
            )
        elif event.text:
            print(event.text)


def _consume_output(
    subagent_id: str,
    agent_name: str,
    proc: subprocess.Popen[str],
) -> str | None:
    prefix = f"[{agent_name}] "
    last_llm: str | None = None
    llm_chunks: list[str] = []
    llm_active = False
    current_kind = "status"

    def finalize_llm() -> None:
        nonlocal last_llm, llm_chunks, llm_active
        if llm_active:
            text = "\n".join(llm_chunks).strip()
            if text:
                last_llm = text
        llm_active = False
        llm_chunks = []

    assert proc.stdout is not None

    for raw_line in proc.stdout:
        line = raw_line.rstrip("\r\n")
        if not line:
            continue

        if line.startswith(prefix):
            line = line[len(prefix):]

        match = _TAG_RE.match(line)
        if match:
            tag, body = match.groups()
            if tag == "stop-reason":
                finalize_llm()
                current_kind = "status"
                continue

            if tag == "permission":
                finalize_llm()
                current_kind = "status"
                _handle_permission_request(proc, body)
                continue

            kind = _TAG_KINDS.get(tag, "status")
            if tag == "llm":
                finalize_llm()
                llm_active = True
                llm_chunks = [body]
            else:
                finalize_llm()

            current_kind = kind
            _publish_output(subagent_id, agent_name, kind, body)
            continue

        if llm_active:
            llm_chunks.append(line)
            _publish_output(subagent_id, agent_name, "assistant", line)
        else:
            _publish_output(subagent_id, agent_name, current_kind, line)

    finalize_llm()
    return last_llm


def _handle_permission_request(
    proc: subprocess.Popen[str],
    body: str,
) -> None:
    try:
        request = json.loads(body)
        tool_name = str(request.get("tool", ""))
        arguments = request.get("args") or {}
        reason = str(request.get("reason", ""))
    except (TypeError, ValueError):
        _respond_to_permission(proc, False)
        return

    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    _respond_to_permission(proc, ask_user(tool_name, arguments, reason))


def _respond_to_permission(
    proc: subprocess.Popen[str],
    allow: bool,
) -> None:
    if proc.stdin is None:
        return
    try:
        proc.stdin.write("y\n" if allow else "n\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _publish_output(
    subagent_id: str,
    agent_name: str,
    kind: str,
    text: str,
) -> None:
    if not text.strip():
        return
    publish_event(
        SubagentEvent(
            subagent_id=subagent_id,
            name=agent_name,
            kind=kind,
            text=text,
        )
    )
