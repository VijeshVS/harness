import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from openai.types.chat import ChatCompletionMessageParam

from config import settings

MARKER = "[tool output is compacted recall if needed]"
TODO_TOOL = "run_todo_write"
WRITE_TOOL = "write_file"
TRUNCATION_HINT = "chars truncated; read"

Message = Dict[str, Any]


@dataclass(frozen=True)
class CompactionStats:
    messages: int
    chars: int


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _call_id(call: Any) -> Optional[str]:
    return _get(call, "id")


def _call_name(call: Any) -> Optional[str]:
    function = _get(call, "function")
    return _get(function, "name") if function is not None else None


def _call_arguments(call: Any) -> Any:
    function = _get(call, "function")
    return _get(function, "arguments") if function is not None else None


def _normalize_call(call: Any) -> Optional[Message]:
    if isinstance(call, dict):
        normalized = dict(call)
        function = normalized.get("function")
        if isinstance(function, dict):
            normalized["function"] = dict(function)
        return normalized
    function = getattr(call, "function", None)
    if function is None:
        return None
    return {
        "id": getattr(call, "id", None),
        "type": getattr(call, "type", "function"),
        "function": {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", ""),
        },
    }


def _has_content(message: Message) -> bool:
    content = message.get("content")
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def _without_tool_calls(message: Message) -> Message:
    return {key: value for key, value in message.items() if key != "tool_calls"}


def _parse_todos(arguments: Any) -> Optional[List[Any]]:
    if not isinstance(arguments, str):
        return None
    try:
        data = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    todos = data.get("todos")
    if not isinstance(todos, list):
        return None
    if not all(isinstance(todo, dict) for todo in todos):
        return None
    return todos


def _todo_key(todos: List[Any]) -> Tuple[str, ...]:
    titles = []
    for todo in todos:
        if isinstance(todo, dict):
            titles.append(str(todo.get("title", "")).strip().lower())
    return tuple(sorted(titles))


def _format_todos(todos: List[Any]) -> str:
    if not todos:
        return "[todos] none"
    parts = []
    for todo in todos:
        if not isinstance(todo, dict):
            continue
        title = str(todo.get("title", "")).strip()
        status = str(todo.get("status", "")).strip()
        parts.append(f"{title} ({status})" if status else title)
    if not parts:
        return "[todos] none"
    return "[todos] " + ", ".join(parts)


def _todo_note(text: str) -> Message:
    return {"role": "assistant", "content": text}


def _collapse_todos(messages: List[Message]) -> List[Message]:
    todo_call_ids = set()
    latest: Dict[Tuple[str, ...], Message] = {}

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if _call_name(call) != TODO_TOOL:
                continue
            call_id = _call_id(call)
            todos = _parse_todos(_call_arguments(call))
            if not call_id or todos is None:
                continue
            todo_call_ids.add(call_id)
            latest[_todo_key(todos)] = {
                "todos": todos,
                "assistant_index": index,
            }

    if not latest:
        return messages

    notes: Dict[int, List[Tuple[Tuple[str, ...], str]]] = {}
    for key in sorted(latest):
        entry = latest[key]
        notes.setdefault(entry["assistant_index"], []).append(
            (key, _format_todos(entry["todos"]))
        )

    output: List[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]

        if message.get("role") == "assistant" and message.get("tool_calls"):
            calls = list(message["tool_calls"])
            call_ids = {_call_id(call) for call in calls}
            end = index + 1
            block: List[Message] = []
            while end < len(messages):
                candidate = messages[end]
                if (
                    candidate.get("role") != "tool"
                    or candidate.get("tool_call_id") not in call_ids
                ):
                    break
                block.append(candidate)
                end += 1

            remaining = [
                call for call in calls if _call_id(call) not in todo_call_ids
            ]
            normalized = [
                call
                for call in (_normalize_call(call) for call in remaining)
                if call is not None
            ]
            surviving_ids = {_call_id(call) for call in normalized}
            inserted_notes = [
                _todo_note(text) for _, text in sorted(notes.get(index, []))
            ]

            if normalized:
                output.append({**message, "tool_calls": normalized})
                output.extend(
                    result
                    for result in block
                    if result.get("tool_call_id") in surviving_ids
                )
                output.extend(inserted_notes)
            elif _has_content(message):
                output.append(_without_tool_calls(message))
                output.extend(inserted_notes)
            else:
                output.extend(inserted_notes)

            index = end
            continue

        if (
            message.get("role") == "tool"
            and message.get("tool_call_id") in todo_call_ids
        ):
            index += 1
            continue
        output.append(message)
        index += 1

    return output


def _truncate_write_arguments(arguments: Any) -> Optional[str]:
    if not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    content = parsed.get("content")
    if not isinstance(content, str) or TRUNCATION_HINT in content:
        return None

    head_chars = settings.compaction.write_head_chars
    tail_chars = settings.compaction.write_tail_chars
    if len(content) <= head_chars + tail_chars:
        return None

    omitted = len(content) - head_chars - tail_chars
    path = parsed.get("path") or "<path>"
    marker = f"\n\n[... {omitted} chars truncated; read {path} to recall ...]\n\n"
    parsed["content"] = content[:head_chars] + marker + content[-tail_chars:]
    return json.dumps(parsed, ensure_ascii=False)


def _truncate_tool_arguments(messages: List[Message]) -> List[Message]:
    output: List[Message] = []
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            output.append(message)
            continue

        calls = []
        for call in message["tool_calls"]:
            normalized = _normalize_call(call)
            if normalized is None:
                calls.append(call)
                continue
            if _call_name(normalized) == WRITE_TOOL:
                arguments = _call_arguments(normalized)
                truncated = _truncate_write_arguments(arguments)
                if truncated is not None and truncated != arguments:
                    normalized["function"]["arguments"] = truncated
            calls.append(normalized)

        output.append({**message, "tool_calls": calls})

    return output


def _compact_old_tool_outputs(messages: List[Message]) -> List[Message]:
    ordered_ids = []
    protected_ids = set()

    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = _call_id(call)
            if not call_id:
                continue
            ordered_ids.append(call_id)
            if _call_name(call) == TODO_TOOL:
                protected_ids.add(call_id)

    recent_count = settings.compaction.recent_tool_calls
    recent_ids = set(ordered_ids[-recent_count:]) if recent_count and ordered_ids else set()

    output: List[Message] = []
    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            content = message.get("content")
            if (
                call_id not in recent_ids
                and call_id not in protected_ids
                and isinstance(content, str)
                and content != MARKER
            ):
                output.append({**message, "content": MARKER})
                continue
        output.append(message)

    return output


def _validate_and_repair(messages: List[Message]) -> List[Message]:
    call_ids = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = _call_id(call)
            if call_id:
                call_ids.add(call_id)

    result_ids = {
        message.get("tool_call_id")
        for message in messages
        if message.get("role") == "tool"
    }
    surviving = call_ids & result_ids

    output: List[Message] = []
    seen_results = set()
    seen_calls = set()
    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            kept = []
            for call in message["tool_calls"]:
                call_id = _call_id(call)
                if call_id in surviving and call_id not in seen_calls:
                    seen_calls.add(call_id)
                    kept.append(call)
            if kept:
                output.append({**message, "tool_calls": kept})
            elif _has_content(message):
                output.append(_without_tool_calls(message))
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id in surviving and call_id not in seen_results:
                seen_results.add(call_id)
                output.append(message)
        else:
            output.append(message)

    return output



def _message_chars(message: Message) -> int:
    chars = 0
    content = message.get("content")
    if isinstance(content, str):
        chars += len(content)
    for call in message.get("tool_calls") or []:
        arguments = _call_arguments(call)
        if isinstance(arguments, str):
            chars += len(arguments)
    return chars


def _snapshot(messages: List[Message]) -> CompactionStats:
    return CompactionStats(
        messages=len(messages),
        chars=sum(_message_chars(message) for message in messages),
    )


def needs_compaction(messages: List[ChatCompletionMessageParam]) -> bool:
    return _snapshot(messages).chars > settings.compaction.max_context_chars


def compact_messages(
    messages: List[ChatCompletionMessageParam],
) -> List[ChatCompletionMessageParam]:
    working: List[Message] = [dict(message) for message in messages]
    before = _snapshot(working)

    working = _collapse_todos(working)
    working = _truncate_tool_arguments(working)
    working = _compact_old_tool_outputs(working)
    working = _validate_and_repair(working)

    after = _snapshot(working)
    message_delta = after.messages - before.messages
    print(
        f"[compact] messages {before.messages}->{after.messages} "
        f"({message_delta:+d}) | chars {before.chars}->{after.chars}"
    )

    return cast(List[ChatCompletionMessageParam], working)
