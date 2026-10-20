import os
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionMessageParam 
from services.client import client
from typing import List

hooks = {
    "Start": [],
    "PostUserInput": [],
    "PreToolCall": [],
    "PostToolCall": [],
    "Stop": [],
}

def register_hook(event_name: str, callback):
    if event_name in hooks:
        hooks[event_name].append(callback)
    else:
        raise ValueError(f"Unknown event name: {event_name}")

def trigger_hooks(event_name: str, *args, **kwargs):
    if event_name in hooks:
        for callback in hooks[event_name]:
            callback(*args, **kwargs)
    else:
        raise ValueError(f"Unknown event name: {event_name}")

def execution_summary(messages: List[ChatCompletionMessageParam]):
    messages.append(ChatCompletionUserMessageParam(role="user", content="Give me summary of conversation we had so far in one line"))
    response = client.chat.completions.create(
        model="gpt-5.6-luna",   
        messages=messages,
        prompt_cache_key=get_session_id(),
        prompt_cache_options={"mode": "implicit"},
        reasoning_effort="none",
    )
    print(f"[summary] {response.choices[0].message.content}")

register_hook("Stop", execution_summary)

_session_id: str | None = None

def start_session():
    global _session_id
    _session_id = os.urandom(8).hex()
    print(f"[session] Session started with ID: {_session_id}")

def get_session_id() -> str:
    if _session_id is None:
        raise RuntimeError("Session not started")
    return _session_id

register_hook("Start", start_session)