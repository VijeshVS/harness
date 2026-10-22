from typing import Callable

OutputListener = Callable[[str, str], None]

_listener: OutputListener | None = None
_output_prefix = ""


def set_output_listener(listener: OutputListener) -> None:
    global _listener
    _listener = listener


def clear_output_listener() -> None:
    global _listener
    _listener = None


def set_output_prefix(prefix: str) -> None:
    global _output_prefix
    _output_prefix = prefix


def emit(kind: str, text: str, prefix: str = "") -> None:
    if _listener is not None:
        _listener(kind, text)
    else:
        print(f"{_output_prefix}{prefix}{text}")
