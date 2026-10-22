import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if not value:
        return default
    return tuple(int(part) for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class LLMSettings:
    model: str = os.getenv("LLM_MODEL", "gpt-5.6-luna")
    max_retries: int = _env_int("LLM_MAX_RETRIES", 5)
    base_delay: float = _env_float("LLM_BASE_DELAY", 1.0)
    max_delay: float = _env_float("LLM_MAX_DELAY", 20.0)
    request_timeout: float = _env_float("LLM_TIMEOUT", 60.0)
    client_max_retries: int = _env_int("LLM_CLIENT_MAX_RETRIES", 1)
    default_max_tokens: int = _env_int("LLM_DEFAULT_MAX_TOKENS", 1024)
    max_completion_tokens: tuple[int, ...] = _env_int_tuple(
        "LLM_MAX_COMPLETION_TOKENS",
        (64, 1024, 2048, 4096, 8192, 16384, 32768),
    )
    starting_token_index: int = _env_int("LLM_STARTING_TOKEN_INDEX", 3)

    def __post_init__(self) -> None:
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be > 0")
        if self.client_max_retries < 0:
            raise ValueError("client_max_retries must be >= 0")
        if self.default_max_tokens < 1:
            raise ValueError("default_max_tokens must be >= 1")
        if not self.max_completion_tokens:
            raise ValueError("max_completion_tokens must not be empty")
        if not 0 <= self.starting_token_index < len(self.max_completion_tokens):
            raise ValueError("starting_token_index out of range")


@dataclass(frozen=True)
class CompactionSettings:
    recent_tool_calls: int = _env_int("COMPACT_RECENT_TOOL_CALLS", 3)
    write_head_chars: int = _env_int("COMPACT_WRITE_HEAD_CHARS", 800)
    write_tail_chars: int = _env_int("COMPACT_WRITE_TAIL_CHARS", 800)
    max_context_chars: int = _env_int("COMPACT_MAX_CONTEXT_CHARS", 1_000_000)

    def __post_init__(self) -> None:
        if self.recent_tool_calls < 0:
            raise ValueError("recent_tool_calls must be >= 0")
        if self.write_head_chars < 0 or self.write_tail_chars < 0:
            raise ValueError("truncation sizes must be >= 0")
        if self.max_context_chars < 1:
            raise ValueError("max_context_chars must be >= 1")


@dataclass(frozen=True)
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    compaction: CompactionSettings = field(default_factory=CompactionSettings)


settings = Settings()
