import io
import json
import re
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from core.agent import run_turn
from core.background import (
    BackgroundEvent,
    add_background_listener,
    format_background_event,
    remove_background_listener,
    terminate_background_task,
    terminate_background_tasks,
)
from core.compact import compact_messages
from core.output import clear_output_listener, set_output_listener
from core.subagent import (
    SubagentEvent,
    add_subagent_listener,
    remove_subagent_listener,
    terminate_subagents,
)
from core.user_input import EXIT_COMMANDS
from services import tools as tools_service
from services.hooks import (
    register_hook,
    trigger_hooks,
    unregister_hook,
)
from services.permission import (
    WORKDIR,
    clear_prompt_handler,
    set_prompt_handler,
)
from services.prompt import assemble_system_prompt
from tui.diff import render_contents, render_diff

PREFIX_KINDS = (
    ("[tool]", "tool"),
    ("[system]", "system"),
    ("[session]", "system"),
    ("[summary]", "summary"),
    ("[compact]", "info"),
    ("[todos]", "todos"),
    ("[command]", "error"),
    ("[llm]", "status"),
    ("[stop-reason]", "debug"),
    ("[error]", "error"),
)

STATUS_ICONS = {
    "pending": ("\u29d7", "dim"),
    "in_progress": ("\u25b8", "bold yellow"),
    "completed": ("\u2713", "green"),
}

COMMANDS = (
    ("/compact", "Compact the conversation context"),
    ("/exit", "Quit the harness"),
    ("/help", "Show available commands"),
    ("/quit", "Quit the harness"),
)

FILE_TOOLS = ("read_file", "write_file")

SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

_COMPACT_STATS = re.compile(
    r"messages (\d+)->(\d+) \(([+-]?\d+)\) \| chars (\d+)->(\d+)"
)
_TAGGED_LINE = re.compile(r"^\[[A-Za-z][\w.-]*\]\s")
_TOOL_CALL_RE = re.compile(r"^[A-Za-z_][\w.]*\(")

SUBAGENT_STATUS_STYLES = {
    "running": ("\u25b8", "bold yellow"),
    "completed": ("\u2713", "green"),
    "error": ("\u2717", "bold red"),
}

BACKGROUND_STATUS_STYLES = {
    "running": ("\u25b8", "bold yellow"),
    "completed": ("\u2713", "green"),
    "failed": ("\u2717", "bold red"),
    "cancelled": ("\u2717", "dim"),
    "cancelling": ("\u25b8", "yellow"),
}


@dataclass
class SubagentEntry:
    kind: str
    text: str
    tool_name: str = ""
    widget: Widget | None = field(default=None, repr=False)
    content_widget: Static | None = field(default=None, repr=False)


@dataclass
class SubagentState:
    id: str
    name: str
    query: str
    started: float = field(default_factory=time.monotonic)
    status: str = "running"
    finished_at: float | None = None
    entries: list[SubagentEntry] = field(default_factory=list)
    card_status: Static | None = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started)


@dataclass
class BackgroundState:
    id: str
    cmd: str
    started: float = field(default_factory=time.monotonic)
    status: str = "running"
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    error: str = ""
    finished_at: float | None = None

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started)


@dataclass
class TurnRequest:
    kind: str
    content: str
    label: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    attempts: int = 0


class _TUILogWriter(io.TextIOBase):
    """Redirects writes from the agent worker into the Textual UI."""

    def __init__(self, emit_lines) -> None:
        self._emit_lines = emit_lines
        self._buffer = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._buffer += text
            if "\n" not in self._buffer:
                return len(text)
            lines = self._buffer.split("\n")
            self._buffer = lines.pop()
            if lines:
                self._emit_lines(lines)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            if self._buffer:
                self._emit_lines([self._buffer])
                self._buffer = ""


class PermissionScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, tool_name: str, args: dict, reason: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = args
        self.reason = reason

    def compose(self) -> ComposeResult:
        body = Text()
        body.append("Permission required\n\n", style="bold")
        body.append("Tool: ", style="bold")
        body.append(f"{self.tool_name}\n", style="bold cyan")
        body.append("Reason: ", style="bold")
        body.append(f"{self.reason}\n\n", style="yellow")
        arguments = json.dumps(
            self.tool_args, indent=2, ensure_ascii=False, default=str
        )
        body.append(_shorten(arguments, 2000), style="dim")
        yield Vertical(
            Static(body, id="permission-message"),
            Horizontal(
                Button("Allow (y)", id="allow", variant="success"),
                Button("Deny (n)", id="deny", variant="error"),
                id="permission-buttons",
            ),
            id="permission-dialog",
        )

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#allow")
    def _allow(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny")
    def _deny(self) -> None:
        self.dismiss(False)


class HarnessApp(App):
    TITLE = "Coding Harness"
    SUB_TITLE = "agent console"

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #history-panel {
        width: 1fr;
        height: 1fr;
        border: round $primary;
    }

    #history {
        height: 1fr;
        padding: 0 1;
        scrollbar-gutter: stable;
        overflow-x: hidden;
    }

    #history.hidden {
        display: none;
    }

    #subagent-view {
        display: none;
        width: 1fr;
        height: 1fr;
        border: round $accent;
    }

    #subagent-view.visible {
        display: block;
    }

    #subagent-header {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }

    #subagent-log {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        scrollbar-gutter: stable;
        overflow-x: hidden;
    }

    #background-view {
        display: none;
        width: 1fr;
        height: 1fr;
        border: round $warning;
    }

    #background-view.visible {
        display: block;
    }

    #background-header {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }

    #background-log {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
        scrollbar-gutter: stable;
        overflow-x: hidden;
    }

    #loading {
        height: 1;
        padding: 0 1;
        display: none;
    }

    #loading.visible {
        display: block;
    }

    #spinner {
        width: 2;
        height: 1;
        content-align: left middle;
        color: $accent;
        text-style: bold;
    }

    #loading-label {
        height: 1;
        width: 1fr;
        padding-left: 1;
        content-align: left middle;
        color: $text-muted;
    }

    #sidebar {
        width: 42;
        height: 1fr;
    }

    #todos {
        width: 1fr;
        height: 1fr;
        border: round $secondary;
        padding: 0 1;
        scrollbar-gutter: stable;
        overflow-x: hidden;
    }

    #subagent-list {
        width: 1fr;
        height: auto;
        max-height: 12;
        border: round $secondary;
        background: $surface;
    }

    #subagent-list.background-hidden {
        display: none;
    }

    #background-list {
        display: none;
        width: 1fr;
        height: auto;
        max-height: 12;
        border: round $warning;
        background: $surface;
    }

    #background-list.visible {
        display: block;
    }

    .subagent-card {
        width: 1fr;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border-left: thick $accent;
        background: $surface;
    }

    .subagent-title {
        height: 1;
    }

    .subagent-query {
        width: 1fr;
        height: auto;
        color: $text-muted;
    }

    .subagent-status {
        width: 1fr;
        height: 1;
    }

    #turn-queue {
        display: none;
        height: auto;
        max-height: 7;
        margin: 0 1;
        padding: 0 1;
        border: round $warning;
        background: $surface;
    }

    #turn-queue.visible {
        display: block;
    }

    #turn-queue-header {
        height: 1;
        color: $warning;
        text-style: bold;
    }

    #turn-queue-list {
        height: auto;
        max-height: 4;
    }

    .turn-queue-item {
        height: 1;
    }

    #suggestions {
        display: none;
        height: auto;
        max-height: 6;
        margin: 0 1;
        border: round $accent;
        background: $surface;
    }

    #suggestions.visible {
        display: block;
    }

    #command-list {
        height: auto;
        max-height: 5;
        background: transparent;
    }

    #query {
        height: 3;
        border: tall $accent;
    }

    .message {
        height: auto;
        margin: 0 0 1 0;
    }

    .role {
        text-style: bold;
    }

    .user-role {
        color: $accent;
    }

    .agent-role {
        color: #7aa2f7;
    }

    .system-message {
        color: $text-muted;
    }

    .error-message {
        color: $error;
        text-style: bold;
    }

    .compact-card {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: round $warning;
        background: $surface;
    }

    .compact-title {
        text-style: bold;
        color: $warning;
    }

    .tool-call {
        margin: 0 0 1 0;
    }

    .tool-details {
        padding: 0 1;
        color: $text-muted;
    }

    .file-view {
        padding: 0 1;
    }

    PermissionScreen {
        align: center middle;
        background: $background 60%;
    }

    #permission-dialog {
        width: 80;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }

    #permission-buttons {
        height: auto;
        align-horizontal: right;
    }

    #permission-buttons Button {
        margin-left: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding(
            "tab",
            "toggle_subagent_view",
            "Subagents",
            priority=True,
        ),
        Binding(
            "shift+tab",
            "toggle_background_view",
            "Background",
            priority=True,
        ),
        Binding(
            "x",
            "stop_background_task",
            "Stop task",
            show=False,
        ),
        Binding("escape", "back_to_main", "Back", show=False),
    ]

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=assemble_system_prompt()
            )
        ]
        self._busy = False
        self._quitting = False
        self._snapshot: list[ChatCompletionMessageParam] = []

        self._last_stream_kind: str | None = None
        self._active_tool_details: Static | None = None
        self._active_tool_lines: list[str] = []
        self._suppress_next_tool: str | None = None
        self._write_snapshots: dict[str, tuple[str, str]] = {}
        self._spinner_index = 0

        self._suggestions_visible = False
        self._suggested_commands: list[str] = []
        self._suggestion_index = 0

        self._permission_events: set[threading.Event] = set()

        self._thread_id = threading.get_ident()
        self._view_mode = "main"
        self._subagents: dict[str, SubagentState] = {}
        self._selected_subagent_id: str | None = None
        self._subagent_option_ids: list[str] = []
        self._subagent_tick = 0

        self._turn_queue: deque[TurnRequest] = deque()
        self._current_request: TurnRequest | None = None
        self._failed_request: TurnRequest | None = None

        self._background: dict[str, BackgroundState] = {}
        self._selected_background_id: str | None = None
        self._background_option_ids: list[str] = []
        self._background_tick = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="history-panel"):
                yield VerticalScroll(id="history")
                with Vertical(id="subagent-view"):
                    yield Static("", id="subagent-header")
                    yield VerticalScroll(id="subagent-log")
                with Vertical(id="background-view"):
                    yield Static("", id="background-header")
                    yield VerticalScroll(id="background-log")
                yield Horizontal(
                    Static(SPINNER_FRAMES[0], id="spinner"),
                    Static("Agent is thinking\u2026", id="loading-label"),
                    id="loading",
                )
            with Vertical(id="sidebar"):
                yield RichLog(
                    id="todos",
                    markup=False,
                    wrap=True,
                    min_width=1,
                    highlight=False,
                    max_lines=500,
                )
                yield OptionList(id="subagent-list", compact=True)
                yield OptionList(id="background-list", compact=True)
        yield Vertical(
            Static("Queued turns", id="turn-queue-header"),
            VerticalScroll(id="turn-queue-list"),
            id="turn-queue",
        )
        yield Vertical(
            OptionList(id="command-list", compact=True),
            id="suggestions",
        )
        yield Input(
            placeholder="Ask the agent... (Enter to send, / for commands)",
            id="query",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#history-panel").border_title = "Conversation"
        self.query_one("#todos", RichLog).border_title = "Todos"
        self.query_one("#subagent-list", OptionList).border_title = "Subagents"
        self.query_one("#background-list", OptionList).border_title = "Background"

        set_output_listener(self._output_event)
        set_prompt_handler(self._ask_permission)
        add_subagent_listener(self._subagent_event)
        add_background_listener(self._background_event)
        register_hook("PreToolCall", self._pre_tool_call)
        register_hook("PostToolCall", self._post_tool_call)
        self.set_interval(0.08, self._tick_spinner)
        self.set_interval(0.5, self._tick_subagents)
        self.set_interval(0.5, self._tick_background_tasks)

        self._refresh_todos()
        self._refresh_subagent_list()
        self._refresh_background_list()
        self._refresh_turn_queue()
        self._append_history(
            Static(
                Text(
                    "Coding harness ready. Ask a question below "
                    "or type / for commands."
                ),
                classes="message system-message",
            )
        )

        query_input = self.query_one("#query", Input)
        query_input.focus()

        if self.args.query:
            self._enqueue_turn(
                TurnRequest(
                    kind="user",
                    content=self.args.query,
                    label=self.args.query,
                )
            )

    def on_unmount(self) -> None:
        clear_output_listener()
        clear_prompt_handler()
        remove_subagent_listener(self._subagent_event)
        remove_background_listener(self._background_event)
        terminate_subagents()
        terminate_background_tasks()
        unregister_hook("PreToolCall", self._pre_tool_call)
        unregister_hook("PostToolCall", self._post_tool_call)
        for event in list(self._permission_events):
            event.set()

    # ------------------------------------------------------------------
    # input and command suggestions
    # ------------------------------------------------------------------
    def on_key(self, event: events.Key) -> None:
        if self._view_mode != "main" or not self._suggestions_visible:
            return

        if event.key == "down":
            event.stop()
            event.prevent_default()
            self._move_suggestion(1)
        elif event.key == "up":
            event.stop()
            event.prevent_default()
            self._move_suggestion(-1)
        elif event.key == "escape":
            event.stop()
            event.prevent_default()
            self._hide_suggestions()

    @on(Input.Changed, "#query")
    def _query_changed(self, event: Input.Changed) -> None:
        self._update_suggestions(event.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        event.input.value = ""

        if self._quitting or not query:
            return

        if self._suggestions_visible and self._suggested_commands:
            command = self._suggested_commands[self._suggestion_index]
            self._hide_suggestions()
            await self._execute_command(command)
            return

        if query.lower() in EXIT_COMMANDS:
            await self.action_quit()
            return

        if query.startswith("/"):
            await self._execute_command(query)
            return

        self._enqueue_turn(
            TurnRequest(kind="user", content=query, label=query)
        )

    async def _execute_command(self, raw: str) -> None:
        command = raw.strip().lower()
        if command == "/compact":
            if self._busy or self._current_request is not None:
                self._write_error(
                    "[command] /compact is unavailable while a turn is running"
                )
                return
            self._start_compact()
        elif command in EXIT_COMMANDS:
            await self.action_quit()
        elif command == "/help":
            self._write_help()
        else:
            self._write_error(f"[command] invalid operation: {raw.strip()}")

    @on(OptionList.OptionSelected, "#command-list")
    async def _suggestion_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        command = str(event.option.id or event.option.prompt)
        self.query_one("#query", Input).value = ""
        self._hide_suggestions()
        await self._execute_command(command)
        self.query_one("#query", Input).focus()

    def _update_suggestions(self, value: str) -> None:
        if not value.startswith("/") or " " in value:
            self._hide_suggestions()
            return

        prefix = value.lower()
        matches = [
            (command, description)
            for command, description in COMMANDS
            if command.startswith(prefix)
        ]
        if not matches:
            self._hide_suggestions()
            return

        self._suggested_commands = [command for command, _ in matches]
        self._suggestion_index = 0

        option_list = self.query_one("#command-list", OptionList)
        option_list.set_options(
            Option(
                Text.assemble(
                    (command, "bold cyan"),
                    (f"  {description}", "dim"),
                ),
                id=command,
            )
            for command, description in matches
        )
        option_list.highlighted = 0

        self._suggestions_visible = True
        self.query_one("#suggestions").add_class("visible")

    def _move_suggestion(self, delta: int) -> None:
        if not self._suggested_commands:
            return
        count = len(self._suggested_commands)
        self._suggestion_index = (self._suggestion_index + delta) % count
        option_list = self.query_one("#command-list", OptionList)
        option_list.highlighted = self._suggestion_index
        option_list.scroll_to_highlight()

    def _hide_suggestions(self) -> None:
        self._suggestions_visible = False
        self._suggested_commands = []
        self._suggestion_index = 0
        if self.is_mounted:
            self.query_one("#suggestions").remove_class("visible")

    def _write_help(self) -> None:
        body = "\n".join(
            f"{command}  {description}"
            for command, description in COMMANDS
        )
        self._append_history(
            Static(Text(body), classes="message system-message")
        )

    # ------------------------------------------------------------------
    # turns
    # ------------------------------------------------------------------
    def _enqueue_turn(self, request: TurnRequest, front: bool = False) -> None:
        if self._quitting:
            return
        if front:
            self._turn_queue.appendleft(request)
        else:
            self._turn_queue.append(request)
        self._refresh_turn_queue()
        self._start_next_turn()

    def _start_next_turn(self) -> None:
        if (
            self._busy
            or self._quitting
            or self._current_request is not None
            or not self._turn_queue
        ):
            return

        request = self._turn_queue.popleft()
        if request.kind == "background":
            batch = [request]
            while self._turn_queue and self._turn_queue[0].kind == "background":
                batch.append(self._turn_queue.popleft())
            if len(batch) > 1:
                request = TurnRequest(
                    kind="background",
                    content="\n\n".join(item.content for item in batch),
                    label=f"{len(batch)} background updates",
                    attempts=max(item.attempts for item in batch),
                )

        self._current_request = request
        self._failed_request = None
        self._busy = True
        self._last_stream_kind = None
        self._active_tool_details = None
        self._active_tool_lines = []
        self._snapshot = list(self.messages)

        if request.kind == "user":
            self._write_user(request.content)
        else:
            self._write_background_update(request.label)

        self.messages.append(
            ChatCompletionUserMessageParam(
                role="user", content=request.content
            )
        )
        self._refresh_turn_queue()
        self._set_loading(True, "Agent is thinking\u2026")
        self.run_worker(self._worker_turn, thread=True)

    def _worker_turn(self) -> None:
        try:
            with self._capture():
                run_turn(self.messages)
        except Exception as error:
            self.messages[:] = self._snapshot
            self._failed_request = self._current_request
            self._emit_lines([f"[error] turn failed: {error}"])
        finally:
            self._finish_turn_from_thread()

    def _start_compact(self) -> None:
        self._busy = True
        self._current_request = None
        self._failed_request = None
        self._set_loading(True, "Compacting conversation\u2026")
        self.run_worker(self._worker_compact, thread=True)

    def _worker_compact(self) -> None:
        try:
            with self._capture():
                self.messages[:] = compact_messages(self.messages)
        except Exception as error:
            self._emit_lines([f"[error] compaction failed: {error}"])
        finally:
            self._finish_turn_from_thread()

    def _finish_turn_from_thread(self) -> None:
        try:
            self.call_from_thread(self._finish_turn)
        except RuntimeError:
            pass

    def _finish_turn(self) -> None:
        self._busy = False
        self._set_loading(False)

        failed = self._failed_request
        self._failed_request = None
        self._current_request = None

        if self._quitting:
            return

        if (
            failed is not None
            and failed.kind == "background"
            and failed.attempts < 1
        ):
            failed.attempts += 1
            self._turn_queue.appendleft(failed)

        self._refresh_turn_queue()
        if self._turn_queue:
            self._start_next_turn()
        else:
            self.query_one("#query", Input).focus()

    def _refresh_turn_queue(self) -> None:
        try:
            container = self.query_one("#turn-queue")
            header = self.query_one("#turn-queue-header", Static)
            queue_list = self.query_one("#turn-queue-list", VerticalScroll)
        except Exception:
            return

        queue_list.remove_children()
        if not self._turn_queue:
            container.remove_class("visible")
            header.update("Queued turns")
            return

        container.add_class("visible")
        header.update(Text(f"Queued turns ({len(self._turn_queue)})"))

        rows = []
        for index, request in enumerate(self._turn_queue, start=1):
            style = "cyan" if request.kind == "user" else "yellow"
            label = _shorten(request.label.replace("\n", " "), 90)
            rows.append(
                Static(
                    Text.assemble(
                        (f"{index}. ", "dim"),
                        (f"[{request.kind}] ", style),
                        (label, ""),
                    ),
                    classes="turn-queue-item",
                )
            )
        if rows:
            queue_list.mount_all(rows)

    async def action_quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        self._turn_queue.clear()
        self._refresh_turn_queue()

        if self._busy:
            terminate_background_tasks()
            self.exit()
            return

        self.query_one("#query", Input).disabled = True
        worker = self.run_worker(self._worker_stop, thread=True)
        try:
            await worker.wait()
        except Exception:
            pass
        terminate_background_tasks()
        self.exit()

    def _worker_stop(self) -> None:
        try:
            with self._capture():
                trigger_hooks("Stop", messages=self.messages)
        except Exception as error:
            self._emit_lines([f"[error] stop hooks failed: {error}"])

    def _set_loading(self, visible: bool, label: str = "") -> None:
        loading = self.query_one("#loading")
        if visible:
            self._spinner_index = 0
            self.query_one("#spinner", Static).update(
                Text(SPINNER_FRAMES[0])
            )
            if label:
                self.query_one("#loading-label", Static).update(label)
            loading.add_class("visible")
            self._scroll_history_end()
        else:
            loading.remove_class("visible")

    def _tick_spinner(self) -> None:
        try:
            loading = self.query_one("#loading")
        except Exception:
            return
        if not loading.has_class("visible"):
            return
        self._spinner_index = (self._spinner_index + 1) % len(
            SPINNER_FRAMES
        )
        self.query_one("#spinner", Static).update(
            Text(SPINNER_FRAMES[self._spinner_index])
        )

    # ------------------------------------------------------------------
    # subagents
    # ------------------------------------------------------------------
    def action_toggle_subagent_view(self) -> None:
        if self._view_mode == "subagents":
            self._show_main_view()
        else:
            self._show_subagent_view()

    def action_toggle_background_view(self) -> None:
        if self._view_mode == "background":
            self._show_main_view()
        else:
            self._show_background_view()

    def action_back_to_main(self) -> None:
        if self._view_mode in {"subagents", "background"}:
            self._show_main_view()

    def action_stop_background_task(self) -> None:
        if self._view_mode != "background":
            return
        state = self._selected_background()
        if state is None or state.status != "running":
            return
        if terminate_background_task(state.id):
            state.status = "cancelling"
            self._refresh_background_list()
            self._render_selected_background()

    def _show_subagent_view(self) -> None:
        self._hide_suggestions()
        self._view_mode = "subagents"
        self.query_one("#history").add_class("hidden")
        self.query_one("#background-view").remove_class("visible")
        self.query_one("#subagent-view").add_class("visible")
        self.query_one("#background-list").remove_class("visible")
        self.query_one("#subagent-list").remove_class("background-hidden")

        if self._subagents:
            self._selected_subagent_id = max(
                self._subagents.values(), key=lambda state: state.started
            ).id

        self._refresh_subagent_list()
        self._select_subagent(self._selected_subagent_id)
        self.call_after_refresh(self._render_selected_subagent)

        try:
            self.query_one("#subagent-list", OptionList).focus()
        except Exception:
            pass

    def _show_background_view(self) -> None:
        self._hide_suggestions()
        self._view_mode = "background"
        self.query_one("#history").add_class("hidden")
        self.query_one("#subagent-view").remove_class("visible")
        self.query_one("#background-view").add_class("visible")
        self.query_one("#subagent-list").add_class("background-hidden")
        self.query_one("#background-list").add_class("visible")

        if self._background:
            self._selected_background_id = max(
                self._background.values(), key=lambda state: state.started
            ).id

        self._refresh_background_list()
        self._select_background(self._selected_background_id)
        self.call_after_refresh(self._render_selected_background)

        try:
            self.query_one("#background-list", OptionList).focus()
        except Exception:
            pass

    def _show_main_view(self) -> None:
        self._view_mode = "main"
        self.query_one("#subagent-view").remove_class("visible")
        self.query_one("#background-view").remove_class("visible")
        self.query_one("#history").remove_class("hidden")
        self.query_one("#subagent-list").remove_class("background-hidden")
        self.query_one("#background-list").remove_class("visible")
        self._scroll_history_end()
        self.query_one("#query", Input).focus()

    def _select_subagent(self, subagent_id: str | None) -> None:
        self._selected_subagent_id = subagent_id
        if subagent_id is None:
            return

        try:
            option_list = self.query_one("#subagent-list", OptionList)
            if subagent_id in self._subagent_option_ids:
                option_list.highlighted = self._subagent_option_ids.index(
                    subagent_id
                )
        except Exception:
            pass

    def _selected_subagent(self) -> SubagentState | None:
        if self._selected_subagent_id is None:
            return None
        return self._subagents.get(self._selected_subagent_id)

    @on(OptionList.OptionHighlighted, "#subagent-list")
    def _subagent_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        option_id = str(event.option.id or "")
        if not option_id or option_id == "__empty__":
            return
        self._selected_subagent_id = option_id
        if self._view_mode == "subagents":
            self._render_selected_subagent()

    def _subagent_event(self, event: SubagentEvent) -> None:
        if threading.get_ident() == self._thread_id:
            self._handle_subagent_event(event)
            return
        try:
            self.call_from_thread(self._handle_subagent_event, event)
        except RuntimeError:
            pass

    def _handle_subagent_event(self, event: SubagentEvent) -> None:
        state = self._subagents.get(event.subagent_id)

        if event.kind == "spawned":
            state = SubagentState(
                id=event.subagent_id,
                name=event.name,
                query=event.query,
            )
            self._subagents[event.subagent_id] = state
            if self._selected_subagent_id is None:
                self._selected_subagent_id = state.id

            self._append_subagent_card(state)
            self._refresh_subagent_list()
            self._update_loading_for_subagents()
            if self._view_mode == "subagents":
                self._select_subagent(state.id)
                self._render_selected_subagent()
            return

        if state is None:
            return

        if event.kind == "finished":
            state.status = event.status or "completed"
            state.finished_at = time.monotonic()
            self._update_subagent_card(state)
            self._refresh_subagent_list()
            if (
                self._view_mode == "subagents"
                and self._selected_subagent_id == state.id
            ):
                self._update_subagent_header()
            self._update_loading_for_subagents()
            return

        entry, created = self._append_subagent_entry(
            state, event.kind, event.text
        )
        if len(state.entries) > 2000:
            removed = state.entries[:1000]
            if (
                self._view_mode == "subagents"
                and self._selected_subagent_id == state.id
            ):
                for old_entry in removed:
                    if old_entry.widget is not None:
                        try:
                            old_entry.widget.remove()
                        except Exception:
                            pass
            del state.entries[:1000]

        if (
            self._view_mode == "subagents"
            and self._selected_subagent_id == state.id
        ):
            if created and entry.widget is not None:
                try:
                    self.query_one(
                        "#subagent-log", VerticalScroll
                    ).mount(entry.widget)
                except Exception:
                    pass
            self.call_after_refresh(self._scroll_subagent_log)
            self._update_subagent_header()


    def _append_subagent_entry(
        self,
        state: SubagentState,
        kind: str,
        text: str,
    ) -> tuple[SubagentEntry, bool]:
        last = state.entries[-1] if state.entries else None

        if kind == "assistant" and last is not None and last.kind == "assistant":
            last.text = f"{last.text}\n{text}" if last.text else text
            if last.content_widget is not None:
                last.content_widget.update(RichMarkdown(last.text))
            return last, False

        if (
            kind == "tool"
            and last is not None
            and last.kind == "tool"
            and not _TOOL_CALL_RE.match(text)
        ):
            last.text = f"{last.text}\n{text}"
            if last.content_widget is not None:
                last.content_widget.update(
                    Text(_shorten(last.text, 4000))
                )
            return last, False

        entry_kind = kind if kind in ("assistant", "tool", "error") else "status"
        tool_name = _tool_name(text) if entry_kind == "tool" else ""
        entry = SubagentEntry(
            kind=entry_kind,
            text=text,
            tool_name=tool_name,
        )
        entry.widget = _build_subagent_entry(entry)
        state.entries.append(entry)
        return entry, True

    def _scroll_subagent_log(self) -> None:
        try:
            self.query_one("#subagent-log", VerticalScroll).scroll_end(
                animate=False
            )
        except Exception:
            pass

    def _append_subagent_card(self, state: SubagentState) -> None:
        title = Text()
        title.append("\u25c6 ", style="bold cyan")
        title.append(state.name, style="bold cyan")
        title.append("  spawned", style="dim")

        query = Static(Text(state.query), classes="subagent-query")
        status = Static(Text("Starting\u2026", style="dim"),
                        classes="subagent-status")
        state.card_status = status

        card = Vertical(
            Static(title, classes="subagent-title"),
            query,
            status,
            classes="message subagent-card",
        )
        self._append_history(card)
        self._update_subagent_card(state)

    def _update_subagent_card(self, state: SubagentState) -> None:
        status = state.card_status
        if status is None:
            return

        if state.status == "running":
            frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            status.update(
                Text(
                    f"{frame} running for {_format_elapsed(state.elapsed)}",
                    style="yellow",
                )
            )
        elif state.status == "completed":
            status.update(
                Text(
                    f"\u2713 completed in {_format_elapsed(state.elapsed)}",
                    style="green",
                )
            )
        else:
            status.update(
                Text(
                    f"\u2717 failed after {_format_elapsed(state.elapsed)}",
                    style="bold red",
                )
            )

    def _tick_subagents(self) -> None:
        if not self._subagents:
            return

        for state in self._subagents.values():
            if state.status == "running":
                self._update_subagent_card(state)

        if self._view_mode == "subagents":
            self._update_subagent_header()

        self._subagent_tick += 1
        if self._subagent_tick % 2 == 0:
            self._refresh_subagent_list()

    def _refresh_subagent_list(self) -> None:
        try:
            option_list = self.query_one("#subagent-list", OptionList)
        except Exception:
            return

        states = sorted(
            self._subagents.values(), key=lambda state: state.started
        )
        ids = [state.id for state in states]

        if not ids:
            self._subagent_option_ids = []
            if option_list.option_count == 0:
                option_list.set_options(
                    [
                        Option(
                            Text("No subagents yet", style="dim"),
                            id="__empty__",
                            disabled=True,
                        )
                    ]
                )
            return

        if ids != self._subagent_option_ids:
            self._subagent_option_ids = ids
            selected = (
                self._selected_subagent_id
                if self._selected_subagent_id in ids
                else ids[0]
            )
            self._selected_subagent_id = selected
            option_list.set_options(
                Option(self._subagent_option_prompt(state), id=state.id)
                for state in states
            )
            if selected in ids:
                option_list.highlighted = ids.index(selected)
            return

        for index, state in enumerate(states):
            try:
                option_list.replace_option_prompt_at_index(
                    index, self._subagent_option_prompt(state)
                )
            except Exception:
                pass

    def _subagent_option_prompt(self, state: SubagentState) -> Text:
        if state.status == "running":
            icon = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            style = "bold yellow"
        else:
            icon, style = SUBAGENT_STATUS_STYLES.get(
                state.status, ("\u2022", "dim")
            )

        return Text.assemble(
            (f"{icon} ", style),
            (state.name, "bold"),
            (f"  {_format_elapsed(state.elapsed)}", "dim"),
        )

    def _update_subagent_header(self) -> None:
        try:
            header = self.query_one("#subagent-header", Static)
        except Exception:
            return

        state = self._selected_subagent()
        if state is None:
            header.update(Text("No subagent selected", style="dim"))
            return

        icon, style = SUBAGENT_STATUS_STYLES.get(
            state.status, ("\u2022", "dim")
        )
        body = Text()
        body.append(f"{icon} ", style=style)
        body.append(state.name, style="bold cyan")
        body.append(f"  {state.status}", style=style)
        body.append(f"  {_format_elapsed(state.elapsed)}", style="dim")
        body.append("\n")
        body.append("Query: ", style="dim")
        body.append(state.query)
        header.update(body)

    def _render_selected_subagent(self) -> None:
        try:
            log = self.query_one("#subagent-log", VerticalScroll)
        except Exception:
            return

        log.remove_children()
        state = self._selected_subagent()
        if state is None:
            log.mount(
                Static(
                    Text("No subagent selected.", style="dim"),
                    classes="message system-message",
                )
            )
            self._update_subagent_header()
            return

        if not state.entries:
            log.mount(
                Static(
                    Text("Waiting for output\u2026", style="dim"),
                    classes="message system-message",
                )
            )
        else:
            widgets = []
            for entry in state.entries:
                entry.widget = _build_subagent_entry(entry)
                widgets.append(entry.widget)
            if widgets:
                log.mount_all(widgets)

        self.call_after_refresh(self._scroll_subagent_log)
        self._update_subagent_header()

    def _update_loading_for_subagents(self) -> None:
        if not self._busy:
            return

        running = [
            state.name
            for state in self._subagents.values()
            if state.status == "running"
        ]
        if running:
            self._set_loading(
                True, f"Waiting for subagent {', '.join(running)}\u2026"
            )
        else:
            self._set_loading(True, "Agent is thinking\u2026")

    # ------------------------------------------------------------------
    # background tasks
    # ------------------------------------------------------------------
    def _background_event(self, event: BackgroundEvent) -> None:
        if threading.get_ident() == self._thread_id:
            self._handle_background_event(event)
            return
        try:
            self.call_from_thread(self._handle_background_event, event)
        except RuntimeError:
            pass

    def _handle_background_event(self, event: BackgroundEvent) -> None:
        state = self._background.get(event.task_id)

        if event.kind == "started":
            if state is None:
                state = BackgroundState(
                    id=event.task_id,
                    cmd=event.cmd,
                    started=time.monotonic(),
                )
                self._background[event.task_id] = state
                if self._selected_background_id is None:
                    self._selected_background_id = state.id
            else:
                state.cmd = event.cmd or state.cmd

            self._refresh_background_list()
            if self._view_mode == "background":
                self._select_background(state.id)
                self._render_selected_background()
            return

        if state is None:
            state = BackgroundState(
                id=event.task_id,
                cmd=event.cmd,
                started=time.monotonic() - event.duration,
            )
            self._background[event.task_id] = state

        state.status = event.status or event.kind
        state.stdout = event.stdout
        state.stderr = event.stderr
        state.return_code = event.return_code
        state.error = event.error
        state.finished_at = time.monotonic()

        self._refresh_background_list()
        if self._view_mode == "background":
            if self._selected_background_id == state.id:
                self._render_selected_background()
            else:
                self._update_background_header()

        if event.kind in {"completed", "failed"} and not self._quitting:
            self._enqueue_turn(
                TurnRequest(
                    kind="background",
                    content=format_background_event(event),
                    label=f"{event.task_id} {event.kind}",
                )
            )

    def _select_background(self, task_id: str | None) -> None:
        self._selected_background_id = task_id
        if task_id is None:
            return

        try:
            option_list = self.query_one("#background-list", OptionList)
            if task_id in self._background_option_ids:
                option_list.highlighted = self._background_option_ids.index(
                    task_id
                )
        except Exception:
            pass

    def _selected_background(self) -> BackgroundState | None:
        if self._selected_background_id is None:
            return None
        return self._background.get(self._selected_background_id)

    @on(OptionList.OptionHighlighted, "#background-list")
    def _background_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        option_id = str(event.option.id or "")
        if not option_id or option_id == "__empty__":
            return
        self._selected_background_id = option_id
        if self._view_mode == "background":
            self._render_selected_background()

    def _tick_background_tasks(self) -> None:
        if not self._background:
            return

        if self._view_mode == "background":
            self._update_background_header()

        self._background_tick += 1
        if self._background_tick % 2 == 0:
            self._refresh_background_list()

    def _refresh_background_list(self) -> None:
        try:
            option_list = self.query_one("#background-list", OptionList)
        except Exception:
            return

        states = sorted(
            self._background.values(), key=lambda state: state.started
        )
        ids = [state.id for state in states]

        if not ids:
            self._background_option_ids = []
            if option_list.option_count == 0:
                option_list.set_options(
                    [
                        Option(
                            Text("No background tasks yet", style="dim"),
                            id="__empty__",
                            disabled=True,
                        )
                    ]
                )
            return

        if ids != self._background_option_ids:
            self._background_option_ids = ids
            selected = (
                self._selected_background_id
                if self._selected_background_id in ids
                else ids[-1]
            )
            self._selected_background_id = selected
            option_list.set_options(
                Option(self._background_option_prompt(state), id=state.id)
                for state in states
            )
            if selected in ids:
                option_list.highlighted = ids.index(selected)
            return

        for index, state in enumerate(states):
            try:
                option_list.replace_option_prompt_at_index(
                    index, self._background_option_prompt(state)
                )
            except Exception:
                pass

    def _background_option_prompt(self, state: BackgroundState) -> Text:
        if state.status == "running":
            icon = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
            style = "bold yellow"
        else:
            icon, style = BACKGROUND_STATUS_STYLES.get(
                state.status, ("\u2022", "dim")
            )

        return Text.assemble(
            (f"{icon} ", style),
            (state.id, "bold"),
            (f"  {_shorten(state.cmd, 28)}", "dim"),
            (f"  {_format_elapsed(state.elapsed)}", "dim"),
        )

    def _update_background_header(self) -> None:
        try:
            header = self.query_one("#background-header", Static)
        except Exception:
            return

        state = self._selected_background()
        if state is None:
            header.update(Text("No background task selected", style="dim"))
            return

        icon, style = BACKGROUND_STATUS_STYLES.get(
            state.status, ("\u2022", "dim")
        )
        body = Text()
        body.append(f"{icon} ", style=style)
        body.append(state.id, style="bold yellow")
        body.append(f"  {state.status}", style=style)
        body.append(f"  {_format_elapsed(state.elapsed)}", style="dim")
        if state.return_code is not None:
            body.append(f"  rc={state.return_code}", style="dim")
        body.append("\n")
        body.append("Command: ", style="dim")
        body.append(state.cmd)
        if state.error:
            body.append("\n")
            body.append("Error: ", style="dim")
            body.append(state.error, style="bold red")
        header.update(body)

    def _render_selected_background(self) -> None:
        try:
            log = self.query_one("#background-log", VerticalScroll)
        except Exception:
            return

        log.remove_children()
        state = self._selected_background()
        if state is None:
            log.mount(
                Static(
                    Text("No background task selected.", style="dim"),
                    classes="message system-message",
                )
            )
            self._update_background_header()
            return

        if state.status in {"running", "cancelling"}:
            message = (
                "Task is still running. Captured output will appear "
                "here when it finishes."
            )
            if state.status == "running":
                message += " Press x to stop it."
            else:
                message = "Stopping task\u2026"
            log.mount(
                Static(
                    Text(message, style="dim"),
                    classes="message system-message",
                )
            )
            self._update_background_header()
            return

        body = Text()
        body.append("Status    ", style="dim")
        body.append(state.status, style="bold")
        body.append("\n")
        body.append("Command   ", style="dim")
        body.append(state.cmd)
        body.append("\n")
        body.append("Duration  ", style="dim")
        body.append(_format_elapsed(state.elapsed))
        body.append("\n")
        body.append("Exit code ", style="dim")
        body.append(
            "unknown" if state.return_code is None else str(state.return_code)
        )
        body.append("\n\nstdout\n", style="bold")
        body.append(state.stdout or "(empty)")
        body.append("\n\nstderr\n", style="bold")
        body.append(state.stderr or "(empty)")
        if state.error:
            body.append("\n\nerror\n", style="bold red")
            body.append(state.error, style="bold red")

        log.mount(
            Static(body, classes="message system-message")
        )
        self.call_after_refresh(self._scroll_background_log)
        self._update_background_header()

    def _scroll_background_log(self) -> None:
        try:
            self.query_one("#background-log", VerticalScroll).scroll_end(
                animate=False
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # history rendering
    # ------------------------------------------------------------------
    def _append_history(self, widget) -> None:
        history = self.query_one("#history", VerticalScroll)
        history.mount(widget)
        self.call_after_refresh(self._scroll_history_end)

    def _scroll_history_end(self) -> None:
        try:
            self.query_one("#history", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _write_user(self, query: str) -> None:
        block = Vertical(
            Static("You", classes="role user-role"),
            Static(Text(query)),
            classes="message user-message",
        )
        self._append_history(block)

    def _write_background_update(self, label: str) -> None:
        block = Vertical(
            Static("Background update", classes="role system-message"),
            Static(Text(label)),
            classes="message system-message",
        )
        self._append_history(block)

    def _write_error(self, message: str) -> None:
        self._append_history(
            Static(
                Text(message),
                classes="message error-message",
            )
        )

    def _write_compaction(self, body: str) -> None:
        stats = _COMPACT_STATS.search(body)
        details = Text()

        if stats:
            before_m, after_m, delta_m, before_c, after_c = stats.groups()
            delta = int(delta_m)
            details.append("Messages    ", style="dim")
            details.append(before_m, style="bold")
            details.append(" \u2192 ", style="dim")
            details.append(after_m, style="bold")
            details.append(
                f"  ({delta:+d})\n",
                style="green" if delta <= 0 else "yellow",
            )
            details.append("Characters  ", style="dim")
            details.append(f"{int(before_c):,}", style="bold")
            details.append(" \u2192 ", style="dim")
            details.append(f"{int(after_c):,}", style="bold")
        else:
            details.append(body, style="dim")

        card = Vertical(
            Static("Context compacted", classes="role compact-title"),
            Static(details),
            classes="message compact-card",
        )
        self._append_history(card)

    def _output_event(self, kind: str, body: str) -> None:
        try:
            self.call_from_thread(self._handle_output_event, kind, body)
        except RuntimeError:
            pass

    def _handle_output_event(self, kind: str, body: str) -> None:
        self._last_stream_kind = None
        self._active_tool_details = None
        self._active_tool_lines = []

        if kind == "assistant":
            block = Vertical(
                Static("Agent", classes="role agent-role"),
                Static(RichMarkdown(body)),
                classes="message agent-message",
            )
            self._append_history(block)
            return

        if kind == "status":
            self._append_history(
                Static(Text(body), classes="message system-message")
            )
            return

        if kind == "info":
            if _COMPACT_STATS.search(body):
                self._write_compaction(body)
            else:
                self._append_history(
                    Static(Text(body), classes="message system-message")
                )
            return

        if kind == "debug":
            return

        if kind == "error":
            self._write_error(body)
            return

        self._append_history(Static(Text(body), classes="message"))

    def _emit_lines(self, lines: list[str]) -> None:
        try:
            self.call_from_thread(self._handle_stream_lines, lines)
        except RuntimeError:
            pass

    def _handle_stream_lines(self, lines: list[str]) -> None:
        for raw_line in lines:
            self._handle_stream_line(raw_line.rstrip("\r"))

    def _handle_stream_line(self, line: str) -> None:
        if not line:
            return

        kind: str | None = None
        body = line
        for prefix, candidate in PREFIX_KINDS:
            if line.startswith(prefix):
                kind = candidate
                body = line[len(prefix):].lstrip()
                break

        had_prefix = kind is not None
        if kind is None:
            if _TAGGED_LINE.match(line):
                kind = "plain"
            else:
                kind = self._last_stream_kind or "plain"
        self._last_stream_kind = kind

        if had_prefix and kind != "tool":
            self._active_tool_details = None
            self._active_tool_lines = []

        if kind in ("todos", "debug", "tool_skip"):
            return

        if kind == "tool":
            if had_prefix:
                name = _tool_name(body)
                if name == self._suppress_next_tool:
                    self._suppress_next_tool = None
                    self._last_stream_kind = "tool_skip"
                    return
                self._start_tool_widget(name, body)
            else:
                self._append_tool_detail(body)
            return

        if kind == "info":
            if _COMPACT_STATS.search(body):
                self._write_compaction(body)
            else:
                self._append_history(
                    Static(Text(body), classes="message system-message")
                )
            return

        body = _shorten(body)

        if kind == "system":
            widget = Static(
                Text(body),
                classes="message system-message",
            )
        elif kind == "summary":
            widget = Static(
                Text(body),
                classes="message system-message",
            )
        elif kind == "status":
            widget = Static(
                Text(body, style="italic"),
                classes="message system-message",
            )
        elif kind == "error":
            widget = Static(
                Text(body),
                classes="message error-message",
            )
        else:
            widget = Static(Text(body), classes="message")

        self._append_history(widget)

    def _start_tool_widget(self, name: str, body: str) -> None:
        if name in {"spawn_subagent", "spawn_subagents"}:
            return

        details = Static(Text(body), classes="tool-details")
        collapsible = Collapsible(
            details,
            title=f"\U0001f527 {name}",
            collapsed=name not in FILE_TOOLS,
            classes="tool-call",
        )
        self._active_tool_details = details
        self._active_tool_lines = [body]
        self._append_history(collapsible)

    def _append_tool_detail(self, line: str) -> None:
        if self._active_tool_details is None:
            return
        self._active_tool_lines.append(line)
        self._active_tool_details.update(
            Text("\n".join(self._active_tool_lines))
        )

    # ------------------------------------------------------------------
    # file tools
    # ------------------------------------------------------------------
    def _pre_tool_call(self, messages=None, tool=None) -> None:
        try:
            self._pre_tool_call_impl(messages, tool)
        except Exception:
            pass

    def _pre_tool_call_impl(self, messages, tool) -> None:
        name = _tool_name_from_call(tool)
        if name != "write_file":
            return

        args = _tool_args(tool)
        path = args.get("path")
        if not isinstance(path, str) or not _inside_workspace(path):
            return

        try:
            old = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            old = ""

        tool_id = getattr(tool, "id", "")
        self._write_snapshots[str(tool_id)] = (path, old)

    def _post_tool_call(self, messages=None, tool=None, result=None) -> None:
        try:
            self._post_tool_call_impl(messages, tool, result)
        except Exception:
            pass

    def _post_tool_call_impl(self, messages, tool, result) -> None:
        name = _tool_name_from_call(tool)

        if name == "run_todo_write":
            self._suppress_next_tool = name
            try:
                self.call_from_thread(self._refresh_todos)
            except RuntimeError:
                pass
            return

        if name not in FILE_TOOLS:
            return

        args = _tool_args(tool)
        path = str(args.get("path", ""))

        if name == "write_file":
            if not _inside_workspace(path):
                return
            tool_id = str(getattr(tool, "id", ""))
            saved_path, old = self._write_snapshots.pop(
                tool_id, (path, "")
            )
            path = saved_path or path
            try:
                new = Path(path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                new = str(args.get("content", ""))
            rendered = render_diff(path, old, new)
            title = f"\U0001f4dd write_file \u00b7 {_shorten(path, 60)}"
        else:
            stdout, error = _parse_read_result(result)
            if error:
                body = Text()
                body.append(error, style="bold red")
                rendered = body
            else:
                offset = args.get("offset", 0)
                try:
                    start_line = int(offset) + 1
                except (TypeError, ValueError):
                    start_line = 1
                rendered = render_contents(path, stdout, start_line)
            title = f"\U0001f4d6 read_file \u00b7 {_shorten(path, 60)}"

        self._suppress_next_tool = name
        try:
            self.call_from_thread(self._append_file_view, title, rendered)
        except RuntimeError:
            pass

    def _append_file_view(self, title: str, rendered: Text) -> None:
        details = Static(rendered, classes="file-view")
        collapsible = Collapsible(
            details,
            title=escape(title),
            collapsed=False,
            classes="tool-call",
        )
        self._append_history(collapsible)

    # ------------------------------------------------------------------
    # todos
    # ------------------------------------------------------------------
    def _refresh_todos(self) -> None:
        todos = list(tools_service.CURRENT_TODOS)
        todos_log = self.query_one("#todos", RichLog)
        todos_log.clear()

        if not todos:
            todos_log.write(Text("No todos yet.", style="dim"))
            return

        for todo in todos:
            status = str(todo.get("status", "pending"))
            icon, style = STATUS_ICONS.get(status, ("\u2022", ""))
            line = Text()
            line.append(f"{icon} ", style=style)
            line.append(str(todo.get("title", "")), style=style)
            todos_log.write(line)

    # ------------------------------------------------------------------
    # permissions
    # ------------------------------------------------------------------
    def _ask_permission(self, tool_name: str, args: dict, reason: str) -> bool:
        done = threading.Event()
        outcome = {"allow": False}
        self._permission_events.add(done)

        def show() -> None:
            self._set_loading(True, "Waiting for approval\u2026")
            screen = PermissionScreen(tool_name, args, reason)

            def handle(result: bool | None) -> None:
                outcome["allow"] = bool(result)
                self._set_loading(True, "Agent is thinking\u2026")
                done.set()

            self.push_screen(screen, handle)

        try:
            self.call_from_thread(show)
            done.wait()
        finally:
            self._permission_events.discard(done)

        return outcome["allow"]

    @contextmanager
    def _capture(self):
        writer = _TUILogWriter(self._emit_lines)
        with redirect_stdout(writer), redirect_stderr(writer):
            try:
                yield
            finally:
                writer.flush()


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, remainder = divmod(total, 60)
    return f"{minutes}m {remainder:02d}s"


def _subagent_tool_title(entry: SubagentEntry) -> tuple[str, bool]:
    match = re.search(r"['\"]path['\"]\s*:\s*['\"]([^'\"]+)", entry.text)
    path = match.group(1) if match else ""

    if entry.tool_name == "read_file":
        title = "\U0001f4d6 read_file"
        if path:
            title += f" \u00b7 {_shorten(path, 60)}"
        return title, True

    if entry.tool_name == "write_file":
        title = "\U0001f4dd write_file"
        if path:
            title += f" \u00b7 {_shorten(path, 60)}"
        return title, False

    return f"\U0001f527 {entry.tool_name}", False


def _build_subagent_entry(entry: SubagentEntry) -> Widget:
    if entry.kind == "assistant":
        content = Static(RichMarkdown(entry.text))
        entry.content_widget = content
        return Vertical(
            Static("Agent", classes="role agent-role"),
            content,
            classes="message agent-message",
        )

    if entry.kind == "tool":
        details = Static(
            Text(_shorten(entry.text, 4000)), classes="tool-details"
        )
        entry.content_widget = details
        title, collapsed = _subagent_tool_title(entry)
        return Collapsible(
            details,
            title=escape(title),
            collapsed=collapsed,
            classes="tool-call",
        )

    if entry.kind == "error":
        return Static(
            Text(_shorten(entry.text, 4000)),
            classes="message error-message",
        )

    return Static(
        Text(_shorten(entry.text, 4000)),
        classes="message system-message",
    )


def _shorten(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\u2026"


def _inside_workspace(path: str) -> bool:
    try:
        return Path(path).resolve().is_relative_to(WORKDIR.resolve())
    except (OSError, ValueError):
        return False


def _tool_name(body: str) -> str:
    name, separator, _ = body.partition("(")
    return name.strip() if separator else body.strip()


def _tool_name_from_call(tool) -> str:
    return str(getattr(getattr(tool, "function", None), "name", ""))


def _tool_args(tool) -> dict:
    raw = getattr(getattr(tool, "function", None), "arguments", "{}")
    try:
        args = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return args if isinstance(args, dict) else {}


def _parse_read_result(result) -> tuple[str, str]:
    try:
        data = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return "", str(result)

    if not isinstance(data, dict):
        return "", str(result)

    stdout = str(data.get("stdout", ""))
    stderr = str(data.get("stderr", ""))
    return_code = data.get("return_code", 0)

    if return_code != 0 and stderr:
        return "", stderr
    return stdout, ""


def run_tui(args) -> None:
    trigger_hooks("Start")
    HarnessApp(args).run()
