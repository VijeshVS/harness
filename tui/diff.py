from difflib import SequenceMatcher

from rich.text import Text

ADDED_STYLE = "on #173b22 #8ee08e"
REMOVED_STYLE = "on #3b1f1f #ff9d9d"
GUTTER_STYLE = "dim"


def _line_number_width(old_lines: list[str], new_lines: list[str]) -> int:
    highest = max(len(old_lines), len(new_lines), 1)
    return max(2, len(str(highest)))


def _append_line(
    text: Text,
    number: int,
    line: str,
    width: int,
    style: str | None,
) -> None:
    text.append(f"{number:>{width}} ", style=style or GUTTER_STYLE)
    text.append(line + "\n", style=style or "")


def _append_no_newline(text: Text, number: int, width: int, style: str) -> None:
    text.append(
        f"{number:>{width}} \\ No newline at end of file\n",
        style=style,
    )


def render_diff(path: str, old: str, new: str, context: int = 3) -> Text:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    width = _line_number_width(old_lines, new_lines)

    text = Text()

    if old == new:
        text.append("No changes.\n", style="dim")
        return text

    old_no_newline = bool(old) and not old.endswith("\n")
    new_no_newline = bool(new) and not new.endswith("\n")

    matcher = SequenceMatcher(None, old_lines, new_lines)
    groups = list(matcher.get_grouped_opcodes(context))

    for group_index, group in enumerate(groups):
        if group_index:
            text.append("  ...\n", style="dim")

        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for offset, line in enumerate(old_lines[i1:i2]):
                    number = i1 + offset + 1
                    _append_line(text, number, line, width, None)
                    if (
                        number == len(old_lines) == len(new_lines)
                        and old_no_newline
                        and new_no_newline
                    ):
                        _append_no_newline(text, number, width, GUTTER_STYLE)
                continue

            if tag == "delete":
                for offset, line in enumerate(old_lines[i1:i2]):
                    number = i1 + offset + 1
                    _append_line(text, number, line, width, REMOVED_STYLE)
                    if old_no_newline and number == len(old_lines):
                        _append_no_newline(text, number, width, REMOVED_STYLE)
                continue

            if tag == "insert":
                for offset, line in enumerate(new_lines[j1:j2]):
                    number = j1 + offset + 1
                    _append_line(text, number, line, width, ADDED_STYLE)
                    if new_no_newline and number == len(new_lines):
                        _append_no_newline(text, number, width, ADDED_STYLE)
                continue

            for offset, line in enumerate(old_lines[i1:i2]):
                number = i1 + offset + 1
                _append_line(text, number, line, width, REMOVED_STYLE)
                if old_no_newline and number == len(old_lines):
                    _append_no_newline(text, number, width, REMOVED_STYLE)

            for offset, line in enumerate(new_lines[j1:j2]):
                number = j1 + offset + 1
                _append_line(text, number, line, width, ADDED_STYLE)
                if new_no_newline and number == len(new_lines):
                    _append_no_newline(text, number, width, ADDED_STYLE)

    return text


def render_contents(path: str, content: str, start_line: int = 1) -> Text:
    lines = content.splitlines()
    width = max(2, len(str(start_line + len(lines))))

    text = Text()

    if not content:
        text.append("(empty file)\n", style="dim")
        return text

    for offset, line in enumerate(lines):
        _append_line(text, start_line + offset, line, width, None)

    if not content.endswith("\n"):
        _append_no_newline(
            text,
            start_line + len(lines) - 1,
            width,
            GUTTER_STYLE,
        )

    return text
