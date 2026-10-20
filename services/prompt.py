from pathlib import Path
from services.skills import SYSTEM_SKILLS

WORKDIR = Path.cwd()

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "workspace": f"Working directory: {WORKDIR}",
    "custom_instructions": ["Before starting, create an actionable task plan and keep it updated as work progresses, ensuring statuses accurately reflect actual progress.",
                            "When calling a tool, pass only the arguments specified in the tool call. Never add, modify, or invent extra arguments yourself.",],
}

def assemble_system_prompt() -> str:
    sections = []

    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["workspace"])
    sections.extend(PROMPT_SECTIONS["custom_instructions"])
    sections.append(SYSTEM_SKILLS)

    return "\n\n".join(sections)