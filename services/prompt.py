from pathlib import Path
from services.skills import SYSTEM_SKILLS

WORKDIR = Path.cwd()

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "workspace": f"Working directory: {WORKDIR}",
    "custom_instructions": ["Before starting, create an actionable task plan and keep it updated as work progresses, ensuring statuses accurately reflect actual progress."],
}

def assemble_system_prompt() -> str:
    sections = []

    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["workspace"])
    sections.extend(PROMPT_SECTIONS["custom_instructions"])
    sections.append(SYSTEM_SKILLS)

    return "\n\n".join(sections)