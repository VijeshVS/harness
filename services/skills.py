from pathlib import Path
import yaml

WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"

SKILL_REGISTRY: dict[str, dict] = {}

def _parse_skill(raw: str) -> dict:
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        return {}
    return data

def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta = _parse_skill(raw)
            if not meta:
                continue
            name = meta.get("name", d.name)
            desc = meta.get("description", "")
            SKILL_REGISTRY[name] = {
                "name": name,
                "description": desc,
                "content": meta.get("instructions", raw),
            }

_scan_skills()

def list_skills() -> str:
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def build_system() -> str:
    catalog = list_skills()
    return (
        f"You are a coding agent at {WORKDIR}. "
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

SYSTEM_SKILLS = build_system()
