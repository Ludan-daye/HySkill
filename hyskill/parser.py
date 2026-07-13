"""Split SKILL.md text into meta / body / code fields."""

import re

_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_code(content: str) -> tuple[str, str]:
    """Return (body_without_fences, concatenated_fenced_code)."""
    code_blocks = _FENCE.findall(content or "")
    body = _FENCE.sub(" ", content or "")
    return body.strip(), "\n".join(b.strip() for b in code_blocks).strip()


def _meta(name: str, description: str) -> str:
    parts = [p.strip() for p in (name, description) if p and p.strip()]
    if len(parts) == 2:
        return parts[0].rstrip(".") + ". " + parts[1]
    return parts[0] if parts else ""


def parse_fields(name: str, description: str, content: str) -> dict:
    """Fields for a corpus skill entry ({skill_id, name, description, content})."""
    body, code = _split_code(content)
    return {"meta": _meta(name, description), "body": body, "code": code}


def parse_generated(md: str) -> dict:
    """Fields for a generated hypothetical SKILL.md (frontmatter optional)."""
    md = md or ""
    name, description = "", ""
    m = _FRONTMATTER.match(md)
    rest = md
    if m:
        rest = md[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                if key == "name":
                    name = val.strip()
                elif key == "description":
                    description = val.strip()
    body, code = _split_code(rest)
    if not name:
        lines = [l for l in rest.splitlines() if l.strip()]
        name = lines[0].strip().lstrip("# ") if lines else ""
        body, code = _split_code("\n".join(lines[1:]))
    return {"meta": _meta(name, description), "body": body, "code": code}
