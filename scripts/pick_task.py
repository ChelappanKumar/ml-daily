#!/usr/bin/env python3
"""Pick today's task from curriculum.yaml, scaffold a dated folder, append progress.

Rotates categories: pipelines -> agents -> notes -> pipelines ...
Avoids repeating a topic seen in the last 30 days (best-effort via progress.md).

Exits 0 with the scaffolded folder path on stdout. Exits non-zero on error.
"""
from __future__ import annotations

import datetime as dt
import os
import random
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "PyYAML is required. Install with: pip install pyyaml\n"
        "(setup.sh handles this in a venv.)\n"
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent
CURRICULUM = REPO_ROOT / "curriculum.yaml"
PROGRESS = REPO_ROOT / "progress.md"
ROTATION = ["pipelines", "agents", "notes"]


def load_curriculum() -> dict:
    with CURRICULUM.open() as f:
        return yaml.safe_load(f)


def recent_slugs(days: int = 30) -> set[str]:
    if not PROGRESS.exists():
        return set()
    cutoff = dt.date.today() - dt.timedelta(days=days)
    slugs: set[str] = set()
    row_re = re.compile(r"^\|\s*(\d{4}-\d{2}-\d{2})\s*\|[^|]*\|[^|]*\|\s*([^|/]+)/([^|]+?)\s*\|")
    for line in PROGRESS.read_text().splitlines():
        m = row_re.match(line)
        if not m:
            continue
        try:
            d = dt.date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if d < cutoff:
            continue
        folder = m.group(3).strip()
        slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", folder)
        slugs.add(slug)
    return slugs


def pick_category(today: dt.date) -> str:
    # Deterministic rotation based on day-of-year so the schedule is predictable.
    return ROTATION[today.toordinal() % len(ROTATION)]


def pick_task(curriculum: dict, category: str, avoid: set[str]) -> dict:
    pool = [t for t in curriculum[category] if t["slug"] not in avoid]
    if not pool:
        pool = curriculum[category]  # all seen recently — accept a repeat
    return random.choice(pool)


def starter_python(task: dict) -> str:
    refs = task.get("refs") or []
    refs_block = "\n".join(f"#   - {r}" for r in refs) if refs else "#   (none)"
    return f'''"""{task["title"]}

Goal:
    {task["goal"]}

References:
{refs_block}
"""
from __future__ import annotations


def main() -> None:
    # TODO: implement the goal above.
    # Suggested structure:
    #   1) Load / generate a small dataset.
    #   2) Build the component (pipeline / agent / experiment).
    #   3) Run it and print a clear result.
    #   4) Write 2-3 lines in NOTES.md below on what you learned.
    raise NotImplementedError("fill me in")


if __name__ == "__main__":
    main()
'''


def starter_markdown(task: dict) -> str:
    refs = task.get("refs") or []
    refs_block = "\n".join(f"- {r}" for r in refs) if refs else "- (add references as you read)"
    return f"""# {task["title"]}

**Goal:** {task["goal"]}

## Summary

_TODO: 1-paragraph plain-English summary._

## Key ideas

- TODO

## Worked example / intuition

_TODO_

## Open questions

- TODO

## References

{refs_block}
"""


def notes_md(task: dict) -> str:
    return f"""# Learning log — {task["title"]}

What worked:
- TODO

What surprised me:
- TODO

What I'd try next:
- TODO
"""


def append_progress(today: dt.date, category: str, task: dict, folder: Path) -> None:
    rel = folder.relative_to(REPO_ROOT)
    row = f"| {today.isoformat()} | {category} | {task['title']} | {rel} |\n"
    with PROGRESS.open("a") as f:
        f.write(row)


def main() -> int:
    curriculum = load_curriculum()
    today = dt.date.today()
    category = pick_category(today)
    task = pick_task(curriculum, category, recent_slugs())
    folder_name = f"{today.isoformat()}-{task['slug']}"
    folder = REPO_ROOT / category / folder_name

    if folder.exists():
        # Already scaffolded today (idempotent).
        print(str(folder))
        return 0

    folder.mkdir(parents=True, exist_ok=False)

    if category == "notes":
        (folder / "README.md").write_text(starter_markdown(task))
    else:
        (folder / "starter.py").write_text(starter_python(task))
        (folder / "NOTES.md").write_text(notes_md(task))

    append_progress(today, category, task, folder)
    print(str(folder))
    return 0


if __name__ == "__main__":
    sys.exit(main())
