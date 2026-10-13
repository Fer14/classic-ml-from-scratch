from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BADGE = "https://img.shields.io/badge/{}-{}-6b4a3a?style=flat-square&labelColor=fafaf5"


def phases():
    for phase in sorted((ROOT / "phases").iterdir()):
        if not phase.is_dir():
            continue
        lines = (phase / "README.md").read_text().splitlines()
        title = lines[0].removeprefix("# ").split(": ", 1)[1]
        motto = next(l for l in lines if l.startswith("> ")).removeprefix("> ")
        lessons = []
        for lesson in sorted(p for p in phase.iterdir() if p.is_dir()):
            lessons.append((lesson, (lesson / "docs" / "en.md").read_text().splitlines()[0].removeprefix("# ")))
        yield phase, title, motto, lessons


def main():
    data = list(phases())
    total = sum(len(l) for *_, l in data)
    table = "\n".join(f"| {p.name[:2]} | [{t}](phases/{p.name}/) | {len(l)} | {m} |" for p, t, m, l in data)
    contents = []
    for phase, title, motto, lessons in data:
        rows = "\n".join(f"| {l.name[:2]} | [{name}](phases/{phase.name}/{l.name}/) |" for l, name in lessons)
        contents.append(f"""<a id="phase-{phase.name[:2]}"></a>
<details>
<summary><strong>Phase {phase.name[:2]} · {title}</strong> · {len(lessons)} lessons</summary>

> {motto}

| # | Lesson |
|---|---|
{rows}

</details>""")
    contents = "\n\n".join(contents)
    readme = f"""<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/header-dark.svg">
    <img src="assets/readme/header-light.svg" alt="Classic ML from Scratch" width="840">
  </picture>
</p>

Build every classic machine learning algorithm yourself. Start from plain Python and the math underneath, implement each model by hand in NumPy, then check your answer against scikit-learn.

> {total} lessons. {len(data)} phases. Python.

<p align="center">
  <a href="LICENSE"><img src="{BADGE.format('license', 'MIT')}" alt="MIT License"></a>
  <a href="#contents"><img src="{BADGE.format('lessons', total)}" alt="{total} lessons"></a>
  <a href="#contents"><img src="{BADGE.format('phases', len(data))}" alt="{len(data)} phases"></a>
</p>

## Phases

| # | Phase | Lessons | What it is about |
|---|---|---|---|
{table}

## Every lesson has the same shape

```
NN-lesson-name/
├── code/main.py   from-scratch implementation, runnable from the repo root
├── docs/en.md     The Problem · The Concept · Build It · Use It · Exercises · Key Terms
└── quiz.json      questions to check yourself before and after
```

## Local setup

```bash
git clone https://github.com/Fer14/classic-ml-from-scratch.git
cd classic-ml-from-scratch
uv sync
uv run python phases/00-python-foundations/01-dev-environment/code/main.py
```

No uv? `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` works too.

Check that every lesson runs: `uv run python scripts/check.py`

<a id="contents"></a>

## Contents

{contents}

## Contributing

New lessons follow [LESSON_TEMPLATE.md](LESSON_TEMPLATE.md). Run `uv run python scripts/check.py` before opening a pull request.

## License

[MIT](LICENSE). Structure and style inspired by [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch).
"""
    (ROOT / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
