# Lesson Template

Use this template when creating a new lesson. Copy the folder structure and fill in the content.

## Folder Structure

```
NN-lesson-name/
├── code/
│   └── main.py            (from-scratch implementation, runnable from the repo root)
├── docs/
│   └── en.md              (lesson documentation)
└── quiz.json              (pre/post check questions)
```

## Documentation Format (docs/en.md)

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

**Type:** Build | Learn
**Languages:** Python
**Prerequisites:** [List prior lessons needed]
**Time:** ~[estimated time] minutes

## The Problem

[2-3 paragraphs. What can't you do without this? Why should you care?
Make it concrete — show a scenario where not knowing this hurts.]

## The Concept

[Explain with diagrams and intuition. No code yet.
Use ASCII diagrams, tables, or link to visuals in the web app.
Build mental models before implementation.]

## Build It

[Step-by-step implementation from scratch.
Start with the simplest version, then add complexity.
Every code block should be runnable on its own.]

### Step 1: [Name]

[Explanation]

    [code block]

### Step 2: [Name]

[Explanation]

    [code block]

[...continue...]

## Use It

[Now show how frameworks/libraries do the same thing.
Compare your from-scratch version to the library version.
This proves the concept and introduces practical tools.]

## Exercises

1. [Easy — reinforce the core concept]
2. [Medium — apply it to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [term] | [common misconception] | [actual definition] |

## Further Reading

- [Resource 1](url) — [why it's worth reading]
- [Resource 2](url) — [why it's worth reading]
```

## Code File Guidelines

- Code must run without errors
- No comments — code should be self-explanatory
- Use the language that fits best for the topic
- Use only the dependencies in `pyproject.toml` (numpy, pandas, matplotlib, scikit-learn, scipy)
- Start simple, build up complexity
- Every function and class should have a clear purpose

## Quiz Format (quiz.json)

```json
[
  {
    "id": "lesson-pre-1",
    "stage": "pre",
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "correct": 1,
    "explanation": "..."
  }
]
```

Run `uv run python scripts/check.py` before committing: every lesson's code must run and every quiz must validate.
