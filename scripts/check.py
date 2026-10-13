import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUIZ_KEYS = {"id", "stage", "question", "options", "correct", "explanation"}


def check_lesson(lesson):
    errors = []
    for required in ("code/main.py", "docs/en.md", "quiz.json"):
        if not (lesson / required).exists():
            errors.append(f"missing {required}")
    quiz = lesson / "quiz.json"
    if quiz.exists():
        for q in json.loads(quiz.read_text()):
            if set(q) != QUIZ_KEYS or len(q["options"]) != 4 or not 0 <= q["correct"] < 4:
                errors.append(f"bad quiz question {q.get('id')}")
    main = lesson / "code" / "main.py"
    if main.exists():
        run = subprocess.run([sys.executable, str(main)], cwd=ROOT, capture_output=True, text=True, timeout=120,
                             env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"})
        if run.returncode:
            errors.append(f"main.py exited {run.returncode}: {run.stderr.strip().splitlines()[-1:]}")
    return errors


def main():
    lessons = sorted(p for p in (ROOT / "phases").glob("*/*") if p.is_dir())
    failed = 0
    for lesson in lessons:
        errors = check_lesson(lesson)
        failed += bool(errors)
        print(f"{'FAIL' if errors else 'ok  '}  {lesson.relative_to(ROOT)}")
        for error in errors:
            print(f"      {error}")
    print(f"\n{len(lessons) - failed}/{len(lessons)} lessons pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
