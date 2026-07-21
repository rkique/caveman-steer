"""Small shared helpers: JSONL IO and a degenerate-output heuristic."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def is_degenerate(text: str) -> bool:
    """Flags empty or heavily repetitive output, a known failure mode of over-strong steering."""
    words = text.split()
    if len(words) < 3:
        return True
    if len(words) >= 10 and len(set(words)) / len(words) < 0.3:
        return True
    return False
