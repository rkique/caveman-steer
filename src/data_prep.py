"""Build train/dev/test splits of (code, reference explanation) pairs from CodeXGLUE code-to-text (Python)."""
import ast
import json
import random
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0
MIN_CODE_LINES = 3
MAX_CODE_LINES = 30
MIN_DOCSTRING_WORDS = 5
N_TRAIN = 180
N_DEV = 50
N_TEST = 180


def is_usable(example: dict) -> bool:
    code_lines = example["code"].strip().count("\n") + 1
    docstring_words = len(example["docstring"].split())
    return (
        MIN_CODE_LINES <= code_lines <= MAX_CODE_LINES
        and docstring_words >= MIN_DOCSTRING_WORDS
        and "TODO" not in example["docstring"]
    )


def strip_docstring(code: str) -> str | None:
    """Remove the docstring (and, via re-emission, comments) so the snippet doesn't leak the reference explanation."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    fn = tree.body[0]
    first_stmt = fn.body[0] if fn.body else None
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        fn.body = fn.body[1:]
    if not fn.body:
        return None
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def main() -> None:
    ds = load_dataset("code_x_glue_ct_code_to_text", "python", split="train")
    ds = ds.filter(is_usable)

    rows: list[dict] = []
    for ex in ds:
        cleaned_code = strip_docstring(ex["code"])
        if cleaned_code is None:
            continue
        code_lines = cleaned_code.count("\n") + 1
        if not (MIN_CODE_LINES <= code_lines <= MAX_CODE_LINES):
            continue
        # cap examples per repo so no single project dominates the splits
        repo_count = sum(1 for r in rows if r["repo"] == ex["repo"])
        if repo_count >= 3:
            continue
        rows.append(
            {
                "id": ex["id"],
                "repo": ex["repo"],
                "func_name": ex["func_name"],
                "code": cleaned_code,
                "reference_explanation": ex["docstring"].strip(),
            }
        )

    rng = random.Random(SEED)
    rng.shuffle(rows)

    n_needed = N_TRAIN + N_DEV + N_TEST
    if len(rows) < n_needed:
        raise ValueError(f"only {len(rows)} usable examples after filtering, need {n_needed}")
    rows = rows[:n_needed]

    splits = {
        "train": rows[:N_TRAIN],
        "dev": rows[N_TRAIN : N_TRAIN + N_DEV],
        "test": rows[N_TRAIN + N_DEV : N_TRAIN + N_DEV + N_TEST],
    }

    DATA_DIR.mkdir(exist_ok=True)
    for name, split_rows in splits.items():
        out_path = DATA_DIR / f"{name}.jsonl"
        with out_path.open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(split_rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
