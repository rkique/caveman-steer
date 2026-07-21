"""Judge every response in results/sweep_dev.jsonl for correctness/coherence. Runs locally, no GPU."""
from data_utils import RESULTS_DIR, read_jsonl, write_jsonl
from judge import KEY_PATH, judge_one
from openai import OpenAI


def main() -> None:
    api_key = KEY_PATH.read_text().strip()
    client = OpenAI(api_key=api_key)
    rows = read_jsonl(RESULTS_DIR / "sweep_dev.jsonl")

    out_rows = []
    for i, row in enumerate(rows):
        score = judge_one(client, row["code"], row["reference_explanation"], row["response"])
        out_rows.append(
            {
                "id": row["id"],
                "layer": row["layer"],
                "coeff": row["coeff"],
                "tokens": row["tokens"],
                "correct": score["correct"],
                "coherent": score["coherent"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} judged")

    write_jsonl(RESULTS_DIR / "judged_sweep_dev.jsonl", out_rows)
    print(f"wrote {len(out_rows)} rows to results/judged_sweep_dev.jsonl")


if __name__ == "__main__":
    main()
