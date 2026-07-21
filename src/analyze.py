"""Aggregate judged results into a summary table and a token-count-vs-correctness plot. Runs locally."""
import argparse
import json

import matplotlib.pyplot as plt

from data_utils import RESULTS_DIR, read_jsonl

CONDITIONS = ["base", "prompt", "const", "prompt_const"]
LABELS = {"base": "Base", "prompt": "Prompt", "const": "Const-steer", "prompt_const": "Prompt+Steer"}
MARKERS = {"base": "o", "prompt": "s", "const": "^", "prompt_const": "D"}
INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    n = len(rows)
    for cond in CONDITIONS:
        avg_tokens = sum(r[f"{cond}_tokens"] for r in rows) / n
        full_correct = sum(1 for r in rows if r[f"{cond}_correct"] == 2) / n
        any_correct = sum(1 for r in rows if r[f"{cond}_correct"] >= 1) / n
        coherent = sum(1 for r in rows if r[f"{cond}_coherent"]) / n
        summary[cond] = {
            "avg_tokens": avg_tokens,
            "full_correct_rate": full_correct,
            "any_correct_rate": any_correct,
            "coherent_rate": coherent,
        }
    return summary


def plot(summary: dict[str, dict], out_path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for cond in CONDITIONS:
        s = summary[cond]
        ax.scatter(
            s["avg_tokens"],
            s["full_correct_rate"] * 100,
            marker=MARKERS[cond],
            s=110,
            color=INK,
            zorder=3,
        )
        ax.annotate(
            LABELS[cond],
            (s["avg_tokens"], s["full_correct_rate"] * 100),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
            color=INK,
        )

    ax.set_xlabel("Average response tokens (lower = cheaper)", color=INK)
    ax.set_ylabel("Fully-correct rate (%)", color=INK)
    ax.set_title("Terseness vs. correctness: Prompt vs. Const-steer vs. PSR", color=INK, fontsize=11)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote plot to {out_path}")


def main(split: str) -> None:
    rows = read_jsonl(RESULTS_DIR / f"judged_{split}.jsonl")
    summary = summarize(rows)

    print(f"{'condition':<14}{'avg_tokens':>12}{'full_correct':>14}{'any_correct':>13}{'coherent':>11}")
    for cond in CONDITIONS:
        s = summary[cond]
        print(
            f"{LABELS[cond]:<14}{s['avg_tokens']:>12.1f}{s['full_correct_rate'] * 100:>13.1f}%"
            f"{s['any_correct_rate'] * 100:>12.1f}%{s['coherent_rate'] * 100:>10.1f}%"
        )

    with (RESULTS_DIR / f"summary_{split}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot(summary, RESULTS_DIR / f"summary_plot_{split}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
