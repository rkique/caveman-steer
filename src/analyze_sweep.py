"""Aggregate the dev sweep into a token-vs-correctness table/plot across every (layer, coeff) grid point."""
import json
from collections import defaultdict

import matplotlib.pyplot as plt

from data_utils import RESULTS_DIR, read_jsonl

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
LAYER_MARKERS = {7: "o", 14: "s", 18: "^", 22: "D"}
# Validated all-pairs safe for a light-mode static scatter (node scripts/validate_palette.js, dataviz skill):
# CVD worst-pair ΔE 9.2, normal-vision worst-pair ΔE 16.3. Aqua's low surface contrast is mitigated by the
# direct labels already on every point (the skill's "relief rule").
LAYER_COLORS = {7: "#2a78d6", 14: "#eb6834", 18: "#1baf7a", 22: "#4a3aa7"}


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["layer"], r["coeff"])].append(r)

    summary = []
    for (layer, coeff), grp in groups.items():
        n = len(grp)
        summary.append(
            {
                "layer": layer,
                "coeff": coeff,
                "avg_tokens": sum(g["tokens"] for g in grp) / n,
                "full_correct_rate": sum(1 for g in grp if g["correct"] == 2) / n,
                "any_correct_rate": sum(1 for g in grp if g["correct"] >= 1) / n,
                "coherent_rate": sum(1 for g in grp if g["coherent"]) / n,
            }
        )
    return sorted(summary, key=lambda s: s["avg_tokens"])


def plot(summary: list[dict], out_path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for s in summary:
        ax.scatter(
            s["avg_tokens"],
            s["full_correct_rate"] * 100,
            marker=LAYER_MARKERS[s["layer"]],
            s=90,
            color=LAYER_COLORS[s["layer"]],
            zorder=3,
        )
        ax.annotate(
            f"c{s['coeff']}",
            (s["avg_tokens"], s["full_correct_rate"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=MUTED,
        )

    handles = [
        plt.Line2D([0], [0], marker=LAYER_MARKERS[l], color=LAYER_COLORS[l], linestyle="", markersize=7)
        for l in LAYER_MARKERS
    ]
    ax.legend(handles, [f"layer {l}" for l in LAYER_MARKERS], frameon=False, labelcolor=INK, loc="lower right")

    ax.set_xlabel("Average response tokens (dev)", color=INK)
    ax.set_ylabel("Fully-correct rate (%)", color=INK)
    ax.set_title("Prompt+Steer grid sweep: token count vs. correctness (dev)", color=INK, fontsize=11)
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote plot to {out_path}")


def main() -> None:
    rows = read_jsonl(RESULTS_DIR / "judged_sweep_dev.jsonl")
    summary = summarize(rows)

    print(f"{'layer':<7}{'coeff':<7}{'avg_tokens':>12}{'full_correct':>14}{'any_correct':>13}{'coherent':>11}")
    for s in summary:
        print(
            f"{s['layer']:<7}{s['coeff']:<7}{s['avg_tokens']:>12.1f}{s['full_correct_rate'] * 100:>13.1f}%"
            f"{s['any_correct_rate'] * 100:>12.1f}%{s['coherent_rate'] * 100:>10.1f}%"
        )

    with (RESULTS_DIR / "summary_sweep_dev.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot(summary, RESULTS_DIR / "summary_sweep_plot_dev.png")


if __name__ == "__main__":
    main()
