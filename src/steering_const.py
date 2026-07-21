"""Stolfo-style constant activation steering, calibrated to answer: does steering further reduce
tokens ON TOP OF the caveman-style terse instruction, beyond what the instruction alone achieves?"""
import json

import torch

from data_utils import DATA_DIR, RESULTS_DIR, is_degenerate, read_jsonl
from model_common import (
    build_prompt,
    generate_response,
    hidden_at_last_token_all_layers,
    layer_indices_from_fractions,
    load_model,
    make_const_hook,
    steering_hook,
    token_count,
)

COEFF_GRID = [6, 12, 20, 28, 36]
MAX_DEGENERATE_RATE = 0.15


def compute_directions(model, tokenizer, train_rows: list[dict], layer_indices: list[int]) -> dict[int, torch.Tensor]:
    diffs = {layer_idx: [] for layer_idx in layer_indices}
    for row in train_rows:
        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
        base_h = hidden_at_last_token_all_layers(model, tokenizer, base_prompt, layer_indices)
        terse_h = hidden_at_last_token_all_layers(model, tokenizer, terse_prompt, layer_indices)
        for layer_idx in layer_indices:
            diffs[layer_idx].append(terse_h[layer_idx] - base_h[layer_idx])
    directions = {}
    for layer_idx, vecs in diffs.items():
        v = torch.stack(vecs).mean(dim=0)
        directions[layer_idx] = v / v.norm()
    return directions


def prompt_dev_tokens(tokenizer, model, dev_rows: list[dict]) -> float:
    """Reference point: how many tokens the caveman-style instruction alone gets to on dev, no steering."""
    counts = []
    for row in dev_rows:
        prompt = build_prompt(tokenizer, row["code"], terse=True)
        response = generate_response(model, tokenizer, prompt)
        counts.append(token_count(tokenizer, response))
    return sum(counts) / len(counts)


def calibrate(model, tokenizer, dev_rows: list[dict], directions: dict[int, torch.Tensor]):
    """Sweeps (layer, coeff) with steering applied ON TOP OF the terse instruction. Picks the config that
    pushes the token count lowest while keeping the degenerate-output rate bounded."""
    best = None
    for layer_idx, direction in directions.items():
        for coeff in COEFF_GRID:
            counts = []
            degenerate = 0
            for row in dev_rows:
                terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
                with steering_hook(model, layer_idx, make_const_hook(direction, coeff)):
                    response = generate_response(model, tokenizer, terse_prompt)
                if is_degenerate(response):
                    degenerate += 1
                counts.append(token_count(tokenizer, response))
            degenerate_rate = degenerate / len(dev_rows)
            avg_tokens = sum(counts) / len(counts)
            print(f"layer={layer_idx} coeff={coeff} prompt+steer avg_tokens={avg_tokens:.1f} degenerate_rate={degenerate_rate:.2f}")
            if degenerate_rate > MAX_DEGENERATE_RATE:
                continue
            if best is None or avg_tokens < best["avg_tokens"]:
                best = {"layer": layer_idx, "coeff": coeff, "avg_tokens": avg_tokens}
    if best is None:
        raise RuntimeError("no (layer, coeff) config avoided degenerate outputs on dev")
    return best


def main() -> None:
    model, tokenizer = load_model()
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")
    layer_indices = layer_indices_from_fractions(model)

    print(f"computing directions for layers {layer_indices} from {len(train_rows)} train pairs")
    directions = compute_directions(model, tokenizer, train_rows, layer_indices)

    print("measuring Prompt-alone (no steering) token count on dev, for reference")
    prompt_tokens = prompt_dev_tokens(tokenizer, model, dev_rows)
    print(f"prompt_dev_tokens={prompt_tokens:.1f}")

    best = calibrate(model, tokenizer, dev_rows, directions)
    print("best config:", best, f"(vs prompt_dev_tokens={prompt_tokens:.1f})")

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(directions, RESULTS_DIR / "const_steer_directions.pt")
    with (RESULTS_DIR / "const_steer_config.json").open("w") as f:
        json.dump({**best, "prompt_dev_tokens": prompt_tokens}, f, indent=2)


if __name__ == "__main__":
    main()
