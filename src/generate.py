"""Generate all conditions (Base, Prompt, Steer, Prompt+Steer) over a data split and save raw outputs."""
import argparse
import json

import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl, write_jsonl
from model_common import (
    build_prompt,
    generate_response,
    make_const_hook,
    load_model,
    steering_hook,
    token_count,
)


def load_steering_config(device: str):
    with (RESULTS_DIR / "const_steer_config.json").open() as f:
        const_config = json.load(f)
    layer_idx = const_config["layer"]
    coeff = const_config["coeff"]
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    direction = directions[layer_idx]
    return layer_idx, direction, coeff


def main(split: str) -> None:
    device = "cuda"
    model, tokenizer = load_model(device)
    rows = read_jsonl(DATA_DIR / f"{split}.jsonl")
    layer_idx, direction, coeff = load_steering_config(device)

    out_rows = []
    for i, row in enumerate(rows):
        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)

        base_resp = generate_response(model, tokenizer, base_prompt)
        prompt_resp = generate_response(model, tokenizer, terse_prompt)
        with steering_hook(model, layer_idx, make_const_hook(direction, coeff)):
            const_resp = generate_response(model, tokenizer, base_prompt)
        with steering_hook(model, layer_idx, make_const_hook(direction, coeff)):
            prompt_const_resp = generate_response(model, tokenizer, terse_prompt)

        out_rows.append(
            {
                "id": row["id"],
                "code": row["code"],
                "reference_explanation": row["reference_explanation"],
                "base_response": base_resp,
                "prompt_response": prompt_resp,
                "const_response": const_resp,
                "prompt_const_response": prompt_const_resp,
                "base_tokens": token_count(tokenizer, base_resp),
                "prompt_tokens": token_count(tokenizer, prompt_resp),
                "const_tokens": token_count(tokenizer, const_resp),
                "prompt_const_tokens": token_count(tokenizer, prompt_const_resp),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(rows)} done")

    RESULTS_DIR.mkdir(exist_ok=True)
    write_jsonl(RESULTS_DIR / f"generations_{split}.jsonl", out_rows)
    print(f"wrote {len(out_rows)} rows to results/generations_{split}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()
    main(args.split)
