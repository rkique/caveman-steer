"""Generate Prompt+Steer at every (layer, coeff) in the calibration grid, on dev, saving every response.
steering_const.py's calibration only checked for degenerate/repetitive output, not actual correctness —
this sweep exists to judge correctness at each grid point and find a real token-vs-correctness operating
point, rather than trusting the single most-aggressive-non-degenerate config it picked."""
import torch

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl, write_jsonl
from model_common import build_prompt, generate_response, load_model, make_const_hook, steering_hook, token_count
from steering_const import COEFF_GRID


def main() -> None:
    model, tokenizer = load_model()
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")

    out_rows = []
    for layer_idx, direction in directions.items():
        for coeff in COEFF_GRID:
            for row in dev_rows:
                terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
                with steering_hook(model, layer_idx, make_const_hook(direction, coeff)):
                    response = generate_response(model, tokenizer, terse_prompt)
                out_rows.append(
                    {
                        "id": row["id"],
                        "code": row["code"],
                        "reference_explanation": row["reference_explanation"],
                        "layer": layer_idx,
                        "coeff": coeff,
                        "response": response,
                        "tokens": token_count(tokenizer, response),
                    }
                )
            print(f"layer={layer_idx} coeff={coeff} done ({len(dev_rows)} dev rows)")

    RESULTS_DIR.mkdir(exist_ok=True)
    write_jsonl(RESULTS_DIR / "sweep_dev.jsonl", out_rows)
    print(f"wrote {len(out_rows)} rows to results/sweep_dev.jsonl")


if __name__ == "__main__":
    main()
