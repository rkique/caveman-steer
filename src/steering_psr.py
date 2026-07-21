"""Train the S-PSR probe: a token-specific steering coefficient matched (via MSE) to the effect of prompting."""
import json

import torch
import torch.nn.functional as F

from data_utils import DATA_DIR, RESULTS_DIR, read_jsonl
from model_common import PSRProbe, build_prompt, generate_response, load_model

N_EPOCHS = 200
LR = 1e-3


@torch.no_grad()
def collect_teacher_forced_pairs(model, tokenizer, rows: list[dict], layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row: generate the terse (Prompt-condition) response, then teacher-force it through both the
    base and terse prompt contexts, collecting layer_idx activations at the response-token positions.
    These pairs (base activation -> instructed activation) are what the probe learns to bridge."""
    xs, ys = [], []
    for row in rows:
        base_prompt = build_prompt(tokenizer, row["code"], terse=False)
        terse_prompt = build_prompt(tokenizer, row["code"], terse=True)
        teacher_response = generate_response(model, tokenizer, terse_prompt)
        resp_ids = tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        if resp_ids.shape[1] == 0:
            continue

        base_ids = tokenizer(base_prompt, return_tensors="pt")["input_ids"].to(model.device)
        instr_ids = tokenizer(terse_prompt, return_tensors="pt")["input_ids"].to(model.device)
        full_base = torch.cat([base_ids, resp_ids], dim=1)
        full_instr = torch.cat([instr_ids, resp_ids], dim=1)

        out_base = model(input_ids=full_base, output_hidden_states=True)
        out_instr = model(input_ids=full_instr, output_hidden_states=True)
        n_resp = resp_ids.shape[1]
        h_base = out_base.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu()
        h_instr = out_instr.hidden_states[layer_idx + 1][0, -n_resp:, :].float().cpu()
        xs.append(h_base)
        ys.append(h_instr)

    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def train_probe(x: torch.Tensor, y: torch.Tensor, direction: torch.Tensor, device: str) -> PSRProbe:
    hidden_size = x.shape[1]
    probe = PSRProbe(hidden_size).to(device)
    x, y, direction = x.to(device), y.to(device), direction.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR)
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        lam = probe(x)
        pred = x + lam * direction
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch={epoch} train_mse={loss.item():.4f}")
    return probe


@torch.no_grad()
def eval_probe(probe: PSRProbe, x: torch.Tensor, y: torch.Tensor, direction: torch.Tensor, device: str) -> float:
    x, y, direction = x.to(device), y.to(device), direction.to(device)
    lam = probe(x)
    pred = x + lam * direction
    return F.mse_loss(pred, y).item()


def main() -> None:
    device = "cuda"
    model, tokenizer = load_model(device)
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    dev_rows = read_jsonl(DATA_DIR / "dev.jsonl")

    with (RESULTS_DIR / "const_steer_config.json").open() as f:
        const_config = json.load(f)
    layer_idx = const_config["layer"]
    directions = torch.load(RESULTS_DIR / "const_steer_directions.pt")
    direction = directions[layer_idx]

    print(f"collecting teacher-forced activation pairs at layer {layer_idx} from {len(train_rows)} train rows")
    x_train, y_train = collect_teacher_forced_pairs(model, tokenizer, train_rows, layer_idx)
    print(f"collected {x_train.shape[0]} token-level training pairs")

    print(f"collecting dev pairs from {len(dev_rows)} rows")
    x_dev, y_dev = collect_teacher_forced_pairs(model, tokenizer, dev_rows, layer_idx)

    baseline_mse = F.mse_loss(x_dev, y_dev).item()
    print(f"dev baseline MSE (no intervention): {baseline_mse:.4f}")

    probe = train_probe(x_train, y_train, direction, device)
    dev_mse = eval_probe(probe, x_dev, y_dev, direction, device)
    print(f"dev MSE after PSR probe: {dev_mse:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save({"probe_state": probe.state_dict(), "layer": layer_idx, "hidden_size": x_train.shape[1]}, RESULTS_DIR / "psr_probe.pt")
    with (RESULTS_DIR / "psr_train_log.json").open("w") as f:
        json.dump({"layer": layer_idx, "baseline_dev_mse": baseline_mse, "final_dev_mse": dev_mse, "n_train_pairs": x_train.shape[0]}, f, indent=2)


if __name__ == "__main__":
    main()
