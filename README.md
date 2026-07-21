# Does activation steering help on top of an already-terse prompt?

Tests whether constant activation steering (Stolfo et al., arXiv 2410.12877 — a diff-in-means
vector added at inference) can push `Qwen2.5-Coder-7B-Instruct`'s code explanations *shorter than
prompting alone already gets them*, rather than treating steering as a standalone replacement for
prompting. Four conditions per test example:

1. **Base** — neutral prompt, no instruction, no steering.
2. **Prompt** — caveman's actual "full" mode instruction, word for word from
   [caveman](https://github.com/juliusbrussee/caveman)'s `skills/caveman/SKILL.md` (opening line,
   Persistence, Rules, no-self-reference paragraph, Pattern line — see `model_common.CAVEMAN_SUFFIX`).
   Auto-Clarity, Boundaries, and the language-preservation paragraph are dropped: none of those
   scenarios (destructive ops, writing commits/PRs, non-English input) exist in this task.
3. **Const-steer** — the diff-in-means vector added at inference, Base prompt (no instruction text).
4. **Prompt+Steer** — the same vector added at inference, *on top of* the Prompt condition — this
   is the primary comparison: does steering add anything once the instruction is already present?

Data: CodeXGLUE code-to-text (Python), docstrings stripped from the code via AST before use as
reference explanations, to avoid leaking the answer into the prompt (`src/data_prep.py`).

## Result

Test set (180 held-out examples), judged by `gpt-4o-mini` for correctness against the original
docstring, blinded to condition:

| Condition | Avg tokens | Fully correct |
|---|---|---|
| Base | 150.0 (censored — see below) | 93.3% |
| Prompt | 56.8 | 91.7% |
| Const-steer alone | 149.9 | 91.7% |
| **Prompt+Steer** | **49.3** | **92.2%** |

**Steering on top of the prompt gets a further ~13% token reduction beyond prompting alone, at no
correctness cost** (92.2% vs 91.7%, within noise). Const-steer alone is a non-result at this
coefficient (149.9, indistinguishable from Base) — the coefficient was calibrated specifically for
the combined regime, not as a standalone replacement for the instruction; this matches the general
pattern (also seen at higher coefficients during calibration) that steering alone is substantially
weaker than prompting.

Base's 150.0 average is right-censored: 95.6% of Base responses hit `MAX_NEW_TOKENS` without the
model choosing to stop, so it's a floor on "at least this many tokens," not a real measurement of
default verbosity — the Prompt vs. Prompt+Steer comparison (both close to fully natural stopping)
is the one to trust.

### Finding the operating point mattered

The first calibration pass picked its config by sweeping (layer, coefficient) on dev and choosing
whichever was most aggressive (lowest tokens) without producing obviously degenerate/repetitive
output — no correctness signal in that loop. That picked layer 14, coeff 36, which looked fine by
the degenerate check but collapsed dev full-correct rate to 40% (e.g. `_descriptor_names`'s Django
descriptor-filtering detail dropped entirely: "List self's descriptors."). Judging the *entire*
grid (`src/sweep_dev.py` + `src/judge_sweep.py` + `src/analyze_sweep.py`, 20 configs × 50 dev
examples) showed why: correctness holds at 86-96% for coefficients 6-12 at layers 14/18, then falls
off a cliff past ~20. Layer 14, coeff 6 (96% dev full-correct, already beating Prompt-alone's dev
token count) was picked from that curve instead of trusting the degenerate-only auto-pick. Full
frontier: `results/summary_sweep_dev.json`, `results/summary_sweep_plot_dev.png`.

## Scope decisions

- PSR (arXiv 2605.03907) is paused for now — `src/steering_psr.py` still exists but isn't run.
  An earlier pass showed the MSE-matching variant underperforming constant steering; revisit later.
- The judge is `gpt-4o-mini` (OpenAI), not Claude — switched to use available OpenAI credits. Reads
  the key from `openai.key` (gitignored, never commit it).
- Task is code **explanation** only, not code generation.

## Running it

Everything in `src/data_prep.py`, `src/judge*.py`, and `src/analyze*.py` runs locally — no GPU
needed. The GPU-bound steps (`steering_const.py`, `sweep_dev.py`, `generate.py`) run on a rented
GPU pod via `infra/run_gpu_pipeline.sh` (or invoked directly, as when re-running just one step).

```
python3 src/data_prep.py                        # local — data/{train,dev,test}.jsonl

# on the GPU pod:
bash infra/setup_runpod.sh
python3 src/steering_const.py                   # -> results/const_steer_{config.json,directions.pt}
python3 src/sweep_dev.py                        # -> results/sweep_dev.jsonl (all grid configs, dev)

# back locally: judge the sweep, inspect the frontier, edit const_steer_config.json's
# layer/coeff to the chosen operating point (see "Finding the operating point mattered" above)
python3 src/judge_sweep.py                      # -> results/judged_sweep_dev.jsonl (needs openai.key)
python3 src/analyze_sweep.py                    # -> results/summary_sweep_dev.json, plot

# back on the GPU pod, with the final chosen config:
python3 src/generate.py --split test            # -> results/generations_test.jsonl

# back locally:
python3 src/judge.py --split test               # -> results/judged_test.jsonl (needs openai.key)
python3 src/analyze.py --split test             # -> results/summary_test.json, summary_plot_test.png
```
