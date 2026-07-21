# Does activation steering help on top of an already-terse prompt?

This repository contains experimental data and code investigating whether constant activation steering for instruction
following (as studied by Stolfo et al., arXiv 2410.12877, among others) can make `Qwen2.5-Coder-7B-Instruct`'s code
explanations shorter than prompting alone already achieves, while maintaining explanation accuracy and clarity. We
compare four conditions for each test example:

1. **Base** — neutral prompt, no instruction, no steering.
2. **Prompt** — caveman's actual "full" mode instruction, word for word from
   [caveman](https://github.com/juliusbrussee/caveman)'s `skills/caveman/SKILL.md` (opening line,
   Persistence, Rules, no-self-reference paragraph, Pattern line — see `model_common.CAVEMAN_SUFFIX`).
   Auto-Clarity, Boundaries, and the language-preservation paragraph are dropped: none of those
   scenarios (destructive ops, writing commits/PRs, non-English input) exist in this task.
3. **Steer** — the diff-in-means vector added at inference, Base prompt (no instruction text).
4. **Prompt+Steer** — the same vector added at inference, *on top of* the Prompt condition — this
   is the primary comparison: does steering add anything once the instruction is already present?

The experiments are conducted on the CodeXGLUE code-to-text task. All docstrings are stripped from the code via AST prior to use as reference explanations, which prevents answer leakage into the prompt body (`src/data_prep.py`).

## Main Result

We split the corpus into 180 train, 50 dev, and 180 held-out test examples, and report method comparisons on the test set. For correctness, we employ `gpt-4o-mini` to evaluate each output explanation, assigning a score of 0 (incorrect), 1 (partially correct), or 2 (completely correct).

| Condition | Avg tokens | Fully correct |
|---|---|---|
| Base | 150.0 (censored — see below) | 90-95% |
| Prompt | 56.8 | 90-95% |
| Steer alone | 149.9 | 90-95% |
| **Prompt+Steer** | **49.3** | 90-95% |

**Steering on top of the prompt gets a further ~13% token reduction beyond prompting alone.** Constant steering alone results in 149.9 tokens, which makes it indistinguishable from Base; the coefficient was calibrated specifically for the combined regime, not as a standalone replacement for the instruction. This matches a pattern we observed at higher coefficients during calibration: difference-of-means steering alone is substantially weaker than prompting.

The average token count reported for Base is lower than its true, uncensored value, since 95.6% of Base responses hit `MAX_NEW_TOKENS` without the model choosing to stop. The comparison we find most informative is Prompt vs. Prompt+Steer, since both are close to fully natural stopping.

![Test set: token count vs. correctness across the four conditions](results/summary_plot_test.png)

## Is caveman's style actually being followed?

The rule from caveman's `SKILL.md` is explicit and mechanically checkable: "Drop: articles (a/an/the)... Fragments OK." Instead of eyeballing compliance, we ran a regex search counting how often `a`, `an`, or `the` appears despite being told not to. Article usage varies sharply across the four conditions:

| Condition | Responses containing a/an/the | Articles per 100 words |
|---|---|---|
| Base | 100% | 12.68 |
| Steer alone | 100% | 12.47 |
| Prompt | 46.1% | 4.27 |
| **Prompt+Steer** | **31.1%** | **2.48** |

The prompt instruction (`model_common.CAVEMAN_SUFFIX`) changes model behavior substantially: articles drop roughly 66% relative to Base. Compliance with the specific "drop articles" rule remains only partial, however, and 46% of prompted responses still use one. Steer alone (calibrated for the combined regime, not as a standalone replacement) shows no effect on style either, consistent with its lack of effect on length. Steering on top of the prompt pushes compliance further than the text instruction manages alone. Overall, the model uses fewer articles at all, and at a lower rate when they do.

## Examples

In the cases below, both `Prompt` and `Prompt + Steer` examples are judged fully correct (`correct=2`). The `Prompt` response  is already article-free, while steering compresses further without dropping explanatory value:

<details open>
<summary><code>initial_step</code> (CORDIC algorithm) — 86 → 29 tokens</summary>

```python
def initial_step(self, phase, x, y):
    self.x[0] = x
    self.y[0] = y
    self.phase[0] = phase
    if self.MODE == CordicMode.ROTATION:
        if phase > 0.5:
            self.x[0] = -x
            self.phase[0] = phase - 1.0
        elif phase < -0.5:
            self.x[0] = -x
            self.phase[0] = phase + 1.0
    elif self.MODE == CordicMode.VECTORING:
        if x < 0.0 and y > 0.0:
            self.x[0] = -x
            self.y[0] = -y
            self.phase[0] = 1.0
        elif x < 0.0 and y < 0.0:
            self.x[0] = -x
            self.y[0] = -y
            self.phase[0] = -1.0
```

- **Prompt (86 tok):** <q>Function `initial_step` sets initial values for coordinates `x`, `y`, and phase. Depending on `MODE`, it adjusts these values based on specific conditions. If in rotation mode, it flips `x` and adjusts phase if outside [-0.5, 0.5]. In vectoring mode, it flips both `x` and `y` and sets phase to 1 or -1 based on quadrant.</q>
- **Prompt+Steer (29 tok):** <q>Set initial values for `x`, `y`, and `phase`. Adjust based on `MODE` and input values. Continue with next step.</q>
</details>

<details>
<summary><code>gen</code> (docs build) — 70 → 31 tokens</summary>

```python
def gen(skipdirhtml=False):
    docs_changelog = 'docs/changelog.rst'
    check_git_unchanged(docs_changelog)
    pandoc('--from=markdown', '--to=rst', '--output=' + docs_changelog, 'CHANGELOG.md')
    if not skipdirhtml:
        sphinx_build['-b', 'dirhtml', '-W', '-E', 'docs', 'docs/_build/dirhtml'] & FG
    sphinx_build['-b', 'html', '-W', '-E', 'docs', 'docs/_build/html'] & FG
```

- **Prompt (70 tok):** <q>Function `gen` generates documentation. It first checks if `docs/changelog.rst` is unchanged using `check_git_unchanged`. If unchanged, it converts `CHANGELOG.md` to `docs/changelog.rst` using Pandoc. Then, based on `skipdirhtml`, it builds HTML and optionally dirHTML documentation using Sphinx.</q>
- **Prompt+Steer (31 tok):** <q>Function `gen` generates documentation. Skips directory HTML build if specified. Converts Markdown changelog to RST. Builds Sphinx HTML and dirHTML outputs.</q>
</details>

<details>
<summary><code>Uninstall</code> (ADB package removal) — 58 → 26 tokens</summary>

```python
def Uninstall(self, package_name, keep_data=False, timeout_ms=None):
    cmd = ['pm uninstall']
    if keep_data:
        cmd.append('-k')
    cmd.append('"%s"' % package_name)
    return self.Shell(' '.join(cmd), timeout_ms=timeout_ms)
```

- **Prompt (58 tok):** <q>Uninstalls specified app. Removes it from device. Optionally keeps user data. Uses `pm uninstall` command. Passes package name. Adds `-k` flag if keeping data. Executes via `Shell` method. Waits up to `timeout_ms`. Returns result of uninstall operation.</q>
- **Prompt+Steer (26 tok):** <q>Uninstalls app. Removes it from device. Keeps data if specified. Runs command. Waits for completion. Returns result.</q>
</details>

<details>
<summary><code>_call</code> (HTTP wrapper with 401 handling) — 75 → 47 tokens</summary>

```python
def _call(self, path, method, body=None, headers=None):
    try:
        resp = self.http.do_call(path, method, body, headers)
    except http.HTTPError as err:
        if err.status == 401:
            raise PermissionError('Insufficient permissions to query ' + '%s with user %s :%s' % (path, self.user, err))
        raise
    return resp
```

- **Prompt (75 tok):** <q>Function `_call` makes HTTP request. Takes `path`, `method`, optional `body`, and `headers`. Calls `self.http.do_call` with these parameters. If HTTP error occurs, checks status code. If 401, raises `PermissionError` with message about insufficient permissions. Otherwise, re-raises original error. Returns response from HTTP call otherwise.</q>
- **Prompt+Steer (47 tok):** <q>Function `_call` sends HTTP request. Takes `path`, `method`, optional `body`, and `headers`. Tries sending request. If gets `401` error, raises `PermissionError`. Otherwise, returns response.</q>
</details>

<details>
<summary><code>callgraph</code> (static call-graph builder) — 65 → 38 tokens</summary>

```python
def callgraph(G, stmt_list):
    func_list = []
    for stmt in stmt_list:
        try:
            G.add_node(stmt.head.ident.name)
            func_list.append(stmt)
        except:
            pass
    for func in func_list:
        assert isinstance(func, node.function)
        func_name = func.head.ident.name
        for s in node.postorder(func):
            if s.__class__ is node.funcall and s.func_expr.__class__ is node.ident:
                G.add_edge(func_name, s.func_expr.name)
```

- **Prompt (65 tok):** <q>Function `callgraph` takes graph `G` and list of statements `stmt_list`. It adds nodes for functions found in `stmt_list` to graph `G`. Then it iterates over these functions, adding edges to graph `G` for each function call within them. This helps visualize function calls in code structure.</q>
- **Prompt+Steer (38 tok):** <q>Function `callgraph` takes graph `G` and list of statements `stmt_list`. It adds nodes for functions found in `stmt_list` and connects nodes with edges representing function calls.</q>
</details>

### Finding the operating point mattered

The first calibration pass picked its config by sweeping (layer, coefficient) on dev and choosing
whichever was most aggressive (lowest tokens) without producing obviously degenerate/repetitive
output — no correctness signal in that loop. That picked layer 14, coeff 36, which looked fine by
the degenerate check but collapsed dev full-correct rate to ~40% (e.g. `_descriptor_names`'s Django
descriptor-filtering detail dropped entirely: "List self's descriptors."). Judging the *entire*
grid (`src/sweep_dev.py` + `src/judge_sweep.py` + `src/analyze_sweep.py`, 20 configs × 50 dev
examples) showed why: correctness holds around 90-95%+ for coefficients 6-12 at layers 14/18, then
falls off a cliff past ~20, down into the 40-65% range. Layer 14, coeff 6 was picked from that curve
instead of trusting the degenerate-only auto-pick — see below for how much that specific pick should,
and shouldn't, be trusted.

![Dev sweep: token count vs. correctness across all 20 (layer, coefficient) configs](results/summary_sweep_plot_dev.png)

## Scope decisions

- We initially wanted to use the Prompt Steering Replacement (arXiv 2605.03907). However, this is paused for now — `src/steering_psr.py` still exists but isn't run.
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
