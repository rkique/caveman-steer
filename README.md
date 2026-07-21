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
docstring, blinded to condition. Correctness is reported as a rounded band, not false-precision
decimals — see [Statistical details](#statistical-details) for why and for the exact counts:

| Condition | Avg tokens | Fully correct |
|---|---|---|
| Base | 150.0 (censored — see below) | 90-95% |
| Prompt | 56.8 | 90-95% |
| Const-steer alone | 149.9 | 90-95% |
| **Prompt+Steer** | **49.3** | **90-95%** |

**Steering on top of the prompt gets a further ~13% token reduction beyond prompting alone, with no
statistically detectable correctness cost.** Const-steer alone is a non-result at this coefficient
(149.9 tokens, indistinguishable from Base) — the coefficient was calibrated specifically for the
combined regime, not as a standalone replacement for the instruction; this matches the general
pattern (also seen at higher coefficients during calibration) that steering alone is substantially
weaker than prompting.

Base's 150.0 average is right-censored: 95.6% of Base responses hit `MAX_NEW_TOKENS` without the
model choosing to stop, so it's a floor on "at least this many tokens," not a real measurement of
default verbosity — the Prompt vs. Prompt+Steer comparison (both close to fully natural stopping)
is the one to trust.

![Test set: token count vs. correctness across the four conditions](results/summary_plot_test.png)

## Is caveman's style actually being followed?

Caveman's rule is explicit and mechanically checkable: "Drop: articles (a/an/the)... Fragments OK."
So rather than eyeballing it, we measured literal compliance — how often `a`/`an`/`the` still shows
up despite being told not to, across all 180 test responses:

| Condition | Responses containing a/an/the | Articles per 100 words |
|---|---|---|
| Base | 100% | 12.68 |
| Const-steer alone | 100% | 12.47 |
| Prompt | 46.1% | 4.27 |
| **Prompt+Steer** | **31.1%** | **2.48** |

The instruction is genuinely in the prompt (verified directly — see `model_common.CAVEMAN_SUFFIX`)
and genuinely changes behavior (articles drop ~66% vs. Base), but compliance with the specific
"drop articles" rule is partial, not total: 46% of Prompt responses still use one. Const-steer alone
(coefficient calibrated for the combined regime, not as a standalone replacement) shows no effect on
style either, consistent with it showing no effect on length. Steering on top of the prompt pushes
compliance further than the text instruction manages alone — fewer responses use an article at all,
and at a lower rate when they do. Since compliance is partial, the examples below were selected for
being representative (Prompt itself already article-free) rather than for the single biggest
token-count drop, so this section doesn't accidentally show only the non-compliant half.

## Examples

Real test-set pairs, Prompt vs. Prompt+Steer, both judged fully correct (`correct=2`) and Prompt
itself already caveman-compliant (article-free) — steering compresses further without dropping
substance:

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

- **Prompt (86 tok):** "Function `initial_step` sets initial values for coordinates `x`, `y`, and phase. Depending on `MODE`, it adjusts these values based on specific conditions. If in rotation mode, it flips `x` and adjusts phase if outside [-0.5, 0.5]. In vectoring mode, it flips both `x` and `y` and sets phase to 1 or -1 based on quadrant."
- **Prompt+Steer (29 tok):** "Set initial values for `x`, `y`, and `phase`. Adjust based on `MODE` and input values. Continue with next step."
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

- **Prompt (70 tok):** "Function `gen` generates documentation. It first checks if `docs/changelog.rst` is unchanged using `check_git_unchanged`. If unchanged, it converts `CHANGELOG.md` to `docs/changelog.rst` using Pandoc. Then, based on `skipdirhtml`, it builds HTML and optionally dirHTML documentation using Sphinx."
- **Prompt+Steer (31 tok):** "Function `gen` generates documentation. Skips directory HTML build if specified. Converts Markdown changelog to RST. Builds Sphinx HTML and dirHTML outputs."
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

- **Prompt (58 tok):** "Uninstalls specified app. Removes it from device. Optionally keeps user data. Uses `pm uninstall` command. Passes package name. Adds `-k` flag if keeping data. Executes via `Shell` method. Waits up to `timeout_ms`. Returns result of uninstall operation."
- **Prompt+Steer (26 tok):** "Uninstalls app. Removes it from device. Keeps data if specified. Runs command. Waits for completion. Returns result."
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

- **Prompt (75 tok):** "Function `_call` makes HTTP request. Takes `path`, `method`, optional `body`, and `headers`. Calls `self.http.do_call` with these parameters. If HTTP error occurs, checks status code. If 401, raises `PermissionError` with message about insufficient permissions. Otherwise, re-raises original error. Returns response from HTTP call otherwise."
- **Prompt+Steer (47 tok):** "Function `_call` sends HTTP request. Takes `path`, `method`, optional `body`, and `headers`. Tries sending request. If gets `401` error, raises `PermissionError`. Otherwise, returns response."
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

- **Prompt (65 tok):** "Function `callgraph` takes graph `G` and list of statements `stmt_list`. It adds nodes for functions found in `stmt_list` to graph `G`. Then it iterates over these functions, adding edges to graph `G` for each function call within them. This helps visualize function calls in code structure."
- **Prompt+Steer (38 tok):** "Function `callgraph` takes graph `G` and list of statements `stmt_list`. It adds nodes for functions found in `stmt_list` and connects nodes with edges representing function calls."
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

## Statistical details

**Is the Prompt+Steer test-set result just cherry-picking?** Two separate risks, checked
separately:

**1. Is "Prompt+Steer ≥ Prompt" on the test set real, or noise?** Test set, n=180, McNemar's exact
test on paired fully-correct outcomes (same 180 examples, both conditions): Prompt 165/180,
Prompt+Steer 166/180 — 4 examples correct only under Prompt, 5 correct only under Prompt+Steer,
**p = 1.0**. That's as close to a coin flip as paired data gets. Read this as: **steering shows no
detectable correctness cost**, not as "steering improves correctness" — the one-example difference
carries no statistical weight either way. What *is* well-supported and doesn't need a significance
test: the ~13% token reduction, since it's a large, consistent shift across the distribution, not a
one-example-sized effect.

**2. Was picking layer 14/coeff 6 out of 20 dev configs cherry-picking?** Partly, yes — and it's
worth being explicit about where. Wilson 95% CIs on the dev full-correct rate (n=50 per config):

| Layer | Coeff | k/50 | Rate | 95% CI |
|---|---|---|---|---|
| 7 | 12 | 48 | 96% | [86.5%, 98.9%] |
| 7 | 28 | 48 | 96% | [86.5%, 98.9%] |
| **14** | **6** | **48** | **96%** | **[86.5%, 98.9%]** |
| 22 | 6 | 48 | 96% | [86.5%, 98.9%] |
| 22 | 12 | 48 | 96% | [86.5%, 98.9%] |
| 22 | 36 | 48 | 96% | [86.5%, 98.9%] |
| 14 | 12 | 44 | 88% | [76.2%, 94.4%] |
| 18 | 20 | 41 | 82% | [69.2%, 90.2%] |
| 14 | 36 | 20 | 40% | [27.6%, 53.8%] |

*(full 20-row table: `results/summary_sweep_dev.json`)*

Six different configs tie at exactly 48/50, and the CI on any single one of them comfortably
overlaps configs as low as 88% or even 82%. With only 50 dev examples per config and 20 configs
swept, picking "the max observed value" has real winner's-curse risk — the specific rank-1 pick
among that top cluster shouldn't be over-trusted, which is why the result above is reported as a
90-95% band rather than "96%." What *is* robust: the top cluster (78-96%, overlapping CIs) and the
collapsed bottom cluster (40-66% at coeff ≥28, layers 14/18) have **non-overlapping** CIs — the
qualitative finding "moderate coefficients are safe, aggressive ones collapse correctness" doesn't
depend on which exact config within the safe cluster you'd pick.

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
