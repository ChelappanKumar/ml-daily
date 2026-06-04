# STUDY.md — 10-minute guide to ml-daily

**How to read this.** Skim the index (30s). Read whichever section is relevant to the question being asked (~1 min each). If someone asks about the *whole repo*, jump to **Cross-cutting patterns** at the end — that's the framing that ties everything together.

**One honest note before you go in.** Some of this code was scaffolded with AI assistance. Before claiming any commit in an interview, make sure you can:
1. Explain why each design choice was made (this file gives you those).
2. Modify the code under pressure (read each file once, then close it and try to rewrite the core class).
3. Spot the bugs that were fixed (each section flags them — they're real interview material).

---

## Index — what's in the repo

| Date | Topic | Category | Substantive code? |
|---|---|---|---|
| 2026-05-20 | RAG eval with Ragas | agents | scaffold only |
| 2026-05-24 | Transformer block internals | notes | scaffold only |
| 2026-05-26 | Minimal MCP server | agents | scaffold only |
| **2026-05-27** | **BERT — paper notes + MiniBERT MLM from scratch** | notes | yes (PyTorch) |
| **2026-05-28** | **Data validation (Great Expectations style)** | pipelines | yes (pandas) |
| **2026-05-30** | **LLM eval frameworks comparison + mini harness** | notes | yes |
| **2026-06-01** | **Code-execution agent (sandboxed) with retry loop** | agents | yes |
| **2026-06-02** | **Bias / variance — polynomial degree sweep** | notes | yes (numpy + sklearn) |
| **2026-06-04** | **Agent eval with traces — 10-case harness** | agents | yes |

Bold = days with real implementations. The three scaffold-only days are TODO stubs — say so honestly if asked.

---

## 2026-05-27 — BERT from scratch

**Problem solved.** Understand BERT well enough to (a) explain it and (b) implement the core training loop without `nn.TransformerEncoderLayer`.

**Mental model.**
- BERT = stack of Transformer **encoder** layers. No decoder.
- Pretrained on two objectives: **Masked Language Modeling (MLM)** + Next Sentence Prediction (NSP).
- **MLM**: hide 15% of tokens, predict them from bidirectional context. Of those 15%: **80% → `[MASK]`, 10% → random token, 10% → unchanged**. The 10%+10% noise stops the model from learning "only attend where you see `[MASK]`" — critical because fine-tuning sees no `[MASK]` tokens, so pure masking would create a distribution mismatch.
- **NSP**: predict whether sentence B follows A. RoBERTa later showed NSP barely contributes; most of BERT's win is MLM.
- Fine-tune by adding a head on top of `[CLS]` (or per-token for NER/QA) and updating all weights with a tiny LR (2e-5 to 5e-5).
- Sizes: BERT-base = 12 layers, 768 hidden, 12 heads, 110M params. BERT-large = 24/1024/16/340M.

**What the code does (`bert_mlm_from_scratch.py`, ~270 lines).**
- Multi-head self-attention from scratch (single QKV linear → split → scaled dot product → concat heads).
- Pre-norm Transformer block (`x = x + drop(attn(norm(x)))`, then `x = x + drop(ffn(norm(x)))`). The paper used post-norm; pre-norm is the modern default and trains more stably at small scale.
- Tied MLM head — reuses the token embedding matrix as the output projection. Halves the head's param count.
- 80/10/10 masking implemented exactly as the paper.
- Trains a 4-layer / 64-hidden / 4-head mini-BERT on a synthetic bigram corpus.

**Likely interview questions.**
- *Why bidirectional?* Left-to-right LMs can't see future context. MLM hides labels so attending bidirectionally is safe.
- *Why 80/10/10?* Stops `[MASK]`-position dependence; reduces pretrain/finetune distribution mismatch.
- *What's `[CLS]` for?* Aggregate representation for sentence-level downstream tasks.
- *Why pre-norm?* More stable gradients at depth than post-norm; converges from worse init.

---

## 2026-05-28 — Data validation (Great Expectations style)

**Problem solved.** ML pipelines fail silently when input data drifts. Catch it at the **data layer** with explicit assertions, before bad data hits the model.

**Mental model.**
- Declare per-column expectations (not null, in set, in range, regex, unique, schema).
- Run them on every batch as a hard gate.
- **Critical** failures halt the pipeline; **warnings** surface in the report but don't halt.
- The output is a structured `SuiteReport` (JSON-serializable), not a bool — so it can stream into observability.

**What the code does (`starter.py`, ~250 lines, just pandas).**
- `Expectation` ABC with `validate(df) -> ExpectationResult`.
- 6 concrete checks: `ExpectColumnsToExist`, `ExpectColumnValuesToNotBeNull`, `ExpectColumnValuesToBeInSet`, `ExpectColumnValuesToBeBetween`, `ExpectColumnValuesToMatchRegex`, `ExpectColumnValuesToBeUnique`.
- `Suite` aggregator + `validate_or_raise(df, suite)` pipeline entrypoint.
- Demo: good batch passes, bad batch raises `DataValidationError`.

**Gotchas worth remembering.**
- `pandas.Series.str.match` anchors at the **start only** — need explicit `^...$` or it silently passes garbage.
- Tuning `mostly=` thresholds is the hard part. Strict `null_rate == 0` breaks too often; `mostly=0.95` ignores real regressions. Real teams tune per-column from historical distribution.
- Prod uses the real library: `pip install great-expectations` — adds checkpoints, data docs, and integrations.

---

## 2026-05-30 — LLM eval frameworks + mini harness

**Problem solved.** Evaluating an LLM app ≠ evaluating a classifier. No single ground-truth label — you have *behavioural properties* (groundedness, relevancy, no PII leaks, correct tool args).

**Mental model.**
- Every test case has 4 fields: `input`, `actual_output`, `expected_output` (optional), `context` (for RAG). Every framework converges on this shape.
- Metrics are **reference-based** (BLEU, exact match, semantic similarity vs. gold) or **reference-free** (faithfulness, answer relevancy — usually run by an LLM judge).
- **LLM-as-judge biases**: position bias (prefers first), length bias (prefers longer), self-preference (prefers own outputs). Mitigate with pairwise + swap, calibration set, multi-judge ensemble.
- **G-Eval pattern** (DeepEval): give judge a rubric in natural language, ask for 1-5 + chain-of-thought rationale.

**Framework cheat sheet.**
| Framework | Sweet spot |
|---|---|
| **Ragas** | RAG-specific (faithfulness, context precision/recall, answer relevancy). |
| **DeepEval** | General-purpose, pytest-like, G-Eval primitive, CI-friendly. |
| **OpenAI evals** | Model-vs-model comparison registries; OpenAI ecosystem. |
| **Promptfoo** | Side-by-side prompt A/B testing, visual diff. |
| **Inspect (UK AISI)** | Capability/safety evals at scale, agent harnesses. |

**What the code does (`mini_eval_harness.py`, ~250 lines).**
- `TestCase` dataclass + `Metric` ABC.
- 4 concrete metrics: `ExactMatch`, `KeywordOverlap`, `ContextGroundedness`, `LLMJudgeMetric` (pluggable `judge_fn`, no API key needed for the bundled stub).
- `Suite.run()` produces per-case + aggregate report.
- `assert_passes(report, min_pass_rate)` — pytest-style gate breaks CI on regression.
- Verified: the deliberately-bad RAG case ("Tomatoes are red" answering a BERT question) is correctly flagged by all 4 metrics.

---

## 2026-06-01 — Code-execution agent (sandboxed)

**Problem solved.** An LLM agent that writes Python to solve math/data tasks needs a sandbox — `exec()` in-process can wipe files, exfiltrate env vars, or hang forever.

**Mental model.**
- Sandbox isolates the child process from the parent.
- Three layers of defense in this implementation:
  1. **Process isolation** (subprocess, not exec).
  2. **Python isolation** (`python -I` skips PYTHONPATH and user site-packages → stdlib only).
  3. **Resource limits** (`setrlimit` on CPU, address space, file handles, core dumps).
- Plus: env allowlist (only `PATH`, `LANG`, etc. — strips API keys), fresh tempdir workdir, stdin closed, wall-clock timeout.
- **Iterate-on-error loop**: LLM proposes code → sandbox runs → on failure, append the (assistant code, "here's the traceback, fix it") pair to messages and re-prompt. Stop on success or `max_steps`.

**What the code does (`starter.py`, ~290 lines).**
- `Sandbox.run(code) -> RunResult` with all the above.
- `CodeAgent.solve(task)` runs the retry loop.
- Stub LLM seeded with a buggy attempt + correct retry so the loop is observable without API keys.
- Verified end-to-end: prime-sum and Fibonacci tasks fail on attempt 1, recover on attempt 2 with correct answers (328, 55).

**Gotchas.**
- `RLIMIT_AS` is **Linux-only** — macOS raises `ValueError`. Wrap each `setrlimit` in try/except so unsupported limits don't kill the sandbox.
- `subprocess.TimeoutExpired` doesn't always kill the process group promptly — fork-bomb code can leave orphans. Send SIGKILL to the group, not the leader.
- `python -I` blocks PYTHONPATH AND user site-packages. To allow numpy/pandas in the sandbox, build a curated venv and point `sys.executable` at it.
- Subprocess isolation **stops accidents, not a determined attacker on the same host**. For untrusted code: Docker with `--network=none --read-only --cap-drop=ALL`, gVisor, nsjail, or Firecracker microVMs (Vercel Sandbox).

---

## 2026-06-02 — Bias / variance tradeoff

**Problem solved.** Make the textbook formula concrete: `E[(y - ŷ)²] = Bias² + Var + σ²`. Measure all three on a known generating function and see the U-curve.

**Mental model.**
- **Bias** = *systematic* error. A linear model fitting `y = sin(x)` is biased no matter how much data — the hypothesis class can't represent the truth.
- **Variance** = sensitivity to the training sample. A degree-15 polynomial on 30 points draws wildly different curves with 3 points swapped.
- **More data shrinks variance, not bias.** Biased model? You need a more expressive class. High-variance model? More data flattens noise-tracking.
- **Regularization trades variance for bias on purpose** (ridge / lasso).
- **Double descent** (Belkin 2019): for over-parameterized models, test error U-shapes and *then drops again* past the interpolation threshold. Reconciles classical theory with deep learning.

**What the code does (`bias_variance_demo.py`, ~140 lines).**
- True function: `y = sin(1.5πx) + N(0, 0.2²)`.
- Sweeps polynomial degrees 1–15.
- Per degree: 200 bootstrap fits, each on n=30. Compute bias² and variance pointwise on a 200-point test grid.
- Prints a sortable table marking the best degree by test MSE.
- Saves a log-y plot: train MSE keeps falling, test MSE U-shapes, bias² falls and variance climbs as degree grows. Crossover around degree 3-5.

**Likely interview questions.**
- *Why does train MSE keep falling?* Higher-degree polynomial can perfectly interpolate the noisy training data; flexibility ≠ generalization.
- *Why doesn't the U-shape help with deep nets?* Beyond the interpolation threshold, double descent kicks in.
- *How would you reduce variance without changing the model class?* More data, regularization, ensembling, early stopping.

---

## 2026-06-04 — Agent evaluation with traces

**Problem solved.** Evaluating an agent ≠ evaluating an LLM call. The agent might reach the right answer via the wrong tool, or call the right tool with bad args, or loop forever. Score the **trajectory**, not just the output.

**Mental model.**
- Capture every step (LLM call, tool call with args + result, final answer, errors) into a `Trace` object with timestamps + durations.
- Score each case on multiple independent dimensions, not one binary.
- Aggregate **per-check pass rates** are the most useful number — tells you whether the agent is failing on routing, on the answer, or on cost.

**The 4-check rubric used here.**
1. `called_required_tools` — did it actually use the tools the task needs?
2. `avoided_forbidden_tools` — did it skip the anti-pattern tools?
3. `answer_contains_expected` — does the final answer include the expected substrings?
4. `within_step_budget` — did it finish in fewer than N steps?

**What the code does (`starter.py`, ~325 lines).**
- 3 tool stubs (`search_web`, `calculate`, `get_weather`).
- `Trace` / `TraceEvent` capture timeline with latency.
- `Agent.run()` is the tool-calling loop with `max_steps` budget.
- 10 `EvalCase` scenarios, scored on the 4 dimensions.
- Evaluator outputs `pass_rate`, `per_check_pass_rate`, `mean_latency_ms`, `tool_call_distribution`, plus a CI gate.
- Verified: ended at 10/10 pass after fixing two real bugs.

**Bugs that were fixed during development (good interview material).**
- Keyword routing over-fired: "what is the capital of France" matched "what is" → routed to calculator → returned `refused: unsafe characters` → that string became the final answer → eval failed. **Fix**: gate calculator on actual numeric expression presence, not English question stems.
- Arithmetic regex `[-+]?\d+(\s*[\d+\-*/().\s]+)?` started matching at the first digit, so `(12 + 8) * 4` became `12 + 8) * 4` (syntax error). **Fix**: find the longest run of arithmetic chars containing both digit + operator.

---

## Cross-cutting patterns (memorize these)

Four patterns recur across the substantive days. If asked about the **whole repo**, lead with these.

### 1. `ABC` + concrete subclasses + a `Suite` runner

Appears in: data validation, eval harness, agent eval.

- `Expectation` / `Suite` (data validation)
- `Metric` / `Suite` (LLM eval)
- `Tool` registry (agent eval)

Each "rule" is one class with one method. The runner is a dumb `[exp.validate(df) for exp in self.expectations]`. The report aggregates without knowing implementation details. This is also how `great_expectations`, `deepeval`, and `ragas` are actually structured — it's the standard shape.

### 2. Structured report objects (never just `bool`)

`SuiteReport`, `MetricResult`, `CaseScore`, `Trace`. All dataclasses with `to_dict()` / `to_json()`. Why:
- JSON-serializable → streams to BigQuery / observability stacks.
- Per-check fields → makes failures debuggable (you see *which* dimension failed, not just "failed").
- Stable schema → safe to diff across runs.

### 3. Retry-on-error loops

In the code-execution agent: LLM proposes → sandbox runs → on failure, append `(assistant_message, "traceback + fix it")` to messages → re-prompt. ~10 lines. Same pattern works for any agent where the environment gives observable feedback (tool error, validator failure, compiler error).

### 4. LLM-as-judge framing

`LLMJudgeMetric` with a pluggable `judge_fn`. Real frameworks (G-Eval, Ragas) are this same shape with better rubrics. Biases to know: position, length, self-preference. Mitigations: pairwise + swap, calibration set, multi-judge ensemble.

---

## If you have 2 minutes left

Memorize these 5 sentences. They cover 80% of what someone might ask about this repo:

1. *"BERT is a Transformer encoder pretrained with MLM (15% of tokens masked using an 80/10/10 strategy to avoid pretrain/finetune mismatch) and fine-tuned by adding a small head on `[CLS]`."*
2. *"Data validation in ML pipelines should be a hard gate before model code — I implement it as expectations per column with critical/warning severity levels, producing a structured report rather than a boolean."*
3. *"LLM eval is reference-based vs reference-free. For reference-free I use an LLM judge with G-Eval-style rubrics, knowing position/length/self-preference biases need pairwise + swap mitigation."*
4. *"Code-execution agents need real isolation — subprocess + setrlimit + env allowlist is a baseline; production uses Docker with `--network=none` or microVM sandboxes like Vercel Sandbox or Firecracker."*
5. *"Agent eval scores the trajectory, not just the output: did it call the right tools, avoid the wrong ones, produce the expected answer, and stay within step budget — surfaced as per-check pass rates, not one binary."*
