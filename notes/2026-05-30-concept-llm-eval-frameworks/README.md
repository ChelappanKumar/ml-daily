# Concept — LLM eval frameworks

**Goal:** Compare Ragas, DeepEval, OpenAI evals — when to use which, and what the moving parts actually are.

## Summary

Evaluating an LLM application is not the same problem as evaluating a classifier. There's no single ground-truth label — instead you have *behavioural properties* you want to hold (the answer is grounded in the retrieved context, the response is on-topic, no PII leaks, the tool call has the right arguments). LLM eval frameworks give you (a) a way to express those properties as **metrics**, (b) a way to run them on a **dataset of cases**, and (c) **judges** — usually another LLM — that score each case. The framework choice is mostly about which metrics ship out-of-the-box, how easily you can write custom ones, and whether you need CI integration vs. an interactive dashboard.

## Key ideas

- **A test case has 4 fields**: `input` (user query), `actual_output` (what the model produced), `expected_output` (optional, for reference-based metrics), `context` (retrieved docs, for RAG). Every framework converges on something close to this shape.
- **Metrics are either reference-based or reference-free.**
  - *Reference-based*: BLEU, ROUGE, exact match, semantic similarity vs. a gold answer. Cheap, deterministic, but you need labelled data.
  - *Reference-free*: faithfulness, answer relevancy, context precision. Run by an LLM judge. No labels needed, but slower and noisier.
- **LLM-as-judge has real biases**: position bias (prefers first answer), length bias (prefers longer), self-preference (a model judges its own outputs higher). Mitigations: pairwise comparison with swapped order, calibration set, multi-judge ensemble.
- **G-Eval pattern** (popularized by DeepEval): give the judge a rubric in natural language, ask it for a 1-5 score + chain-of-thought rationale. The CoT improves consistency. Implemented as a generic primitive — write any metric in a paragraph.

## When to use which

| Framework | Sweet spot | Skip if |
|-----------|------------|---------|
| **Ragas** | RAG-specific metrics (faithfulness, context precision/recall, answer relevancy). Best when you have retrieval in the loop. | You're not doing RAG; their non-RAG support is thinner. |
| **DeepEval** | General-purpose. pytest-like syntax, G-Eval primitive, broad metric library, CI-friendly. | You want a hosted dashboard with no setup — DeepEval is library-first. |
| **OpenAI evals** | Large model-vs-model comparison registries. Good if you live in the OpenAI ecosystem and want their public eval suites. | You're testing application logic, not model capability. |
| **Promptfoo** | Side-by-side prompt-template A/B testing, visual diff. | You need programmatic eval inside Python tests. |
| **Inspect (UK AISI)** | Capability/safety evals at scale. Good tools, agent harnesses. | You want a 30-minute setup. |

## Worked example — see `mini_eval_harness.py`

A 200-line eval harness from scratch that demonstrates the same architecture:
- A `TestCase` dataclass matching the standard 4-field shape.
- Three metrics: exact-match (reference-based), keyword-overlap (reference-free heuristic), and a stub `LLMJudgeMetric` (the only piece a real framework adds — call out to a model).
- A `Suite.run(model)` loop that produces a per-case + aggregate report.
- A pytest-style assertion API so failing eval cases can break CI.

## Open questions

- How do you keep the judge calibrated as you swap judge models? Anchored rubrics (with worked examples in the prompt) seem to help — measure.
- For agent evals (multi-step, tool calls), how do you score *trajectory* vs. just final output? Inspect has trajectory metrics worth studying.
- Cost: at 10k cases × 3 judge metrics × $0.01/judge call you're at $300/eval run. What's the sampling strategy that catches regressions without re-judging everything?

## References

- https://docs.ragas.io/
- https://docs.confident-ai.com/ (DeepEval)
- https://github.com/openai/evals
- https://www.promptfoo.dev/
- https://inspect.aisi.org.uk/
- https://arxiv.org/abs/2303.16634 (G-Eval paper)
