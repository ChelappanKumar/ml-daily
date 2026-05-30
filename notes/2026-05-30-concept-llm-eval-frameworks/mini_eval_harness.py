"""Mini LLM eval harness — the same shape as Ragas / DeepEval, from scratch.

Problem this solves:
    You shipped a RAG chatbot. You want to know: did the last prompt change make
    the answers better, worse, or just different? You need a repeatable way to
    score outputs across a fixed dataset, with both reference-based metrics
    (compare to gold) and reference-free metrics (LLM-as-judge or heuristic).

This file demonstrates the standard architecture:
    1. TestCase dataclass (input / actual_output / expected_output / context)
    2. Metric ABC — each metric scores ONE case to a float in [0, 1].
    3. Three concrete metrics:
         - ExactMatch        (reference-based)
         - KeywordOverlap    (reference-free heuristic, no LLM)
         - LLMJudgeMetric    (reference-free, calls a `judge_fn` — swap in any model)
    4. Suite + run loop producing per-case + aggregate report.
    5. pytest-style assert_passes(threshold) so eval can break CI.

Run with the bundled stub judge (no API key needed):
    python mini_eval_harness.py

Plug in a real judge:
    Pass a `judge_fn(prompt: str) -> str` that calls Anthropic/OpenAI/etc.
"""
from __future__ import annotations

import json
import re
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable


# ---------- data model ----------

@dataclass
class TestCase:
    input: str
    actual_output: str
    expected_output: str | None = None
    context: list[str] = field(default_factory=list)
    case_id: str = ""


@dataclass
class MetricResult:
    metric: str
    case_id: str
    score: float        # in [0, 1]
    passed: bool
    reason: str = ""


@dataclass
class SuiteReport:
    threshold: float
    per_metric_mean: dict[str, float]
    pass_rate: float
    results: list[MetricResult]

    def to_json(self) -> str:
        d = {
            "threshold": self.threshold,
            "per_metric_mean": self.per_metric_mean,
            "pass_rate": self.pass_rate,
            "results": [asdict(r) for r in self.results],
        }
        return json.dumps(d, indent=2)


# ---------- metric base ----------

class Metric(ABC):
    name: str = "metric"
    threshold: float = 0.5  # default pass threshold

    @abstractmethod
    def score(self, case: TestCase) -> tuple[float, str]:
        """Return (score in [0,1], short reason)."""

    def evaluate(self, case: TestCase) -> MetricResult:
        score, reason = self.score(case)
        score = max(0.0, min(1.0, score))
        return MetricResult(
            metric=self.name, case_id=case.case_id,
            score=score, passed=score >= self.threshold, reason=reason,
        )


# ---------- concrete metrics ----------

class ExactMatch(Metric):
    name = "exact_match"
    threshold = 1.0

    def score(self, case: TestCase) -> tuple[float, str]:
        if case.expected_output is None:
            return 0.0, "no expected_output provided"
        a = case.actual_output.strip().lower()
        b = case.expected_output.strip().lower()
        return (1.0, "match") if a == b else (0.0, "no match")


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


class KeywordOverlap(Metric):
    """Reference-free heuristic — answer should mention the key terms from the input.

    Crude but cheap. A real implementation might use embeddings + cosine.
    """
    name = "keyword_overlap"
    threshold = 0.4

    def score(self, case: TestCase) -> tuple[float, str]:
        q = _tokens(case.input)
        a = _tokens(case.actual_output)
        if not q:
            return 0.0, "empty input"
        overlap = len(q & a) / len(q)
        return overlap, f"{len(q & a)}/{len(q)} input tokens echoed"


class ContextGroundedness(Metric):
    """RAG-specific: did the answer reuse terms from the retrieved context?

    Approximates Ragas 'faithfulness' without calling an LLM. Real faithfulness
    uses a judge to verify each claim, but this catches gross hallucinations.
    """
    name = "context_groundedness"
    threshold = 0.3

    def score(self, case: TestCase) -> tuple[float, str]:
        if not case.context:
            return 0.0, "no context provided"
        ctx_tokens = set()
        for c in case.context:
            ctx_tokens |= _tokens(c)
        ans_tokens = _tokens(case.actual_output)
        if not ans_tokens:
            return 0.0, "empty answer"
        grounded = len(ans_tokens & ctx_tokens) / len(ans_tokens)
        return grounded, f"{len(ans_tokens & ctx_tokens)}/{len(ans_tokens)} answer tokens in context"


JudgeFn = Callable[[str], str]


class LLMJudgeMetric(Metric):
    """G-Eval-style: judge model scores 1-5 from a rubric, we rescale to [0, 1].

    Pass a `judge_fn(prompt) -> str`. The judge must return a single digit 1-5.
    """
    name = "llm_judge"
    threshold = 0.6  # i.e. 4/5

    RUBRIC = (
        "You are a strict evaluator. Score the assistant's answer from 1 to 5 "
        "for helpfulness AND factual correctness, with 1 being unusable and 5 being excellent. "
        "Respond with ONLY the single digit (1, 2, 3, 4, or 5)."
    )

    def __init__(self, judge_fn: JudgeFn, name: str = "llm_judge", threshold: float = 0.6):
        self.judge_fn = judge_fn
        self.name = name
        self.threshold = threshold

    def score(self, case: TestCase) -> tuple[float, str]:
        prompt = (
            f"{self.RUBRIC}\n\n"
            f"USER QUESTION:\n{case.input}\n\n"
            f"ASSISTANT ANSWER:\n{case.actual_output}\n\n"
            f"Score (1-5):"
        )
        raw = self.judge_fn(prompt).strip()
        m = re.search(r"[1-5]", raw)
        if not m:
            return 0.0, f"judge returned unparseable: {raw!r}"
        rating = int(m.group(0))
        return (rating - 1) / 4.0, f"judge gave {rating}/5"


# ---------- suite ----------

@dataclass
class Suite:
    name: str
    cases: list[TestCase]
    metrics: list[Metric]

    def run(self) -> SuiteReport:
        results: list[MetricResult] = []
        for case in self.cases:
            for metric in self.metrics:
                results.append(metric.evaluate(case))

        # Aggregate per metric.
        per_metric: dict[str, list[float]] = {}
        for r in results:
            per_metric.setdefault(r.metric, []).append(r.score)
        means = {k: round(statistics.mean(v), 4) for k, v in per_metric.items()}

        pass_rate = round(sum(r.passed for r in results) / len(results), 4) if results else 0.0
        return SuiteReport(threshold=0.0, per_metric_mean=means, pass_rate=pass_rate, results=results)


def assert_passes(report: SuiteReport, min_pass_rate: float) -> None:
    """pytest-style gate. Raises AssertionError so CI fails on regression."""
    if report.pass_rate < min_pass_rate:
        raise AssertionError(
            f"eval failed: pass_rate {report.pass_rate} < threshold {min_pass_rate}\n"
            + report.to_json()
        )


# ---------- demo ----------

def _stub_judge(prompt: str) -> str:
    """Deterministic fake judge so the file runs with no API key.

    Heuristic: longer answers containing 'because' or 'specifically' get 4-5,
    short or generic answers get 2-3. This is obviously not a real judge —
    swap with Anthropic/OpenAI for production.
    """
    answer_line = prompt.rsplit("ASSISTANT ANSWER:", 1)[-1].split("Score")[0]
    n_words = len(answer_line.split())
    has_evidence = any(w in answer_line.lower() for w in ("because", "specifically", "according to"))
    if n_words > 25 and has_evidence:
        return "5"
    if n_words > 15:
        return "4"
    if n_words > 5:
        return "3"
    return "2"


def _demo_cases() -> list[TestCase]:
    return [
        TestCase(
            case_id="rag-01",
            input="What is the capital of France?",
            actual_output="The capital of France is Paris, specifically located on the Seine river.",
            expected_output="Paris",
            context=["France is a country in Europe. Its capital is Paris, located on the Seine."],
        ),
        TestCase(
            case_id="rag-02",
            input="When was BERT published?",
            actual_output="BERT was introduced in 2018 by researchers at Google, specifically Devlin et al.",
            expected_output="2018",
            context=["BERT was published by Devlin et al. in 2018 at Google AI."],
        ),
        TestCase(
            case_id="rag-03-bad",
            input="What loss does BERT use for pretraining?",
            actual_output="Tomatoes are red.",  # obviously wrong
            expected_output="Masked Language Modeling",
            context=["BERT pretrains with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)."],
        ),
    ]


def main() -> None:
    suite = Suite(
        name="rag_smoke_eval",
        cases=_demo_cases(),
        metrics=[
            ExactMatch(),
            KeywordOverlap(),
            ContextGroundedness(),
            LLMJudgeMetric(judge_fn=_stub_judge),
        ],
    )

    report = suite.run()
    print(report.to_json())

    print(f"\npass_rate = {report.pass_rate}")
    print("per_metric_mean =", report.per_metric_mean)

    # Gate: fail the script if the suite regresses below 50% pass rate.
    try:
        assert_passes(report, min_pass_rate=0.5)
        print("\nCI gate PASSED.")
    except AssertionError as e:
        print(f"\nCI gate FAILED: {e}")


if __name__ == "__main__":
    main()
