"""Agent evaluation with traces — record tool calls, score them, compute pass rate.

Problem this solves:
    Evaluating an agent isn't the same as evaluating a single LLM call. The
    agent might reach the right answer via the wrong tool, or call the right
    tool with bad arguments, or loop forever. You need to capture the full
    trajectory (every tool call + its result) and score it against task-level
    expectations: did it call the expected tools, in roughly the right order,
    with arguments that match the rubric, and arrive at the right final answer?

What's here:
    1. `Tool` protocol + a tiny tool registry (`search_web`, `calculate`,
       `get_weather`) — stubbed deterministic implementations so the file runs
       without network or API keys.
    2. `Trace` / `TraceEvent` data classes capturing the full execution timeline:
       LLM call, tool call (with args + result), final answer, latency, errors.
    3. `Agent.run(task)` — a simple tool-calling loop with retry, max-step cap,
       and trace recording.
    4. `EvalCase` + 10 scripted tasks with expectations:
         - `must_call_tools`: tools that must appear in the trace.
         - `must_not_call_tools`: tools that must NOT appear (anti-patterns).
         - `expected_answer_contains`: substrings the final answer must include.
         - `max_steps_allowed`: trajectory budget.
    5. `evaluate(cases)` — runs all 10, scores each on 4 dimensions, prints a
       per-case breakdown + an aggregate report with pass rate, mean latency,
       tool-call distribution.

Plug a real LLM:
    Replace `scripted_llm` with a function that calls Anthropic / OpenAI and
    returns the same `LlmDecision` shape (either a tool call or a final answer).

Run:
    python starter.py
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal


# ---------- tool registry ----------

ToolFn = Callable[[dict[str, Any]], Any]


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn


def _t_search_web(args: dict[str, Any]) -> str:
    """Stub web search: deterministic mock results keyed off the query."""
    q = (args.get("query") or "").lower()
    if "capital of france" in q:
        return "Paris is the capital of France."
    if "tallest mountain" in q:
        return "Mount Everest is the tallest mountain at 8,848 m."
    if "year bert" in q or "bert published" in q:
        return "BERT was published by Devlin et al. in 2018."
    if "speed of light" in q:
        return "The speed of light in vacuum is 299,792,458 m/s."
    return "No relevant results."


def _t_calculate(args: dict[str, Any]) -> float | str:
    """Tiny safe-ish calculator. Real prod would use a parser, not eval."""
    expr = str(args.get("expression", ""))
    if not expr or any(c not in "0123456789+-*/(). " for c in expr):
        return f"refused: unsafe characters in {expr!r}"
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — scoped, char-filtered
    except Exception as e:
        return f"error: {e}"


def _t_get_weather(args: dict[str, Any]) -> str:
    """Stub weather: deterministic by city."""
    city = (args.get("city") or "").lower()
    table = {"paris": "Paris: 18°C, light rain.", "tokyo": "Tokyo: 24°C, clear."}
    return table.get(city, f"No weather data for {city!r}.")


TOOLS: dict[str, Tool] = {
    t.name: t for t in [
        Tool("search_web", "Look up a fact on the web. args: {query: str}", _t_search_web),
        Tool("calculate", "Evaluate an arithmetic expression. args: {expression: str}", _t_calculate),
        Tool("get_weather", "Get current weather. args: {city: str}", _t_get_weather),
    ]
}


# ---------- trace ----------

EventKind = Literal["llm_call", "tool_call", "final_answer", "error"]


@dataclass
class TraceEvent:
    kind: EventKind
    timestamp: float
    duration_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    task_id: str
    events: list[TraceEvent] = field(default_factory=list)
    final_answer: str | None = None
    succeeded: bool = False  # set by evaluator, not by agent

    def tool_call_names(self) -> list[str]:
        return [e.payload["tool"] for e in self.events if e.kind == "tool_call"]

    def total_latency_ms(self) -> float:
        return sum(e.duration_ms for e in self.events)

    def step_count(self) -> int:
        return sum(1 for e in self.events if e.kind in ("llm_call", "tool_call"))


# ---------- agent ----------

@dataclass
class LlmDecision:
    """Two-shape union: either a tool call OR a final answer."""
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None


LlmFn = Callable[[str, list[TraceEvent]], LlmDecision]


class Agent:
    def __init__(self, llm_fn: LlmFn, max_steps: int = 6):
        self.llm_fn = llm_fn
        self.max_steps = max_steps

    def run(self, task_id: str, task: str) -> Trace:
        trace = Trace(task_id=task_id)
        for _ in range(self.max_steps):
            # LLM decides next action.
            t0 = time.perf_counter()
            try:
                decision = self.llm_fn(task, trace.events)
            except Exception as e:
                trace.events.append(TraceEvent(kind="error", timestamp=time.time(),
                                                duration_ms=(time.perf_counter() - t0) * 1000,
                                                payload={"stage": "llm", "error": str(e)}))
                return trace
            trace.events.append(TraceEvent(kind="llm_call", timestamp=time.time(),
                                            duration_ms=(time.perf_counter() - t0) * 1000,
                                            payload={"decision": asdict(decision)}))

            if decision.final_answer is not None:
                trace.final_answer = decision.final_answer
                trace.events.append(TraceEvent(kind="final_answer", timestamp=time.time(),
                                                payload={"answer": decision.final_answer}))
                return trace

            if decision.tool is None or decision.tool not in TOOLS:
                trace.events.append(TraceEvent(kind="error", timestamp=time.time(),
                                                payload={"reason": "unknown_or_missing_tool",
                                                         "requested": decision.tool}))
                return trace

            # Execute tool.
            tool = TOOLS[decision.tool]
            t0 = time.perf_counter()
            try:
                result = tool.fn(decision.args)
                err = None
            except Exception as e:
                result = None
                err = str(e)
            trace.events.append(TraceEvent(
                kind="tool_call", timestamp=time.time(),
                duration_ms=(time.perf_counter() - t0) * 1000,
                payload={"tool": tool.name, "args": decision.args, "result": result, "error": err},
            ))

        # Out of steps.
        trace.events.append(TraceEvent(kind="error", timestamp=time.time(),
                                        payload={"reason": "max_steps_reached"}))
        return trace


# ---------- scripted LLM (deterministic, no API key needed) ----------

_ARITH_RUN = re.compile(r"[\d+\-*/().\s]{2,}")


def _extract_arith(s: str) -> str | None:
    """Return the longest substring that looks like an arithmetic expression.

    Strategy: strip everything that isn't a digit, operator, paren, dot, or
    whitespace. Then split on long whitespace runs and pick the longest chunk
    that contains both a digit and an operator (or is a bare number).
    """
    candidates: list[str] = []
    for m in _ARITH_RUN.finditer(s):
        chunk = m.group(0).strip()
        if not chunk:
            continue
        has_digit = any(c.isdigit() for c in chunk)
        has_op = any(c in "+-*/" for c in chunk)
        if has_digit and (has_op or chunk.isdigit()):
            candidates.append(chunk)
    return max(candidates, key=len) if candidates else None


def scripted_llm(task: str, events: list[TraceEvent]) -> LlmDecision:
    """Routes by simple keyword heuristics. Deterministic so eval is reproducible.

    The point isn't the routing quality — it's that we can score the trace.
    Order matters: weather > arithmetic > web search.
    """
    seen_tools = [e.payload["tool"] for e in events if e.kind == "tool_call"]
    last_result = next(
        (e.payload.get("result") for e in reversed(events) if e.kind == "tool_call"),
        None,
    )
    t = task.lower()

    # If a tool already returned something useful, summarize as final answer.
    if isinstance(last_result, str) and last_result and last_result != "No relevant results."         and not last_result.startswith(("refused:", "error:")):
        return LlmDecision(final_answer=last_result)
    if last_result is not None and not isinstance(last_result, str):
        return LlmDecision(final_answer=f"The answer is {last_result}.")

    # Weather has the strongest signal — check first.
    if "weather" in t and "get_weather" not in seen_tools:
        # Pull the city after "in".
        city = t.split(" in ", 1)[-1].strip(" ?.").title() if " in " in t else ""
        return LlmDecision(tool="get_weather", args={"city": city})

    # Arithmetic: only if there's an actual numeric expression in the prompt.
    expr = _extract_arith(t)
    if expr is not None and "calculate" not in seen_tools:
        return LlmDecision(tool="calculate", args={"expression": expr})

    # Everything else: web search.
    if "search_web" not in seen_tools:
        return LlmDecision(tool="search_web", args={"query": task})

    return LlmDecision(final_answer="I could not find an answer.")


# ---------- eval cases ----------

@dataclass
class EvalCase:
    id: str
    task: str
    must_call_tools: list[str] = field(default_factory=list)
    must_not_call_tools: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    max_steps_allowed: int = 6


CASES: list[EvalCase] = [
    EvalCase("c1",  "What is the capital of France?",
             must_call_tools=["search_web"], expected_answer_contains=["Paris"]),
    EvalCase("c2",  "What is 17 * 23?",
             must_call_tools=["calculate"], must_not_call_tools=["search_web"],
             expected_answer_contains=["391"]),
    EvalCase("c3",  "What is the weather in Paris?",
             must_call_tools=["get_weather"], expected_answer_contains=["Paris", "18"]),
    EvalCase("c4",  "What year was BERT published?",
             must_call_tools=["search_web"], expected_answer_contains=["2018"]),
    EvalCase("c5",  "Calculate (12 + 8) * 4",
             must_call_tools=["calculate"], expected_answer_contains=["80"]),
    EvalCase("c6",  "What is the tallest mountain on Earth?",
             must_call_tools=["search_web"], expected_answer_contains=["Everest"]),
    EvalCase("c7",  "What is the weather in Tokyo?",
             must_call_tools=["get_weather"], expected_answer_contains=["Tokyo", "24"]),
    EvalCase("c8",  "What is the speed of light?",
             must_call_tools=["search_web"], expected_answer_contains=["299"]),
    EvalCase("c9",  "What is 100 / 4?",
             must_call_tools=["calculate"], must_not_call_tools=["search_web"],
             expected_answer_contains=["25"]),
    EvalCase("c10", "Compute 2 * (3 + 5)",
             must_call_tools=["calculate"], expected_answer_contains=["16"]),
]


# ---------- evaluator ----------

@dataclass
class CaseScore:
    case_id: str
    passed: bool
    checks: dict[str, bool]
    trace_summary: dict[str, Any]


def score_case(case: EvalCase, trace: Trace) -> CaseScore:
    called = trace.tool_call_names()
    answer = (trace.final_answer or "").lower()

    checks = {
        "called_required_tools": all(t in called for t in case.must_call_tools),
        "avoided_forbidden_tools": not any(t in called for t in case.must_not_call_tools),
        "answer_contains_expected": all(s.lower() in answer for s in case.expected_answer_contains),
        "within_step_budget": trace.step_count() <= case.max_steps_allowed,
    }
    passed = all(checks.values())
    trace.succeeded = passed
    return CaseScore(
        case_id=case.id, passed=passed, checks=checks,
        trace_summary={
            "tools_called": called,
            "step_count": trace.step_count(),
            "total_latency_ms": round(trace.total_latency_ms(), 3),
            "final_answer": trace.final_answer,
        },
    )


def evaluate(cases: list[EvalCase], agent: Agent) -> dict[str, Any]:
    scores: list[CaseScore] = []
    for case in cases:
        trace = agent.run(case.id, case.task)
        scores.append(score_case(case, trace))

    pass_rate = sum(s.passed for s in scores) / len(scores)
    latencies = [s.trace_summary["total_latency_ms"] for s in scores]
    tool_counter: Counter[str] = Counter()
    for s in scores:
        tool_counter.update(s.trace_summary["tools_called"])

    # Per-check pass rates surface WHICH dimension is failing across the suite.
    per_check: dict[str, float] = {}
    for check in ("called_required_tools", "avoided_forbidden_tools",
                  "answer_contains_expected", "within_step_budget"):
        per_check[check] = sum(s.checks[check] for s in scores) / len(scores)

    return {
        "pass_rate": round(pass_rate, 4),
        "per_check_pass_rate": {k: round(v, 4) for k, v in per_check.items()},
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "tool_call_distribution": dict(tool_counter),
        "cases": [asdict(s) for s in scores],
    }


def main() -> None:
    agent = Agent(llm_fn=scripted_llm, max_steps=6)
    report = evaluate(CASES, agent)

    # Top-line summary first, then a per-case detail block.
    summary_keys = {k: v for k, v in report.items() if k != "cases"}
    print(json.dumps(summary_keys, indent=2))

    print("\nPer-case detail:")
    for case in report["cases"]:
        mark = "PASS" if case["passed"] else "FAIL"
        failed_checks = [k for k, ok in case["checks"].items() if not ok]
        detail = f" failed_checks={failed_checks}" if failed_checks else ""
        print(f"  {case['case_id']}: {mark} "
              f"tools={case['trace_summary']['tools_called']} "
              f"answer={case['trace_summary']['final_answer']!r}{detail}")

    if report["pass_rate"] < 0.8:
        print(f"\nCI gate: FAIL (pass rate {report['pass_rate']} < 0.80)")
    else:
        print(f"\nCI gate: PASS (pass rate {report['pass_rate']})")


if __name__ == "__main__":
    main()
