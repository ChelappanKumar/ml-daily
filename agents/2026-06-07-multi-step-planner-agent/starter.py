"""Multi-step planner agent vs. single-shot agent — head-to-head on the same tasks.

Problem this solves:
    Composite tasks like "search for the population of France, then search for
    Germany, then compute the ratio" break a naive single-shot agent: it tries
    to do everything in one tool call and gets confused, or it grabs one fact
    and stops. The planner-then-execute pattern fixes this by separating
    *what* to do (a plan) from *how* to do each step (tool calls). The plan
    is a cheap LLM call producing structured subtasks; execution is then a
    deterministic loop over those subtasks.

What this file does:
    1. Tool registry (search_facts, calculate, lookup_population) — deterministic
       stubs so the whole script runs without API keys.
    2. SingleShotAgent — one LLM call per step, no planning. Tries to solve in
       <= max_steps tool calls.
    3. PlannerAgent — two phases: (a) PLAN: LLM writes an ordered list of
       subtasks with declared tool + args, (b) EXECUTE: run each subtask,
       store its result in a scratchpad, the final subtask reads scratchpad
       results and produces the answer.
    4. 5 composite eval tasks scored on correctness.
    5. Side-by-side comparison table at the end.

Plug a real LLM:
    Replace `single_shot_llm` and `planner_llm` with functions that call
    Anthropic / OpenAI. Both functions are pure — given a task (and current
    context), return a structured decision. Easy to swap.

Run:
    python starter.py
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


# ---------- tool registry ----------

ToolFn = Callable[[dict[str, Any]], Any]


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn


# Stubbed knowledge base. Real version would call out to search APIs / Wikidata.
POPULATION_M = {"france": 68.0, "germany": 84.0, "italy": 59.0, "spain": 48.0, "uk": 67.0}
FACTS = {
    "capital of france": "Paris",
    "capital of germany": "Berlin",
    "founder of openai": "Sam Altman, Greg Brockman, Ilya Sutskever, Elon Musk, and others",
    "year bert published": "2018",
}


def _t_search_facts(args: dict[str, Any]) -> str:
    q = (args.get("query") or "").lower().strip(" ?.")
    for k, v in FACTS.items():
        if k in q:
            return v
    return "No fact found."


def _t_lookup_population(args: dict[str, Any]) -> float | str:
    country = (args.get("country") or "").lower().strip()
    if country in POPULATION_M:
        return POPULATION_M[country]
    return f"unknown country: {country}"


def _t_calculate(args: dict[str, Any]) -> float | str:
    expr = str(args.get("expression", ""))
    if not expr or any(c not in "0123456789+-*/(). " for c in expr):
        return f"refused: unsafe characters in {expr!r}"
    try:
        return eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as e:
        return f"error: {e}"


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_facts", "Look up a fact. args: {query: str}", _t_search_facts),
    Tool("lookup_population", "Get a country's population in millions. args: {country: str}", _t_lookup_population),
    Tool("calculate", "Evaluate an arithmetic expression. args: {expression: str}", _t_calculate),
]}


# ---------- shared types ----------

@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class AgentRun:
    task: str
    plan: list[dict[str, Any]] = field(default_factory=list)  # PlannerAgent only
    calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    succeeded: bool | None = None  # filled by evaluator


def run_tool(name: str, args: dict[str, Any]) -> ToolCall:
    if name not in TOOLS:
        return ToolCall(tool=name, args=args, error=f"unknown tool {name!r}")
    try:
        result = TOOLS[name].fn(args)
        return ToolCall(tool=name, args=args, result=result)
    except Exception as e:
        return ToolCall(tool=name, args=args, error=str(e))


# ---------- single-shot agent ----------

class SingleShotAgent:
    """Naive baseline: one LLM call per step, no planning, no scratchpad.

    Stops on the first non-error tool result or after max_steps.
    """

    def __init__(self, llm_fn: Callable[[str, list[ToolCall]], dict[str, Any]], max_steps: int = 3):
        self.llm_fn = llm_fn
        self.max_steps = max_steps

    def run(self, task: str) -> AgentRun:
        run = AgentRun(task=task)
        for _ in range(self.max_steps):
            decision = self.llm_fn(task, run.calls)
            if "final_answer" in decision:
                run.final_answer = str(decision["final_answer"])
                return run
            call = run_tool(decision["tool"], decision.get("args", {}))
            run.calls.append(call)
            # Naive: first non-error result becomes the answer.
            if call.error is None and call.result not in (None, "No fact found.")                     and not (isinstance(call.result, str) and call.result.startswith(("refused:", "error:", "unknown"))):
                run.final_answer = str(call.result)
                return run
        return run


# ---------- planner agent ----------

class PlannerAgent:
    """Plan-then-execute:

      1. PLAN  — LLM produces an ordered list of subtasks. Each subtask
         declares its tool, args, and an output key that goes into the
         scratchpad (so later subtasks can reference earlier results).
      2. EXECUTE — iterate the plan. For each subtask, substitute scratchpad
         placeholders into args, run the tool, store the result under the
         output key. The last subtask uses `final: true` to set the answer.
    """

    def __init__(
        self,
        plan_fn: Callable[[str], list[dict[str, Any]]],
        max_subtasks: int = 6,
    ):
        self.plan_fn = plan_fn
        self.max_subtasks = max_subtasks

    def run(self, task: str) -> AgentRun:
        run = AgentRun(task=task)
        plan = self.plan_fn(task)[: self.max_subtasks]
        run.plan = plan
        scratchpad: dict[str, Any] = {}

        for step in plan:
            args = {k: self._substitute(v, scratchpad) for k, v in step.get("args", {}).items()}
            tool_name = step.get("tool")
            if tool_name == "format_answer":
                # Pseudo-tool: build the final answer from scratchpad values.
                run.final_answer = str(args.get("answer"))
                continue
            call = run_tool(tool_name, args)
            run.calls.append(call)
            out_key = step.get("output")
            if out_key:
                scratchpad[out_key] = call.result if call.error is None else None

        # If the plan never declared a format_answer step, fall back to the last
        # tool result. Real production would force a final synthesis step.
        if run.final_answer is None and run.calls:
            run.final_answer = str(run.calls[-1].result)
        return run

    @staticmethod
    def _substitute(value: Any, scratchpad: dict[str, Any]) -> Any:
        """Replace `{{key}}` placeholders in string args with scratchpad values."""
        if not isinstance(value, str):
            return value
        def repl(m: re.Match) -> str:
            key = m.group(1).strip()
            v = scratchpad.get(key)
            return str(v) if v is not None else m.group(0)
        return re.sub(r"\{\{([^}]+)\}\}", repl, value)


# ---------- scripted "LLMs" (deterministic, no API key) ----------

_FRANCE_GERMANY = re.compile(r"france.*germany|germany.*france", re.IGNORECASE)


def single_shot_llm(task: str, prior: list[ToolCall]) -> dict[str, Any]:
    """The naive routing strategy: grab the first thing that looks relevant.

    Deliberately weak on composite tasks so the planner wins the comparison.
    """
    t = task.lower()
    seen = [c.tool for c in prior]

    # If we have a non-error result already, summarize it.
    last = next((c for c in reversed(prior) if c.error is None), None)
    if last and last.result not in (None, "No fact found.")             and not (isinstance(last.result, str) and last.result.startswith(("refused:", "error:", "unknown"))):
        return {"final_answer": str(last.result)}

    if "population" in t and "lookup_population" not in seen:
        # Single-shot picks ONE country — the first one it sees.
        for country in POPULATION_M:
            if country in t:
                return {"tool": "lookup_population", "args": {"country": country}}
    if any(op in t for op in ("+", "-", "*", "/", "ratio", "sum", "compute")) and "calculate" not in seen:
        # Strips out non-arithmetic chars — usually too little context to solve.
        expr = "".join(c for c in t if c in "0123456789+-*/(). ").strip()
        return {"tool": "calculate", "args": {"expression": expr or "0"}}
    if "search_facts" not in seen:
        return {"tool": "search_facts", "args": {"query": task}}
    return {"final_answer": "I could not solve this in one shot."}


def planner_llm(task: str) -> list[dict[str, Any]]:
    """The planner: writes a structured multi-step plan.

    Pattern recognition kept simple so the file is deterministic. In a real
    system, this is a single LLM call returning JSON.
    """
    t = task.lower()

    # Ratio-of-populations pattern.
    m = re.search(r"ratio.*population.*?(\w+).*?(\w+)", t) or _FRANCE_GERMANY.search(t)
    if m and "population" in t and "ratio" in t:
        countries = re.findall(r"\b(france|germany|italy|spain|uk)\b", t)
        if len(countries) >= 2:
            a, b = countries[0], countries[1]
            return [
                {"tool": "lookup_population", "args": {"country": a}, "output": "pop_a"},
                {"tool": "lookup_population", "args": {"country": b}, "output": "pop_b"},
                {"tool": "calculate", "args": {"expression": "{{pop_a}} / {{pop_b}}"}, "output": "ratio"},
                {"tool": "format_answer", "args": {"answer": "ratio = {{ratio}}"}},
            ]

    # Sum-of-populations pattern.
    if "population" in t and ("sum" in t or "total" in t or "combined" in t):
        countries = re.findall(r"\b(france|germany|italy|spain|uk)\b", t)
        if len(countries) >= 2:
            steps = []
            keys = []
            for i, c in enumerate(countries[:3]):
                key = f"pop_{i}"
                steps.append({"tool": "lookup_population", "args": {"country": c}, "output": key})
                keys.append("{{" + key + "}}")
            expr = " + ".join(keys)
            steps.append({"tool": "calculate", "args": {"expression": expr}, "output": "total"})
            steps.append({"tool": "format_answer", "args": {"answer": "total = {{total}} million"}})
            return steps

    # Fact-then-confirm pattern: look up a fact, then format it.
    return [
        {"tool": "search_facts", "args": {"query": task}, "output": "fact"},
        {"tool": "format_answer", "args": {"answer": "{{fact}}"}},
    ]


# ---------- eval ----------

@dataclass
class EvalCase:
    id: str
    task: str
    expected_substrings: list[str]


CASES: list[EvalCase] = [
    EvalCase("e1", "What is the ratio of population of France to Germany?", ["0.8"]),  # 68/84 ≈ 0.809
    EvalCase("e2", "What is the sum of populations of France, Germany, and Italy?", ["211"]),  # 68+84+59
    EvalCase("e3", "What is the capital of France?", ["Paris"]),
    EvalCase("e4", "What is the ratio of population of UK to Spain?", ["1.39"]),  # 67/48 ≈ 1.396
    EvalCase("e5", "What is the sum of populations of Spain, UK, and Italy?", ["174"]),  # 48+67+59
]


def score(case: EvalCase, run: AgentRun) -> bool:
    if run.final_answer is None:
        return False
    ans = run.final_answer.lower()
    return all(sub.lower() in ans for sub in case.expected_substrings)


def evaluate(name: str, runs: list[AgentRun]) -> dict[str, Any]:
    passed = sum(r.succeeded for r in runs)
    return {
        "agent": name,
        "pass_rate": round(passed / len(runs), 4),
        "passed": passed,
        "total": len(runs),
        "mean_steps": round(sum(len(r.calls) for r in runs) / len(runs), 2),
    }


def main() -> None:
    single = SingleShotAgent(single_shot_llm, max_steps=3)
    planner = PlannerAgent(planner_llm, max_subtasks=6)

    single_runs: list[AgentRun] = []
    planner_runs: list[AgentRun] = []
    for case in CASES:
        sr = single.run(case.task)
        sr.succeeded = score(case, sr)
        single_runs.append(sr)

        pr = planner.run(case.task)
        pr.succeeded = score(case, pr)
        planner_runs.append(pr)

    print("=" * 72)
    print("Per-case comparison")
    print("=" * 72)
    print(f"{'id':<5}{'task':<55}{'single':<10}{'planner':<10}")
    for case, sr, pr in zip(CASES, single_runs, planner_runs):
        s_mark = "PASS" if sr.succeeded else "FAIL"
        p_mark = "PASS" if pr.succeeded else "FAIL"
        print(f"{case.id:<5}{case.task[:53]:<55}{s_mark:<10}{p_mark:<10}")

    print("\n" + "=" * 72)
    print("Aggregate")
    print("=" * 72)
    print(json.dumps(evaluate("single_shot", single_runs), indent=2))
    print(json.dumps(evaluate("planner", planner_runs), indent=2))

    # One example plan, expanded — so the trace is visible.
    print("\n" + "=" * 72)
    print(f"Example plan for: {CASES[0].task}")
    print("=" * 72)
    print(json.dumps(planner_runs[0].plan, indent=2))
    print(f"\nfinal_answer: {planner_runs[0].final_answer!r}")
    print(f"calls: {[asdict(c) for c in planner_runs[0].calls]}")


if __name__ == "__main__":
    main()
