"""ReAct vs CoT-only vs Act-only — head-to-head on 5 eval tasks, no API key.

What this demonstrates (see README.md for the paper summary):
    ReAct interleaves Thought → Action → Observation steps.
    CoT-only reasons without tools (parametric knowledge only).
    Act-only calls tools without articulating why (no Thought steps).

    The eval surfaces three failure modes:
      1. CoT hallucination on niche facts  — ReAct wins (grounds in obs)
      2. Act-only looping on multi-hop     — ReAct wins (re-thinks after bad obs)
      3. Simple one-hop queries            — all three tie (Thought is wasted overhead)

    All LLMs are scripted (deterministic), so the whole file runs without network
    or API keys. Replace react_llm / cot_llm / act_llm with real LLM calls to
    reproduce the paper's HotpotQA results.

Run:
    python react_demo.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Tool registry (same stub design as planner-agent, 2026-06-07)
# ─────────────────────────────────────────────────────────────────────────────

POPULATION_M: dict[str, float] = {
    "france": 68.0, "germany": 84.0, "italy": 59.0,
    "spain": 48.0, "uk": 67.0, "united states": 331.0,
}
FACTS: dict[str, str] = {
    "capital of france":          "Paris",
    "capital of germany":         "Berlin",
    "eiffel tower location":      "The Eiffel Tower is located in Paris, France.",
    "eiffel tower country":       "France",
    "bert language model origin": "BERT was published by Google Brain in 2018.",
    "bert published country":     "United States",
    "founder of openai":          "Sam Altman, Greg Brockman, Ilya Sutskever, and others",
    "year bert published":        "2018",
    "attention is all you need":  "Transformer architecture paper published by Google in 2017.",
    "hotpotqa dataset":           "HotpotQA is a multi-hop QA dataset from CMU and Stanford.",
}


def tool_search(args: dict[str, Any]) -> str:
    q = (args.get("query") or "").lower().strip(" ?.")
    for k, v in FACTS.items():
        if k in q or q in k:
            return v
    return "No result found."


def tool_lookup_population(args: dict[str, Any]) -> float | str:
    c = (args.get("country") or "").lower().strip()
    return POPULATION_M.get(c, f"unknown country: {c}")


def tool_calculate(args: dict[str, Any]) -> float | str:
    expr = str(args.get("expression", ""))
    safe = re.fullmatch(r"[\d+\-*/().\s]+", expr)
    if not safe:
        return f"refused: unsafe expression {expr!r}"
    try:
        return round(eval(expr, {"__builtins__": {}}, {}), 4)  # noqa: S307
    except Exception as e:
        return f"error: {e}"


TOOLS = {
    "search":            tool_search,
    "lookup_population": tool_lookup_population,
    "calculate":         tool_calculate,
}


def run_tool(name: str, args: dict[str, Any]) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"[unknown tool: {name}]"
    result = fn(args)
    return str(result)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    kind: str       # "thought" | "action" | "observation" | "answer"
    text: str


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    succeeded: bool | None = None

    def add(self, kind: str, text: str) -> None:
        self.steps.append(Step(kind=kind, text=text))

    def thought_count(self) -> int:
        return sum(1 for s in self.steps if s.kind == "thought")

    def action_count(self) -> int:
        return sum(1 for s in self.steps if s.kind == "action")


# ─────────────────────────────────────────────────────────────────────────────
# ReAct agent
# ─────────────────────────────────────────────────────────────────────────────

# A "scripted LLM" returns either:
#   {"thought": "...", "action": {"tool": "...", "args": {...}}}   — continue
#   {"thought": "...", "finish": "answer text"}                   — done
ReactDecision = dict[str, Any]


def react_llm(task: str, history: list[Step]) -> ReactDecision:
    """Deterministic ReAct policy — encodes enough patterns to pass all 5 tasks."""
    t = task.lower()
    observations = [s.text for s in history if s.kind == "observation"]
    actions_done = [s.text for s in history if s.kind == "action"]
    n_obs = len(observations)

    # ── Task e1: ratio of populations ────────────────────────────────────────
    if "ratio" in t and "population" in t:
        countries = re.findall(r"\b(france|germany|italy|spain|uk|united states)\b", t)
        if len(countries) >= 2:
            a, b = countries[0], countries[1]
            if n_obs == 0:
                return {"thought": f"I need population of {a} and {b}. Look up {a} first.",
                        "action": {"tool": "lookup_population", "args": {"country": a}}}
            if n_obs == 1:
                pop_a = observations[0]
                return {"thought": f"Got {a}={pop_a}. Now look up {b}.",
                        "action": {"tool": "lookup_population", "args": {"country": b}}}
            if n_obs == 2:
                pop_a, pop_b = observations[0], observations[1]
                return {"thought": f"Both populations retrieved. Compute {pop_a} / {pop_b}.",
                        "action": {"tool": "calculate", "args": {"expression": f"{pop_a} / {pop_b}"}}}
            if n_obs == 3:
                return {"thought": f"Ratio computed: {observations[2]}.",
                        "finish": f"ratio = {observations[2]}"}

    # ── Task e2: multi-hop (Eiffel Tower country → BERT country → population ratio) ─
    if "eiffel" in t or ("tower" in t and "bert" in t):
        if n_obs == 0:
            return {"thought": "Need country for Eiffel Tower. Search for it.",
                    "action": {"tool": "search", "args": {"query": "eiffel tower country"}}}
        if n_obs == 1:
            country_a = observations[0].lower()
            if "france" in country_a:
                return {"thought": "Eiffel Tower is in France. Now find BERT's origin country.",
                        "action": {"tool": "search", "args": {"query": "bert published country"}}}
            # Re-think: bad observation
            return {"thought": f"Unexpected result: {observations[0]!r}. Try a different query.",
                    "action": {"tool": "search", "args": {"query": "eiffel tower location"}}}
        if n_obs == 2:
            # Extract country from BERT observation
            country_b = "United States" if "united states" in observations[1].lower() else "unknown"
            return {"thought": f"BERT country = {country_b}. Look up France population.",
                    "action": {"tool": "lookup_population", "args": {"country": "france"}}}
        if n_obs == 3:
            return {"thought": "Got France pop. Now US population.",
                    "action": {"tool": "lookup_population", "args": {"country": "united states"}}}
        if n_obs == 4:
            pop_a, pop_b = observations[2], observations[3]
            return {"thought": f"Compute {pop_a} / {pop_b}.",
                    "action": {"tool": "calculate", "args": {"expression": f"{pop_a} / {pop_b}"}}}
        if n_obs == 5:
            return {"thought": f"Done. ratio ≈ {observations[4]}.",
                    "finish": f"ratio France/US population = {observations[4]}"}

    # ── Task e3: simple fact lookup ───────────────────────────────────────────
    if "capital" in t:
        country = re.search(r"capital of (\w+)", t)
        if country:
            c = country.group(1)
            if n_obs == 0:
                return {"thought": f"Look up capital of {c}.",
                        "action": {"tool": "search", "args": {"query": f"capital of {c}"}}}
            return {"thought": f"Answer is {observations[0]}.", "finish": observations[0]}

    # ── Task e4: fact + year ──────────────────────────────────────────────────
    if "bert" in t and ("year" in t or "when" in t or "published" in t):
        if n_obs == 0:
            return {"thought": "Search for when BERT was published.",
                    "action": {"tool": "search", "args": {"query": "year bert published"}}}
        return {"thought": f"BERT was published in {observations[0]}.",
                "finish": observations[0]}

    # ── Task e5: founder lookup ───────────────────────────────────────────────
    if "founder" in t or "openai" in t:
        if n_obs == 0:
            return {"thought": "Search for OpenAI founders.",
                    "action": {"tool": "search", "args": {"query": "founder of openai"}}}
        return {"thought": f"Got founders: {observations[0]}.", "finish": observations[0]}

    # Fallback
    return {"thought": "I don't know how to solve this.", "finish": "unknown"}


class ReActAgent:
    """Interleaves Thought → Action → Observation until Finish."""

    def __init__(self, llm_fn: Callable, max_steps: int = 10) -> None:
        self.llm_fn = llm_fn
        self.max_steps = max_steps

    def run(self, task: str) -> Trajectory:
        traj = Trajectory(task=task)
        for _ in range(self.max_steps):
            decision = self.llm_fn(task, traj.steps)
            thought = decision.get("thought", "")
            if thought:
                traj.add("thought", thought)
            if "finish" in decision:
                traj.final_answer = decision["finish"]
                traj.add("answer", traj.final_answer)
                return traj
            action = decision.get("action", {})
            tool_str = f"{action.get('tool')}[{action.get('args')}]"
            traj.add("action", tool_str)
            obs = run_tool(action["tool"], action.get("args", {}))
            traj.add("observation", obs)
        return traj


# ─────────────────────────────────────────────────────────────────────────────
# CoT-only agent (reasons but cannot call tools)
# ─────────────────────────────────────────────────────────────────────────────

def cot_llm(task: str) -> dict[str, str]:
    """Parametric knowledge only — no tool calls. Hallucination risk on niche facts."""
    t = task.lower()
    if "ratio" in t and "population" in t:
        countries = re.findall(r"\b(france|germany|italy|spain|uk|united states)\b", t)
        if len(countries) >= 2:
            # Simulate plausible parametric recall (correct here, but unreliable in general)
            pop = {"france": 68.0, "germany": 84.0, "italy": 59.0,
                   "spain": 48.0, "uk": 67.0, "united states": 331.0}
            a, b = countries[0], countries[1]
            pa, pb = pop.get(a, 70.0), pop.get(b, 80.0)
            return {"thought": f"From memory: {a}≈{pa}M, {b}≈{pb}M. Ratio≈{pa/pb:.3f}.",
                    "answer": f"ratio ≈ {round(pa/pb, 3)}"}
    if "eiffel" in t or ("tower" in t and "bert" in t):
        # CoT hallucinates: guesses BERT is from the UK
        return {"thought": "Eiffel Tower is in France. BERT is from Google DeepMind (UK). "
                           "France≈68M, UK≈67M, ratio≈1.01.",
                "answer": "ratio ≈ 1.015  [WRONG — BERT is US, not UK]"}
    if "capital" in t:
        country = re.search(r"capital of (\w+)", t)
        if country:
            caps = {"france": "Paris", "germany": "Berlin"}
            c = country.group(1).lower()
            ans = caps.get(c, f"[unknown capital of {c}]")
            return {"thought": f"Capital of {c} is {ans}.", "answer": ans}
    if "bert" in t and ("year" in t or "published" in t):
        return {"thought": "BERT was published around 2018.", "answer": "2018"}
    if "founder" in t or "openai" in t:
        return {"thought": "OpenAI was co-founded by Sam Altman, Greg Brockman, Ilya Sutskever.",
                "answer": "Sam Altman, Greg Brockman, Ilya Sutskever"}
    return {"thought": "Unknown.", "answer": "unknown"}


class CoTAgent:
    def run(self, task: str) -> Trajectory:
        result = cot_llm(task)
        traj = Trajectory(task=task)
        traj.add("thought", result["thought"])
        traj.final_answer = result["answer"]
        traj.add("answer", traj.final_answer)
        return traj


# ─────────────────────────────────────────────────────────────────────────────
# Act-only agent (calls tools without Thought steps)
# ─────────────────────────────────────────────────────────────────────────────

def act_llm(task: str, history: list[Step]) -> dict[str, Any]:
    """No Thought steps — just pattern-matches task to tool calls."""
    t = task.lower()
    actions_done = [s.text for s in history if s.kind == "action"]
    observations = [s.text for s in history if s.kind == "observation"]
    n_obs = len(observations)

    if "ratio" in t and "population" in t:
        countries = re.findall(r"\b(france|germany|italy|spain|uk|united states)\b", t)
        if len(countries) >= 2:
            a, b = countries[0], countries[1]
            if n_obs == 0:
                return {"action": {"tool": "lookup_population", "args": {"country": a}}}
            if n_obs == 1:
                return {"action": {"tool": "lookup_population", "args": {"country": b}}}
            if n_obs == 2:
                return {"action": {"tool": "calculate",
                                   "args": {"expression": f"{observations[0]} / {observations[1]}"}}}
            return {"finish": f"ratio = {observations[2]}"}

    # Multi-hop: Act-only guesses wrong on BERT origin and loops
    if "eiffel" in t or ("tower" in t and "bert" in t):
        if n_obs == 0:
            return {"action": {"tool": "search", "args": {"query": "eiffel tower country"}}}
        if n_obs == 1:
            # No thought step → doesn't realise it needs to confirm BERT origin; guesses
            return {"action": {"tool": "search", "args": {"query": "bert language model origin"}}}
        if n_obs == 2:
            # Doesn't parse "Google Brain" → "United States"; issues wrong country
            return {"action": {"tool": "lookup_population", "args": {"country": "france"}}}
        if n_obs == 3:
            # Blindly uses "United Kingdom" (no reasoning to correct it)
            return {"action": {"tool": "lookup_population", "args": {"country": "uk"}}}
        if n_obs == 4:
            return {"action": {"tool": "calculate",
                               "args": {"expression": f"{observations[2]} / {observations[3]}"}}}
        return {"finish": f"ratio = {observations[4]}  [WRONG — used UK instead of US]"}

    if "capital" in t:
        country = re.search(r"capital of (\w+)", t)
        if country:
            if n_obs == 0:
                return {"action": {"tool": "search",
                                   "args": {"query": f"capital of {country.group(1)}"}}}
            return {"finish": observations[0]}

    if "bert" in t and ("year" in t or "published" in t):
        if n_obs == 0:
            return {"action": {"tool": "search", "args": {"query": "year bert published"}}}
        return {"finish": observations[0]}

    if "founder" in t or "openai" in t:
        if n_obs == 0:
            return {"action": {"tool": "search", "args": {"query": "founder of openai"}}}
        return {"finish": observations[0]}

    return {"finish": "unknown"}


class ActOnlyAgent:
    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    def run(self, task: str) -> Trajectory:
        traj = Trajectory(task=task)
        for _ in range(self.max_steps):
            decision = act_llm(task, traj.steps)
            if "finish" in decision:
                traj.final_answer = decision["finish"]
                traj.add("answer", traj.final_answer)
                return traj
            action = decision["action"]
            tool_str = f"{action['tool']}[{action.get('args')}]"
            traj.add("action", tool_str)
            obs = run_tool(action["tool"], action.get("args", {}))
            traj.add("observation", obs)
        return traj


# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    task: str
    expected: list[str]   # substrings that must appear in the answer
    note: str = ""        # what failure mode this tests


CASES: list[EvalCase] = [
    EvalCase("e1", "What is the ratio of the population of France to Germany?",
             ["0.809", "0.81"],
             note="multi-step, clean numbers — all should pass"),
    EvalCase("e2", "What is the ratio of the population of the country where the Eiffel Tower "
             "is located to the country where BERT was published?",
             ["0.205", "0.206"],
             note="multi-hop: CoT hallucinates BERT=UK; Act-only uses wrong country"),
    EvalCase("e3", "What is the capital of France?",
             ["Paris"],
             note="single-hop — all should pass (Thought is wasted overhead here)"),
    EvalCase("e4", "In what year was BERT published?",
             ["2018"],
             note="single-hop factual — all pass"),
    EvalCase("e5", "Who founded OpenAI?",
             ["Altman", "Brockman", "Sutskever"],
             note="multi-key fact — tests partial match"),
]


def score(ans: str | None, expected: list[str]) -> bool:
    if ans is None:
        return False
    return any(e.lower() in ans.lower() for e in expected)


def print_traj(traj: Trajectory) -> None:
    for s in traj.steps:
        prefix = {"thought": "  Thought  :", "action": "  Action   :",
                  "observation": "  Obs      :", "answer": "  Answer   :"}.get(s.kind, "  ?:")
        print(f"{prefix} {s.text[:90]}")


def main() -> None:
    react  = ReActAgent(react_llm, max_steps=10)
    cot    = CoTAgent()
    act    = ActOnlyAgent(max_steps=8)

    react_trajs = [react.run(c.task) for c in CASES]
    cot_trajs   = [cot.run(c.task)   for c in CASES]
    act_trajs   = [act.run(c.task)   for c in CASES]

    for traj, case in zip(react_trajs, CASES):
        traj.succeeded = score(traj.final_answer, case.expected)
    for traj, case in zip(cot_trajs, CASES):
        traj.succeeded = score(traj.final_answer, case.expected)
    for traj, case in zip(act_trajs, CASES):
        traj.succeeded = score(traj.final_answer, case.expected)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"{'id':<4}{'task':<45}{'ReAct':<8}{'CoT':<8}{'Act':<8}")
    print("-" * 72)
    for i, case in enumerate(CASES):
        r = "PASS" if react_trajs[i].succeeded else "FAIL"
        c = "PASS" if cot_trajs[i].succeeded else "FAIL"
        a = "PASS" if act_trajs[i].succeeded else "FAIL"
        print(f"{case.id:<4}{case.task[:43]:<45}{r:<8}{c:<8}{a:<8}")
    print()
    r_pass = sum(t.succeeded for t in react_trajs)
    c_pass = sum(t.succeeded for t in cot_trajs)
    a_pass = sum(t.succeeded for t in act_trajs)
    print(f"{'TOTAL':<4}{'':<45}{r_pass}/5{'':<5}{c_pass}/5{'':<5}{a_pass}/5")

    # ── Avg steps ─────────────────────────────────────────────────────────────
    print()
    print(f"Avg actions/query  ReAct={sum(t.action_count() for t in react_trajs)/5:.1f}"
          f"  Act={sum(t.action_count() for t in act_trajs)/5:.1f}")
    print(f"Avg thoughts/query ReAct={sum(t.thought_count() for t in react_trajs)/5:.1f}"
          f"  CoT={sum(t.thought_count() for t in cot_trajs)/5:.1f}")

    # ── Show the interesting failure: e2 traces ───────────────────────────────
    print("\n" + "=" * 72)
    print(f"TRACE: task e2 (multi-hop, Eiffel Tower + BERT) — the key failure case")
    print("=" * 72)
    print("\n--- CoT-only ---")
    print_traj(cot_trajs[1])
    print("\n--- Act-only ---")
    print_traj(act_trajs[1])
    print("\n--- ReAct ---")
    print_traj(react_trajs[1])


if __name__ == "__main__":
    main()
