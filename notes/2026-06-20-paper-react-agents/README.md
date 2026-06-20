# Paper notes — ReAct: Synergizing Reasoning and Acting in Language Models

**Goal:** Understand how interleaving reasoning traces with tool actions beats either alone; know when to use ReAct vs a plan-then-execute planner.

## Summary

ReAct (Yao et al., 2022) is a prompting strategy that makes a language model alternate between **Thought** steps (free-form reasoning in natural language) and **Act** steps (structured tool calls that return **Observation**s). The key insight is that pure Chain-of-Thought (CoT) hallucinates facts because it never checks them against the world, while pure action agents (Act-only) make poor multi-step plans because they never articulate why they're doing each step. ReAct fixes both: the Thought step lets the model reason about what it has observed and plan the next action; the Act step grounds that reasoning in real retrieved evidence. On HotpotQA (multi-hop factual reasoning) and Fever (fact verification), ReAct outperforms both CoT and Act-only baselines. The other major finding is **interpretability**: the Thought traces expose the model's internal plan, making errors easy to diagnose and correct with human feedback.

## Key ideas

### 1. The Thought–Act–Observe loop

Each step in a ReAct trajectory is one of three token types:

```
Thought t: ... (free-form reasoning — never executed, never seen by tools)
Action  t: tool_name[argument]
Obs     t: <result returned by the tool>
```

The model generates Thought and Action tokens autoregressively; Observation tokens are injected by the environment after the action runs. The loop repeats until the model emits a special `Finish[answer]` action.

### 2. Why CoT-only fails

CoT generates a reasoning chain but produces all tokens from the model's parametric knowledge — no external retrieval. On questions that require recent or niche facts ("Which city was the 1996 Olympic marathon held in?"), CoT confidently hallucinates. There is no feedback mechanism: a wrong intermediate step silently propagates.

### 3. Why Act-only fails

An Act-only agent issues tool calls without narrating its intent. On multi-hop tasks ("Who was the director of the 2003 film starring the actor who played Legolas?"), the agent often issues redundant searches or gets stuck in a loop because it has no way to track what sub-questions remain open.

### 4. ReAct's compound advantage

| Capability | CoT | Act-only | ReAct |
|---|:---:|:---:|:---:|
| External knowledge retrieval | ✗ | ✓ | ✓ |
| Recovers from wrong intermediate result | ✗ | ✗ | **✓** |
| Human-interpretable reasoning | ✓ | ✗ | **✓** |
| Multi-hop planning | ✓ (in-weights) | ✗ | **✓** |
| Works without scratchpad in context | ✓ | ✓ | ✗ |

### 5. Error recovery via re-thinking

When an Observation is unexpected (e.g. a search returns irrelevant results), the Thought step lets the model **detect the failure** and issue a different action. This is impossible in Act-only: the agent just picks the next action from history with no notion of "this result was wrong." In the paper's ablation, roughly 20% of ReAct's correct answers required at least one re-think after a bad observation.

### 6. Human-in-the-loop editing

Because Thoughts are readable text, a human can intercept the trajectory at any Thought step and correct it ("Actually, you need to search for the *director*, not the *actor*"). The paper shows this "ReAct + 1 human edit" variant closes almost all remaining errors on Fever.

## ReAct vs Plan-then-Execute (from 2026-06-07)

Both are multi-step agent patterns. They differ in when planning happens:

| | ReAct | Planner (2026-06-07) |
|---|---|---|
| **When plan is made** | Interleaved — one step at a time | Upfront — full plan before any action |
| **Plan can change mid-run** | Yes (re-think after bad obs) | No (plan is locked; re-plan on error is a separate loop) |
| **Visibility into sub-goals** | Implicit in Thought traces | Explicit structured subtask list |
| **Best for** | Open-ended tasks where intermediate results change the next step | Predictable composite tasks where structure is known upfront |
| **LLM calls** | 1 per step (Thought + Action generated together) | 1 up front (plan) + 0 per step (execution is deterministic) |

Rule of thumb: **use ReAct when the result of step N changes what step N+1 should be**. Use a planner when the full structure is predictable and you want cost-efficient execution.

## Worked example — multi-hop question

Task: *"What is the ratio of the population of the country that hosts the Eiffel Tower to the country where BERT was published?"*

**CoT-only (fails):**
```
Thought: The Eiffel Tower is in France. BERT was published by Google, which is in the US.
Answer: 68 / 331 ≈ 0.21
```
Possibly correct here, but relies entirely on parametric knowledge. If the question were about a less-famous landmark, CoT would hallucinate.

**Act-only (struggles):**
```
Action: search["Eiffel Tower country"]  → "France"
Action: search["BERT published country"] → "United States"
Action: lookup_population["France"]      → 68.0
Action: calculate["68 / 331"]           → 0.205
```
Gets the right answer but has no way to detect if `search["BERT published country"]` returned noise. Also, it doesn't know it needs to look up population until after searching — it may issue a redundant search first.

**ReAct (correct + interpretable):**
```
Thought 1: I need two facts: (a) country of Eiffel Tower, (b) country where BERT was created.
           Let me search for (a) first.
Action  1: search["Eiffel Tower location"]
Obs     1: "The Eiffel Tower is located in Paris, France."

Thought 2: France confirmed. Now find BERT's origin.
Action  2: search["BERT language model origin"]
Obs     2: "BERT was published by Google Brain in 2018."

Thought 3: Google Brain is a US organisation. Now I have France and US. Look up populations.
Action  3: lookup_population["France"]
Obs     3: 68.0

Action  4: lookup_population["United States"]
Obs     4: 331.0

Thought 4: Both populations retrieved. Compute the ratio.
Action  5: calculate["68 / 331"]
Obs     5: 0.2054…

Thought 5: Done. ratio ≈ 0.21
Finish["ratio = 0.205"]
```

The Thought steps make each decision auditable. If Obs 2 had returned a wrong country, Thought 3 would have caught the inconsistency and issued a corrective search.

## Open questions

- **When does ReAct hurt vs help?** On simple factual questions (single-hop), the extra Thought tokens waste context and latency. Is there a cheap classifier that routes "easy" questions to direct answer and "hard" ones to ReAct?
- **Thought quality bottleneck.** ReAct is only as good as the LLM's ability to reason in its Thought steps. Small models produce low-quality Thoughts and often loop. Is there a minimum model size below which ReAct degrades to Act-only?
- **Comparison with self-consistency + CoT.** Self-consistency (sample K CoT chains, take the majority answer) is another strong CoT baseline. The ReAct paper doesn't compare against it. On tasks where hallucination is rare, does self-consistency match ReAct's accuracy at lower latency?
- **Memory.** In the demo (see `react_demo.py`) the full Thought–Act–Observe history stays in the context window. For long tasks this blows the context limit. MemGPT / hierarchical context management is one solution — but how much recall degrades when old observations are compressed?
- **ReAct + retrieval-augmented generation (RAG).** Each Action in ReAct can be a RAG lookup. Does the Thought step help RAG decide *which* query to issue, or is the gain already captured by good query rewriting?

## See also

- `react_demo.py` — ReAct, CoT-only, and Act-only agents running head-to-head on 5 eval tasks. Deterministic (no API key). Shows re-think recovery on the task where Act-only loops.
- `agents/2026-06-07-multi-step-planner-agent/` — planner vs single-shot; complement to this.

## References

- Yao et al. 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models* — https://arxiv.org/abs/2210.03629
- Wei et al. 2022 — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* — https://arxiv.org/abs/2201.11903
- Yao et al. 2023 — *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* — https://arxiv.org/abs/2305.10601
- Shinn et al. 2023 — *Reflexion: Language Agents with Verbal Reinforcement Learning* — https://arxiv.org/abs/2303.11366
