# Learning log — Multi-step planner agent

What worked:
- Separating PLAN from EXECUTE was the key abstraction. The plan is a pure
  function of the task (cheap, one LLM call, no tools). Execution is a
  deterministic loop. This split also makes the plan inspectable and
  cacheable — same task, same plan, same trace.
- Scratchpad with `{{key}}` placeholders is a tiny pattern (`re.sub` + dict
  lookup) but it carries 80% of the value of fancier "state machine" agent
  frameworks. Each subtask declares an `output` key, later subtasks reference
  it. Result on `e1`: planner produced `expression: "68.0 / 84.0"` from
  the two prior lookups without an LLM in the loop.
- Adding a pseudo-tool `format_answer` for the final synthesis step let the
  plan stay declarative end-to-end — no "and then summarize" hand-waving.
- The head-to-head result on the 5 cases (single-shot 1/5 = 20%, planner 5/5 =
  100%) reproduces the published intuition: composite tasks separate the two
  approaches sharply. Single-step tasks (e3 capital-of-France) tie.

What surprised me:
- The naive single-shot agent isn't even close on composite tasks. Mean
  steps = 1.0 because it latched onto the first non-error tool result and
  stopped. Adding a "are we done?" check to single-shot would help, but
  the deeper problem is it has no concept of subgoals.
- Plan quality bottlenecks the whole system. If `planner_llm` misses a
  pattern (e.g. "ratio" with non-country words around it), execution does
  the wrong thing perfectly. Real systems mitigate this with (a) better
  plan prompts, (b) plan validation before execution, (c) re-plan on
  execution failure.
- The substitution string `"{{pop_a}} / {{pop_b}}"` becoming `"68.0 / 84.0"`
  works because the calculator accepts decimals — but if a lookup returned
  `unknown country: x`, substitution would yield a syntax-error expression
  and the calculator would refuse. Need a "guard" subtask between lookup
  and calculate, or per-step error propagation.

What I'd try next:
- Re-plan loop: if a tool returns an error mid-execution, ask the planner
  for a new plan that takes the failure into account. This is where
  LangGraph / pydantic-graph state machines start earning their keep.
- Cost compare: planner adds one extra LLM call (the plan) up front but
  avoids many wasted tool calls on composite tasks. Plot cost vs accuracy
  for both agents across N tasks.
- Plan validation step: parse the plan, check each `tool` exists, each
  `{{key}}` references something declared earlier, no cycles. Reject and
  re-plan if invalid. Catches most LLM hallucinations before execution.
- Compare against ReAct (interleaved thought-action-observation, no upfront
  plan). My guess: ReAct wins when intermediate results change the next
  action choice, planner wins when the structure is predictable upfront.
