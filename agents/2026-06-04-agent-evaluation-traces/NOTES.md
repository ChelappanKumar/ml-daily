# Learning log — Agent evaluation with traces

What worked:
- Splitting the score into 4 independent checks (`called_required_tools`,
  `avoided_forbidden_tools`, `answer_contains_expected`, `within_step_budget`)
  meant the failing dimension was always obvious. The aggregate per-check
  pass-rate is the single most useful number — it tells you whether the
  agent is mostly failing on routing, on the answer itself, or on cost.
- Recording every event (LLM call + tool call + final answer + error) with a
  timestamp + duration_ms gave latency and step-count metrics basically for
  free. The 4 checks all read off the same `Trace` object.
- The "stop iterating once a tool returns a useful result" check needed to
  exclude `refused:` and `error:` strings — otherwise the agent latches onto a
  failed tool result as its answer and never tries another tool. Caught this
  in the first eval run when 5/10 cases failed identically.

What surprised me:
- Routing on keyword presence (`"what is " in task`) overshoots badly.
  "What is the capital of France?" looked like a calculator task. Fix was
  to require an actual numeric expression in the prompt before picking the
  calculator. Lesson: in real agents, prefer model-level tool selection over
  hand-rolled regex routing — but if you DO hand-roll, gate on hard signals
  (digits, operators), not soft ones (English question stems).
- The arithmetic-extractor regex took two passes to get right. First
  version `[-+]?\d+(\s*[\d+\-*/().\s]+)?` started matching at the first digit
  and missed leading parens, so `(12 + 8) * 4` became `12 + 8) * 4` and
  failed eval. Switched to "find longest run of arithmetic chars containing
  both a digit and an operator." Got 10/10.

What I'd try next:
- Add a `must_call_tools_in_order` check for multi-step tasks (search, then
  calculate). Trajectory order matters once tasks get composite.
- Hook this into a real LLM (Anthropic / OpenAI) and re-run the same cases.
  The expectation is that the keyword-routing failure modes disappear but
  new ones show up — wrong arg parsing, made-up tool names, repeated calls.
- Save traces as JSONL and load them into a notebook. Eyeballing 10 traces
  in stdout is fine; eyeballing 1,000 isn't. The right shape is "JSONL on
  disk + a small viewer."
