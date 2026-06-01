# Learning log — Code-execution agent (sandboxed)

What worked:
- Subprocess + `python -I` (isolated mode) + `setrlimit` is a surprisingly
  decent baseline. Stops infinite loops (CPU rlimit), stops memory bombs
  (AS rlimit), stops `os.environ.get("ANTHROPIC_API_KEY")` exfiltration
  (env allowlist). Took ~80 lines.
- The retry loop is short because the structure is obvious: run, on failure
  append the (assistant message + user "here's the traceback, fix it") pair
  and re-prompt. Most production agents I've read about look exactly like this.
- Stripping the code block with a single regex (`re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)`)
  is good enough for 99% of model outputs. Falling back to the raw text catches the rest.

What surprised me:
- `preexec_fn` is Unix-only. On Windows the `resource` module doesn't even
  import. A real production sandbox needs platform branching or to outsource
  isolation entirely (Docker, Firecracker, Vercel Sandbox).
- `subprocess.TimeoutExpired` doesn't always actually kill the child process
  promptly — orphaned processes after a timeout are a real risk. Need to send
  SIGKILL to the process group, not just the leader, for fork-bomb-style code.
- `python -I` skips `PYTHONPATH` AND user site-packages, which is great for
  isolation but means the sandbox can ONLY use stdlib. To allow numpy/pandas
  in the sandbox, build a curated venv and point `[sys.executable]` at it.

What I'd try next:
- Replace the subprocess sandbox with a Docker call: `docker run --rm --network=none
  --read-only --memory=256m --cpus=1 --cap-drop=ALL python:3.12-slim`. Much
  stronger isolation, ~200ms cold-start overhead.
- Add Vercel Sandbox as a second backend — same `Sandbox.run()` interface, just
  swap implementations. The microVM start time (claimed ~250ms) is competitive.
- Stream stdout back to the agent during execution instead of capture-then-show.
  Useful when the code is doing something long-running like training.
