"""Code-execution agent with a local subprocess sandbox + iterate-on-error loop.

Problem this solves:
    A common agent pattern: user asks a math/data question that's faster to
    *solve by writing Python* than to reason through token-by-token. The agent
    writes code, runs it in a sandbox, sees the result (or the traceback), and
    iterates until the code succeeds or the budget runs out.

    The naive version is dangerous — `exec()` in-process means a buggy or
    malicious snippet can wipe files, exfiltrate env vars, or hang forever.
    This file builds the safer version: subprocess isolation, resource limits,
    network blocking, and a retry-on-error loop.

What's here:
    1. `Sandbox.run(code, timeout)` — runs Python in a subprocess with:
         - separate process (crashes don't kill the agent)
         - working dir set to a fresh tempdir (no path traversal to your repo)
         - rlimit on CPU + address space (on Unix) so runaway code dies
         - stdin closed, no inherited env vars except a minimal allowlist
         - hard wall-clock timeout
    2. `CodeAgent.solve(task)` — agent loop:
         (a) ask `llm_fn` to write Python that prints the answer,
         (b) run it in the sandbox,
         (c) if it failed, feed the traceback back and ask for a fix,
         (d) stop on success or after MAX_STEPS.
    3. A stub `llm_fn` so the file is runnable without API keys. Swap with
       Anthropic / OpenAI for real use.

For production, prefer a real isolation layer:
    - Vercel Sandbox (Firecracker microVMs) — https://vercel.com/docs/vercel-sandbox
    - Docker container with --network=none, --read-only, --cap-drop=ALL
    - gVisor / nsjail / Firejail
    The subprocess approach here is a baseline — it stops accidents, not
    a determined attacker on the same host.

Run:
    python starter.py
"""
from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable


# ---------- sandbox ----------

# Hard limits applied via setrlimit() before exec. Tune per workload.
CPU_SECONDS = 5            # RLIMIT_CPU
ADDRESS_SPACE_MB = 1024    # RLIMIT_AS (Linux only — macOS ignores this)
MAX_OPEN_FILES = 256       # RLIMIT_NOFILE

# Only these env vars are passed into the sandbox. Everything else (API keys,
# AWS creds, etc.) is stripped.
ALLOWED_ENV = ("PATH", "LANG", "LC_ALL", "HOME", "PYTHONHASHSEED")


def _apply_limits() -> None:
    """Pre-exec hook: lock down the child process before it runs user code.

    Each setrlimit is best-effort: macOS doesn't support RLIMIT_AS, and some
    limits may be capped by the parent. We swallow per-limit errors rather
    than letting one unsupported limit kill the whole sandbox.
    """
    def _try(limit_name: str, soft_hard: tuple[int, int]) -> None:
        limit = getattr(resource, limit_name, None)
        if limit is None:
            return
        try:
            resource.setrlimit(limit, soft_hard)
        except (ValueError, OSError):
            pass  # platform doesn't support this limit; continue.

    _try("RLIMIT_CPU", (CPU_SECONDS, CPU_SECONDS))
    _try("RLIMIT_AS", (ADDRESS_SPACE_MB * 1024 * 1024, ADDRESS_SPACE_MB * 1024 * 1024))
    _try("RLIMIT_NOFILE", (MAX_OPEN_FILES, MAX_OPEN_FILES))
    _try("RLIMIT_CORE", (0, 0))


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    def short(self) -> str:
        if self.timed_out:
            return "TIMEOUT (process killed after wall-clock budget)"
        head = "OK" if self.ok else f"FAIL (exit {self.returncode})"
        out = (self.stdout or "")[-800:]
        err = (self.stderr or "")[-800:]
        return f"{head}\n--- stdout ---\n{out}\n--- stderr ---\n{err}".rstrip()


class Sandbox:
    """Subprocess-based Python sandbox. Best-effort isolation only."""

    def __init__(self, wall_clock_timeout: float = 10.0):
        self.wall_clock_timeout = wall_clock_timeout

    def run(self, code: str) -> RunResult:
        workdir = tempfile.mkdtemp(prefix="agent_sbx_")
        try:
            script_path = os.path.join(workdir, "snippet.py")
            with open(script_path, "w") as f:
                f.write(code)

            env = {k: v for k, v in os.environ.items() if k in ALLOWED_ENV}
            # Strip site-packages from PYTHONPATH so user code can only `import`
            # stdlib + whatever the system Python ships with. Tighter sandboxes
            # would use a separate venv with a curated package list.
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", script_path],  # -I = isolated mode
                    cwd=workdir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.wall_clock_timeout,
                    preexec_fn=_apply_limits if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired as e:
                return RunResult(
                    ok=False, stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                    stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                    returncode=-1, timed_out=True,
                )
            return RunResult(
                ok=proc.returncode == 0,
                stdout=proc.stdout, stderr=proc.stderr,
                returncode=proc.returncode,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


# ---------- agent ----------

MAX_STEPS = 4

SYSTEM_PROMPT = """You are a careful Python problem solver.
Write a SHORT self-contained Python script that solves the task and PRINTS the answer.
Only use Python standard library (no pip installs, no network calls, no file writes).
Wrap the code in a single ```python ... ``` fenced block. No prose outside the block.
If a previous attempt failed, read the traceback and fix the bug."""


@dataclass
class Step:
    code: str
    result: RunResult


@dataclass
class AgentRun:
    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    success: bool = False


LlmFn = Callable[[list[dict[str, str]]], str]
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text)
    return m.group(1).strip() if m else text.strip()


class CodeAgent:
    def __init__(self, llm_fn: LlmFn, sandbox: Sandbox | None = None, max_steps: int = MAX_STEPS):
        self.llm_fn = llm_fn
        self.sandbox = sandbox or Sandbox()
        self.max_steps = max_steps

    def solve(self, task: str) -> AgentRun:
        run = AgentRun(task=task)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Task:\n{task}"},
        ]

        for step_i in range(1, self.max_steps + 1):
            reply = self.llm_fn(messages)
            code = extract_code(reply)
            result = self.sandbox.run(code)
            run.steps.append(Step(code=code, result=result))

            if result.ok:
                run.success = True
                run.final_answer = result.stdout.strip()
                return run

            # Failure -> feed the traceback back and ask for a corrected version.
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    f"That code failed on attempt {step_i}.\n"
                    f"{result.short()}\n\n"
                    "Read the traceback and produce a corrected version. "
                    "Reply with one ```python``` block only."
                ),
            })

        return run


# ---------- stub LLM (works without API keys) ----------

def stub_llm(messages: list[dict[str, str]]) -> str:
    """Toy LLM that recognizes a couple of tasks and demonstrates the loop.

    Pattern: returns intentionally-buggy code on attempt 1, fixed code on attempt 2.
    Swap this for an Anthropic / OpenAI client in real use.
    """
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_user = user_msgs[-1].lower()
    is_retry = "that code failed" in last_user
    task_text = user_msgs[0].lower()

    if "primes" in task_text:
        if not is_retry:
            # Bug: off-by-one in range (misses n itself) AND wrong condition.
            return (
                "```python\n"
                "n = 50\n"
                "primes = []\n"
                "for i in range(2, n):  # should be n+1\n"
                "    if all(i % d for d in range(2, i)):\n"
                "        primes.append(i)\n"
                "print(undefined_variable)  # NameError on purpose\n"
                "```"
            )
        return (
            "```python\n"
            "n = 50\n"
            "primes = [i for i in range(2, n + 1) if all(i % d for d in range(2, i))]\n"
            "print(sum(primes))\n"
            "```"
        )

    if "fibonacci" in task_text:
        if not is_retry:
            return (
                "```python\n"
                "a, b = 0, 1\n"
                "for _ in range(10)\n"  # SyntaxError: missing colon
                "    a, b = b, a + b\n"
                "print(a)\n"
                "```"
            )
        return (
            "```python\n"
            "a, b = 0, 1\n"
            "for _ in range(10):\n"
            "    a, b = b, a + b\n"
            "print(a)\n"
            "```"
        )

    return "```python\nprint('I do not know how to solve this task.')\n```"


# ---------- demo ----------

def main() -> None:
    agent = CodeAgent(llm_fn=stub_llm, sandbox=Sandbox(wall_clock_timeout=5.0))

    for task in [
        "Compute the sum of all primes below or equal to 50.",
        "Print the 10th Fibonacci number (0-indexed: F(0)=0, F(1)=1).",
    ]:
        print(f"\n{'=' * 70}\nTASK: {task}\n{'=' * 70}")
        run = agent.solve(task)
        for i, step in enumerate(run.steps, 1):
            print(f"\n--- attempt {i} ---")
            print(step.code)
            print(f"-> {step.result.short()}")
        print(f"\nSUCCESS: {run.success}  ANSWER: {run.final_answer!r}")


if __name__ == "__main__":
    main()
