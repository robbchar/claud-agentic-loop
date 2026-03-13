"""
Claude Code subprocess client.

Shells out to `claude -p` so agents inherit the user's full local Claude Code
setup — MCP servers, model config, everything — while keeping tool access
restricted via --allowedTools to preserve the swarm's isolation model.

Agents that need MCP access (Dev → Context7, QA → Chrome DevTools) use this
instead of claude_client.py. Agents that don't (PM, Reviewer) keep using the
direct Anthropic API to avoid the subprocess overhead.

Isolation contract
------------------
Each call passes --allowedTools with only the MCPs that agent needs.
Bash, Edit, Write, and other filesystem/shell tools are never included,
so a misbehaving agent cannot delete files or run arbitrary commands.
The file writer boundary (writer.py) stays the only path to disk.

Environment variables
---------------------
SWARM_CC_AGENTS   : comma-separated list of agents that use this client.
                    Default: "dev,qa"
                    Example override: SWARM_CC_AGENTS=dev,qa,reviewer
                    Set to empty string to disable for all agents.
SWARM_CC_TIMEOUT  : seconds before a subprocess call is killed (default: 300).
"""

import json
import os
import subprocess
import threading
import time


# Tools each agent is allowed to use. Restricting to specific MCP namespaces
# means agents can look things up (read-only) but cannot touch the filesystem.
AGENT_ALLOWED_TOOLS: dict[str, list[str]] = {
    "dev": ["mcp__chrome-devtools__*", "mcp__context7__*"],
    "qa": ["mcp__chrome-devtools__*", "mcp__context7__*"],
    "reviewer": ["mcp__context7__*"],
    "pm": [],  # PM never shells out; entry here is just for completeness
}


def _build_cmd(
    system_prompt: str,
    user_message: str,
    allowed_tools: list[str],
) -> list[str]:
    cmd = ["claude", "-p", user_message, "--system-prompt", system_prompt]
    if allowed_tools:
        cmd += ["--allowedTools"] + allowed_tools
    else:
        # Explicit empty list — deny all tools (belt-and-suspenders)
        cmd += ["--allowedTools", "none"]
    return cmd


_TIMEOUT = int(os.environ.get("SWARM_CC_TIMEOUT", "300"))


def _run(cmd: list[str], label: str = "agent", spinner=None) -> str:
    """
    Run `claude -p` and stream its stdout to the terminal in real time
    while also collecting it for the return value. Kills the process and
    raises RuntimeError if it exceeds _TIMEOUT seconds.

    If a Spinner is passed, it is cleared as soon as the first output chunk
    arrives so streaming output is not interleaved with the spinner.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    collected: list[str] = []
    start = time.monotonic()
    _first = True

    def _stream() -> None:
        nonlocal _first
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(64), ""):
            if _first and spinner is not None:
                spinner.clear()
                _first = False
            collected.append(chunk)
            print(chunk, end="", flush=True)

    reader = threading.Thread(target=_stream, daemon=True)
    reader.start()

    try:
        proc.wait(timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(timeout=2)
        elapsed = time.monotonic() - start
        raise RuntimeError(
            f"claude subprocess timed out after {elapsed:.0f}s "
            f"(SWARM_CC_TIMEOUT={_TIMEOUT})"
        )
    except KeyboardInterrupt:
        print(f"\n\n⛔  Interrupted — killing [{label}] subprocess...", flush=True)
        proc.kill()
        reader.join(timeout=2)
        raise

    reader.join()
    elapsed = time.monotonic() - start
    print(f"\n  ⏱  [{label}] finished in {elapsed:.1f}s", flush=True)

    if proc.returncode != 0:
        assert proc.stderr is not None
        raise RuntimeError(
            f"claude subprocess exited {proc.returncode}:\n{proc.stderr.read().strip()}"
        )

    return "".join(collected).strip()


def call_claude_cc(
    system_prompt: str,
    user_message: str,
    agent_name: str,
    expect_json: bool = False,
    spinner=None,
) -> str:
    """
    Single-turn call via `claude -p`. Returns the text response.

    Args:
        system_prompt: The agent's persona + instructions.
        user_message:  The task payload (assembled from SwarmState by the agent).
        agent_name:    Used to look up the correct --allowedTools list.
        expect_json:   If True, appends a JSON reminder and strips markdown fences.
    """
    if expect_json:
        system_prompt += (
            "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no backticks."
        )

    allowed = AGENT_ALLOWED_TOOLS.get(agent_name, [])
    cmd = _build_cmd(system_prompt, user_message, allowed)
    text = _run(cmd, label=agent_name, spinner=spinner)

    if expect_json:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    return text


def call_claude_cc_messages(
    system_prompt: str,
    messages: list[dict],
    agent_name: str,
    spinner=None,
) -> str:
    """
    Multi-turn variant. Flattens the messages list into a single prompt string
    since `claude -p` takes a single input. The full conversation history is
    preserved in the text so the model still has prior context.
    """
    parts = []
    for m in messages:
        label = "USER" if m["role"] == "user" else "ASSISTANT"
        parts.append(f"[{label}]\n{m['content']}")
    user_message = "\n\n---\n\n".join(parts)

    return call_claude_cc(system_prompt, user_message, agent_name, expect_json=False, spinner=spinner)


def call_claude_cc_json(
    system_prompt: str,
    user_message: str,
    agent_name: str,
    spinner=None,
) -> dict:
    """Convenience wrapper that returns a parsed dict."""
    raw = call_claude_cc(system_prompt, user_message, agent_name, expect_json=True, spinner=spinner)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"claude subprocess returned invalid JSON:\n{raw}\n\nError: {e}")
