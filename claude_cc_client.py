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


from models import BillingError


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
    allowed_tools: list[str],
) -> list[str]:
    # Prompt is passed via stdin (not as a CLI argument) to avoid Windows'
    # 32,767-character command line limit when project context is large.
    # --output-format stream-json emits newline-delimited JSON events so we
    # can parse tool calls and update the spinner in real time.
    cmd = [
        "claude", "-p",
        "--output-format", "stream-json",
        "--system-prompt", system_prompt,
    ]
    if allowed_tools:
        cmd += ["--allowedTools"] + allowed_tools
    else:
        cmd += ["--allowedTools", "none"]
    return cmd


_TIMEOUT = int(os.environ.get("SWARM_CC_TIMEOUT", "600"))


def _tool_label(tool_name: str, tool_input: dict) -> str:
    """Convert an MCP tool name + input into a short human-readable status."""
    if "context7" in tool_name:
        if "resolve" in tool_name:
            lib = tool_input.get("libraryName", "")
            return f"Looking up '{lib}'" if lib else "Looking up library"
        if "get-library-docs" in tool_name:
            topic = tool_input.get("topic", "")
            return f"Reading docs: {topic}" if topic else "Reading docs"
        return "Checking docs"
    if "chrome-devtools" in tool_name:
        return "Checking browser"
    # Generic: strip mcp__ prefix + namespace, humanise the verb
    clean = tool_name.removeprefix("mcp__")
    parts = clean.split("__", 1)
    verb = parts[-1].replace("-", " ")
    return f"Using {verb}"


def _run(cmd: list[str], user_message: str, label: str = "agent", spinner=None) -> str:
    """
    Run `claude -p --output-format stream-json` and parse the event stream.

    - Tool-use events update the spinner so the user sees e.g.
      "💻 [Dev Agent] Writing code [3.4] — Looking up 'react'... (1m 12s)"
    - Text content is printed to the terminal as it streams.
    - The final `result` event provides the canonical return value.
    - On timeout, any text collected so far is returned as partial output
      rather than discarding everything.

    The prompt is written to stdin to avoid Windows' 32 767-char CLI limit.
    """
    popen_kwargs: dict = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        **popen_kwargs,
    )

    def _write_stdin() -> None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(user_message)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    threading.Thread(target=_write_stdin, daemon=True).start()

    collected: list[str] = []   # text chunks streamed so far (for partial return)
    final_result: list[str] = []  # populated from the result event
    _text_started = False

    def _stream() -> None:
        nonlocal _text_started
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Shouldn't happen with stream-json but pass non-JSON through
                if not _text_started:
                    if spinner is not None:
                        spinner.clear()
                    _text_started = True
                print(line, flush=True)
                continue

            etype = event.get("type")

            if etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    btype = block.get("type")
                    if btype == "tool_use":
                        if spinner is not None and not _text_started:
                            spinner.update(_tool_label(
                                block.get("name", ""),
                                block.get("input") or {},
                            ))
                    elif btype == "text":
                        text = block.get("text", "")
                        if text:
                            if not _text_started:
                                if spinner is not None:
                                    spinner.clear()
                                _text_started = True
                            collected.append(text)
                            print(text, end="", flush=True)

            elif etype == "result":
                result_text = event.get("result") or ""
                if result_text:
                    final_result.append(result_text)

    start = time.monotonic()
    reader = threading.Thread(target=_stream, daemon=True)
    reader.start()

    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - start
            if elapsed > _TIMEOUT:
                proc.kill()
                reader.join(timeout=2)
                partial = (final_result[0] if final_result else "".join(collected)).strip()
                if partial:
                    print(
                        f"\n\n⚠️  [{label}] timed out after {elapsed:.0f}s "
                        f"(SWARM_CC_TIMEOUT={_TIMEOUT}) — returning partial output.",
                        flush=True,
                    )
                    return partial
                raise RuntimeError(
                    f"claude subprocess timed out after {elapsed:.0f}s "
                    f"(SWARM_CC_TIMEOUT={_TIMEOUT}) — no output was produced"
                )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n\n⛔  Interrupted — killing [{label}] subprocess...", flush=True)
        proc.kill()
        if proc.stdout:
            proc.stdout.close()
        reader.join(timeout=2)
        raise

    reader.join()
    elapsed = time.monotonic() - start
    print(f"\n  ⏱  [{label}] finished in {elapsed:.1f}s", flush=True)

    if proc.returncode != 0:
        assert proc.stderr is not None
        stderr_text = proc.stderr.read().strip()
        # result event may already hold an error message
        stdout_text = (final_result[0] if final_result else "".join(collected)).strip()
        combined = f"{stdout_text}\n{stderr_text}".lower()
        if "credit balance" in combined or "insufficient" in combined or "billing" in combined:
            raise BillingError(
                "Claude Code credit balance is too low.\n"
                "Top up at: https://claude.ai/settings/billing"
            )
        raise RuntimeError(
            f"claude subprocess exited {proc.returncode}:\n{stderr_text or stdout_text}"
        )

    return (final_result[0] if final_result else "".join(collected)).strip()


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
    cmd = _build_cmd(system_prompt, allowed)
    text = _run(cmd, user_message=user_message, label=agent_name, spinner=spinner)

    if expect_json:
        text = text.strip()
        # Extract JSON from a fenced block even when prose precedes it.
        extracted = False
        for fence in ("```json\n", "```\n"):
            if fence in text:
                after = text.split(fence, 1)[1]
                text = after.rsplit("```", 1)[0].strip()
                extracted = True
                break
        if not extracted and text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        # Last-resort: strip any prose before the first { or [
        if not (text.startswith('{') or text.startswith('[')):
            for marker in ('{', '['):
                idx = text.find(marker)
                if idx > 0:
                    text = text[idx:]
                    break
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
