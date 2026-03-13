"""Tests for claude_cc_client.py — subprocess wrapper for `claude -p`."""

import json
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

import claude_cc_client
from claude_cc_client import (
    AGENT_ALLOWED_TOOLS,
    _build_cmd,
    _run,
    call_claude_cc,
    call_claude_cc_json,
    call_claude_cc_messages,
)


# ---------------------------------------------------------------------------
# _build_cmd
# ---------------------------------------------------------------------------

def test_build_cmd_includes_system_and_message():
    cmd = _build_cmd("sys", "hello", [])
    assert "claude" in cmd
    assert "-p" in cmd
    assert "hello" in cmd
    assert "--system-prompt" in cmd
    assert "sys" in cmd


def test_build_cmd_with_allowed_tools():
    cmd = _build_cmd("sys", "msg", ["mcp__context7__*"])
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "mcp__context7__*"


def test_build_cmd_empty_tools_passes_none():
    cmd = _build_cmd("sys", "msg", [])
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "none"


def test_build_cmd_multiple_tools():
    tools = ["mcp__context7__*", "mcp__chrome-devtools__*"]
    cmd = _build_cmd("sys", "msg", tools)
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "mcp__context7__*"
    assert cmd[idx + 2] == "mcp__chrome-devtools__*"


# ---------------------------------------------------------------------------
# AGENT_ALLOWED_TOOLS defaults
# ---------------------------------------------------------------------------

def test_dev_gets_chrome_and_context7():
    tools = AGENT_ALLOWED_TOOLS["dev"]
    assert "mcp__chrome-devtools__*" in tools
    assert "mcp__context7__*" in tools


def test_qa_gets_chrome_and_context7():
    tools = AGENT_ALLOWED_TOOLS["qa"]
    assert "mcp__chrome-devtools__*" in tools
    assert "mcp__context7__*" in tools


def test_reviewer_only_gets_context7():
    assert AGENT_ALLOWED_TOOLS["reviewer"] == ["mcp__context7__*"]


def test_pm_gets_no_tools():
    assert AGENT_ALLOWED_TOOLS["pm"] == []


# ---------------------------------------------------------------------------
# _run — Popen-level tests
# ---------------------------------------------------------------------------

def _make_popen_mock(output: str = "response text", returncode: int = 0, stderr_text: str = "") -> MagicMock:
    """Build a mock Popen process that streams `output` in 64-byte chunks."""
    chunks = [output[i:i + 64] for i in range(0, max(len(output), 1), 64)] + [""]
    proc = MagicMock()
    proc.stdout.read.side_effect = chunks
    proc.returncode = returncode
    proc.stderr.read.return_value = stderr_text
    # poll() returns None while running, then returncode when done.
    # We simulate "already finished" so the polling loop exits immediately.
    proc.poll.return_value = returncode
    return proc


@patch("claude_cc_client.subprocess.Popen")
def test_run_returns_stripped_output(mock_popen):
    mock_popen.return_value = _make_popen_mock(output="  hello world  ")
    result = _run(["claude", "-p", "test"], label="dev")
    assert result == "hello world"


@patch("claude_cc_client.subprocess.Popen")
def test_run_raises_on_nonzero_exit(mock_popen):
    mock_popen.return_value = _make_popen_mock(returncode=1, stderr_text="bad things")
    with pytest.raises(RuntimeError, match="exited 1"):
        _run(["claude", "-p", "test"])


@patch("claude_cc_client.subprocess.Popen")
def test_run_raises_on_timeout(mock_popen):
    proc = _make_popen_mock()
    # poll() keeps returning None so the timeout loop fires
    proc.poll.return_value = None
    mock_popen.return_value = proc
    with patch("claude_cc_client._TIMEOUT", 0):
        with pytest.raises(RuntimeError, match="timed out"):
            _run(["claude", "-p", "test"])
    proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# call_claude_cc — happy path (mock _run directly)
# ---------------------------------------------------------------------------

@patch("claude_cc_client._run", return_value="hello world")
def test_call_claude_cc_returns_output(mock_run):
    result = call_claude_cc("sys", "msg", "dev")
    assert result == "hello world"


@patch("claude_cc_client._run", return_value="")
def test_call_claude_cc_passes_correct_agent_tools(mock_run):
    call_claude_cc("sys", "msg", "dev")
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--allowedTools")
    assert "mcp__chrome-devtools__*" in cmd[idx + 1:]
    assert "mcp__context7__*" in cmd[idx + 1:]


@patch("claude_cc_client._run", return_value="")
def test_call_claude_cc_unknown_agent_gets_no_tools(mock_run):
    call_claude_cc("sys", "msg", "unknown_agent")
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "none"


# ---------------------------------------------------------------------------
# call_claude_cc — JSON stripping
# ---------------------------------------------------------------------------

@patch("claude_cc_client._run", return_value='```json\n{"key": "val"}\n```')
def test_call_claude_cc_strips_json_fences(mock_run):
    result = call_claude_cc("sys", "msg", "qa", expect_json=True)
    assert result == '{"key": "val"}'


@patch("claude_cc_client._run", return_value='{"key": "val"}')
def test_call_claude_cc_plain_json_unchanged(mock_run):
    result = call_claude_cc("sys", "msg", "qa", expect_json=True)
    assert result == '{"key": "val"}'


# ---------------------------------------------------------------------------
# call_claude_cc_json
# ---------------------------------------------------------------------------

@patch("claude_cc_client._run")
def test_call_claude_cc_json_returns_dict(mock_run):
    payload = {"passed": True, "summary": "ok", "issues": []}
    mock_run.return_value = json.dumps(payload)
    result = call_claude_cc_json("sys", "msg", "qa")
    assert result == payload


@patch("claude_cc_client._run", return_value="not json at all")
def test_call_claude_cc_json_raises_on_invalid_json(mock_run):
    with pytest.raises(ValueError, match="invalid JSON"):
        call_claude_cc_json("sys", "msg", "qa")


# ---------------------------------------------------------------------------
# call_claude_cc_messages — flattening
# ---------------------------------------------------------------------------

@patch("claude_cc_client._run", return_value="response")
def test_call_claude_cc_messages_flattens_conversation(mock_run):
    messages = [
        {"role": "user", "content": "write me a function"},
        {"role": "assistant", "content": "def foo(): pass"},
        {"role": "user", "content": "add tests"},
    ]
    call_claude_cc_messages("sys", messages, "dev")
    cmd = mock_run.call_args[0][0]
    prompt_idx = cmd.index("-p") + 1
    flattened = cmd[prompt_idx]
    assert "[USER]" in flattened
    assert "[ASSISTANT]" in flattened
    assert "write me a function" in flattened
    assert "def foo(): pass" in flattened
    assert "add tests" in flattened


# ---------------------------------------------------------------------------
# CC_AGENTS env var parsing
# ---------------------------------------------------------------------------

def test_cc_agents_default_includes_dev_and_qa():
    import importlib
    with patch.dict("os.environ", {"SWARM_CC_AGENTS": "dev,qa"}):
        importlib.reload(claude_cc_client)
        import agents
        importlib.reload(agents)
        from agents import CC_AGENTS
        assert "dev" in CC_AGENTS
        assert "qa" in CC_AGENTS
        assert "pm" not in CC_AGENTS
        assert "reviewer" not in CC_AGENTS


def test_cc_agents_empty_string_disables_all():
    import importlib
    with patch.dict("os.environ", {"SWARM_CC_AGENTS": ""}):
        importlib.reload(claude_cc_client)
        import agents
        importlib.reload(agents)
        from agents import CC_AGENTS
        assert len(CC_AGENTS) == 0


def test_cc_agents_custom_includes_reviewer():
    import importlib
    with patch.dict("os.environ", {"SWARM_CC_AGENTS": "dev,qa,reviewer"}):
        importlib.reload(claude_cc_client)
        import agents
        importlib.reload(agents)
        from agents import CC_AGENTS
        assert "reviewer" in CC_AGENTS
