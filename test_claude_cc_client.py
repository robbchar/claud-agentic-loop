"""Tests for claude_cc_client.py — subprocess wrapper for `claude -p`."""

import json
from unittest.mock import MagicMock, patch

import pytest

import claude_cc_client
from claude_cc_client import (
    AGENT_ALLOWED_TOOLS,
    _build_cmd,
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

def test_dev_only_gets_context7():
    assert AGENT_ALLOWED_TOOLS["dev"] == ["mcp__context7__*"]


def test_qa_gets_chrome_and_context7():
    tools = AGENT_ALLOWED_TOOLS["qa"]
    assert "mcp__chrome-devtools__*" in tools
    assert "mcp__context7__*" in tools


def test_reviewer_only_gets_context7():
    assert AGENT_ALLOWED_TOOLS["reviewer"] == ["mcp__context7__*"]


def test_pm_gets_no_tools():
    assert AGENT_ALLOWED_TOOLS["pm"] == []


# ---------------------------------------------------------------------------
# call_claude_cc — happy path
# ---------------------------------------------------------------------------

def _make_proc(stdout="response text", returncode=0, stderr=""):
    proc = MagicMock()
    proc.stdout = stdout
    proc.returncode = returncode
    proc.stderr = stderr
    return proc


@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_returns_stdout(mock_run):
    mock_run.return_value = _make_proc(stdout="  hello world  ")
    result = call_claude_cc("sys", "msg", "dev")
    assert result == "hello world"


@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_passes_correct_agent_tools(mock_run):
    mock_run.return_value = _make_proc()
    call_claude_cc("sys", "msg", "dev")
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "mcp__context7__*"


@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_unknown_agent_gets_no_tools(mock_run):
    mock_run.return_value = _make_proc()
    call_claude_cc("sys", "msg", "unknown_agent")
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "none"


# ---------------------------------------------------------------------------
# call_claude_cc — JSON stripping
# ---------------------------------------------------------------------------

@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_strips_json_fences(mock_run):
    raw = '```json\n{"key": "val"}\n```'
    mock_run.return_value = _make_proc(stdout=raw)
    result = call_claude_cc("sys", "msg", "qa", expect_json=True)
    assert result == '{"key": "val"}'


@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_plain_json_unchanged(mock_run):
    raw = '{"key": "val"}'
    mock_run.return_value = _make_proc(stdout=raw)
    result = call_claude_cc("sys", "msg", "qa", expect_json=True)
    assert result == '{"key": "val"}'


# ---------------------------------------------------------------------------
# call_claude_cc — error handling
# ---------------------------------------------------------------------------

@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_raises_on_nonzero_exit(mock_run):
    mock_run.return_value = _make_proc(returncode=1, stderr="something went wrong")
    with pytest.raises(RuntimeError, match="exited 1"):
        call_claude_cc("sys", "msg", "dev")


# ---------------------------------------------------------------------------
# call_claude_cc_json
# ---------------------------------------------------------------------------

@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_json_returns_dict(mock_run):
    payload = {"passed": True, "summary": "ok", "issues": []}
    mock_run.return_value = _make_proc(stdout=json.dumps(payload))
    result = call_claude_cc_json("sys", "msg", "qa")
    assert result == payload


@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_json_raises_on_invalid_json(mock_run):
    mock_run.return_value = _make_proc(stdout="not json at all")
    with pytest.raises(ValueError, match="invalid JSON"):
        call_claude_cc_json("sys", "msg", "qa")


# ---------------------------------------------------------------------------
# call_claude_cc_messages — flattening
# ---------------------------------------------------------------------------

@patch("claude_cc_client.subprocess.run")
def test_call_claude_cc_messages_flattens_conversation(mock_run):
    mock_run.return_value = _make_proc(stdout="response")
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
    # Default should be dev and qa; reload with no override
    with patch.dict("os.environ", {}, clear=False):
        # Re-evaluate the module-level constant with default env
        import importlib
        # Patch the env to the default value explicitly
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
