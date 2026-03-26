"""Tests for claude_cc_client.py — subprocess wrapper for `claude -p`."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import claude_cc_client
from claude_cc_client import (
    AGENT_ALLOWED_TOOLS,
    _build_cmd,
    _run,
    _tool_label,
    call_claude_cc,
    call_claude_cc_json,
    call_claude_cc_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_lines(text: str = "response text") -> list[str]:
    """Build the stream-json lines that _run expects for a successful response."""
    lines = []
    if text:
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        }) + "\n")
    lines.append(json.dumps({
        "type": "result",
        "result": text,
        "is_error": False,
    }) + "\n")
    return lines


def _make_popen_mock(text: str = "response text", returncode: int = 0, stderr_text: str = "") -> MagicMock:
    """Build a mock Popen process that yields stream-json lines."""
    proc = MagicMock()
    proc.stdout = iter(_stream_lines(text))
    proc.returncode = returncode
    proc.stderr.read.return_value = stderr_text
    proc.poll.return_value = returncode
    return proc


# ---------------------------------------------------------------------------
# _build_cmd
# ---------------------------------------------------------------------------

def test_build_cmd_includes_system_prompt():
    cmd = _build_cmd("sys", [])
    assert "claude" in cmd
    assert "-p" in cmd
    assert "--system-prompt" in cmd
    assert "sys" in cmd


def test_build_cmd_includes_stream_json_output_format():
    cmd = _build_cmd("sys", [])
    assert "--output-format" in cmd
    idx = cmd.index("--output-format")
    assert cmd[idx + 1] == "stream-json"


def test_build_cmd_includes_verbose():
    # stream-json requires --verbose with -p
    cmd = _build_cmd("sys", [])
    assert "--verbose" in cmd


def test_build_cmd_does_not_include_user_message():
    cmd = _build_cmd("sys", [])
    assert "hello" not in cmd


def test_build_cmd_with_allowed_tools():
    cmd = _build_cmd("sys", ["mcp__context7__*"])
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "mcp__context7__*"


def test_build_cmd_empty_tools_passes_none():
    cmd = _build_cmd("sys", [])
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1] == "none"


def test_build_cmd_multiple_tools():
    tools = ["mcp__context7__*", "mcp__chrome-devtools__*"]
    cmd = _build_cmd("sys", tools)
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
# _tool_label
# ---------------------------------------------------------------------------

def test_tool_label_context7_resolve():
    label = _tool_label("mcp__context7__resolve-library-id", {"libraryName": "react"})
    assert "react" in label.lower()


def test_tool_label_context7_resolve_no_input():
    label = _tool_label("mcp__context7__resolve-library-id", {})
    assert "library" in label.lower()


def test_tool_label_context7_get_docs():
    label = _tool_label("mcp__context7__get-library-docs", {"topic": "hooks"})
    assert "hooks" in label.lower()


def test_tool_label_context7_get_docs_no_topic():
    label = _tool_label("mcp__context7__get-library-docs", {})
    assert label  # non-empty


def test_tool_label_chrome_devtools():
    label = _tool_label("mcp__chrome-devtools__screenshot", {})
    assert "browser" in label.lower()


def test_tool_label_unknown_tool():
    label = _tool_label("mcp__some-server__do-thing", {})
    assert label  # non-empty, doesn't crash


# ---------------------------------------------------------------------------
# _run — stream-json parsing
# ---------------------------------------------------------------------------

@patch("claude_cc_client.subprocess.Popen")
def test_run_returns_result_event_text(mock_popen):
    mock_popen.return_value = _make_popen_mock(text="hello world")
    result = _run(["claude", "-p"], user_message="hi", label="dev")
    assert result == "hello world"


@patch("claude_cc_client.subprocess.Popen")
def test_run_strips_whitespace(mock_popen):
    mock_popen.return_value = _make_popen_mock(text="  hello world  ")
    result = _run(["claude", "-p"], user_message="hi", label="dev")
    assert result == "hello world"


@patch("claude_cc_client.subprocess.Popen")
def test_run_falls_back_to_collected_text_if_no_result_event(mock_popen):
    """If stream ends without a result event, collected text chunks are returned."""
    proc = MagicMock()
    proc.stdout = iter([
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "partial"}]}}) + "\n",
        # no result event
    ])
    proc.returncode = 0
    proc.stderr.read.return_value = ""
    proc.poll.return_value = 0
    mock_popen.return_value = proc
    result = _run(["claude", "-p"], user_message="hi")
    assert result == "partial"


@patch("claude_cc_client.subprocess.Popen")
def test_run_spinner_updated_on_tool_use(mock_popen):
    proc = MagicMock()
    proc.stdout = iter([
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "mcp__context7__resolve-library-id", "input": {"libraryName": "react"}},
        ]}}) + "\n",
        json.dumps({"type": "result", "result": "done", "is_error": False}) + "\n",
    ])
    proc.returncode = 0
    proc.stderr.read.return_value = ""
    proc.poll.return_value = 0
    mock_popen.return_value = proc

    spinner = MagicMock()
    _run(["claude", "-p"], user_message="hi", spinner=spinner)
    spinner.update.assert_called_once()
    call_arg = spinner.update.call_args[0][0]
    assert "react" in call_arg.lower()


@patch("claude_cc_client.subprocess.Popen")
def test_run_spinner_cleared_when_text_starts(mock_popen):
    mock_popen.return_value = _make_popen_mock(text="some code")
    spinner = MagicMock()
    _run(["claude", "-p"], user_message="hi", spinner=spinner)
    spinner.clear.assert_called_once()


@patch("claude_cc_client.subprocess.Popen")
def test_run_raises_on_nonzero_exit(mock_popen):
    mock_popen.return_value = _make_popen_mock(returncode=1, stderr_text="bad things")
    with pytest.raises(RuntimeError, match="exited 1"):
        _run(["claude", "-p"], user_message="test")


@patch("claude_cc_client.subprocess.Popen")
def test_run_raises_on_timeout_with_no_output(mock_popen):
    proc = MagicMock()
    proc.stdout = iter([])   # no output at all
    proc.returncode = None
    proc.poll.return_value = None
    mock_popen.return_value = proc
    with patch("claude_cc_client._TIMEOUT", 0):
        with pytest.raises(RuntimeError, match="timed out"):
            _run(["claude", "-p"], user_message="test")
    proc.kill.assert_called_once()


@patch("claude_cc_client.subprocess.Popen")
def test_run_returns_partial_on_timeout_with_output(mock_popen):
    proc = MagicMock()
    proc.stdout = iter([
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "partial code"}]}}) + "\n",
        # no result event — process will be killed
    ])
    proc.returncode = None
    proc.poll.return_value = None
    mock_popen.return_value = proc
    with patch("claude_cc_client._TIMEOUT", 0):
        result = _run(["claude", "-p"], user_message="test")
    assert result == "partial code"


@patch("claude_cc_client.subprocess.Popen")
def test_run_raises_billing_error_on_credit_message(mock_popen):
    mock_popen.return_value = _make_popen_mock(returncode=1, stderr_text="credit balance too low")
    with pytest.raises(Exception) as exc_info:
        _run(["claude", "-p"], user_message="test")
    assert "billing" in type(exc_info.value).__name__.lower() or "credit" in str(exc_info.value).lower()


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


@patch("claude_cc_client._run", return_value='Here is my verdict.\n{"key": "val"}')
def test_call_claude_cc_strips_preamble_before_json(mock_run):
    result = call_claude_cc("sys", "msg", "qa", expect_json=True)
    assert result == '{"key": "val"}'


@patch("claude_cc_client._run", return_value='Some explanation.\n[1, 2, 3]')
def test_call_claude_cc_strips_preamble_before_json_array(mock_run):
    result = call_claude_cc("sys", "msg", "qa", expect_json=True)
    assert result == '[1, 2, 3]'


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
    flattened = mock_run.call_args.kwargs["user_message"]
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
