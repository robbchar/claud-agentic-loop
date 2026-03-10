"""Tests for agents.py — all four agents, with Claude API calls mocked out."""

from unittest.mock import patch, MagicMock
import pytest
from models import SwarmState, AgentResult


# ---------------------------------------------------------------------------
# PM Agent
# ---------------------------------------------------------------------------

PM_JSON_RESPONSE = {
    "milestone": "M1 — Core",
    "tasks": [
        {
            "id": "T1",
            "summary": "Implement login",
            "acceptance_criteria": ["user can log in", "invalid creds rejected"],
            "warnings": [],
        },
        {
            "id": "T2",
            "summary": "Add rate limiting",
            "acceptance_criteria": ["max 5 req/s"],
            "warnings": ["rate limit value not specified in spec"],
        },
    ],
    "global_constraints": ["Python 3.12", "no network in tests"],
}


class TestPMAgent:
    def _run(self, state: SwarmState):
        from agents import pm_agent
        with patch("agents.call_claude_json", return_value=PM_JSON_RESPONSE) as mock:
            result = pm_agent.run(state)
        return result, mock

    def test_returns_agent_result(self):
        state = SwarmState(architecture="arch", tasks_doc="tasks")
        result, _ = self._run(state)
        assert isinstance(result, AgentResult)
        assert result.passed is True

    def test_prompt_uses_spec_docs(self):
        state = SwarmState(architecture="ARCH CONTENT", tasks_doc="TASKS CONTENT")
        _, mock = self._run(state)
        _, prompt = mock.call_args.args
        assert "ARCH CONTENT" in prompt
        assert "TASKS CONTENT" in prompt

    def test_output_contains_milestone(self):
        state = SwarmState(architecture="a", tasks_doc="t")
        result, _ = self._run(state)
        assert "M1 — Core" in result.output

    def test_output_contains_task_ids(self):
        state = SwarmState(architecture="a", tasks_doc="t")
        result, _ = self._run(state)
        assert "T1" in result.output
        assert "T2" in result.output

    def test_output_contains_acceptance_criteria(self):
        state = SwarmState(architecture="a", tasks_doc="t")
        result, _ = self._run(state)
        assert "user can log in" in result.output
        assert "invalid creds rejected" in result.output

    def test_output_contains_warnings(self):
        state = SwarmState(architecture="a", tasks_doc="t")
        result, _ = self._run(state)
        assert "rate limit value not specified in spec" in result.output

    def test_output_contains_global_constraints(self):
        state = SwarmState(architecture="a", tasks_doc="t")
        result, _ = self._run(state)
        assert "Python 3.12" in result.output
        assert "no network in tests" in result.output

    def test_empty_warnings_not_printed(self):
        state = SwarmState(architecture="a", tasks_doc="t")
        result, _ = self._run(state)
        # T1 has no warnings — no warning block should appear for it
        # (just verify output doesn't raise and warnings section only shows for T2)
        assert result.output.count("Warnings:") == 1

    def test_no_global_constraints_shows_none(self):
        response = {**PM_JSON_RESPONSE, "global_constraints": []}
        state = SwarmState(architecture="a", tasks_doc="t")
        from agents import pm_agent
        with patch("agents.call_claude_json", return_value=response):
            result = pm_agent.run(state)
        assert "None" in result.output


# ---------------------------------------------------------------------------
# Dev Agent
# ---------------------------------------------------------------------------

class TestDevAgent:
    def test_first_iteration_seeds_messages(self):
        state = SwarmState()
        state.requirements = "Build X"
        snapshots = []

        def capture(system, messages):
            snapshots.append(list(messages))  # copy before the assistant turn is appended
            return "def foo(): pass"

        from agents import dev_agent
        with patch("agents.call_claude_messages", side_effect=capture):
            dev_agent.run(state)

        assert len(snapshots[0]) == 1
        assert snapshots[0][0]["role"] == "user"
        assert "Build X" in snapshots[0][0]["content"]

    def test_first_iteration_stores_assistant_turn(self):
        state = SwarmState()
        state.requirements = "Build X"
        from agents import dev_agent
        with patch("agents.call_claude_messages", return_value="def foo(): pass"):
            dev_agent.run(state)

        assert len(state.dev_messages) == 2
        assert state.dev_messages[1]["role"] == "assistant"
        assert state.dev_messages[1]["content"] == "def foo(): pass"

    def test_subsequent_iteration_appends_feedback(self):
        state = SwarmState()
        state.requirements = "Build X"
        state.dev_messages = [
            {"role": "user", "content": "REQUIREMENTS:\nBuild X"},
            {"role": "assistant", "content": "def foo(): pass"},
        ]
        state.feedback = "Add error handling"
        snapshots = []

        def capture(system, messages):
            snapshots.append(list(messages))  # copy before the assistant turn is appended
            return "def foo(): raise ..."

        from agents import dev_agent
        with patch("agents.call_claude_messages", side_effect=capture):
            dev_agent.run(state)

        assert snapshots[0][-1]["role"] == "user"
        assert "Add error handling" in snapshots[0][-1]["content"]

    def test_returns_code_as_output(self):
        state = SwarmState()
        state.requirements = "Build X"
        from agents import dev_agent
        with patch("agents.call_claude_messages", return_value="def foo(): pass"):
            result = dev_agent.run(state)
        assert result.output == "def foo(): pass"
        assert result.passed is True


# ---------------------------------------------------------------------------
# QA Agent
# ---------------------------------------------------------------------------

QA_PASS_RESPONSE = {
    "passed": True,
    "summary": "All criteria met",
    "issues": [],
    "feedback_for_dev": "",
}

QA_FAIL_RESPONSE = {
    "passed": False,
    "summary": "Critical bug found",
    "issues": [{"severity": "critical", "description": "crashes on empty input"}],
    "feedback_for_dev": "Handle empty input case",
}


class TestQAAgent:
    def _make_state(self):
        state = SwarmState()
        state.requirements = "validate emails"
        state.code = "def validate(email): return True"
        return state

    def test_pass_result(self):
        state = self._make_state()
        from agents import qa_agent
        with patch("agents.call_claude_json", return_value=QA_PASS_RESPONSE):
            result = qa_agent.run(state)
        assert result.passed is True
        assert result.feedback is None
        assert "PASS" in result.output

    def test_fail_result(self):
        state = self._make_state()
        from agents import qa_agent
        with patch("agents.call_claude_json", return_value=QA_FAIL_RESPONSE):
            result = qa_agent.run(state)
        assert result.passed is False
        assert result.feedback == "Handle empty input case"
        assert "FAIL" in result.output

    def test_issues_formatted_in_output(self):
        state = self._make_state()
        from agents import qa_agent
        with patch("agents.call_claude_json", return_value=QA_FAIL_RESPONSE):
            result = qa_agent.run(state)
        assert "CRITICAL" in result.output
        assert "crashes on empty input" in result.output

    def test_no_issues_shows_none(self):
        state = self._make_state()
        from agents import qa_agent
        with patch("agents.call_claude_json", return_value=QA_PASS_RESPONSE):
            result = qa_agent.run(state)
        assert "None" in result.output

    def test_prompt_includes_requirements_and_code(self):
        state = self._make_state()
        from agents import qa_agent
        with patch("agents.call_claude_json", return_value=QA_PASS_RESPONSE) as mock:
            qa_agent.run(state)
        _, prompt = mock.call_args.args
        assert "validate emails" in prompt
        assert "def validate(email)" in prompt

    def test_empty_feedback_string_becomes_none(self):
        state = self._make_state()
        from agents import qa_agent
        with patch("agents.call_claude_json", return_value=QA_PASS_RESPONSE):
            result = qa_agent.run(state)
        assert result.feedback is None


# ---------------------------------------------------------------------------
# Reviewer Agent
# ---------------------------------------------------------------------------

REVIEWER_APPROVED_RESPONSE = {
    "approved": True,
    "summary": "Clean, well-structured code",
    "comments": [{"type": "nit", "description": "minor naming thing"}],
    "feedback_for_dev": "",
}

REVIEWER_CHANGES_RESPONSE = {
    "approved": False,
    "summary": "Needs work",
    "comments": [{"type": "blocking", "description": "missing type hints"}],
    "feedback_for_dev": "Add type hints to all public functions",
}


class TestReviewerAgent:
    def _make_state(self):
        state = SwarmState()
        state.requirements = "validate emails"
        state.code = "def validate(email): return True"
        state.qa_report = "QA RESULT: PASS"
        return state

    def test_approved(self):
        state = self._make_state()
        from agents import reviewer_agent
        with patch("agents.call_claude_json", return_value=REVIEWER_APPROVED_RESPONSE):
            result = reviewer_agent.run(state)
        assert result.passed is True
        assert result.feedback is None
        assert "APPROVED" in result.output

    def test_changes_requested(self):
        state = self._make_state()
        from agents import reviewer_agent
        with patch("agents.call_claude_json", return_value=REVIEWER_CHANGES_RESPONSE):
            result = reviewer_agent.run(state)
        assert result.passed is False
        assert result.feedback == "Add type hints to all public functions"
        assert "CHANGES REQUESTED" in result.output

    def test_comments_in_output(self):
        state = self._make_state()
        from agents import reviewer_agent
        with patch("agents.call_claude_json", return_value=REVIEWER_CHANGES_RESPONSE):
            result = reviewer_agent.run(state)
        assert "BLOCKING" in result.output
        assert "missing type hints" in result.output

    def test_prompt_includes_all_three_inputs(self):
        state = self._make_state()
        from agents import reviewer_agent
        with patch("agents.call_claude_json", return_value=REVIEWER_APPROVED_RESPONSE) as mock:
            reviewer_agent.run(state)
        _, prompt = mock.call_args.args
        assert "validate emails" in prompt
        assert "QA RESULT: PASS" in prompt
        assert "def validate(email)" in prompt

    def test_empty_feedback_string_becomes_none(self):
        state = self._make_state()
        from agents import reviewer_agent
        with patch("agents.call_claude_json", return_value=REVIEWER_APPROVED_RESPONSE):
            result = reviewer_agent.run(state)
        assert result.feedback is None
