"""Tests for orchestrator.py — loop control, PM skip, iteration logic."""

from unittest.mock import patch, MagicMock
import pytest
from models import SwarmState, AgentResult
from orchestrator import run_swarm


def _make_pm_result(output="requirements text"):
    return AgentResult(output=output, passed=True)

def _make_dev_result(output="def foo(): pass"):
    return AgentResult(output=output, passed=True)

def _make_qa_pass():
    return AgentResult(output="QA PASS", passed=True)

def _make_qa_fail(feedback="fix the bug"):
    return AgentResult(output="QA FAIL", passed=False, feedback=feedback)

def _make_reviewer_approved():
    return AgentResult(output="APPROVED", passed=True)

def _make_reviewer_changes(feedback="add types"):
    return AgentResult(output="CHANGES REQUESTED", passed=False, feedback=feedback)


class TestRunSwarmPMPhase:
    def test_pm_runs_when_requirements_empty(self):
        state = SwarmState()
        with (
            patch("orchestrator.pm_agent.run", return_value=_make_pm_result("reqs")) as pm_mock,
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False)
        pm_mock.assert_called_once()

    def test_pm_sets_requirements_on_state(self):
        state = SwarmState()
        with (
            patch("orchestrator.pm_agent.run", return_value=_make_pm_result("generated reqs")),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result_state = run_swarm(state=state, verbose=False)
        assert result_state.requirements == "generated reqs"

    def test_pm_skipped_when_requirements_already_set(self):
        state = SwarmState()
        state.requirements = "pre-existing requirements"
        with (
            patch("orchestrator.pm_agent.run") as pm_mock,
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False)
        pm_mock.assert_not_called()

    def test_pm_skipped_preserves_existing_requirements(self):
        state = SwarmState()
        state.requirements = "pre-existing requirements"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result_state = run_swarm(state=state, verbose=False)
        assert result_state.requirements == "pre-existing requirements"

    def test_pm_recorded_in_history_when_run(self):
        state = SwarmState()
        with (
            patch("orchestrator.pm_agent.run", return_value=_make_pm_result("reqs")),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result_state = run_swarm(state=state, verbose=False)
        pm_entries = [h for h in result_state.history if h["agent"] == "pm"]
        assert len(pm_entries) == 1

    def test_pm_not_recorded_in_history_when_skipped(self):
        state = SwarmState()
        state.requirements = "pre-existing"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result_state = run_swarm(state=state, verbose=False)
        pm_entries = [h for h in result_state.history if h["agent"] == "pm"]
        assert len(pm_entries) == 0


class TestRunSwarmLoop:
    def test_approved_on_first_iteration(self):
        state = SwarmState()
        state.requirements = "reqs"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result = run_swarm(state=state, verbose=False)
        assert result.approved is True
        dev_calls = [h for h in result.history if h["agent"] == "dev"]
        assert len(dev_calls) == 1

    def test_qa_fail_loops_back_to_dev(self):
        state = SwarmState()
        state.requirements = "reqs"
        qa_results = [_make_qa_fail("fix X"), _make_qa_pass()]
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", side_effect=qa_results),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result = run_swarm(state=state, verbose=False)
        dev_calls = [h for h in result.history if h["agent"] == "dev"]
        assert len(dev_calls) == 2

    def test_qa_fail_sets_feedback_on_state(self):
        state = SwarmState()
        state.requirements = "reqs"
        # Fail once then pass, then reviewer approves
        qa_results = [_make_qa_fail("fix X"), _make_qa_pass()]
        captured_feedback = []

        def dev_side_effect(s):
            captured_feedback.append(s.feedback)
            return _make_dev_result()

        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", side_effect=dev_side_effect),
            patch("orchestrator.qa_agent.run", side_effect=qa_results),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False)
        # Second dev call should have received the QA feedback
        assert captured_feedback[1] == "fix X"

    def test_reviewer_changes_loops_back_to_dev(self):
        state = SwarmState()
        state.requirements = "reqs"
        reviewer_results = [_make_reviewer_changes("add types"), _make_reviewer_approved()]
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", side_effect=reviewer_results),
        ):
            result = run_swarm(state=state, verbose=False)
        assert result.approved is True
        dev_calls = [h for h in result.history if h["agent"] == "dev"]
        assert len(dev_calls) == 2

    def test_reviewer_not_called_when_qa_fails(self):
        state = SwarmState()
        state.requirements = "reqs"
        # QA always fails — reviewer should never be called
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_fail()),
            patch("orchestrator.reviewer_agent.run") as reviewer_mock,
        ):
            result = run_swarm(state=state, verbose=False)
        reviewer_mock.assert_not_called()
        assert result.approved is False

    def test_max_iterations_stops_loop(self):
        from orchestrator import MAX_ITERATIONS
        state = SwarmState()
        state.requirements = "reqs"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_fail()),
            patch("orchestrator.reviewer_agent.run"),
        ):
            result = run_swarm(state=state, verbose=False)
        dev_calls = [h for h in result.history if h["agent"] == "dev"]
        assert len(dev_calls) == MAX_ITERATIONS
        assert result.approved is False

    def test_approved_clears_feedback(self):
        state = SwarmState()
        state.requirements = "reqs"
        state.feedback = "old feedback"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result = run_swarm(state=state, verbose=False)
        assert result.feedback is None

    def test_code_stored_on_state(self):
        state = SwarmState()
        state.requirements = "reqs"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result("def bar(): pass")),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            result = run_swarm(state=state, verbose=False)
        assert result.code == "def bar(): pass"
