"""Tests for orchestrator.py — loop control, PM skip, iteration logic, per-task splitting."""

import contextlib
import json
from unittest.mock import patch, MagicMock
import pytest
from models import SwarmState, AgentResult
from orchestrator import run_swarm, _split_tasks, _mark_task_complete, MAX_ITERATIONS


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


# ---------------------------------------------------------------------------
# _split_tasks
# ---------------------------------------------------------------------------

PM_OUTPUT_TWO_TASKS = """\
MILESTONE: M1 — Core

TASKS:
  [1.1] Set up scaffold
  Acceptance criteria:
    - package.json exists
    - server/index.js exists

  [1.2] Add deck service
  Acceptance criteria:
    - server/services/deckService.js exists

GLOBAL CONSTRAINTS:
  - Node.js backend
  - JSON storage"""


class TestSplitTasks:
    def test_simple_string_returns_single_task(self):
        result = _split_tasks("build a rate limiter")
        assert result == ["build a rate limiter"]

    def test_splits_two_tasks(self):
        result = _split_tasks(PM_OUTPUT_TWO_TASKS)
        assert len(result) == 2

    def test_each_task_contains_task_block(self):
        result = _split_tasks(PM_OUTPUT_TWO_TASKS)
        assert "1.1" in result[0]
        assert "1.2" in result[1]

    def test_each_task_contains_global_constraints(self):
        result = _split_tasks(PM_OUTPUT_TWO_TASKS)
        assert "GLOBAL CONSTRAINTS:" in result[0]
        assert "Node.js backend" in result[0]
        assert "GLOBAL CONSTRAINTS:" in result[1]

    def test_each_task_contains_milestone(self):
        result = _split_tasks(PM_OUTPUT_TWO_TASKS)
        assert "MILESTONE: M1" in result[0]
        assert "MILESTONE: M1" in result[1]

    def test_tasks_do_not_contain_other_task_blocks(self):
        result = _split_tasks(PM_OUTPUT_TWO_TASKS)
        assert "1.2" not in result[0]
        assert "1.1" not in result[1]

    def test_no_tasks_section_returns_single(self):
        result = _split_tasks("MILESTONE: M1\n\nsome freeform text")
        assert len(result) == 1
        assert result[0] == "MILESTONE: M1\n\nsome freeform text"

    def test_empty_string_returns_single(self):
        result = _split_tasks("")
        assert result == [""]


# ---------------------------------------------------------------------------
# PM Phase
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Single-task loop (backward-compatible behaviour)
# ---------------------------------------------------------------------------

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
        qa_results = [_make_qa_fail("fix X"), _make_qa_pass()]
        captured_feedback = []

        def dev_side_effect(s, spinner=None):
            captured_feedback.append(s.feedback)
            return _make_dev_result()

        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", side_effect=dev_side_effect),
            patch("orchestrator.qa_agent.run", side_effect=qa_results),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False)
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


# ---------------------------------------------------------------------------
# Multi-task behaviour
# ---------------------------------------------------------------------------

class TestRunSwarmMultiTask:
    def _run_two_tasks(self, state, dev_side_effect=None, qa_side_effect=None):
        """Helper: run two-task swarm with standard mocks, return (result, scout_mock)."""
        scout_mock = MagicMock(return_value="project context")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("orchestrator.pm_agent.run"))
            stack.enter_context(patch("orchestrator.dev_agent.run",
                side_effect=dev_side_effect or (lambda s, spinner=None: _make_dev_result())))
            stack.enter_context(patch("orchestrator.qa_agent.run",
                side_effect=qa_side_effect or (lambda s, spinner=None: _make_qa_pass())))
            stack.enter_context(patch("orchestrator.reviewer_agent.run",
                return_value=_make_reviewer_approved()))
            stack.enter_context(patch("orchestrator.scan_project", scout_mock))
            result = run_swarm(state=state, verbose=False)
        return result, scout_mock

    def test_two_tasks_both_run(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        result, _ = self._run_two_tasks(state)
        dev_calls = [h for h in result.history if h["agent"] == "dev"]
        assert len(dev_calls) == 2

    def test_two_tasks_both_completed(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        result, _ = self._run_two_tasks(state)
        assert len(result.completed_tasks) == 2

    def test_approved_true_when_all_tasks_complete(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        result, _ = self._run_two_tasks(state)
        assert result.approved is True

    def test_project_rescanned_between_tasks(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        _, scout_mock = self._run_two_tasks(state)
        # re-scan called once (after task 1, before task 2); not after the last task
        assert scout_mock.call_count == 1

    def test_dev_messages_reset_between_tasks(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        dev_message_lengths = []

        def capture_dev(s, spinner=None):
            dev_message_lengths.append(len(s.dev_messages))
            return _make_dev_result()

        result, _ = self._run_two_tasks(state, dev_side_effect=capture_dev)
        assert dev_message_lengths == [0, 0]

    def test_failed_task_does_not_block_next(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        # First task always fails QA; second passes
        call_count = {"n": 0}
        def qa_side_effect(s, spinner=None):
            call_count["n"] += 1
            # First MAX_ITERATIONS calls fail (task 1); then pass (task 2)
            if call_count["n"] <= MAX_ITERATIONS:
                return _make_qa_fail()
            return _make_qa_pass()

        result, _ = self._run_two_tasks(state, qa_side_effect=qa_side_effect)
        assert len(result.completed_tasks) == 1   # only task 2 completes
        assert result.approved is True

    def test_no_rescan_after_last_task(self):
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        _, scout_mock = self._run_two_tasks(state)
        assert scout_mock.call_count == 1


# ---------------------------------------------------------------------------
# Checkpoint writing
# ---------------------------------------------------------------------------

TASKS_DOC = """\
## Milestone 1

### Task 1.1 - Scaffold
**Status:** pending

### Task 1.2 - Service
**Status:** pending

### Task 1.3 - Routes
**Status:** pending
"""


class TestMarkTaskComplete:
    def test_marks_task_complete(self, tmp_path):
        f = tmp_path / "TASKS.md"
        f.write_text(TASKS_DOC)
        _mark_task_complete(str(f), "1.1")
        assert "**Status:** complete" in f.read_text()

    def test_only_marks_the_right_task(self, tmp_path):
        f = tmp_path / "TASKS.md"
        f.write_text(TASKS_DOC)
        _mark_task_complete(str(f), "1.2")
        content = f.read_text()
        lines = content.splitlines()
        statuses = [l for l in lines if "**Status:**" in l]
        assert statuses[0] == "**Status:** pending"   # 1.1 unchanged
        assert statuses[1] == "**Status:** complete"  # 1.2 updated
        assert statuses[2] == "**Status:** pending"   # 1.3 unchanged

    def test_noop_when_path_empty(self, tmp_path):
        # Should not raise even when path is empty string
        _mark_task_complete("", "1.1")

    def test_noop_when_file_missing(self, tmp_path):
        _mark_task_complete(str(tmp_path / "nope.md"), "1.1")

    def test_noop_when_task_id_not_found(self, tmp_path):
        f = tmp_path / "TASKS.md"
        f.write_text(TASKS_DOC)
        _mark_task_complete(str(f), "9.9")
        assert "**Status:** pending" in f.read_text()


class TestCheckpoint:
    def test_checkpoint_written_after_task_approval(self, tmp_path):
        checkpoint = tmp_path / "run.json"
        state = SwarmState()
        state.requirements = "build a thing"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False, checkpoint_path=str(checkpoint))
        assert checkpoint.exists()

    def test_checkpoint_contains_completed_task(self, tmp_path):
        checkpoint = tmp_path / "run.json"
        state = SwarmState()
        state.requirements = "build a thing"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False, checkpoint_path=str(checkpoint))
        data = json.loads(checkpoint.read_text())
        assert len(data["completed_tasks"]) == 1

    def test_checkpoint_pending_empty_after_completion(self, tmp_path):
        checkpoint = tmp_path / "run.json"
        state = SwarmState()
        state.requirements = "build a thing"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False, checkpoint_path=str(checkpoint))
        data = json.loads(checkpoint.read_text())
        assert data["pending_tasks"] == []

    def test_checkpoint_preserves_original_requirements(self, tmp_path):
        checkpoint = tmp_path / "run.json"
        state = SwarmState()
        state.requirements = PM_OUTPUT_TWO_TASKS
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
            patch("orchestrator.scan_project", return_value="project context"),
        ):
            run_swarm(state=state, verbose=False, checkpoint_path=str(checkpoint))
        data = json.loads(checkpoint.read_text())
        # requirements should be the full PM output, not just the last task string
        assert "MILESTONE" in data["requirements"]
        assert "GLOBAL CONSTRAINTS" in data["requirements"]

    def test_no_checkpoint_when_path_is_none(self, tmp_path):
        state = SwarmState()
        state.requirements = "build a thing"
        with (
            patch("orchestrator.pm_agent.run"),
            patch("orchestrator.dev_agent.run", return_value=_make_dev_result()),
            patch("orchestrator.qa_agent.run", return_value=_make_qa_pass()),
            patch("orchestrator.reviewer_agent.run", return_value=_make_reviewer_approved()),
        ):
            run_swarm(state=state, verbose=False, checkpoint_path=None)
        # No file should be written in the tmp_path (nothing to assert — just no crash)
