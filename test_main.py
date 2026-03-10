"""Tests for main.py — CLI argument parsing, file loading, dry-run, and spec doc loading."""

import sys
from unittest.mock import patch, MagicMock
import pytest

from models import SwarmState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_run_swarm(captured: dict):
    """Returns a run_swarm side_effect that captures the state passed to it."""
    def _side_effect(state: SwarmState, verbose: bool = True) -> SwarmState:
        captured["state"] = state
        state.history = []
        return state
    return _side_effect


def _call_main(argv: list[str], extra_patches: list = ()):
    """
    Call main.main() with sys.argv set to argv.
    Always patches json.dump to prevent swarm_run.json from being written.
    Returns captured stdout via capsys — callers should use capsys fixture directly.
    """
    import main as m
    with (
        patch("sys.argv", ["main.py"] + argv),
        patch("main.json.dump"),   # prevent writing swarm_run.json
        *extra_patches,
    ):
        try:
            m.main()
        except SystemExit:
            pass


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_does_not_call_run_swarm(self, tmp_path):
        arch = tmp_path / "ARCHITECTURE.md"
        tasks = tmp_path / "TASKS.md"
        arch.write_text("arch content")
        tasks.write_text("tasks content")

        run_swarm_mock = MagicMock()
        import main as m
        with (
            patch("main.run_swarm", run_swarm_mock),
            patch("sys.argv", ["main.py",
                               "--architecture", str(arch),
                               "--tasks", str(tasks),
                               "--dry-run"]),
        ):
            m.main()

        run_swarm_mock.assert_not_called()

    def test_dry_run_prints_loaded_char_counts(self, tmp_path, capsys):
        arch = tmp_path / "ARCHITECTURE.md"
        tasks = tmp_path / "TASKS.md"
        arch.write_text("x" * 42)
        tasks.write_text("y" * 17)

        import main as m
        with (
            patch("main.run_swarm"),
            patch("sys.argv", ["main.py",
                               "--architecture", str(arch),
                               "--tasks", str(tasks),
                               "--dry-run"]),
        ):
            m.main()

        out = capsys.readouterr().out
        assert "42 chars" in out
        assert "17 chars" in out

    def test_dry_run_prints_state_block(self, tmp_path, capsys):
        arch = tmp_path / "ARCHITECTURE.md"
        tasks = tmp_path / "TASKS.md"
        arch.write_text("arch")
        tasks.write_text("tasks")

        import main as m
        with (
            patch("main.run_swarm"),
            patch("sys.argv", ["main.py",
                               "--architecture", str(arch),
                               "--tasks", str(tasks),
                               "--dry-run"]),
        ):
            m.main()

        out = capsys.readouterr().out
        assert "DRY RUN STATE" in out
        assert "No API calls made" in out

    def test_dry_run_without_docs_shows_not_loaded(self, tmp_path, capsys):
        import main as m
        with (
            patch("main.run_swarm"),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(tmp_path / "missing.md")),
            patch("main.DEFAULT_TASKS_PATH", str(tmp_path / "missing2.md")),
            patch("sys.argv", ["main.py", "--dry-run"]),
        ):
            m.main()

        out = capsys.readouterr().out
        assert "not loaded" in out


# ---------------------------------------------------------------------------
# Spec doc loading — use dry-run to avoid needing run_swarm or json.dump
# ---------------------------------------------------------------------------

class TestSpecDocLoading:
    def test_explicit_architecture_flag_loads_file(self, tmp_path, capsys):
        arch = tmp_path / "arch.md"
        arch.write_text("my architecture content")

        import main as m
        with (
            patch("main.run_swarm"),
            patch("main.DEFAULT_TASKS_PATH", str(tmp_path / "missing.md")),
            patch("sys.argv", ["main.py", "--architecture", str(arch), "--dry-run"]),
        ):
            m.main()

        out = capsys.readouterr().out
        assert f"{len('my architecture content')} chars" in out

    def test_explicit_tasks_flag_loads_file(self, tmp_path, capsys):
        tasks = tmp_path / "tasks.md"
        tasks.write_text("my tasks content")

        import main as m
        with (
            patch("main.run_swarm"),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(tmp_path / "missing.md")),
            patch("sys.argv", ["main.py", "--tasks", str(tasks), "--dry-run"]),
        ):
            m.main()

        out = capsys.readouterr().out
        assert f"{len('my tasks content')} chars" in out

    def test_both_docs_loaded_into_state(self, tmp_path):
        arch = tmp_path / "ARCHITECTURE.md"
        tasks = tmp_path / "TASKS.md"
        arch.write_text("arch content")
        tasks.write_text("tasks content")

        captured = {}
        import main as m
        with (
            patch("main.run_swarm", side_effect=_fake_run_swarm(captured)),
            patch("main.json.dump"),
            patch("sys.argv", ["main.py",
                               "--architecture", str(arch),
                               "--tasks", str(tasks)]),
        ):
            m.main()

        assert captured["state"].architecture == "arch content"
        assert captured["state"].tasks_doc == "tasks content"

    def test_missing_explicit_architecture_exits_with_error(self, tmp_path):
        import main as m
        with (
            patch("main.run_swarm"),
            patch("sys.argv", ["main.py", "--architecture", str(tmp_path / "nope.md")]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                m.main()
        assert exc_info.value.code == 1

    def test_missing_explicit_tasks_exits_with_error(self, tmp_path):
        import main as m
        with (
            patch("main.run_swarm"),
            patch("sys.argv", ["main.py", "--tasks", str(tmp_path / "nope.md")]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                m.main()
        assert exc_info.value.code == 1

    def test_default_paths_silently_skipped_when_absent(self, tmp_path):
        captured = {}
        import main as m
        with (
            patch("main.run_swarm", side_effect=_fake_run_swarm(captured)),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(tmp_path / "missing.md")),
            patch("main.DEFAULT_TASKS_PATH", str(tmp_path / "missing2.md")),
            patch("main.json.dump"),
            patch("sys.argv", ["main.py"]),
        ):
            m.main()

        assert captured["state"].architecture == ""
        assert captured["state"].tasks_doc == ""

    def test_default_paths_auto_loaded_when_present(self, tmp_path):
        arch = tmp_path / "ARCHITECTURE.md"
        tasks = tmp_path / "TASKS.md"
        arch.write_text("default arch")
        tasks.write_text("default tasks")

        captured = {}
        import main as m
        with (
            patch("main.run_swarm", side_effect=_fake_run_swarm(captured)),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(arch)),
            patch("main.DEFAULT_TASKS_PATH", str(tasks)),
            patch("main.json.dump"),
            patch("sys.argv", ["main.py"]),
        ):
            m.main()

        assert captured["state"].architecture == "default arch"
        assert captured["state"].tasks_doc == "default tasks"


# ---------------------------------------------------------------------------
# Feature request sourcing
# ---------------------------------------------------------------------------

class TestFeatureRequestSourcing:
    def test_request_flag_sets_feature_request(self, tmp_path):
        captured = {}
        import main as m
        with (
            patch("main.run_swarm", side_effect=_fake_run_swarm(captured)),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(tmp_path / "missing.md")),
            patch("main.DEFAULT_TASKS_PATH", str(tmp_path / "missing2.md")),
            patch("main.json.dump"),
            patch("sys.argv", ["main.py", "--request", "build a cache"]),
        ):
            m.main()

        assert captured["state"].feature_request == "build a cache"

    def test_file_flag_reads_feature_request(self, tmp_path):
        req_file = tmp_path / "request.txt"
        req_file.write_text("build a rate limiter")
        captured = {}

        import main as m
        with (
            patch("main.run_swarm", side_effect=_fake_run_swarm(captured)),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(tmp_path / "missing.md")),
            patch("main.DEFAULT_TASKS_PATH", str(tmp_path / "missing2.md")),
            patch("main.json.dump"),
            patch("sys.argv", ["main.py", "--file", str(req_file)]),
        ):
            m.main()

        assert captured["state"].feature_request == "build a rate limiter"

    def test_no_request_uses_default(self, tmp_path):
        captured = {}
        import main as m
        with (
            patch("main.run_swarm", side_effect=_fake_run_swarm(captured)),
            patch("main.DEFAULT_ARCHITECTURE_PATH", str(tmp_path / "missing.md")),
            patch("main.DEFAULT_TASKS_PATH", str(tmp_path / "missing2.md")),
            patch("main.json.dump"),
            patch("sys.argv", ["main.py"]),
        ):
            m.main()

        assert "email" in captured["state"].feature_request.lower()

    def test_missing_file_exits_with_error(self, tmp_path):
        import main as m
        with (
            patch("main.run_swarm"),
            patch("sys.argv", ["main.py", "--file", str(tmp_path / "nope.txt")]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                m.main()
        assert exc_info.value.code == 1
