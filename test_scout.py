"""Tests for scout.py — project discovery, framework detection, file scanning."""

import sys
from pathlib import Path
from unittest.mock import patch
import pytest

from scout import detect_framework, scan_project, FILE_SIZE_THRESHOLD


class TestDetectFramework:
    def test_angular(self, tmp_path):
        (tmp_path / "angular.json").write_text("{}")
        assert "Angular" in detect_framework(tmp_path)

    def test_nextjs_ts(self, tmp_path):
        (tmp_path / "next.config.ts").write_text("")
        assert "Next.js" in detect_framework(tmp_path)

    def test_nextjs_js(self, tmp_path):
        (tmp_path / "next.config.js").write_text("")
        assert "Next.js" in detect_framework(tmp_path)

    def test_vite(self, tmp_path):
        (tmp_path / "vite.config.ts").write_text("")
        assert "Vite" in detect_framework(tmp_path)

    def test_python_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        assert "Python" in detect_framework(tmp_path)

    def test_python_requirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("")
        assert "Python" in detect_framework(tmp_path)

    def test_node_fallback(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        result = detect_framework(tmp_path)
        assert "Node.js" in result or "JavaScript" in result

    def test_unknown(self, tmp_path):
        assert detect_framework(tmp_path) == "Unknown"

    def test_indicator_filename_in_output(self, tmp_path):
        (tmp_path / "angular.json").write_text("{}")
        assert "angular.json" in detect_framework(tmp_path)


class TestScanProject:
    def test_empty_dir_shows_greenfield(self, tmp_path):
        result = scan_project(str(tmp_path), interactive=False)
        assert "greenfield" in result

    def test_includes_small_file_contents(self, tmp_path):
        (tmp_path / "hello.py").write_text("print('hello')")
        result = scan_project(str(tmp_path), interactive=False)
        assert "print('hello')" in result

    def test_includes_file_in_tree(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        result = scan_project(str(tmp_path), interactive=False)
        assert "main.py" in result

    def test_framework_in_output(self, tmp_path):
        (tmp_path / "angular.json").write_text("{}")
        result = scan_project(str(tmp_path), interactive=False)
        assert "Angular" in result

    def test_ignores_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "some-pkg"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        result = scan_project(str(tmp_path), interactive=False)
        # Check neither the package dir nor the file contents appear
        assert "some-pkg" not in result
        assert "module.exports" not in result

    def test_ignores_git_dir(self, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main")
        result = scan_project(str(tmp_path), interactive=False)
        assert ".git" not in result

    def test_ignores_pycache(self, tmp_path):
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        (pc / "foo.pyc").write_bytes(b"\x00\x01\x02")
        result = scan_project(str(tmp_path), interactive=False)
        assert "__pycache__" not in result

    def test_large_file_skipped_non_interactive(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * (FILE_SIZE_THRESHOLD // 6 + 100))
        result = scan_project(str(tmp_path), interactive=False)
        # Path appears in tree but contents are NOT included
        assert "big.py" in result
        assert "x = 1" not in result

    def test_large_file_skipped_on_s_choice(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_text("secret = 42\n" * (FILE_SIZE_THRESHOLD // 12 + 100))
        with patch("builtins.input", return_value="s"):
            result = scan_project(str(tmp_path), interactive=True)
        assert "big.py" in result
        assert "secret = 42" not in result

    def test_large_file_quits_on_q_choice(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * (FILE_SIZE_THRESHOLD // 6 + 100))
        with patch("builtins.input", return_value="q"):
            with pytest.raises(SystemExit):
                scan_project(str(tmp_path), interactive=True)

    def test_large_file_reprompts_on_invalid_input(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * (FILE_SIZE_THRESHOLD // 6 + 100))
        # First answer is invalid, second is 's'
        with patch("builtins.input", side_effect=["z", "s"]):
            result = scan_project(str(tmp_path), interactive=True)
        assert "big.py" in result

    def test_file_separator_format_in_contents(self, tmp_path):
        (tmp_path / "app.py").write_text("def main(): pass")
        result = scan_project(str(tmp_path), interactive=False)
        assert "--- FILE: app.py ---" in result

    def test_nested_files_in_tree(self, tmp_path):
        src = tmp_path / "src" / "components"
        src.mkdir(parents=True)
        (src / "Button.tsx").write_text("export const Button = () => <button />;")
        result = scan_project(str(tmp_path), interactive=False)
        assert "Button.tsx" in result

    def test_hidden_files_excluded(self, tmp_path):
        (tmp_path / ".env").write_text("SECRET=abc")
        result = scan_project(str(tmp_path), interactive=False)
        assert "SECRET=abc" not in result

    def test_swarm_run_json_excluded(self, tmp_path):
        (tmp_path / "swarm_run.json").write_text('{"history": []}')
        result = scan_project(str(tmp_path), interactive=False)
        assert "swarm_run.json" not in result
