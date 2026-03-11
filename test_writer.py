"""Tests for writer.py — FILE block parsing and file writing."""

from pathlib import Path
import pytest

from writer import parse_files, write_files


SINGLE_FILE_CODE = """\
--- FILE: src/main.py ---
def hello():
    return "hello"
"""

MULTI_FILE_CODE = """\
--- FILE: src/app.py ---
from flask import Flask
app = Flask(__name__)

--- FILE: src/models.py ---
class User:
    pass

--- FILE: tests/test_app.py ---
def test_placeholder():
    assert True
"""

NO_SEPARATOR_CODE = """\
def hello():
    return "hello"
"""


class TestParseFiles:
    def test_single_file(self):
        files = parse_files(SINGLE_FILE_CODE)
        assert len(files) == 1
        assert files[0][0] == "src/main.py"
        assert 'def hello():' in files[0][1]

    def test_multiple_files(self):
        files = parse_files(MULTI_FILE_CODE)
        assert len(files) == 3

    def test_paths_correct(self):
        files = parse_files(MULTI_FILE_CODE)
        paths = [f[0] for f in files]
        assert "src/app.py" in paths
        assert "src/models.py" in paths
        assert "tests/test_app.py" in paths

    def test_contents_correct(self):
        files = parse_files(MULTI_FILE_CODE)
        by_path = dict(files)
        assert "Flask" in by_path["src/app.py"]
        assert "class User" in by_path["src/models.py"]
        assert "test_placeholder" in by_path["tests/test_app.py"]

    def test_no_separators_returns_empty(self):
        assert parse_files(NO_SEPARATOR_CODE) == []

    def test_empty_string_returns_empty(self):
        assert parse_files("") == []

    def test_path_whitespace_stripped(self):
        code = "--- FILE:   src/foo.py   ---\nx = 1\n"
        files = parse_files(code)
        assert files[0][0] == "src/foo.py"

    def test_content_trailing_newlines_stripped(self):
        code = "--- FILE: foo.py ---\nx = 1\n\n\n"
        files = parse_files(code)
        assert not files[0][1].endswith("\n")

    def test_windows_style_path(self):
        # Paths should be stored as-is; writer handles platform joining
        code = "--- FILE: src/components/Button.tsx ---\nexport const Button = () => null;\n"
        files = parse_files(code)
        assert files[0][0] == "src/components/Button.tsx"


class TestWriteFiles:
    def test_writes_single_file(self, tmp_path):
        written = write_files(SINGLE_FILE_CODE, str(tmp_path))
        assert len(written) == 1
        dest = tmp_path / "src" / "main.py"
        assert dest.exists()

    def test_file_contents_correct(self, tmp_path):
        write_files(SINGLE_FILE_CODE, str(tmp_path))
        content = (tmp_path / "src" / "main.py").read_text()
        assert "def hello():" in content

    def test_writes_multiple_files(self, tmp_path):
        written = write_files(MULTI_FILE_CODE, str(tmp_path))
        assert len(written) == 3

    def test_creates_parent_directories(self, tmp_path):
        write_files(MULTI_FILE_CODE, str(tmp_path))
        assert (tmp_path / "src").is_dir()
        assert (tmp_path / "tests").is_dir()

    def test_overwrites_existing_file_silently(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# old content")
        write_files(SINGLE_FILE_CODE, str(tmp_path))
        content = (tmp_path / "src" / "main.py").read_text()
        assert "# old content" not in content
        assert "def hello():" in content

    def test_returns_written_paths(self, tmp_path):
        written = write_files(MULTI_FILE_CODE, str(tmp_path))
        assert all(Path(p).exists() for p in written)

    def test_no_separators_writes_nothing(self, tmp_path):
        written = write_files(NO_SEPARATOR_CODE, str(tmp_path))
        assert written == []
        assert list(tmp_path.iterdir()) == []

    def test_empty_string_writes_nothing(self, tmp_path):
        written = write_files("", str(tmp_path))
        assert written == []

    def test_default_output_dir_is_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_files(SINGLE_FILE_CODE)
        assert (tmp_path / "src" / "main.py").exists()

    def test_deeply_nested_path(self, tmp_path):
        code = "--- FILE: a/b/c/d/deep.py ---\nx = 1\n"
        write_files(code, str(tmp_path))
        assert (tmp_path / "a" / "b" / "c" / "d" / "deep.py").exists()
