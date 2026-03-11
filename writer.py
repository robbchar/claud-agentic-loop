"""
Post-approval file writer.

Parses --- FILE: path --- blocks from the Dev agent's approved output
and writes each file to the output directory.
"""

import re
from pathlib import Path


# Matches:   --- FILE: some/path/file.ext ---
# Captures everything after it up to the next separator or end of string.
_FILE_BLOCK_RE = re.compile(
    r"---\s*FILE:\s*(?P<path>[^\n]+?)\s*---\n(?P<content>.*?)(?=\n---\s*FILE:|$)",
    re.DOTALL,
)


def parse_files(code: str) -> list[tuple[str, str]]:
    """
    Extract (relative_path, content) pairs from FILE-separated code output.
    Returns an empty list if no FILE separators are present.
    """
    return [
        (m.group("path").strip(), m.group("content").rstrip("\n"))
        for m in _FILE_BLOCK_RE.finditer(code)
    ]


def write_files(code: str, output_dir: str = ".") -> list[str]:
    """
    Parse FILE blocks from code and write each to output_dir.

    - Creates parent directories as needed.
    - Overwrites existing files silently.
    - Returns the list of absolute paths written.
    - Returns an empty list if no FILE blocks are found (no files written).
    """
    files = parse_files(code)
    if not files:
        return []

    root = Path(output_dir).resolve()
    written: list[str] = []

    for rel_path, content in files:
        dest = root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        written.append(str(dest))

    return written
