"""
Project discovery — scans a directory to give the Dev agent context about
the project it is generating code for.

No API calls. Pure filesystem inspection.
"""

import os
import sys
from pathlib import Path


FILE_SIZE_THRESHOLD = 10 * 1024  # 10 KB

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "target", ".turbo", ".cache",
}

IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", ".env", ".env.local", ".env.production",
    "swarm_run.json",
}

FRAMEWORK_INDICATORS: dict[str, str] = {
    "angular.json":       "Angular",
    "next.config.js":     "Next.js",
    "next.config.ts":     "Next.js",
    "next.config.mjs":    "Next.js",
    "vite.config.ts":     "Vite",
    "vite.config.js":     "Vite",
    "svelte.config.js":   "SvelteKit",
    "nuxt.config.ts":     "Nuxt",
    "remix.config.js":    "Remix",
    "astro.config.mjs":   "Astro",
    "pyproject.toml":     "Python",
    "requirements.txt":   "Python",
    "setup.py":           "Python",
    "Cargo.toml":         "Rust",
    "go.mod":             "Go",
    "pom.xml":            "Java/Maven",
    "build.gradle":       "Java/Gradle",
}


def detect_framework(root: Path) -> str:
    for indicator, framework in FRAMEWORK_INDICATORS.items():
        if (root / indicator).exists():
            return f"{framework} (detected from {indicator})"
    if (root / "package.json").exists():
        return "Node.js/JavaScript (detected from package.json)"
    return "Unknown"


def _prompt_large_file(rel_path: str, size_kb: float) -> bool:
    """
    Prompt the user when a file exceeds FILE_SIZE_THRESHOLD.
    Returns True if the file should be skipped.
    Exits the process if the user wants to refactor first.
    """
    print(f"\n  Large file: {rel_path} ({size_kb:.1f} KB)")
    print("  Files over 10 KB are a code smell — consider breaking this down first.")
    print("  [s] Skip  — Dev agent won't see its contents (listed in tree only)")
    print("  [q] Quit  — exit now and refactor the file first (recommended)")
    while True:
        choice = input("  Choice [s/q]: ").strip().lower()
        if choice == "s":
            return True
        if choice == "q":
            print("Good call. Break it down, then re-run.")
            sys.exit(0)
        print("  Please enter 's' or 'q'.")


def scan_project(root_dir: str = ".", interactive: bool = True) -> str:
    """
    Walk root_dir and produce a project context string for the Dev agent.

    For files under FILE_SIZE_THRESHOLD: includes full contents.
    For files over FILE_SIZE_THRESHOLD: prompts the user (interactive=True)
      or skips silently (interactive=False, used in tests / dry-run).

    Returns a formatted string ready to be embedded in the Dev agent prompt.
    """
    root = Path(root_dir).resolve()
    framework = detect_framework(root)

    tree_lines: list[str] = []
    file_contents: list[str] = []

    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)
        # Prune ignored and hidden dirs in-place so os.walk skips them
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in IGNORE_DIRS and not d.startswith(".")
        )

        rel_dir = dirpath.relative_to(root)
        depth = len(rel_dir.parts)
        indent = "  " * depth

        if depth > 0:
            tree_lines.append(f"{indent}{rel_dir.name}/")

        for filename in sorted(filenames):
            if filename in IGNORE_FILES or filename.startswith("."):
                continue

            file_path = dirpath / filename
            rel_path = str(file_path.relative_to(root))
            size = file_path.stat().st_size
            size_kb = size / 1024

            tree_lines.append(f"{'  ' * (depth + 1)}{filename}")

            if size > FILE_SIZE_THRESHOLD:
                if interactive:
                    skipped = _prompt_large_file(rel_path, size_kb)
                    if skipped:
                        continue
                else:
                    # Non-interactive: skip silently, path already in tree
                    continue
            else:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    file_contents.append(f"--- FILE: {rel_path} ---\n{content}")
                except Exception:
                    pass  # binary files, permission errors

    tree_str = "\n".join(tree_lines) or "  (empty — greenfield project)"
    contents_str = "\n\n".join(file_contents) if file_contents else "  (no existing source files)"

    return (
        f"PROJECT CONTEXT\n"
        f"{'=' * 40}\n"
        f"Framework : {framework}\n"
        f"Root      : {root}\n\n"
        f"FILE TREE:\n{tree_str}\n\n"
        f"EXISTING FILE CONTENTS:\n{contents_str}"
    )
