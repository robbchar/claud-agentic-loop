"""
Entry point. Run this to kick off the swarm.

Usage:
    python main.py
    python main.py --request "build a rate limiter"
    python main.py --file request.txt
    python main.py --file request.txt --quiet

Spec-driven mode (reads docs instead of a free-form request):
    python main.py --architecture docs/ARCHITECTURE.md --tasks docs/TASKS.md
    python main.py  # uses default paths docs/ARCHITECTURE.md + docs/TASKS.md if they exist

Dry-run (verify file loading without calling the Claude API):
    python main.py --architecture docs/ARCHITECTURE.md --tasks docs/TASKS.md --dry-run
"""

import argparse
import json
import os
import sys
from models import SwarmState
from orchestrator import run_swarm


DEFAULT_REQUEST = """
Build a Python function that validates email addresses. It should:
- Check format using regex
- Verify the domain has MX records (using dnspython)
- Return a result object with is_valid, reason, and domain fields
"""

DEFAULT_ARCHITECTURE_PATH = "docs/ARCHITECTURE.md"
DEFAULT_TASKS_PATH = "docs/TASKS.md"


def _read_file(path: str, label: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the agent swarm")
    request_group = parser.add_mutually_exclusive_group()
    request_group.add_argument("--request", type=str, help="Feature request as a string")
    request_group.add_argument("--file", type=str, help="Path to a file containing the feature request")
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help=f"Path to architecture doc for spec-driven mode (default: {DEFAULT_ARCHITECTURE_PATH})",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=f"Path to tasks doc for spec-driven mode (default: {DEFAULT_TASKS_PATH})",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print state then exit without calling the Claude API",
    )
    args = parser.parse_args()

    if args.file:
        feature_request = _read_file(args.file, "--file")
    else:
        feature_request = args.request or DEFAULT_REQUEST

    # Validate sandbox at startup so failures are obvious, not cryptic mid-run
    if os.environ.get("SWARM_SANDBOX", "false").lower() == "true":
        from sandbox import check_sandbox_available
        available, msg = check_sandbox_available()
        if not available:
            print(f"Error: sandbox requested but not available:\n  {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"Sandbox: {msg}")

    state = SwarmState(feature_request=feature_request)

    # Populate spec docs. Explicit flags always load (error if missing).
    # Default paths are loaded silently only if the file exists.
    arch_path = args.architecture
    tasks_path = args.tasks

    if arch_path is not None:
        state.architecture = _read_file(arch_path, "--architecture")
        print(f"[spec] Loaded architecture doc: {arch_path} ({len(state.architecture)} chars)")
    elif os.path.exists(DEFAULT_ARCHITECTURE_PATH):
        state.architecture = _read_file(DEFAULT_ARCHITECTURE_PATH, "--architecture")
        print(f"[spec] Loaded architecture doc: {DEFAULT_ARCHITECTURE_PATH} ({len(state.architecture)} chars)")

    if tasks_path is not None:
        state.tasks_doc = _read_file(tasks_path, "--tasks")
        print(f"[spec] Loaded tasks doc: {tasks_path} ({len(state.tasks_doc)} chars)")
    elif os.path.exists(DEFAULT_TASKS_PATH):
        state.tasks_doc = _read_file(DEFAULT_TASKS_PATH, "--tasks")
        print(f"[spec] Loaded tasks doc: {DEFAULT_TASKS_PATH} ({len(state.tasks_doc)} chars)")

    if args.dry_run:
        print("\n--- DRY RUN STATE ---")
        print(f"feature_request: {state.feature_request[:120]!r}{'...' if len(state.feature_request) > 120 else ''}")
        print(f"architecture:    {len(state.architecture)} chars loaded" if state.architecture else "architecture:    (not loaded)")
        print(f"tasks_doc:       {len(state.tasks_doc)} chars loaded" if state.tasks_doc else "tasks_doc:       (not loaded)")
        print("No API calls made.")
        return

    state = run_swarm(state=state, verbose=not args.quiet)

    print("\n" + "="*50)
    print("FINAL STATE SUMMARY")
    print("="*50)
    print(f"Approved: {state.approved}")
    print(f"Iterations: {len([h for h in state.history if h['agent'] == 'dev'])}")

    if state.approved and state.code:
        print("\n--- FINAL CODE ---")
        print(state.code)

    # Dump full history to file for inspection
    with open("swarm_run.json", "w") as f:
        json.dump(state.history, f, indent=2)
    print("\n📄 Full history written to swarm_run.json")


if __name__ == "__main__":
    main()
