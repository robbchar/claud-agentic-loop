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

Write output to a specific directory (defaults to current directory):
    python main.py --output-dir ../my-project

Dry-run (verify file loading and project discovery without calling the Claude API):
    python main.py --architecture docs/ARCHITECTURE.md --tasks docs/TASKS.md --dry-run

MCP / Claude Code subprocess mode:
    By default, the Dev and QA agents shell out to `claude -p` instead of calling
    the Anthropic API directly. This gives them access to your locally configured
    MCP servers (e.g. Context7 for library docs, Chrome DevTools for UI verification)
    while keeping tool access locked down via --allowedTools (no Bash, no file writes).

    Override which agents use the subprocess client:
        SWARM_CC_AGENTS=dev,qa python main.py              # default
        SWARM_CC_AGENTS=dev,qa,reviewer python main.py     # reviewer on subscription too
        SWARM_CC_AGENTS=dev,qa,reviewer,pm python main.py  # all on subscription (no API cost)
        SWARM_CC_AGENTS= python main.py                    # disable for all (pure API mode)

    Note: PM produces structured JSON — the subprocess path is slightly less reliable
    than the direct API for this. Recommended only if minimising API spend.

    Requires: `claude` CLI in PATH and an active Claude Code session.
"""

import argparse
import json
import os
import sys
from models import SwarmState
from orchestrator import run_swarm
from scout import scan_project


DEFAULT_REQUEST = """
Build a Python function that validates email addresses. It should:
- Check format using regex
- Verify the domain has MX records (using dnspython)
- Return a result object with is_valid, reason, and domain fields
"""

DEFAULT_ARCHITECTURE_PATH = "docs/ARCHITECTURE.md"
DEFAULT_TASKS_PATH = "docs/TASKS.md"


def _milestone_name(state) -> str:
    """Extract the milestone name from the first completed task string."""
    for task in state.completed_tasks:
        for line in task.splitlines():
            if line.startswith("MILESTONE:"):
                return line.replace("MILESTONE:", "").strip()
    return "Milestone"


def _checkpoint_section(tasks_doc: str) -> str:
    """
    Pull the first CHECKPOINT section out of a TASKS.md-style doc.
    Returns an empty string if none is found.
    """
    if not tasks_doc or "CHECKPOINT" not in tasks_doc:
        return ""
    lines = tasks_doc.splitlines()
    collecting = False
    result = []
    for line in lines:
        if not collecting:
            if "CHECKPOINT" in line:
                collecting = True
        else:
            # Stop at the next top-level heading (## Milestone N+1)
            if line.startswith("## ") and "CHECKPOINT" not in line:
                break
            result.append(line)
    return "\n".join(result).strip()


def _next_run_cmd(args) -> str:
    """Rebuild the main.py invocation from the current args (without --resume)."""
    parts = ["python main.py"]
    if args.file:
        parts.append(f'--file "{args.file}"')
    elif args.request:
        parts.append(f'--request "{args.request}"')
    if args.architecture:
        parts.append(f'--architecture "{args.architecture}"')
    if args.tasks:
        parts.append(f'--tasks "{args.tasks}"')
    if args.output_dir and args.output_dir != ".":
        parts.append(f'--output-dir "{args.output_dir}"')
    if args.quiet:
        parts.append("--quiet")
    return " ".join(parts)


def _startup_commands(output_dir: str, project_context: str) -> list[str]:
    """
    Infer the commands needed to start the project for manual verification.
    Returns a list of shell command strings.
    """
    ctx = project_context.lower()
    out = os.path.abspath(output_dir)
    cmds = []

    if "express" in ctx or "node" in ctx or "package.json" in ctx:
        cmds.append(f"cd {out}")
        if "npm install" in ctx or "node_modules" not in ctx:
            cmds.append("npm install")
        cmds.append("node server/index.js")

    if "vite" in ctx or "react" in ctx:
        cmds.append("# (second terminal) cd client && npm run dev")

    return cmds


def _print_milestone_complete(state, args) -> None:
    name = _milestone_name(state)
    width = 60
    print("\n" + "=" * width)
    print(f"  🎉  MILESTONE COMPLETE: {name}")
    print("=" * width)
    print(f"\n  {len(state.completed_tasks)} task(s) approved and written to disk.")

    # Show startup commands so the user knows how to bring the project up
    startup = _startup_commands(state.output_dir, state.project_context)
    if startup:
        print("\n🚀  Start the project first:\n")
        for cmd in startup:
            print(f"    {cmd}")

    checkpoint = _checkpoint_section(state.tasks_doc)
    if checkpoint:
        print("\n📋  Then verify manually:\n")
        print(checkpoint)
    else:
        print("\n📋  Then spot-check the output:")
        for task in state.completed_tasks:
            for line in task.splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    print(f"   • {stripped}")
                    break

    print(f"\n▶   When ready, start the next milestone:\n")
    print(f"    {_next_run_cmd(args)}\n")
    print("=" * width)


def _read_file(path: str, label: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)


def _check_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=your_key_here", file=sys.stderr)
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write generated files into (default: current directory)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="swarm_run.json",
        metavar="CHECKPOINT",
        help="Resume from a previous run's checkpoint file (default: swarm_run.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load state and scan project, then exit without calling the Claude API",
    )
    args = parser.parse_args()

    if not args.dry_run:
        from agents import CC_AGENTS
        _api_agents = {"dev", "qa", "reviewer", "pm"} - CC_AGENTS
        if _api_agents:
            _check_env()

    # Validate sandbox at startup so failures are obvious, not cryptic mid-run
    if os.environ.get("SWARM_SANDBOX", "false").lower() == "true":
        from sandbox import check_sandbox_available
        available, msg = check_sandbox_available()
        if not available:
            print(f"Error: sandbox requested but not available:\n  {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"Sandbox: {msg}")

    checkpoint_path = "swarm_run.json"

    if args.resume:
        # --- Resume from checkpoint ---
        try:
            with open(args.resume, encoding="utf-8") as f:
                checkpoint = json.load(f)
        except FileNotFoundError:
            print(f"Error: checkpoint file not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: checkpoint file is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)

        completed = checkpoint.get("completed_tasks", [])
        pending = checkpoint.get("pending_tasks", [])
        # Skipped tasks are re-queued so --resume retries them.
        skipped_entries = checkpoint.get("skipped_tasks", [])
        skipped = [e["task"] if isinstance(e, dict) else e for e in skipped_entries]
        retry = skipped + pending  # skipped first so they run before any remaining tasks

        print(f"[resume] Loaded checkpoint: {args.resume}")
        print(f"[resume] {len(completed)} task(s) done, {len(skipped)} skipped (will retry), {len(pending)} pending")

        if not retry:
            print("[resume] No pending or skipped tasks — nothing to do.")
            return

        if skipped:
            for e in skipped_entries:
                task_text = e["task"] if isinstance(e, dict) else e
                reason = e.get("reason", "unknown") if isinstance(e, dict) else "unknown"
                import re as _re
                m = _re.search(r'\[(\d+\.\d+\w*)\]', task_text)
                label = f"[{m.group(1)}]" if m else "(unknown)"
                print(f"[resume]   retrying {label}  ← was skipped: {reason}")

        state = SwarmState(
            feature_request=checkpoint.get("feature_request", ""),
            output_dir=checkpoint.get("output_dir", args.output_dir),
            requirements=checkpoint.get("requirements") or "resumed",
            completed_tasks=completed,
            pending_tasks=retry,
        )
    else:
        # --- Normal startup ---
        if args.file:
            feature_request = _read_file(args.file, "--file")
        else:
            feature_request = args.request or DEFAULT_REQUEST

        state = SwarmState(feature_request=feature_request, output_dir=args.output_dir)

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
            state.tasks_doc_path = os.path.abspath(tasks_path)
            print(f"[spec] Loaded tasks doc: {tasks_path} ({len(state.tasks_doc)} chars)")
        elif os.path.exists(DEFAULT_TASKS_PATH):
            state.tasks_doc = _read_file(DEFAULT_TASKS_PATH, "--tasks")
            state.tasks_doc_path = os.path.abspath(DEFAULT_TASKS_PATH)
            print(f"[spec] Loaded tasks doc: {DEFAULT_TASKS_PATH} ({len(state.tasks_doc)} chars)")

    # Project discovery — scan the output directory so the Dev agent knows
    # what it's writing into. Interactive prompts for large files unless dry-run.
    print(f"\n[scout] Scanning project: {os.path.abspath(state.output_dir)}")
    state.project_context = scan_project(
        state.output_dir,
        interactive=not args.dry_run,
    )
    # Print just the framework line so the user gets quick feedback
    for line in state.project_context.splitlines():
        if line.startswith("Framework"):
            print(f"[scout] {line}")
            break

    if args.dry_run:
        print("\n--- DRY RUN STATE ---")
        print(f"feature_request:  {state.feature_request[:120]!r}{'...' if len(state.feature_request) > 120 else ''}")
        print(f"architecture:     {len(state.architecture)} chars loaded" if state.architecture else "architecture:     (not loaded)")
        print(f"tasks_doc:        {len(state.tasks_doc)} chars loaded" if state.tasks_doc else "tasks_doc:        (not loaded)")
        print(f"project_context:  {len(state.project_context)} chars loaded")
        print(f"output_dir:       {os.path.abspath(state.output_dir)}")
        print("No API calls made.")
        return

    state = run_swarm(state=state, verbose=not args.quiet, checkpoint_path=checkpoint_path)

    print("\n" + "=" * 50)
    print("FINAL STATE SUMMARY")
    print("=" * 50)
    print(f"Approved:   {state.approved}")
    print(f"Iterations: {len([h for h in state.history if h['agent'] == 'dev'])}")

    if state.approved and state.code and not state.code.strip().startswith("--- FILE:"):
        # Only print raw code if it wasn't already written to files
        print("\n--- FINAL CODE ---")
        print(state.code)

    # Final checkpoint write — atomic replace so a failure here doesn't wipe
    # the incremental checkpoint written after each completed task.
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "feature_request": state.feature_request,
            "output_dir": state.output_dir,
            "requirements": state.requirements or "",
            "completed_tasks": state.completed_tasks,
            "pending_tasks": state.pending_tasks,
            "skipped_tasks": [
                {"task": t, "reason": state._skip_reasons[i] if i < len(state._skip_reasons) else "unknown"}
                for i, t in enumerate(state._skipped_tasks)
            ],
            "history": state.history,
        }, f, indent=2)
    os.replace(tmp, checkpoint_path)
    print(f"\nFull history written to {checkpoint_path}")

    if state.approved:
        _print_milestone_complete(state, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔  Interrupted.", flush=True)
        sys.exit(1)
    except Exception as e:
        from models import BillingError
        if isinstance(e, BillingError):
            print(f"\n\n💳  {e}", flush=True)
            print("Once topped up, resume where you left off:", flush=True)
            print("  python main.py --resume", flush=True)
            sys.exit(2)
        raise
