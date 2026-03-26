"""
Agent Swarm Orchestrator
Manages the loop: Requirements → per-task Dev → QA → Reviewer → (repeat or done)
"""

import json
import os
import re
from agents import pm_agent, dev_agent, qa_agent, reviewer_agent
from models import BillingError
from models import SwarmState, AgentResult
from scout import scan_project
from spinner import Spinner
from writer import write_files


MAX_ITERATIONS = 5


def _mark_task_complete(tasks_doc_path: str, task_id: str) -> None:
    """
    Update the **Status:** line for `task_id` in the TASKS.md file to 'complete'.
    No-ops silently if the file doesn't exist or the task ID isn't found.
    """
    if not tasks_doc_path:
        return
    try:
        with open(tasks_doc_path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return

    lines = content.splitlines(keepends=True)
    task_heading_re = re.compile(rf'\b{re.escape(task_id)}\b')
    for i, line in enumerate(lines):
        if line.startswith("#") and task_heading_re.search(line):
            # Found the task heading — update the first **Status:** in the next ~10 lines
            for j in range(i + 1, min(i + 10, len(lines))):
                if "**Status:**" in lines[j]:
                    lines[j] = re.sub(r"\*\*Status:\*\*\s*\S+", "**Status:** complete", lines[j])
                    with open(tasks_doc_path, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    return
                if lines[j].startswith("#"):
                    break  # hit the next section without finding a status line


def _write_checkpoint(state: SwarmState, original_requirements: str, path: str) -> None:
    """Persist enough state to resume a crashed run via --resume.

    Writes to a temp file first, then renames atomically so a failed write
    never leaves the checkpoint file truncated to 0 bytes.
    """
    data = {
        "version": 1,
        "feature_request": state.feature_request,
        "output_dir": state.output_dir,
        "requirements": original_requirements,
        "completed_tasks": state.completed_tasks,
        "pending_tasks": state.pending_tasks,
        "skipped_tasks": [
            {"task": t, "reason": state._skip_reasons[i] if i < len(state._skip_reasons) else "unknown"}
            for i, t in enumerate(state._skipped_tasks)
        ],
        "history": state.history,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _split_tasks(pm_output: str) -> list[str]:
    """
    Parse PM output into individual per-task requirement strings.

    Each returned string contains the milestone header, one task block,
    and the global constraints — enough context for the Dev agent to work
    on a single task without seeing the entire milestone at once.

    Falls back to [pm_output] for simple/free-form inputs that don't
    follow the structured PM output format (e.g. in tests).
    """
    if "TASKS:" not in pm_output or not re.search(r'\[\d+\.\d+\]', pm_output):
        return [pm_output]

    # Milestone header (first line starting with MILESTONE:)
    milestone = next(
        (l for l in pm_output.splitlines() if l.startswith("MILESTONE:")), ""
    )

    # Global constraints block (everything after GLOBAL CONSTRAINTS:)
    constraints = ""
    if "GLOBAL CONSTRAINTS:" in pm_output:
        constraints = "\nGLOBAL CONSTRAINTS:" + pm_output.split("GLOBAL CONSTRAINTS:", 1)[1]

    # Tasks section (between TASKS: and GLOBAL CONSTRAINTS:)
    tasks_text = pm_output.split("TASKS:", 1)[1]
    if "GLOBAL CONSTRAINTS:" in tasks_text:
        tasks_text = tasks_text.split("GLOBAL CONSTRAINTS:", 1)[0]

    # Split on blank line immediately before a task marker "  [X.Y]"
    blocks = re.split(r'\n\n(?=  \[\d)', tasks_text.strip())

    result = []
    for block in blocks:
        block = block.strip()
        if block:
            result.append(f"{milestone}\n\nTASK:\n  {block}{constraints}")

    return result if result else [pm_output]


def run_swarm(state: SwarmState, verbose: bool = True, checkpoint_path: str | None = None) -> SwarmState:
    """
    Main entry point. Accepts a pre-built SwarmState, runs the full pipeline,
    and returns the final state.

    Phase 1: PM agent produces requirements and splits them into a task queue.
    Phase 2: Each task runs its own Dev → QA → Reviewer loop independently.
             Files are written after each task approval, and the project is
             re-scanned so the next task's Dev agent sees what was just built.
    """
    def log(msg: str):
        if verbose:
            print(msg)

    # --- Phase 1: Requirements (runs once, unless already populated) ---
    if state.requirements:
        log("\n⏭  [PM Agent] Requirements already set — skipping.")
    else:
        with Spinner("\n🧠 [PM Agent] Generating requirements"):
            result = pm_agent.run(state)
        state.requirements = result.output
        state.history.append({"agent": "pm", "output": result.output})
        log(f"✅ Requirements done.\n{result.output}\n")

    # Capture full PM output before the per-task loop overwrites state.requirements
    original_requirements: str = state.requirements or ""

    # Populate task queue from PM output (only if not already set — allows
    # callers to inject pending_tasks directly for testing or resumption)
    if not state.pending_tasks:
        state.pending_tasks = _split_tasks(state.requirements)
        if len(state.pending_tasks) > 1:
            log(f"\n📋 Split into {len(state.pending_tasks)} tasks — running each through Dev→QA→Reviewer separately.")

    # --- Phase 2: per-task loop ---
    while state.pending_tasks:
        current_task = state.pending_tasks.pop(0)
        task_num = len(state.completed_tasks) + 1
        total_tasks = task_num + len(state.pending_tasks)

        log(f"\n{'='*50}")
        log(f"📋 Task {task_num}/{total_tasks}")
        log(f"{'='*50}")
        log(current_task)

        # Reset per-task state so Dev starts fresh on each task
        state.requirements = current_task
        state.dev_messages = []
        state.code = None
        state.qa_report = None
        state.review = None
        state.feedback = None

        task_approved = False
        task_id_match = re.search(r'\[(\d+\.\d+)\]', current_task)
        task_label = f" [{task_id_match.group(1)}]" if task_id_match else ""

        for iteration in range(1, MAX_ITERATIONS + 1):
            log(f"\n🔁 Iteration {iteration}")

            # Dev
            try:
                with Spinner(f"\n💻 [Dev Agent] Writing code{task_label}") as sp:
                    result = dev_agent.run(state, spinner=sp)
            except BillingError:
                raise
            except RuntimeError as e:
                reason = f"Dev agent failed: {e}"
                log(f"\n❌ {reason}")
                log("Skipping task.")
                state._skipped_tasks.append(current_task)
                state._skip_reasons.append(reason)
                state.history.append({"agent": "dev", "iteration": iteration, "task": task_num, "error": str(e), "skipped": True})
                break
            state.code = result.output
            state.history.append({"agent": "dev", "iteration": iteration, "task": task_num, "output": result.output})
            log(f"✅ Code written.\n{result.output}\n")

            # QA
            try:
                with Spinner(f"\n🧪 [QA Agent] Testing code{task_label}") as sp:
                    result = qa_agent.run(state, spinner=sp)
            except BillingError:
                raise
            except RuntimeError as e:
                reason = f"QA agent failed: {e}"
                log(f"\n❌ {reason}")
                log("Skipping task.")
                state._skipped_tasks.append(current_task)
                state._skip_reasons.append(reason)
                break
            state.qa_report = result.output
            state.history.append({"agent": "qa", "iteration": iteration, "task": task_num, "output": result.output})
            log(f"✅ QA report:\n{result.output}\n")

            if not result.passed:
                log("❌ QA failed. Sending feedback to Dev agent for next iteration.")
                state.feedback = result.feedback
                continue

            # Reviewer (only if QA passed)
            try:
                with Spinner(f"\n🔍 [Reviewer Agent] Reviewing code{task_label}") as sp:
                    result = reviewer_agent.run(state, spinner=sp)
            except BillingError:
                raise
            except RuntimeError as e:
                reason = f"Reviewer agent failed: {e}"
                log(f"\n❌ {reason}")
                log("Skipping task.")
                state._skipped_tasks.append(current_task)
                state._skip_reasons.append(reason)
                break
            state.review = result.output
            state.history.append({"agent": "reviewer", "iteration": iteration, "task": task_num, "output": result.output})
            log(f"✅ Review:\n{result.output}\n")

            if result.passed:
                log(f"\n✅ Task {task_num}/{total_tasks} approved after {iteration} iteration(s).")
                task_approved = True
                state.feedback = None

                written = write_files(state.code, state.output_dir)
                if written:
                    log(f"\n[writer] Wrote {len(written)} file(s):")
                    for path in written:
                        log(f"  {path}")
                else:
                    log("\n[writer] No FILE blocks found in output — code printed above only.")

                state.completed_tasks.append(current_task)

                # Mark the task complete in TASKS.md on disk so the next run's
                # PM agent doesn't re-process it
                task_id_match = re.search(r'\[(\d+\.\d+)\]', current_task)
                if task_id_match:
                    _mark_task_complete(state.tasks_doc_path, task_id_match.group(1))

                if checkpoint_path:
                    _write_checkpoint(state, original_requirements, checkpoint_path)

                # Re-scan project so the next task's Dev agent sees files just written
                if state.pending_tasks:
                    state.project_context = scan_project(state.output_dir, interactive=False)
                    log(f"\n[scout] Re-scanned project ({len(state.project_context):,} chars).")

                break
            else:
                log("🔄 Reviewer requested changes. Sending feedback to Dev agent.")
                state.feedback = result.feedback
        else:
            reason = f"Exhausted {MAX_ITERATIONS} iterations without approval"
            log(f"\n⚠️  Task {task_num}/{total_tasks} failed after {MAX_ITERATIONS} iterations — skipping.")
            state._skipped_tasks.append(current_task)
            state._skip_reasons.append(reason)

    # --- Done ---
    total = len(state.completed_tasks) + len(state._skipped_tasks)
    skipped = len(state._skipped_tasks)

    if state.completed_tasks and skipped == 0:
        log(f"\n🎉 Done! All {len(state.completed_tasks)} task(s) completed.")
        state.approved = True
    elif state.completed_tasks and skipped > 0:
        log(f"\n⚠️  Partial completion: {len(state.completed_tasks)}/{total} task(s) completed, {skipped} skipped.")
        log("   Skipped tasks:")
        for i, t in enumerate(state._skipped_tasks):
            m = re.search(r'\[(\d+\.\d+)\]', t)
            label = f"[{m.group(1)}]" if m else "(unknown)"
            reason = state._skip_reasons[i] if i < len(state._skip_reasons) else "unknown"
            log(f"   • {label}  ← {reason}")
        log("\n   Re-run to retry skipped tasks, or check the output and fix manually.")
        state.approved = False
    else:
        log(f"\n⚠️  No tasks completed.")

    return state
