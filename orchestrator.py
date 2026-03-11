"""
Agent Swarm Orchestrator
Manages the loop: Requirements → Dev → QA → Reviewer → (repeat or done)
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from agents import pm_agent, dev_agent, qa_agent, reviewer_agent
from models import SwarmState, AgentResult
from writer import write_files


MAX_ITERATIONS = 5


def run_swarm(state: SwarmState, verbose: bool = True) -> SwarmState:
    """
    Main entry point. Accepts a pre-built SwarmState (so callers can populate
    spec docs or pre-set requirements before running), runs the full pipeline,
    and returns the final state (including all artifacts produced).
    """
    def log(msg: str):
        if verbose:
            print(msg)

    # --- Phase 1: Requirements (runs once, unless already populated) ---
    if state.requirements:
        log("\n⏭  [PM Agent] Requirements already set — skipping.")
    else:
        log("\n🧠 [PM Agent] Generating requirements...")
        result = pm_agent.run(state)
        state.requirements = result.output
        state.history.append({"agent": "pm", "output": result.output})
        log(f"✅ Requirements done.\n{result.output}\n")

    # --- Phase 2: Dev → QA → Reviewer loop ---
    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"\n{'='*50}")
        log(f"🔁 Iteration {iteration}")
        log(f"{'='*50}")

        # Dev
        log("\n💻 [Dev Agent] Writing code...")
        result = dev_agent.run(state)
        state.code = result.output
        state.history.append({"agent": "dev", "iteration": iteration, "output": result.output})
        log(f"✅ Code written.\n{result.output}\n")

        # QA
        log("\n🧪 [QA Agent] Testing code...")
        result = qa_agent.run(state)
        state.qa_report = result.output
        state.history.append({"agent": "qa", "iteration": iteration, "output": result.output})
        log(f"✅ QA report:\n{result.output}\n")

        if not result.passed:
            log("❌ QA failed. Sending feedback to Dev agent for next iteration.")
            state.feedback = result.feedback
            continue  # loop back to Dev with QA feedback

        # Reviewer (only if QA passed)
        log("\n🔍 [Reviewer Agent] Reviewing code...")
        result = reviewer_agent.run(state)
        state.review = result.output
        state.history.append({"agent": "reviewer", "iteration": iteration, "output": result.output})
        log(f"✅ Review:\n{result.output}\n")

        if result.passed:
            log(f"\n🎉 Code approved after {iteration} iteration(s)! Ready for PR.")
            state.approved = True
            state.feedback = None
            written = write_files(state.code, state.output_dir)
            if written:
                log(f"\n[writer] Wrote {len(written)} file(s):")
                for path in written:
                    log(f"  {path}")
            else:
                log("\n[writer] No FILE blocks found in output — code printed above only.")
            break
        else:
            log(f"🔄 Reviewer requested changes. Sending feedback to Dev agent.")
            state.feedback = result.feedback
    else:
        log(f"\n⚠️  Max iterations ({MAX_ITERATIONS}) reached without approval.")

    return state
