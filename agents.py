"""
The four agents: PM, Dev, QA, Reviewer.

Each agent follows the same pattern:
  1. A SYSTEM_PROMPT that defines its persona and output format
  2. A build_prompt(state) function that assembles the user message from SwarmState
  3. A run(state) function that calls Claude and parses the result into AgentResult

This separation is intentional — it makes each agent independently testable,
and is exactly what LangGraph would model as individual graph nodes.
"""

import os

from claude_client import call_claude, call_claude_json, call_claude_messages
from models import SwarmState, AgentResult

# Sandbox is opt-in — set SWARM_SANDBOX=true to enable container execution.
# If false, QA does static analysis only (no code execution).
SANDBOX_ENABLED = os.environ.get("SWARM_SANDBOX", "false").lower() == "true"


# =============================================================================
# PM AGENT — Spec-driven: reads architecture + task docs and produces structured
#             acceptance criteria per milestone task
# =============================================================================

PM_SYSTEM = """
You are a senior product manager working from existing specification documents.
Your job is NOT to invent requirements — it is to read the provided architecture
and task documents and produce structured, developer-ready acceptance criteria.

Rules:
- Only include tasks from the CURRENT milestone (the first milestone that has
  pending/incomplete tasks).
- For each task produce a list of concrete, testable acceptance criteria derived
  directly from the spec documents.
- If anything in the spec is ambiguous or contradictory, record it as a warning
  on the relevant task. Do NOT block or refuse — output what you can.
- global_constraints should capture cross-cutting concerns from the architecture
  doc (e.g. language, runtime, security requirements).

Output a JSON object with EXACTLY this shape (no extra keys):
{
  "milestone": "milestone name or id",
  "tasks": [
    {
      "id": "task id from the tasks doc",
      "summary": "one sentence description of the task",
      "acceptance_criteria": ["criterion 1", "criterion 2"],
      "warnings": ["ambiguity or gap noted in spec"]
    }
  ],
  "global_constraints": ["constraint 1", "constraint 2"]
}
"""

class pm_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        prompt = (
            f"ARCHITECTURE DOCUMENT:\n{state.architecture}\n\n"
            f"TASKS DOCUMENT:\n{state.tasks_doc}"
        )
        result = call_claude_json(PM_SYSTEM, prompt)

        tasks_text = ""
        for t in result.get("tasks", []):
            criteria = "\n".join(f"    - {c}" for c in t.get("acceptance_criteria", []))
            warnings = "\n".join(f"    ⚠ {w}" for w in t.get("warnings", []))
            tasks_text += (
                f"\n  [{t['id']}] {t['summary']}\n"
                f"  Acceptance criteria:\n{criteria}\n"
            )
            if warnings:
                tasks_text += f"  Warnings:\n{warnings}\n"

        constraints_text = "\n".join(
            f"  - {c}" for c in result.get("global_constraints", [])
        ) or "  None"

        output = (
            f"MILESTONE: {result.get('milestone', 'unknown')}\n\n"
            f"TASKS:{tasks_text}\n"
            f"GLOBAL CONSTRAINTS:\n{constraints_text}"
        )
        return AgentResult(output=output, passed=True)


# =============================================================================
# DEV AGENT — Writes code given requirements (and optional feedback)
# =============================================================================

DEV_SYSTEM = """
You are a senior software engineer. Write clean, production-quality Python code.

- Include docstrings and type hints
- Handle edge cases mentioned in requirements
- If feedback is provided, address every point explicitly
- Output ONLY the code. No explanation before or after.

RUNTIME ENVIRONMENT:
Your code runs inside an isolated container. The constraints are:
- No network access
- No filesystem access outside of /tmp
- Available third-party packages: requests, dnspython, pydantic, numpy, pandas, httpx, pytest
- Only use packages from the list above or the Python standard library
"""

class dev_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        if not state.dev_messages:
            # First iteration: seed the conversation with requirements
            state.dev_messages = [{"role": "user", "content": f"REQUIREMENTS:\n{state.requirements}"}]
        else:
            # Subsequent iterations: append just the feedback — model already has the code
            # in its prior assistant turn, so we don't re-send it
            state.dev_messages.append({
                "role": "user",
                "content": f"FEEDBACK TO ADDRESS:\n{state.feedback}",
            })

        code = call_claude_messages(DEV_SYSTEM, state.dev_messages)

        # Store the assistant turn so the next iteration has full context without re-sending code
        state.dev_messages.append({"role": "assistant", "content": code})

        return AgentResult(output=code, passed=True)


# =============================================================================
# QA AGENT — Reviews code against requirements, produces pass/fail + report
# =============================================================================

QA_SYSTEM = """
You are a QA engineer. Review the provided code against the requirements.

Output a JSON object with this exact shape:
{
  "passed": true or false,
  "summary": "one sentence verdict",
  "issues": [
    {"severity": "critical|major|minor", "description": "..."}
  ],
  "feedback_for_dev": "clear, actionable instructions for what to fix (empty string if passed)"
}

Be strict. Fail if any acceptance criteria are not met or if there are critical bugs.
If actual execution results are provided, treat runtime errors as critical issues.
"""

class qa_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        # If sandbox is enabled, actually execute the code and include
        # the real output in the QA prompt. This gives Claude real
        # runtime behavior rather than just static analysis.
        execution_block = ""
        if SANDBOX_ENABLED and state.code:
            from sandbox import run_in_sandbox
            print("  Executing code in sandbox...")
            exec_result = run_in_sandbox(state.code)
            status = "exited 0 (success)" if exec_result.success else f"exited {exec_result.exit_code} (failure)"
            execution_block = (
                f"\n\nACTUAL EXECUTION RESULTS ({status}):\n"
                f"STDOUT:\n{exec_result.stdout or '(empty)'}\n"
                f"STDERR:\n{exec_result.stderr or '(empty)'}\n"
                f"TIMED OUT: {exec_result.timed_out}"
            )

        prompt = (
            f"REQUIREMENTS:\n{state.requirements}\n\n"
            f"CODE TO REVIEW:\n{state.code}"
            f"{execution_block}"
        )
        result = call_claude_json(QA_SYSTEM, prompt)

        issues_text = "\n".join(
            f"  [{i['severity'].upper()}] {i['description']}"
            for i in result.get("issues", [])
        ) or "  None"

        output = (
            f"QA RESULT: {'✅ PASS' if result['passed'] else '❌ FAIL'}\n"
            f"SUMMARY: {result['summary']}\n\n"
            f"ISSUES:\n{issues_text}"
        )

        return AgentResult(
            output=output,
            passed=result["passed"],
            feedback=result.get("feedback_for_dev") or None,
        )


# =============================================================================
# REVIEWER AGENT — Code review for quality, style, security, architecture
# =============================================================================

REVIEWER_SYSTEM = """
You are a principal engineer doing a code review. Evaluate the code for:
- Correctness and completeness against requirements
- Code quality (naming, structure, readability)
- Security issues
- Missing tests or edge case handling
- Anything that would block a PR merge

Output a JSON object with this exact shape:
{
  "approved": true or false,
  "summary": "one sentence verdict",
  "comments": [
    {"type": "blocking|suggestion|nit", "description": "..."}
  ],
  "feedback_for_dev": "specific changes required before approval (empty string if approved)"
}
"""

class reviewer_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        prompt = (
            f"REQUIREMENTS:\n{state.requirements}\n\n"
            f"QA REPORT:\n{state.qa_report}\n\n"
            f"CODE TO REVIEW:\n{state.code}"
        )
        result = call_claude_json(REVIEWER_SYSTEM, prompt)

        comments_text = "\n".join(
            f"  [{c['type'].upper()}] {c['description']}"
            for c in result.get("comments", [])
        ) or "  None"

        output = (
            f"REVIEW RESULT: {'✅ APPROVED' if result['approved'] else '🔄 CHANGES REQUESTED'}\n"
            f"SUMMARY: {result['summary']}\n\n"
            f"COMMENTS:\n{comments_text}"
        )

        return AgentResult(
            output=output,
            passed=result["approved"],
            feedback=result.get("feedback_for_dev") or None,
        )
