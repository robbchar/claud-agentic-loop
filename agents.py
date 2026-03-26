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
from claude_cc_client import call_claude_cc, call_claude_cc_json, call_claude_cc_messages
from models import SwarmState, AgentResult

# Sandbox is opt-in — set SWARM_SANDBOX=true to enable container execution.
# If false, QA does static analysis only (no code execution).
SANDBOX_ENABLED = os.environ.get("SWARM_SANDBOX", "false").lower() == "true"

# Agents that shell out to `claude -p` to inherit local MCP servers.
# Default: dev and qa. Override at runtime: SWARM_CC_AGENTS=dev,qa,reviewer
# Set to empty string to disable for all agents: SWARM_CC_AGENTS=
_cc_agents_raw = os.environ.get("SWARM_CC_AGENTS", "dev,qa")
CC_AGENTS: frozenset[str] = frozenset(
    a.strip().lower() for a in _cc_agents_raw.split(",") if a.strip()
)


def _use_cc(agent_name: str) -> bool:
    return agent_name in CC_AGENTS


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
        if _use_cc("pm"):
            raw = call_claude_cc_json(PM_SYSTEM, prompt, "pm")
            result = raw
        else:
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
You are a senior software engineer. Write clean, production-quality code.

- Handle edge cases mentioned in requirements
- If feedback is provided, address every point explicitly

CRITICAL — HEADLESS / NON-INTERACTIVE MODE:
You are running via `claude --print` with no terminal attached.
You MUST NOT use Write, Edit, Read, Bash, or any other tool.
Any tool call will stall indefinitely because there is no one to approve it.
Your ONLY output mechanism is plain text. The orchestrator writes files to disk.

FILE OUTPUT FORMAT:
You MUST wrap every file you generate in a separator line using this exact format:

--- FILE: relative/path/to/file.ext ---
<file contents>

Use paths relative to the project root. Output ONLY FILE blocks — no explanatory prose, no preamble,
no tool calls.

SCOPE — CRITICAL:
- Only output files that are explicitly listed in the task's "Produce:" section.
- Do NOT output files for other tasks, even if they seem related.
- Do NOT output docs/ARCHITECTURE.md, docs/TASKS.md, or any other planning document — these are managed by the orchestrator, not the dev agent.
- The README may be updated only if it is listed in "Produce:" or the task explicitly requires it.

If existing file contents are provided in the project context, read them
carefully — update or extend them rather than rewriting from scratch where
appropriate. If a file needs to be replaced entirely, output the full new
contents inside its FILE block.

TESTING REQUIREMENTS:
Every task must include tests. For frontend code specifically:
- Every component MUST have a "renders without crashing" test as a baseline
- Tests must wrap components in ALL required providers (React Router, Context, etc.)
  — a component that crashes on mount due to missing provider is a critical failure
- Test the wiring: verify that context values flow correctly to child components
- Use vitest + @testing-library/react for React components

RUNTIME ENVIRONMENT (when no project context is provided):
Your code runs inside an isolated container. The constraints are:
- No network access
- No filesystem access outside of /tmp
- Available third-party packages: requests, dnspython, pydantic, numpy, pandas, httpx, pytest
- Only use packages from the list above or the Python standard library
"""

class dev_agent:
    @staticmethod
    def run(state: SwarmState, spinner=None) -> AgentResult:
        if not state.dev_messages:
            # First iteration: seed the conversation with requirements (+ project context)
            user_content = f"REQUIREMENTS:\n{state.requirements}"
            if state.project_context:
                user_content += f"\n\nPROJECT CONTEXT:\n{state.project_context}"
            state.dev_messages = [{"role": "user", "content": user_content}]
        else:
            # Subsequent iterations: append just the feedback — model already has the code
            # in its prior assistant turn, so we don't re-send it
            state.dev_messages.append({
                "role": "user",
                "content": f"FEEDBACK TO ADDRESS:\n{state.feedback}",
            })

        if _use_cc("dev"):
            code = call_claude_cc_messages(DEV_SYSTEM, state.dev_messages, "dev", spinner=spinner)
        else:
            code = call_claude_messages(DEV_SYSTEM, state.dev_messages)

        # Store the assistant turn so the next iteration has full context without re-sending code
        state.dev_messages.append({"role": "assistant", "content": code})

        return AgentResult(output=code, passed=True)


# =============================================================================
# QA AGENT — Reviews code against requirements, produces pass/fail + report
# =============================================================================

QA_SYSTEM = """
You are a QA engineer. Your job is to verify that code actually works, not just
that it looks correct. Static analysis is not enough — you must reason about
runtime behaviour.

Output a JSON object with this exact shape:
{
  "passed": true or false,
  "summary": "one sentence verdict",
  "issues": [
    {"severity": "critical|major|minor", "description": "..."}
  ],
  "feedback_for_dev": "clear, actionable instructions for what to fix (empty string if passed)"
}

SCOPE:
- Only evaluate files explicitly listed in the task's "Produce:" section.
- Do NOT fail the task for issues in files outside that scope.
- Pre-existing stubs, placeholders, or missing files that are not in the Produce
  section are out of scope — note them as informational only, never critical/major.
- If the full test suite has failures in unrelated test files, ignore those when
  deciding pass/fail for this task.

GENERAL RULES:
- Be strict. Fail if any acceptance criteria are not met or if there are critical bugs.
- If actual execution results are provided, treat runtime errors as critical issues.
- Placeholder implementations, stub functions, or "TODO: implement later" comments
  are CRITICAL failures — the code must be complete and functional.

FRONTEND-SPECIFIC CHECKS (apply whenever the task involves React/UI):
- CRITICAL: Every component must have a "renders without crashing" test. If it is
  missing, fail immediately.
- CRITICAL: Check that every component using a React hook (useContext, useReducer,
  etc.) is wrapped in the required provider in both the app entry point AND in its
  tests. A missing provider causes a runtime crash — this is not a minor issue.
- CRITICAL: Verify the app entry point (main.jsx or index.jsx) correctly composes
  all providers and the router around the component tree. Trace the import chain.
- MAJOR: Tests that render a component without its required providers will pass
  locally but crash in the real app — flag this as a major issue.
- Check that routing is wired correctly and navigation between pages works.
- Verify that API calls use the correct base paths and will reach the Express server.
"""

class qa_agent:
    @staticmethod
    def run(state: SwarmState, spinner=None) -> AgentResult:
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
        if _use_cc("qa"):
            result = call_claude_cc_json(QA_SYSTEM, prompt, "qa", spinner=spinner)
        else:
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
    def run(state: SwarmState, spinner=None) -> AgentResult:
        prompt = (
            f"REQUIREMENTS:\n{state.requirements}\n\n"
            f"QA REPORT:\n{state.qa_report}\n\n"
            f"CODE TO REVIEW:\n{state.code}"
        )
        if _use_cc("reviewer"):
            result = call_claude_cc_json(REVIEWER_SYSTEM, prompt, "reviewer", spinner=spinner)
        else:
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
