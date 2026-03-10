"""
The four agents: PM, Dev, QA, Reviewer.

Each agent follows the same pattern:
  1. A SYSTEM_PROMPT that defines its persona and output format
  2. A build_prompt(state) function that assembles the user message from SwarmState
  3. A run(state) function that calls Claude and parses the result into AgentResult

This separation is intentional — it makes each agent independently testable,
and is exactly what LangGraph would model as individual graph nodes.
"""

from claude_client import call_claude, call_claude_json
from models import SwarmState, AgentResult


# =============================================================================
# PM AGENT — Produces structured requirements from a vague feature request
# =============================================================================

PM_SYSTEM = """
You are a senior product manager. Your job is to take a feature request and produce
clear, developer-ready requirements.

Output a JSON object with this exact shape:
{
  "summary": "one sentence description",
  "acceptance_criteria": ["criterion 1", "criterion 2", ...],
  "edge_cases": ["edge case 1", ...],
  "out_of_scope": ["thing 1", ...],
  "tech_notes": "any relevant implementation constraints or suggestions"
}
"""

class pm_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        prompt = f"Feature request:\n{state.feature_request}"
        result = call_claude_json(PM_SYSTEM, prompt)
        # Format it nicely for downstream agents
        output = (
            f"SUMMARY: {result['summary']}\n\n"
            f"ACCEPTANCE CRITERIA:\n" +
            "\n".join(f"  - {c}" for c in result["acceptance_criteria"]) +
            f"\n\nEDGE CASES:\n" +
            "\n".join(f"  - {e}" for e in result["edge_cases"]) +
            f"\n\nOUT OF SCOPE:\n" +
            "\n".join(f"  - {o}" for o in result["out_of_scope"]) +
            f"\n\nTECH NOTES: {result['tech_notes']}"
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
"""

class dev_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        prompt = f"REQUIREMENTS:\n{state.requirements}"

        if state.feedback:
            prompt += f"\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{state.feedback}"

        if state.code:
            prompt += f"\n\nYOUR PREVIOUS CODE (revise this):\n{state.code}"

        code = call_claude(DEV_SYSTEM, prompt)
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
"""

class qa_agent:
    @staticmethod
    def run(state: SwarmState) -> AgentResult:
        prompt = (
            f"REQUIREMENTS:\n{state.requirements}\n\n"
            f"CODE TO REVIEW:\n{state.code}"
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
