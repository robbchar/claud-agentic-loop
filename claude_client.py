"""
Raw Anthropic API wrapper.

This is the only place that actually calls the Claude API.
Every agent goes through here. This is what LangGraph's node execution replaces
(plus its retry logic, streaming support, and state checkpointing).
"""

import json
import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-opus-4-5"  # swap to sonnet for speed/cost during dev

# Context window sizes per model (tokens)
_CONTEXT_WINDOWS = {
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
}


def _warn_context_usage(input_tokens: int, model: str) -> None:
    """Print a warning when a prompt is at 25/50/75% of the context window."""
    limit = _CONTEXT_WINDOWS.get(model, 200_000)
    pct = input_tokens / limit
    if pct >= 0.75:
        print(f"⚠️  Context at {pct:.0%} ({input_tokens:,}/{limit:,} tokens) — approaching limit")
    elif pct >= 0.50:
        print(f"⚠️  Context at {pct:.0%} ({input_tokens:,}/{limit:,} tokens) — halfway")
    elif pct >= 0.25:
        print(f"ℹ️  Context at {pct:.0%} ({input_tokens:,}/{limit:,} tokens)")


def _create(system_prompt: str, messages: list) -> str:
    """Core API call with error handling and context usage warnings."""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )
    except anthropic.BadRequestError as e:
        msg = str(e).lower()
        if "prompt is too long" in msg or "context" in msg or "token" in msg:
            raise RuntimeError(
                f"Context window exceeded: the combined prompt is too long for {MODEL}. "
                "Consider shortening requirements or breaking the task into smaller pieces."
            ) from e
        raise

    _warn_context_usage(response.usage.input_tokens, MODEL)
    return response.content[0].text


def call_claude(
    system_prompt: str,
    user_message: str,
    expect_json: bool = False,
) -> str:
    """
    Single-turn Claude API call. Returns the text response.

    Args:
        system_prompt: The agent's persona + instructions
        user_message:  The actual task payload (built by each agent from SwarmState)
        expect_json:   If True, appends a JSON reminder and strips markdown fences

    Returns:
        Raw string response from Claude
    """
    if expect_json:
        system_prompt += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no backticks."

    text = _create(system_prompt, [{"role": "user", "content": user_message}])

    if expect_json:
        # Strip ```json ... ``` fences if the model adds them anyway
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    return text


def call_claude_messages(system_prompt: str, messages: list) -> str:
    """Multi-turn Claude call with a full messages list. Returns text response."""
    return _create(system_prompt, messages)


def call_claude_json(system_prompt: str, user_message: str) -> dict:
    """Convenience wrapper that returns a parsed dict."""
    raw = call_claude(system_prompt, user_message, expect_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON:\n{raw}\n\nError: {e}")
