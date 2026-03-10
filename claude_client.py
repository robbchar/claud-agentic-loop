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


def call_claude(
    system_prompt: str,
    user_message: str,
    expect_json: bool = False,
) -> str:
    """
    Single Claude API call. Returns the text response.

    Args:
        system_prompt: The agent's persona + instructions
        user_message:  The actual task payload (built by each agent from SwarmState)
        expect_json:   If True, appends a JSON reminder and strips markdown fences

    Returns:
        Raw string response from Claude
    """
    if expect_json:
        system_prompt += "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, no backticks."

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    text = response.content[0].text

    if expect_json:
        # Strip ```json ... ``` fences if the model adds them anyway
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    return text


def call_claude_json(system_prompt: str, user_message: str) -> dict:
    """Convenience wrapper that returns a parsed dict."""
    raw = call_claude(system_prompt, user_message, expect_json=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON:\n{raw}\n\nError: {e}")
