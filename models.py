"""
Shared data models.

SwarmState is the single object that flows through every agent.
Each agent reads what it needs and writes its output back to it.
This is the thing LangGraph would turn into a typed state graph node.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentResult:
    """
    Standardized return type from every agent.run() call.
    - output: the raw text artifact (requirements, code, report, review)
    - passed: whether this agent gives a green light to proceed
    - feedback: structured feedback to pass to the next iteration (if not passed)
    """
    output: str
    passed: bool = True
    feedback: Optional[str] = None


@dataclass
class SwarmState:
    """
    The shared state object. Think of this as the 'memory' of the pipeline.
    Passed by reference through every agent — agents mutate it via the orchestrator.
    """
    feature_request: str

    # Artifacts produced by each agent
    requirements: Optional[str] = None
    code: Optional[str] = None
    qa_report: Optional[str] = None
    review: Optional[str] = None

    # Feedback from QA or Reviewer → consumed by Dev on next iteration
    feedback: Optional[str] = None

    # Set to True when Reviewer approves
    approved: bool = False

    # Conversation history for multi-turn dev agent (avoids re-sending growing code each iteration)
    dev_messages: list = field(default_factory=list)

    # Full audit trail of every agent call
    history: list = field(default_factory=list)
