# Agent Swarm — Raw Claude API Implementation

A multi-agent pipeline that takes a feature request and iterates through
requirements → development → QA → code review until the code is approved.

## Architecture

```
feature_request
      │
      ▼
 [PM Agent]          (runs once)
      │ requirements
      ▼
 [Dev Agent]  ◄──────────────────┐
      │ code                     │ feedback
      ▼                          │
 [QA Agent]                      │
      │                          │
      ├── FAIL ──────────────────┘
      │
      │ pass
      ▼
[Reviewer Agent]
      │
      ├── CHANGES REQUESTED ─────┘ (sends feedback back to Dev)
      │
      └── APPROVED → done ✅
```

## File Structure

```
agent_swarm/
├── main.py           # Entry point / CLI
├── orchestrator.py   # Loop control logic — the "brain"
├── agents.py         # All 4 agents (PM, Dev, QA, Reviewer)
├── models.py         # SwarmState and AgentResult dataclasses
├── claude_client.py  # Raw Anthropic API wrapper
└── README.md
```

### What each file is responsible for

| File | Responsibility |
|------|---------------|
| `models.py` | Shared state. The single object passed between all agents. |
| `claude_client.py` | The only place that calls `anthropic.Anthropic()`. Handles JSON extraction. |
| `agents.py` | System prompts + prompt assembly + response parsing for each agent. |
| `orchestrator.py` | The loop: decides when to advance, retry, or stop. |
| `main.py` | CLI glue. Passes args, prints summary, dumps history. |

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python main.py
python main.py --request "build a JWT auth middleware"
```

## Key Design Decisions

### SwarmState as shared memory
Every agent receives the full `SwarmState` and reads only what it needs.
This makes it trivial to add a new agent — it just reads from state, writes back.

### Structured JSON outputs from agents
QA and Reviewer return `{ passed: bool, feedback_for_dev: string }`.
This gives the orchestrator a **machine-readable signal** to loop or exit,
rather than trying to parse freeform prose.

### Feedback propagation
When QA or Reviewer rejects, the orchestrator stores `feedback` on `state`.
The Dev agent checks `state.feedback` at the top of its prompt on the next iteration.
This is the core of why the loop improves over time.

### Audit trail
`state.history` records every agent call and output. Written to `swarm_run.json`
after each run so you can inspect what happened at each iteration.

## What LangGraph Would Add

Once you understand this raw implementation, here's what LangGraph abstracts:

| Raw (this code) | LangGraph equivalent |
|-----------------|---------------------|
| `SwarmState` dataclass | Typed `StateGraph` schema |
| `if result.passed: ...` in orchestrator | Conditional edges between nodes |
| `for iteration in range(MAX_ITERATIONS)` | Built-in cycle support + checkpointing |
| Manual `state.history.append(...)` | Automatic step persistence |
| Try/catch around API calls | Built-in retry + error node routing |
| `run_swarm()` function | `graph.invoke()` or `graph.stream()` |

The mental model is identical — LangGraph just formalizes it and adds
observability, persistence, and async support out of the box.

## Extending This

**Add a new agent** (e.g. a Security Agent between QA and Reviewer):
1. Add its system prompt + `run()` to `agents.py`
2. Add a call to it in `orchestrator.py` after the QA pass block

**Add tool use** (e.g. Dev agent actually runs the code):
1. Give the Dev agent a sandboxed shell (via `subprocess` or E2B)
2. Pass the execution output into the QA agent's prompt

**Parallelize** (e.g. run QA and Security review simultaneously):
1. Use `asyncio.gather()` with async versions of `call_claude`
2. Merge results before passing to Reviewer
# claud-agentic-loop
