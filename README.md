# Agent Swarm — Raw Claude API Implementation

A multi-agent pipeline that takes a feature request (or existing spec documents)
and iterates through requirements → development → QA → code review until the
code is approved.

## Architecture

```
feature_request  ──OR──  docs/ARCHITECTURE.md + docs/TASKS.md
      │                              │
      └──────────────┬───────────────┘
                     ▼
              [PM Agent]          (runs once; skipped if requirements pre-set)
                     │ requirements
                     ▼
              [Dev Agent]  ◄──────────────────┐
                     │ code                   │ feedback
                     ▼                        │
              [QA Agent]                      │
                     │                        │
                     ├── FAIL ────────────────┘
                     │
                     │ pass
                     ▼
           [Reviewer Agent]
                     │
                     ├── CHANGES REQUESTED ───┘ (sends feedback back to Dev)
                     │
                     └── APPROVED → done ✅
```

### Two modes

| Mode | How it works |
|------|-------------|
| **Free-form** | Pass a feature request string or file. PM agent generates requirements from scratch. |
| **Spec-driven** | Pass `--architecture` and `--tasks` docs. PM agent reads the specs and produces structured acceptance criteria per milestone task — no requirements invented. |

## File Structure

```
agent_swarm/
├── main.py              # Entry point / CLI
├── orchestrator.py      # Loop control logic — the "brain"
├── agents.py            # All 4 agents (PM, Dev, QA, Reviewer)
├── models.py            # SwarmState and AgentResult dataclasses
├── claude_client.py     # Raw Anthropic API wrapper
├── sandbox.py           # Optional Podman-based code execution sandbox
├── Dockerfile.sandbox   # Container image for sandbox execution
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
| `sandbox.py` | Runs generated code in an isolated Podman container. Off by default. |
| `Dockerfile.sandbox` | Defines the sandbox image with pre-installed packages for generated code. |

## Setup

### Getting an Anthropic API key

1. Go to [console.anthropic.com](https://console.anthropic.com) and sign up or log in
2. Navigate to **API Keys** in the left sidebar
3. Click **Create Key**, give it a name, and copy it — you won't be able to see it again
4. Add credits to your account under **Billing** (the API is pay-per-use, no subscription required)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here

# Use the default built-in request
python main.py

# Pass a request inline
python main.py --request "build a JWT auth middleware"

# Read the request from a file
python main.py --file request.txt

# Suppress verbose output
python main.py --file request.txt --quiet
```

### Spec-driven mode

Instead of a free-form feature request, point the swarm at your existing
architecture and task documents. The PM agent will read them and produce
structured acceptance criteria for each task in the current milestone.

```bash
# Explicit paths
python main.py --architecture docs/ARCHITECTURE.md --tasks docs/TASKS.md

# Default paths (docs/ARCHITECTURE.md + docs/TASKS.md) — just drop the files
# in the right place and run without flags
python main.py
```

The PM agent will:
- Extract the **current milestone** (first milestone with pending tasks)
- Produce per-task acceptance criteria derived directly from the spec
- Flag ambiguities as warnings without blocking the pipeline
- Surface global architecture constraints for the Dev agent

**Expected doc conventions:**

`docs/ARCHITECTURE.md` — Describe the system: tech stack, constraints, module
boundaries, security requirements.

`docs/TASKS.md` — List tasks grouped by milestone. Mark completed tasks so
the agent knows which milestone is current. Any format works as long as
milestones and task IDs are clear.

### Dry-run

Verify that your doc files are found and loaded correctly without making any
Claude API calls:

```bash
python main.py --architecture docs/ARCHITECTURE.md --tasks docs/TASKS.md --dry-run
```

Output example:
```
📐 Loaded architecture doc: docs/ARCHITECTURE.md (1842 chars)
📋 Loaded tasks doc: docs/TASKS.md (743 chars)

--- DRY RUN STATE ---
feature_request: '\nBuild a Python function that validates email addresses...'
architecture:    1842 chars loaded
tasks_doc:       743 chars loaded
No API calls made.
```

Works with any combination of flags — useful for CI pre-checks or before
spending API credits on a long run.

## Sandbox (optional)

By default, QA does **static analysis only**. To have QA actually execute the
generated code, enable the sandbox:

```bash
# Windows
winget install RedHat.Podman
podman machine init
podman machine start

# Build the sandbox image once (only needed after install or Dockerfile changes)
podman build -f Dockerfile.sandbox -t swarm-sandbox .

# Then run with sandbox enabled
SWARM_SANDBOX=true python main.py --request "build a rate limiter"
```

When `SWARM_SANDBOX=true`, the QA agent:
1. Writes the generated code to a temp file
2. Mounts it read-only into a fresh `swarm-sandbox` container
3. Runs it with no network access, 256MB RAM cap, and a 30s timeout
4. Passes the real stdout/stderr/exit code back to Claude for review
5. Treats runtime errors as critical failures

The container is automatically removed after each run. Your machine's
filesystem is never touched by the generated code.

### Available packages in the sandbox

The Dev agent is told it can only use these packages (plus the standard library):

| Package | Use case |
|---------|----------|
| `requests` | HTTP calls |
| `dnspython` | DNS lookups (e.g. MX record validation) |
| `pydantic` | Data validation and modelling |
| `numpy` | Numerical computing |
| `pandas` | Data manipulation |
| `httpx` | Async HTTP client |
| `pytest` | Testing |

To add a package: update `Dockerfile.sandbox`, rebuild the image, and add it
to the `RUNTIME ENVIRONMENT` section of `DEV_SYSTEM` in `agents.py`.

## Key Design Decisions

### SwarmState as shared memory
Every agent receives the full `SwarmState` and reads only what it needs.
This makes it trivial to add a new agent — it just reads from state, writes back.
`architecture` and `tasks_doc` fields are populated before `run_swarm()` is
called, so the PM agent can consume them without any orchestrator changes.

### Spec-driven vs free-form PM agent
In free-form mode the PM agent invents requirements from a vague prompt.
In spec-driven mode it reads `state.architecture` and `state.tasks_doc` and
derives acceptance criteria from existing decisions — no hallucinated scope.
The orchestrator also skips the PM phase entirely if `state.requirements` is
already set, which lets callers inject pre-built requirements for testing.

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

**Enable sandbox execution** (QA runs the code, not just static analysis):
1. Install Podman and set `SWARM_SANDBOX=true` (see Sandbox section above)
2. Real stdout/stderr/exit codes are passed into the QA agent's prompt

**Parallelize** (e.g. run QA and Security review simultaneously):
1. Use `asyncio.gather()` with async versions of `call_claude`
2. Merge results before passing to Reviewer
# claud-agentic-loop
