"""
Sandbox executor.

Runs agent-generated code in a throwaway Podman container.
The container is created, executes the code, returns stdout/stderr, then dies.
Your machine is never touched by the generated code.

Architecture:
  1. Write the generated code to a temp file on your machine
  2. Mount that file into a fresh container as read-only
  3. Run it, capture output
  4. Container is automatically removed (--rm flag)
  5. Return structured result to QA agent

Why Podman and not subprocess directly:
  - Generated code could do anything: delete files, make network calls,
    install packages, fork-bomb, etc.
  - Inside the container it can go wild — it can't touch your machine,
    can't escape the filesystem mount, gets killed on timeout.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass


# Custom image built from Dockerfile.sandbox. Extends python:3.11-slim with
# the packages the Dev agent is allowed to use (requests, dnspython, pydantic,
# numpy, pandas, httpx, pytest). Build once with:
#   podman build -f Dockerfile.sandbox -t swarm-sandbox .
CONTAINER_IMAGE = "swarm-sandbox"

# Hard timeout in seconds. Generated code that hangs gets killed after this.
EXECUTION_TIMEOUT = 30


@dataclass
class ExecutionResult:
    success: bool       # True if exit code was 0
    stdout: str         # Captured standard output
    stderr: str         # Captured standard error
    exit_code: int      # Raw exit code
    timed_out: bool = False


def run_in_sandbox(code: str, timeout: int = EXECUTION_TIMEOUT) -> ExecutionResult:
    """
    Write code to a temp file, mount it into a Podman container, run it.

    Args:
        code:    The Python source code string to execute
        timeout: Seconds before the container is forcibly killed

    Returns:
        ExecutionResult with stdout, stderr, exit code
    """
    # Write code to a temp file — this is the only thing that touches
    # your real filesystem, and it's just a .py file in /tmp
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix="swarm_generated_"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        container_path = "/sandbox/code.py"

        cmd = [
            "podman", "run",
            "--rm",                                       # auto-remove container when done
            "--network", "none",                          # no network access — important!
            "--read-only",                                # read-only root filesystem
            "--tmpfs", "/tmp",                            # writable /tmp for code that needs it
            "--memory", "256m",                           # memory cap
            "--cpus", "1",                                # cpu cap
            "-v", f"{tmp_path}:{container_path}:ro",     # mount code as read-only
            CONTAINER_IMAGE,
            "python", container_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds",
            exit_code=-1,
            timed_out=True,
        )
    except FileNotFoundError:
        # Podman isn't installed or not in PATH
        raise RuntimeError(
            "Podman not found. Install it with: "
            "winget install RedHat.Podman  (then restart terminal)"
        )
    finally:
        os.unlink(tmp_path)


def check_sandbox_available() -> tuple[bool, str]:
    """
    Check if Podman is installed and the container image is available.
    Call this at startup so you get a clear error instead of a cryptic failure
    mid-run.

    Returns:
        (available: bool, message: str)
    """
    try:
        result = subprocess.run(
            ["podman", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return False, "Podman installed but returned an error"
    except FileNotFoundError:
        return False, (
            "Podman not found. Install: winget install RedHat.Podman\n"
            "Or disable sandbox: set SWARM_SANDBOX=false in your environment."
        )
    except subprocess.TimeoutExpired:
        return False, "Podman timed out — is the Podman machine running?"

    # Check/pull the image
    try:
        result = subprocess.run(
            ["podman", "image", "exists", CONTAINER_IMAGE],
            capture_output=True, timeout=5
        )
        if result.returncode != 0:
            print(f"Pulling container image {CONTAINER_IMAGE} (first time only)...")
            pull = subprocess.run(
                ["podman", "pull", CONTAINER_IMAGE],
                timeout=120  # pulling can take a while
            )
            if pull.returncode != 0:
                return False, f"Failed to pull image {CONTAINER_IMAGE}"
    except subprocess.TimeoutExpired:
        return False, "Image pull timed out"

    return True, f"Sandbox ready ({CONTAINER_IMAGE})"
