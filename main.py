"""
Entry point. Run this to kick off the swarm.

Usage:
    python main.py
    python main.py --request "build a rate limiter"
    python main.py --file request.txt
    python main.py --file request.txt --quiet
"""

import argparse
import json
import sys
from orchestrator import run_swarm


DEFAULT_REQUEST = """
Build a Python function that validates email addresses. It should:
- Check format using regex
- Verify the domain has MX records (using dnspython)
- Return a result object with is_valid, reason, and domain fields
"""


def main():
    parser = argparse.ArgumentParser(description="Run the agent swarm")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--request", type=str, help="Feature request as a string")
    group.add_argument("--file", type=str, help="Path to a file containing the feature request")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file) as f:
                feature_request = f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    else:
        feature_request = args.request or DEFAULT_REQUEST

    state = run_swarm(
        feature_request=feature_request,
        verbose=not args.quiet,
    )

    print("\n" + "="*50)
    print("FINAL STATE SUMMARY")
    print("="*50)
    print(f"Approved: {state.approved}")
    print(f"Iterations: {len([h for h in state.history if h['agent'] == 'dev'])}")

    if state.approved and state.code:
        print("\n--- FINAL CODE ---")
        print(state.code)

    # Dump full history to file for inspection
    with open("swarm_run.json", "w") as f:
        json.dump(state.history, f, indent=2)
    print("\n📄 Full history written to swarm_run.json")


if __name__ == "__main__":
    main()
