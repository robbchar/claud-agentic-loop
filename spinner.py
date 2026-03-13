"""
A simple animated spinner for long-running agent calls.

Usage:
    with Spinner("💻 [Dev Agent] Writing code"):
        result = dev_agent.run(state)
"""

import itertools
import sys
import threading
import time


class Spinner:
    _FRAMES = [".", "..", "..."]

    def __init__(self, message: str, interval: float = 0.5):
        self.message = message
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            print(f"\r{self.message}{frame}   ", end="", flush=True)
            time.sleep(self.interval)

    def clear(self) -> None:
        """Erase the spinner line — called before streaming output begins."""
        width = len(self.message) + len(self._FRAMES[-1]) + 3
        print(f"\r{' ' * width}\r", end="", flush=True)

    def __enter__(self) -> "Spinner":
        # Print any leading newline once so it doesn't repeat on every frame
        if self.message.startswith("\n"):
            print()
            self.message = self.message.lstrip("\n")
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()
        self.clear()
