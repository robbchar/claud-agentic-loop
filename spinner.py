"""
A simple animated spinner for long-running agent calls.

Usage:
    with Spinner("💻 [Dev Agent] Writing code") as sp:
        sp.update("Looking up 'react'")   # update activity mid-spin
        result = dev_agent.run(state)
"""

import itertools
import threading
import time


class Spinner:
    _FRAMES = [".  ", ".. ", "..."]

    def __init__(self, message: str, interval: float = 0.5):
        self.message = message
        self.interval = interval
        self._activity = ""        # updated live by the stream reader
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._last_width = 0

    def update(self, activity: str) -> None:
        """Change the activity label shown after the base message (thread-safe)."""
        self._activity = activity

    def _spin(self) -> None:
        start = time.monotonic()
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            elapsed = time.monotonic() - start
            mins, secs = divmod(int(elapsed), 60)
            timer = f" ({mins}m {secs:02d}s)" if mins else f" ({secs}s)"
            activity = self._activity
            activity_str = f" — {activity}" if activity else ""
            line = f"\r{self.message}{activity_str}{frame}{timer}   "
            print(line, end="", flush=True)
            self._last_width = len(line)
            time.sleep(self.interval)

    def clear(self) -> None:
        """Erase the spinner line — called before streaming output begins."""
        width = self._last_width or len(self.message) + 40
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
