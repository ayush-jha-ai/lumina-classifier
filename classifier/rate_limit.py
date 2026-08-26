"""Client-side rate limiting for calls to the Anthropic API.

This is a public repo — anyone who clones it and points distill.py or
run_eval.py at more than a handful of examples makes one API call per
question, in a loop, with no built-in pause. The SDK retries 429s after
the fact; this throttles proactively so a naive loop doesn't hit them (or
run up unexpected cost) in the first place.

Configure via LUMINA_RATE_LIMIT_CALLS_PER_MINUTE (default: 20).
"""

from __future__ import annotations

import os
import threading
import time


class RateLimiter:
    """Blocks callers so calls to `acquire()` are spaced at least
    `60 / calls_per_minute` seconds apart. Thread-safe.
    """

    def __init__(self, calls_per_minute: float):
        if calls_per_minute <= 0:
            raise ValueError("calls_per_minute must be positive")
        self._min_interval = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call is not None:
                wait = self._last_call + self._min_interval - now
                if wait > 0:
                    time.sleep(wait)
            self._last_call = time.monotonic()


_default_limiter: RateLimiter | None = None
_default_limiter_lock = threading.Lock()


def default_limiter() -> RateLimiter:
    """A process-wide limiter shared by every caller, sized from
    LUMINA_RATE_LIMIT_CALLS_PER_MINUTE so one env var controls every
    entrypoint (baseline.py, and everything that routes through it:
    extraction.py's LLM fallback, distill.py).
    """
    global _default_limiter
    with _default_limiter_lock:
        if _default_limiter is None:
            calls_per_minute = float(os.environ.get("LUMINA_RATE_LIMIT_CALLS_PER_MINUTE", "20"))
            _default_limiter = RateLimiter(calls_per_minute)
        return _default_limiter
