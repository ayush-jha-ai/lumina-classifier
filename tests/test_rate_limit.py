import time

from classifier.rate_limit import RateLimiter


def test_first_acquire_does_not_block():
    limiter = RateLimiter(calls_per_minute=60)  # 1s min interval
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.05


def test_enforces_minimum_interval_between_calls():
    limiter = RateLimiter(calls_per_minute=600)  # 0.1s min interval
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # 3 calls -> 2 gaps of >= 0.1s each
    assert elapsed >= 0.2 - 0.02
