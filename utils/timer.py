import time


class Timer:
    def __init__(self):
        self._start_ns = 0
        self._end_ns = 0

    def __enter__(self):
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self._end_ns = time.perf_counter_ns()

    @property
    def elapsed_seconds(self) -> float:
        return (self._end_ns - self._start_ns) / 1e9

    @property
    def elapsed_ns(self) -> int:
        return self._end_ns - self._start_ns


def time_it(func, iterations: int = 1) -> float:
    """Time func() for `iterations` times, return minimum time in seconds."""
    best = float("inf")
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        elapsed = (end - start) / 1e9
        if elapsed < best:
            best = elapsed
    return best
