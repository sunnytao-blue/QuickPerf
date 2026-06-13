import time
from cases.base import BenchmarkCase
from cases.utils import get_dtype, create_array
from config import DurationMode, Precision


class SaxpyCase(BenchmarkCase):
    name = "SAXPY"

    def get_flops(self, size: int) -> int:
        return 2 * size

    def get_size(self, mode: DurationMode, target: str) -> int:
        if target == "CPU":
            return {
                DurationMode.QUICK: 10_000_000,
                DurationMode.NORMAL: 50_000_000,
            }[mode]
        else:
            return {
                DurationMode.QUICK: 100_000_000,
                DurationMode.NORMAL: 200_000_000,
            }[mode]

    def run_cpu(self, size: int, precision: Precision) -> float:
        dtype = get_dtype(precision)
        x = create_array(size, precision)
        y = create_array(size, precision)
        alpha = dtype(2)

        _ = alpha * x + y

        niters = 5
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            _ = alpha * x + y
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best

    def run_gpu(self, size: int, precision: Precision, backend) -> float:
        dtype = get_dtype(precision)
        x_host = create_array(size, precision)
        y_host = create_array(size, precision)
        alpha = dtype(2)

        x_dev = backend.to_device(x_host, precision)
        y_dev = backend.to_device(y_host, precision)

        backend.synchronize()
        _ = backend.saxpy(alpha, x_dev, y_dev)
        backend.synchronize()

        niters = 5
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            _ = backend.saxpy(alpha, x_dev, y_dev)
            backend.synchronize()
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best
