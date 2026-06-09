import time
import numpy as np
from cases.base import BenchmarkCase
from config import DurationMode, Precision

CPU_DTYPE = {
    Precision.FP64: np.float64,
    Precision.FP32: np.float32,
    Precision.FP16: np.float32,
    Precision.BF16: np.float32,
}


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
                DurationMode.NORMAL: 500_000_000,
            }[mode]

    def run_cpu(self, size: int, precision: Precision) -> float:
        dtype = CPU_DTYPE[precision]
        rng = np.random.default_rng(42)
        x = rng.random(size, dtype=dtype)
        y = rng.random(size, dtype=dtype)
        alpha = dtype(2.0)

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
        dtype = CPU_DTYPE[precision]
        rng = np.random.default_rng(42)
        x_host = rng.random(size, dtype=dtype)
        y_host = rng.random(size, dtype=dtype)
        alpha = dtype(2.0)

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
