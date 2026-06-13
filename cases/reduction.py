import time
import numpy as np
from cases.base import BenchmarkCase
from cases.utils import get_dtype, create_array
from config import DurationMode, Precision


class ReductionCase(BenchmarkCase):
    name = "Reduction"

    def get_flops(self, size: int) -> int:
        return size - 1

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
        arr = create_array(size, precision)

        _ = np.sum(arr)

        niters = 5
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            _ = np.sum(arr)
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best

    def run_gpu(self, size: int, precision: Precision, backend) -> float:
        dtype = get_dtype(precision)
        arr_host = create_array(size, precision)

        arr_dev = backend.to_device(arr_host, precision)

        backend.synchronize()
        _ = backend.sum(arr_dev)
        backend.synchronize()

        niters = 5
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            _ = backend.sum(arr_dev)
            backend.synchronize()
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best
