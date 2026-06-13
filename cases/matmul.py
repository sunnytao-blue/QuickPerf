import time
from cases.base import BenchmarkCase
from cases.utils import get_dtype, create_array
from config import DurationMode, Precision


class MatMulCase(BenchmarkCase):
    name = "MatMul"

    def get_flops(self, size: int) -> int:
        return 2 * size * size * size

    def get_size(self, mode: DurationMode, target: str) -> int:
        if target == "CPU":
            return {DurationMode.QUICK: 1024, DurationMode.NORMAL: 2048}[mode]
        else:
            return {DurationMode.QUICK: 512, DurationMode.NORMAL: 1024}[mode]

    def run_cpu(self, size: int, precision: Precision) -> float:
        dtype = get_dtype(precision)
        A = create_array((size, size), precision)
        B = create_array((size, size), precision)

        _ = A @ B

        niters = 3
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            _ = A @ B
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best

    def run_gpu(self, size: int, precision: Precision, backend) -> float:
        dtype = get_dtype(precision)
        A_host = create_array((size, size), precision)
        B_host = create_array((size, size), precision)

        A_dev = backend.to_device(A_host, precision)
        B_dev = backend.to_device(B_host, precision)

        backend.synchronize()
        _ = backend.matmul(A_dev, B_dev)
        backend.synchronize()

        niters = 3
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            _ = backend.matmul(A_dev, B_dev)
            backend.synchronize()
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best
