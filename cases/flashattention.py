import math
import time
import numpy as np
from cases.base import BenchmarkCase
from cases.utils import get_dtype, create_array
from config import DurationMode, Precision


class FlashAttentionCase(BenchmarkCase):
    name = "FlashAttention"

    def __init__(self):
        self._head_dim = 64

    def get_flops(self, size: int) -> int:
        N = size
        d = self._head_dim
        return 4 * N * N * d + 5 * N * N

    def get_size(self, mode: DurationMode, target: str) -> int:
        if target == "CPU":
            return {DurationMode.QUICK: 256, DurationMode.NORMAL: 512}[mode]
        else:
            return {DurationMode.QUICK: 512, DurationMode.NORMAL: 1024}[mode]

    def run_cpu(self, size: int, precision: Precision) -> float:
        N = size
        d = self._head_dim
        scale = 1.0 / math.sqrt(d)
        dtype = get_dtype(precision)

        Q = create_array((N, d), precision)
        K = create_array((N, d), precision)
        V = create_array((N, d), precision)
        K_t = np.ascontiguousarray(K.T)

        Q_s = Q * scale
        K_s = K * scale
        K_s_t = np.ascontiguousarray(K_s.T)

        _ = Q_s @ K_s_t
        S_max = np.max(_, axis=1, keepdims=True)
        _ = np.exp(_ - S_max)
        _ = _ / np.sum(_, axis=1, keepdims=True)
        _ = _ @ V

        niters = 3
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            S = Q_s @ K_s_t
            S_max = np.max(S, axis=1, keepdims=True)
            S_exp = np.exp(S - S_max)
            P = S_exp / np.sum(S_exp, axis=1, keepdims=True)
            _ = P @ V
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best

    def run_gpu(self, size: int, precision: Precision, backend) -> float:
        N = size
        d = self._head_dim
        scale = 1.0 / math.sqrt(d)
        dtype = get_dtype(precision)

        Q_host = create_array((N, d), precision)
        K_host = create_array((N, d), precision)
        V_host = create_array((N, d), precision)
        K_t_host = np.ascontiguousarray(K_host.T)

        if precision not in (Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8):
            Q_host = (Q_host.astype(np.float64) * scale).astype(dtype)
            K_host = (K_host.astype(np.float64) * scale).astype(dtype)
            K_t_host = np.ascontiguousarray(K_host.T)

        Q_dev = backend.to_device(Q_host, precision)
        K_t_dev = backend.to_device(K_t_host, precision)
        V_dev = backend.to_device(V_host, precision)

        backend.synchronize()
        S = backend.matmul(Q_dev, K_t_dev)
        P = backend.softmax(S)
        _ = backend.matmul(P, V_dev)
        backend.synchronize()

        niters = 3
        best = float("inf")
        for _ in range(niters):
            start = time.perf_counter_ns()
            S = backend.matmul(Q_dev, K_t_dev)
            P = backend.softmax(S)
            O = backend.matmul(P, V_dev)
            backend.synchronize()
            end = time.perf_counter_ns()
            elapsed = (end - start) / 1e9
            if elapsed < best:
                best = elapsed
        return best
