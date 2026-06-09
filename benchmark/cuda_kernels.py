import numpy as np

from benchmark.gpu_backend import GpuBackend
from config import GpuBackendType, Precision


CPU_TO_CUPY_DTYPE = {
    np.float64: "float64",
    np.float32: "float32",
}

PRECISION_TO_CUPY = {
    Precision.FP64: "float64",
    Precision.FP32: "float32",
    Precision.FP16: "float16",
    Precision.BF16: "bfloat16",
}


class CudaBackend(GpuBackend):
    backend_type = GpuBackendType.CUDA

    def __init__(self, gpu_name: str):
        import cupy as cp
        self.cp = cp
        self.gpu_name = gpu_name
        self._stream = cp.cuda.Stream.null

    def _to_cupy_dtype(self, arr: np.ndarray):
        dtype_key = arr.dtype.type
        cupy_str = CPU_TO_CUPY_DTYPE.get(dtype_key, "float32")
        return getattr(self.cp, cupy_str)

    def _precision_to_dtype(self, precision: Precision):
        cupy_str = PRECISION_TO_CUPY.get(precision, "float32")
        return getattr(self.cp, cupy_str)

    def to_device(self, arr: np.ndarray, precision=None):
        if precision is not None:
            cupy_str = PRECISION_TO_CUPY.get(precision, "float32")
            target_dtype = getattr(self.cp, cupy_str)
        else:
            target_dtype = self._to_cupy_dtype(arr)
        return self.cp.asarray(arr, dtype=target_dtype)

    def from_device(self, arr) -> np.ndarray:
        return self.cp.asnumpy(arr)

    def synchronize(self):
        self._stream.synchronize()

    def matmul(self, A, B):
        return self.cp.matmul(A, B)

    def saxpy(self, alpha, x, y):
        y += alpha * x
        return y

    def sum(self, arr):
        return self.cp.sum(arr)

    def dispose(self):
        self.cp.get_default_memory_pool().free_all_blocks()
