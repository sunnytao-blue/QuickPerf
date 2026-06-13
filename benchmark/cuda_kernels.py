import numpy as np

from benchmark.gpu_backend import GpuBackend
from config import GpuBackendType, Precision


CPU_TO_CUPY_DTYPE = {
    np.float64: "float64",
    np.float32: "float32",
    np.int64: "int64",
    np.int32: "int32",
    np.int16: "int16",
    np.int8: "int8",
}

PRECISION_TO_CUPY = {
    Precision.FP64: "float64",
    Precision.FP32: "float32",
    Precision.FP16: "float16",
    Precision.BF16: "bfloat16",
    Precision.INT64: "int64",
    Precision.INT32: "int32",
    Precision.INT16: "int16",
    Precision.INT8: "int8",
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
        if dtype_key in CPU_TO_CUPY_DTYPE:
            cupy_str = CPU_TO_CUPY_DTYPE[dtype_key]
        else:
            cupy_str = str(arr.dtype)
        return getattr(self.cp, cupy_str, None) or self.cp.float32

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
        if not hasattr(self, '_saxpy_kernels'):
            self._saxpy_kernels = {}
        dtype_key = str(x.dtype)
        if dtype_key not in self._saxpy_kernels:
            self._saxpy_kernels[dtype_key] = self.cp.ElementwiseKernel(
                f'{dtype_key} x, {dtype_key} alpha', f'{dtype_key} y',
                'y = alpha * x + y', 'saxpy')
        self._saxpy_kernels[dtype_key](x, self.cp.dtype(dtype_key).type(alpha), y)
        return y

    def sum(self, arr):
        return self.cp.sum(arr)

    def dispose(self):
        self.cp.get_default_memory_pool().free_all_blocks()
