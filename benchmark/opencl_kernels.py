import os
import numpy as np

from benchmark.gpu_backend import GpuBackend
from config import GpuBackendType, Precision

_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "opencl")


def _load_kernel(filename: str) -> str:
    with open(os.path.join(_KERNEL_DIR, filename), "r") as f:
        return f.read()


SAXPY_FLOAT = _load_kernel("saxpy.cl")
SAXPY_DOUBLE = _load_kernel("saxpy_double.cl")
MATMUL_FLOAT = _load_kernel("matmul.cl")
MATMUL_DOUBLE = _load_kernel("matmul_double.cl")
REDUCE_FLOAT = _load_kernel("reduce.cl")
REDUCE_DOUBLE = _load_kernel("reduce_double.cl")
SOFTMAX_FLOAT = _load_kernel("softmax.cl")
SOFTMAX_DOUBLE = _load_kernel("softmax_double.cl")

PRECISION_TO_OCL = {
    Precision.FP64: np.float64,
    Precision.FP32: np.float32,
    Precision.FP16: np.float32,
    Precision.BF16: np.float32,
    Precision.INT64: np.int64,
    Precision.INT32: np.int32,
    Precision.INT16: np.int16,
    Precision.INT8: np.int8,
}


class OpenCLBackend(GpuBackend):
    backend_type = GpuBackendType.OPENCL

    def __init__(self, gpu_name: str):
        import pyopencl as cl
        self.cl = cl
        self.gpu_name = gpu_name

        platforms = cl.get_platforms()
        self._ctx = None
        self._queue = None
        self._device = None
        for p in platforms:
            devices = p.get_devices(device_type=cl.device_type.GPU)
            if devices:
                self._device = devices[0]
                self._ctx = cl.Context([self._device])
                self._queue = cl.CommandQueue(self._ctx)
                break

        if self._ctx is None:
            raise RuntimeError("No OpenCL GPU device available")

        extensions = self._device.extensions if hasattr(self._device, 'extensions') else ''
        self._has_fp64 = 'cl_khr_fp64' in extensions

        self._saxpy_prog = cl.Program(self._ctx, SAXPY_FLOAT).build()
        self._saxpy_kernel = self._saxpy_prog.saxpy
        self._matmul_prog = cl.Program(self._ctx, MATMUL_FLOAT).build()
        self._matmul_kernel = self._matmul_prog.matmul_naive
        self._reduce_prog = cl.Program(self._ctx, REDUCE_FLOAT).build()
        self._reduce_kernel = self._reduce_prog.reduce_sum
        self._softmax_prog = cl.Program(self._ctx, SOFTMAX_FLOAT).build()
        self._softmax_kernel = self._softmax_prog.softmax_row

        if self._has_fp64:
            self._saxpy_double_prog = cl.Program(self._ctx, SAXPY_DOUBLE).build()
            self._saxpy_double_kernel = self._saxpy_double_prog.saxpy_double
            self._matmul_double_prog = cl.Program(self._ctx, MATMUL_DOUBLE).build()
            self._matmul_double_kernel = self._matmul_double_prog.matmul_naive_double
            self._reduce_double_prog = cl.Program(self._ctx, REDUCE_DOUBLE).build()
            self._reduce_double_kernel = self._reduce_double_prog.reduce_sum_double
            self._softmax_double_prog = cl.Program(self._ctx, SOFTMAX_DOUBLE).build()
            self._softmax_double_kernel = self._softmax_double_prog.softmax_row_double
        else:
            self._saxpy_double_prog = None
            self._saxpy_double_kernel = None
            self._matmul_double_prog = None
            self._matmul_double_kernel = None
            self._reduce_double_prog = None
            self._reduce_double_kernel = None
            self._softmax_double_prog = None
            self._softmax_double_kernel = None

        self._current_precision = Precision.FP32

    @property
    def supports_fp64(self) -> bool:
        return self._has_fp64

    def _check_fp64(self):
        if self._current_precision == Precision.FP64 and not self._has_fp64:
            raise RuntimeError(
                f"GPU '{self.gpu_name}' 不支持双精度 (cl_khr_fp64)，"
                "请选择 FP32/FP16/BF16 精度"
            )

    def to_device(self, arr: np.ndarray, precision=None):
        mf = self.cl.mem_flags
        arr = np.ascontiguousarray(arr)
        if precision is not None and precision in PRECISION_TO_OCL:
            target_dtype = PRECISION_TO_OCL[precision]
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
        buf = self.cl.Buffer(self._ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr)
        buf._ocl_shape = arr.shape
        return buf

    def from_device(self, buf) -> np.ndarray:
        shape = getattr(buf, "_ocl_shape", None)
        if shape is None:
            return np.array([])
        dtype = PRECISION_TO_OCL.get(self._current_precision, np.float32)
        result = np.empty(shape, dtype=dtype)
        self.cl.enqueue_copy(self._queue, result, buf)
        return result

    def synchronize(self):
        self._queue.finish()

    def matmul(self, A, B):
        self._check_fp64()
        shape_a = getattr(A, "_ocl_shape", (2, 2))
        shape_b = getattr(B, "_ocl_shape", (2, 2))
        M = shape_a[0]
        K = shape_a[1]
        N = shape_b[1] if len(shape_b) > 1 else shape_b[0]
        out_shape = (M, N)
        dtype = PRECISION_TO_OCL.get(self._current_precision, np.float32)
        C = self.cl.Buffer(self._ctx, self.cl.mem_flags.READ_WRITE, size=M * N * np.dtype(dtype).itemsize)
        C._ocl_shape = out_shape

        if self._current_precision == Precision.FP64:
            kernel = self._matmul_double_kernel
        else:
            kernel = self._matmul_kernel
        kernel(self._queue, (M, N), None, A, B, C, np.int32(K), np.int32(N))
        return C

    def saxpy(self, alpha, x, y):
        self._check_fp64()
        N = x._ocl_shape[0] if hasattr(x, "_ocl_shape") else 0
        if N == 0:
            return y
        WORKGROUP_SIZE = 256
        MAX_WORK_ITEMS = 65536
        num_work_items = min(N, MAX_WORK_ITEMS)
        num_work_items = max(WORKGROUP_SIZE, (num_work_items + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE * WORKGROUP_SIZE)

        if self._current_precision == Precision.FP64:
            kernel = self._saxpy_double_kernel
        else:
            kernel = self._saxpy_kernel
        dtype = PRECISION_TO_OCL.get(self._current_precision, np.float32)
        alpha_typed = dtype(alpha)
        kernel(self._queue, (num_work_items,), (WORKGROUP_SIZE,), x, y, alpha_typed, np.int32(N))
        return y

    def sum(self, arr):
        self._check_fp64()
        N = arr._ocl_shape[0] if hasattr(arr, "_ocl_shape") else 0
        if N == 0:
            return 0
        dtype = PRECISION_TO_OCL.get(self._current_precision, np.float32)
        itemsize = np.dtype(dtype).itemsize
        WORKGROUP_SIZE = 256
        MAX_WORK_ITEMS = 65536

        current = arr
        remaining = N

        if self._current_precision == Precision.FP64:
            kernel = self._reduce_double_kernel
        else:
            kernel = self._reduce_kernel

        while remaining > 1:
            if remaining <= MAX_WORK_ITEMS:
                num_work_items = max(WORKGROUP_SIZE, (remaining + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE * WORKGROUP_SIZE)
            else:
                num_work_items = MAX_WORK_ITEMS

            num_groups = num_work_items // WORKGROUP_SIZE

            partial_buf = self.cl.Buffer(
                self._ctx, self.cl.mem_flags.READ_WRITE, size=num_groups * itemsize
            )

            kernel(self._queue, (num_work_items,), (WORKGROUP_SIZE,),
                   current, partial_buf, self.cl.LocalMemory(WORKGROUP_SIZE * itemsize), np.int32(remaining))

            current = partial_buf
            remaining = num_groups

        result = np.empty(1, dtype=dtype)
        self.cl.enqueue_copy(self._queue, result, current)
        return result[0]

    def softmax(self, arr):
        shape = getattr(arr, "_ocl_shape", (2, 2))
        N = shape[1] if len(shape) > 1 else shape[0]
        dtype = PRECISION_TO_OCL.get(self._current_precision, np.float32)
        out_buf = self.cl.Buffer(self._ctx, self.cl.mem_flags.READ_WRITE, size=N * N * np.dtype(dtype).itemsize)
        out_buf._ocl_shape = (N, N)

        if self._current_precision == Precision.FP64 and self._has_fp64:
            kernel = self._softmax_double_kernel
        else:
            kernel = self._softmax_kernel
        kernel(self._queue, (N, N), None, arr, out_buf, np.int32(N))
        return out_buf

    def dispose(self):
        self._queue = None
        self._ctx = None
