import numpy as np
import warnings

from benchmark.gpu_backend import GpuBackend
from config import GpuBackendType, Precision

SAXPY_FLOAT = """
__kernel void saxpy(__global const float *x, __global float *y, float alpha, int N) {
    int i = get_global_id(0);
    int stride = get_global_size(0);
    for (; i < N; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}
"""

SAXPY_DOUBLE = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void saxpy_double(__global const double *x, __global double *y, double alpha, int N) {
    int i = get_global_id(0);
    int stride = get_global_size(0);
    for (; i < N; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}
"""

MATMUL_FLOAT = """
__kernel void matmul_naive(__global const float *A,
                           __global const float *B,
                           __global float *C,
                           int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""

MATMUL_DOUBLE = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matmul_naive_double(__global const double *A,
                                   __global const double *B,
                                   __global double *C,
                                   int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= N || col >= N) return;
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"""

REDUCE_FLOAT = """
__kernel void reduce_sum(__global const float *input,
                          __global float *output,
                          __local float *local_sum,
                          int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gsize = get_global_size(0);
    int group_size = get_local_size(0);

    float acc = 0.0f;
    for (int i = gid; i < N; i += gsize) {
        acc += input[i];
    }
    local_sum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_sum[lid] += local_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = local_sum[0];
    }
}
"""

REDUCE_DOUBLE = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void reduce_sum_double(__global const double *input,
                                 __global double *output,
                                 __local double *local_sum,
                                 int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gsize = get_global_size(0);
    int group_size = get_local_size(0);

    double acc = 0.0;
    for (int i = gid; i < N; i += gsize) {
        acc += input[i];
    }
    local_sum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_sum[lid] += local_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        output[get_group_id(0)] = local_sum[0];
    }
}
"""

PRECISION_TO_OCL = {
    Precision.FP64: np.float64,
    Precision.FP32: np.float32,
    Precision.FP16: np.float32,
    Precision.BF16: np.float32,
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

        if self._has_fp64:
            self._saxpy_double_prog = cl.Program(self._ctx, SAXPY_DOUBLE).build()
            self._saxpy_double_kernel = self._saxpy_double_prog.saxpy_double
            self._matmul_double_prog = cl.Program(self._ctx, MATMUL_DOUBLE).build()
            self._matmul_double_kernel = self._matmul_double_prog.matmul_naive_double
            self._reduce_double_prog = cl.Program(self._ctx, REDUCE_DOUBLE).build()
            self._reduce_double_kernel = self._reduce_double_prog.reduce_sum_double
        else:
            self._saxpy_double_prog = None
            self._saxpy_double_kernel = None
            self._matmul_double_prog = None
            self._matmul_double_kernel = None
            self._reduce_double_prog = None
            self._reduce_double_kernel = None

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
        shape = getattr(A, "_ocl_shape", (2, 2))
        if len(shape) == 2:
            N = shape[0]
        else:
            N = int(np.sqrt(shape[0]))
            shape = (N, N)
        dtype = PRECISION_TO_OCL.get(self._current_precision, np.float32)
        C = self.cl.Buffer(self._ctx, self.cl.mem_flags.READ_WRITE, size=N * N * np.dtype(dtype).itemsize)
        C._ocl_shape = shape

        if self._current_precision == Precision.FP64:
            kernel = self._matmul_double_kernel
        else:
            kernel = self._matmul_kernel
        kernel(self._queue, (N, N), None, A, B, C, np.int32(N))
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

    def dispose(self):
        self._queue = None
        self._ctx = None
