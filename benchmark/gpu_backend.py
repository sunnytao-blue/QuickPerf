from abc import ABC, abstractmethod
from config import GpuBackendType, Precision

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class GpuBackend(ABC):
    backend_type: GpuBackendType
    gpu_name: str

    _current_precision = None  # set by runner before each case

    @abstractmethod
    def to_device(self, arr: "np.ndarray", precision=None):
        ...

    @abstractmethod
    def from_device(self, arr) -> "np.ndarray":
        ...

    @abstractmethod
    def synchronize(self):
        ...

    @abstractmethod
    def matmul(self, A, B):
        ...

    @abstractmethod
    def saxpy(self, alpha, x, y):
        ...

    @abstractmethod
    def sum(self, arr):
        ...

    @abstractmethod
    def softmax(self, arr):
        ...

    @abstractmethod
    def dispose(self):
        ...


def create_backend(backend_type: GpuBackendType, gpu_name: str) -> GpuBackend:
    if backend_type == GpuBackendType.CUDA:
        from benchmark.cuda_kernels import CudaBackend
        return CudaBackend(gpu_name)
    elif backend_type == GpuBackendType.OPENCL:
        from benchmark.opencl_kernels import OpenCLBackend
        return OpenCLBackend(gpu_name)
    else:
        raise RuntimeError(f"Unknown backend type: {backend_type}")
