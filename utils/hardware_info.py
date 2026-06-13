from typing import List
from config import Precision, GpuBackendType


def get_supported_precisions(gpu_type: GpuBackendType, gpu_name: str = "") -> List[Precision]:
    cpu_precisions = _get_cpu_supported_precisions()

    if gpu_type == GpuBackendType.NONE:
        return cpu_precisions

    if gpu_type == GpuBackendType.CUDA:
        return _get_cuda_supported_precisions(cpu_precisions)

    return _get_opencl_supported_precisions(cpu_precisions)


def _get_cpu_supported_precisions() -> List[Precision]:
    return [
        Precision.FP64,
        Precision.FP32,
        Precision.FP16,
        Precision.BF16,
        Precision.INT64,
        Precision.INT32,
        Precision.INT16,
        Precision.INT8,
    ]


def _get_cuda_supported_precisions(cpu_precisions: List[Precision]) -> List[Precision]:
    try:
        import cupy
        prop = cupy.cuda.runtime.getDeviceProperties(0)
        major = prop.get("major", 7)
        minor = prop.get("minor", 5)
        cc = major * 10 + minor
    except Exception:
        return [Precision.FP64, Precision.FP32, Precision.FP16, Precision.BF16,
                Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8]

    supported = [Precision.FP64, Precision.FP32]

    if cc >= 53:
        if _is_cupy_dtype_available("float16"):
            supported.append(Precision.FP16)

    if cc >= 80:
        if _is_cupy_dtype_available("bfloat16"):
            supported.append(Precision.BF16)

    for prec in [Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8]:
        prec_str = prec.value.lower()
        if _is_cupy_dtype_available(prec_str):
            supported.append(prec)

    return supported


def _is_cupy_dtype_available(dtype_str: str) -> bool:
    try:
        import cupy
        dtype = getattr(cupy, dtype_str, None)
        if dtype is None:
            dtype = getattr(cupy, f"cupy_{dtype_str}", None)
        if dtype is None:
            return False
        cupy.empty(1, dtype=dtype)
        return True
    except Exception:
        return False


def _get_opencl_supported_precisions(cpu_precisions: List[Precision]) -> List[Precision]:
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for p in platforms:
            devices = p.get_devices(device_type=cl.device_type.GPU)
            if devices:
                extensions = devices[0].extensions if hasattr(devices[0], "extensions") else ""
                has_fp64 = "cl_khr_fp64" in extensions
                has_fp16 = "cl_khr_fp16" in extensions

                result = []
                for prec in cpu_precisions:
                    if prec == Precision.FP64 and not has_fp64:
                        continue
                    if prec == Precision.FP16 and not has_fp16:
                        continue
                    result.append(prec)
                return result
    except Exception:
        pass

    return [p for p in cpu_precisions if p != Precision.FP64]


def format_precision_list(precisions: List[Precision]) -> str:
    parts = []
    for prec in precisions:
        parts.append(f"{prec.value}")
    return ", ".join(parts)
