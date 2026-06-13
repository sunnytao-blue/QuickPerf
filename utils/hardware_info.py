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
        compute_cap = (prop.get("major", 7), prop.get("minor", 5))
        cc = compute_cap[0] * 10 + compute_cap[1]
    except Exception:
        return cpu_precisions

    supported = list(cpu_precisions)

    for p in Precision:
        prec_str = p.value.lower()
        if prec_str in ("fp64", "fp32"):
            continue
        if _is_cupy_dtype_available(prec_str):
            if p not in supported:
                supported.append(p)

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
