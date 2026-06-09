from config import GpuBackendType


def detect_gpu():
    try:
        import cupy
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return GpuBackendType.CUDA, name
    except Exception:
        pass

    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                return GpuBackendType.OPENCL, devices[0].name
    except Exception:
        pass

    return GpuBackendType.NONE, None


def get_cpu_info() -> str:
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", info.get("brand", "Unknown CPU"))
    except Exception:
        import platform
        return platform.processor() or "Unknown CPU"


def get_cuda_version() -> str:
    try:
        import cupy
        return str(cupy.cuda.runtime.runtimeGetVersion())
    except Exception:
        return ""
