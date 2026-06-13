import warnings
from typing import List, Tuple
from config import GpuBackendType

warnings.filterwarnings("ignore", message="CUDA path could not be detected")


class GpuInfo:
    def __init__(self, backend_type: GpuBackendType, name: str, index: int):
        self.backend_type = backend_type
        self.name = name
        self.index = index

    def __repr__(self):
        return f"GpuInfo({self.backend_type.value}, {self.name}, idx={self.index})"


def detect_gpu() -> Tuple[GpuBackendType, str]:
    gpus = detect_all_gpus()
    if gpus:
        return gpus[0].backend_type, gpus[0].name
    return GpuBackendType.NONE, None


def detect_all_gpus() -> List[GpuInfo]:
    gpus = []
    idx = 0

    # NVIDIA GPUs via cupy
    try:
        import cupy
        count = cupy.cuda.runtime.getDeviceCount()
        for i in range(count):
            props = cupy.cuda.runtime.getDeviceProperties(i)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode()
            gpus.append(GpuInfo(GpuBackendType.CUDA, name, idx))
            idx += 1
    except Exception:
        pass

    # OpenCL GPUs (non-NVIDIA, or all if cupy not used)
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            for device in devices:
                # Skip if already detected via CUDA (same GPU shown via both APIs)
                if _is_already_detected(gpus, device.name):
                    continue
                gpus.append(GpuInfo(GpuBackendType.OPENCL, device.name, idx))
                idx += 1
    except Exception:
        pass

    return gpus


def _is_already_detected(gpus: List[GpuInfo], name: str) -> bool:
    name_lower = name.lower()
    for g in gpus:
        g_lower = g.name.lower()
        if name_lower == g_lower:
            return True
        if "nvidia" in name_lower and "nvidia" in g_lower:
            return True
    return False


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
