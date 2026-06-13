from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict


class Precision(Enum):
    FP64 = "FP64"
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT64 = "INT64"
    INT32 = "INT32"
    INT16 = "INT16"
    INT8 = "INT8"


class TestTarget(Enum):
    CPU = "CPU"
    GPU = "GPU"
    BOTH = "BOTH"


class DurationMode(Enum):
    QUICK = "quick"
    NORMAL = "normal"


class GpuBackendType(Enum):
    CUDA = "CUDA"
    OPENCL = "OpenCL"
    NONE = "NONE"


INTEGER_PRECISIONS = {Precision.INT64, Precision.INT32, Precision.INT16, Precision.INT8}
FLOAT_SIMULATED = {Precision.FP16, Precision.BF16}

PRECISION_NOTE: Dict[Precision, str] = {
    Precision.FP16: "FP16(float32 simulate)",
    Precision.BF16: "BF16(float32 simulate)",
}

PRECISION_DISPLAY: Dict[Precision, str] = {
    Precision.FP64: "FP64 (双精度浮点)",
    Precision.FP32: "FP32 (单精度浮点)",
    Precision.FP16: "FP16 (半精度, CPU模拟)",
    Precision.BF16: "BF16 (BFloat16, CPU模拟)",
    Precision.INT64: "INT64 (64位整数)",
    Precision.INT32: "INT32 (32位整数)",
    Precision.INT16: "INT16 (16位整数)",
    Precision.INT8: "INT8 (8位整数)",
}

PRECISION_TO_CPU_DTYPE: Dict[Precision, str] = {
    Precision.FP64: "float64",
    Precision.FP32: "float32",
    Precision.FP16: "float32",
    Precision.BF16: "float32",
    Precision.INT64: "int64",
    Precision.INT32: "int32",
    Precision.INT16: "int16",
    Precision.INT8: "int8",
}


@dataclass
class CaseResult:
    case_name: str
    precision: str
    target: str
    problem_size: str
    time_seconds: float
    flops: int
    tflops: float
    iterations: int
    note: str = ""
    avg_power_watts: float = 0.0
    energy_joules: float = 0.0

    @property
    def gflops(self) -> float:
        return self.tflops * 1000.0

    @property
    def efficiency_gflops_per_watt(self) -> float:
        if self.avg_power_watts > 0:
            return self.gflops / self.avg_power_watts
        return 0.0


@dataclass
class SystemInfo:
    cpu_name: str
    gpu_name: Optional[str] = None
    gpu_backend: Optional[str] = None
    os_info: str = ""
    cuda_version: Optional[str] = None
    supported_precisions: str = ""


@dataclass
class RunnerConfig:
    targets: List[TestTarget] = field(default_factory=list)
    cases: List[str] = field(default_factory=list)
    precisions: List[Precision] = field(default_factory=list)
    duration_mode: DurationMode = DurationMode.QUICK
