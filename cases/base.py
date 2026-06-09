from abc import ABC, abstractmethod
from config import DurationMode, Precision


class BenchmarkCase(ABC):
    name: str = ""

    @abstractmethod
    def get_flops(self, size: int) -> int:
        ...

    @abstractmethod
    def get_size(self, mode: DurationMode, target: str) -> int:
        ...

    @abstractmethod
    def run_cpu(self, size: int, precision: Precision) -> float:
        ...

    @abstractmethod
    def run_gpu(self, size: int, precision: Precision, backend) -> float:
        ...
