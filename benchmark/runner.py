import time
from typing import List

from config import RunnerConfig, CaseResult, TestTarget, Precision
from benchmark.cpu_backend import CpuBackend
from benchmark.gpu_backend import GpuBackend


class Runner:
    def __init__(self, config: RunnerConfig, gpu_backends: List[GpuBackend] = None):
        self.config = config
        self.cpu = CpuBackend()
        self.gpu_backends = gpu_backends or []

    def run(self, progress_callback=None) -> List[CaseResult]:
        results = []

        total_tasks = 0
        if TestTarget.CPU in self.config.targets or TestTarget.BOTH in self.config.targets:
            total_tasks += len(self.config.cases) * len(self.config.precisions)
        if (TestTarget.GPU in self.config.targets or TestTarget.BOTH in self.config.targets):
            total_tasks += len(self.config.cases) * len(self.config.precisions) * len(self.gpu_backends)

        completed = 0

        if TestTarget.CPU in self.config.targets or TestTarget.BOTH in self.config.targets:
            results.extend(self._run_cpu_tests(progress_callback, total_tasks, completed))
            completed += len(self.config.cases) * len(self.config.precisions)

        if (TestTarget.GPU in self.config.targets or TestTarget.BOTH in self.config.targets):
            for backend in self.gpu_backends:
                results.extend(self._run_gpu_tests(backend, progress_callback, total_tasks, completed))
                completed += len(self.config.cases) * len(self.config.precisions)

        return results

    def _run_cpu_tests(self, progress_callback, total: int, completed: int) -> List[CaseResult]:
        results = []
        case_instances = self._get_cases()

        for case in case_instances:
            for prec in self.config.precisions:
                size = case.get_size(self.config.duration_mode, "CPU")
                if progress_callback:
                    progress_callback(completed, total,
                                      f"[CPU] {case.name:12s} ({prec.value:4s}) ... running")

                result = self.cpu.run_case(case, size, prec)
                results.append(result)

                completed += 1
                if progress_callback:
                    progress_callback(completed, total,
                                      f"[CPU] {case.name:12s} ({prec.value:4s}) ... {result.time_seconds:.4f}s  {result.tflops:.4f} TFLOPS")

        return results

    def _run_gpu_tests(self, backend: GpuBackend, progress_callback, total: int, completed: int) -> List[CaseResult]:
        results = []
        case_instances = self._get_cases()
        gpu_label = backend.gpu_name or "GPU"

        for case in case_instances:
            for prec in self.config.precisions:
                size = case.get_size(self.config.duration_mode, "GPU")
                if progress_callback:
                    progress_callback(completed, total,
                                      f"[GPU] {case.name:12s} ({prec.value:4s}) ... running [{gpu_label}]")

                try:
                    result = self._run_gpu_case(case, size, prec, backend)
                    results.append(result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total,
                                          f"[GPU] {case.name:12s} ({prec.value:4s}) ... {result.time_seconds:.4f}s  {result.tflops:.4f} TFLOPS  [{gpu_label}]")
                except Exception as e:
                    completed += 1
                    result = CaseResult(
                        case_name=case.name,
                        precision=prec.value,
                        target="GPU",
                        problem_size=self._format_size(case, size),
                        time_seconds=0,
                        flops=0,
                        tflops=0,
                        iterations=0,
                        note=f"SKIP: {e}",
                        gpu_name=gpu_label,
                    )
                    results.append(result)
                    if progress_callback:
                        progress_callback(completed, total,
                                          f"[GPU] {case.name:12s} ({prec.value:4s}) ... SKIP ({e}) [{gpu_label}]")

        return results

    def _run_gpu_case(self, case, size: int, precision: Precision, backend: GpuBackend) -> CaseResult:
        note = ""
        if precision == Precision.FP16:
            if backend.backend_type.value == "OpenCL":
                note = "FP16(float32 simulate)"
        elif precision == Precision.BF16:
            note = "BF16(float32 simulate)"

        backend._current_precision = precision

        from utils.power_monitor import PowerMonitorContext
        with PowerMonitorContext(sample_interval=0.05) as monitor:
            elapsed = case.run_gpu(size, precision, backend)
        _, gpu_power = monitor.stop()

        flops = case.get_flops(size)
        tflops = (flops / elapsed) / 1e12 if elapsed > 0 else 0.0
        energy_joules = gpu_power * elapsed

        return CaseResult(
            case_name=case.name,
            precision=precision.value,
            target="GPU",
            problem_size=self._format_size(case, size),
            time_seconds=elapsed,
            flops=flops,
            tflops=tflops,
            iterations=1,
            note=note,
            avg_power_watts=gpu_power,
            energy_joules=energy_joules,
            gpu_name=backend.gpu_name,
        )

    def _get_cases(self):
        from cases.matmul import MatMulCase
        from cases.saxpy import SaxpyCase
        from cases.reduction import ReductionCase

        registry = {
            "matmul": MatMulCase,
            "saxpy": SaxpyCase,
            "reduction": ReductionCase,
        }
        return [registry[name]() for name in self.config.cases if name in registry]

    @staticmethod
    def _format_size(case, size: int) -> str:
        name = case.name
        if name == "MatMul":
            return f"{size}x{size}"
        elif size >= 1_000_000:
            return f"{size / 1_000_000:.0f}M elem"
        elif size >= 1_000:
            return f"{size / 1_000:.0f}K elem"
        return str(size)
