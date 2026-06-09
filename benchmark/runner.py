import time
from typing import List

from config import RunnerConfig, CaseResult, TestTarget, Precision
from benchmark.cpu_backend import CpuBackend
from benchmark.gpu_backend import GpuBackend


class Runner:
    def __init__(self, config: RunnerConfig, gpu_backend: GpuBackend = None):
        self.config = config
        self.cpu = CpuBackend()
        self.gpu = gpu_backend

    def run(self, progress_callback=None) -> List[CaseResult]:
        results = []

        total_tasks = 0
        if TestTarget.CPU in self.config.targets or TestTarget.BOTH in self.config.targets:
            total_tasks += len(self.config.cases) * len(self.config.precisions)
        if (TestTarget.GPU in self.config.targets or TestTarget.BOTH in self.config.targets) and self.gpu:
            total_tasks += len(self.config.cases) * len(self.config.precisions)

        completed = 0

        if TestTarget.CPU in self.config.targets or TestTarget.BOTH in self.config.targets:
            results.extend(self._run_cpu_tests(progress_callback, total_tasks, completed))
            completed += len(self.config.cases) * len(self.config.precisions)

        if (TestTarget.GPU in self.config.targets or TestTarget.BOTH in self.config.targets) and self.gpu:
            results.extend(self._run_gpu_tests(progress_callback, total_tasks, completed))

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

    def _run_gpu_tests(self, progress_callback, total: int, completed: int) -> List[CaseResult]:
        results = []
        case_instances = self._get_cases()

        for case in case_instances:
            for prec in self.config.precisions:
                size = case.get_size(self.config.duration_mode, "GPU")
                if progress_callback:
                    progress_callback(completed, total,
                                      f"[GPU] {case.name:12s} ({prec.value:4s}) ... running")

                try:
                    result = self._run_gpu_case(case, size, prec)
                    results.append(result)

                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total,
                                          f"[GPU] {case.name:12s} ({prec.value:4s}) ... {result.time_seconds:.4f}s  {result.tflops:.4f} TFLOPS")
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
                    )
                    results.append(result)
                    if progress_callback:
                        progress_callback(completed, total,
                                          f"[GPU] {case.name:12s} ({prec.value:4s}) ... SKIP ({e})")

        return results

    def _run_gpu_case(self, case, size: int, precision: Precision) -> CaseResult:
        note = ""
        if precision == Precision.FP16:
            if self.gpu.backend_type.value == "OpenCL":
                note = "FP16(float32 simulate)"
        elif precision == Precision.BF16:
            note = "BF16(float32 simulate)"

        self.gpu._current_precision = precision

        elapsed = case.run_gpu(size, precision, self.gpu)

        flops = case.get_flops(size)
        tflops = (flops / elapsed) / 1e12 if elapsed > 0 else 0.0

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
