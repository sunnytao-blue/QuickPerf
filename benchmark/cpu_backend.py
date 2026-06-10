from config import Precision, CaseResult


class CpuBackend:
    def get_name(self) -> str:
        from utils.gpu_detect import get_cpu_info
        return get_cpu_info()

    def run_case(self, case, size: int, precision: Precision) -> CaseResult:
        import time
        note = ""
        if precision in (Precision.FP16, Precision.BF16):
            note = f"{precision.value}(float32 simulate)"

        from utils.power_monitor import PowerMonitorContext
        with PowerMonitorContext(sample_interval=0.05) as monitor:
            elapsed = case.run_cpu(size, precision)
        cpu_power, _ = monitor.stop()

        flops = case.get_flops(size)
        tflops = (flops / elapsed) / 1e12 if elapsed > 0 else 0.0
        energy_joules = cpu_power * elapsed

        return CaseResult(
            case_name=case.name,
            precision=precision.value,
            target="CPU",
            problem_size=self._format_size(case, size),
            time_seconds=elapsed,
            flops=flops,
            tflops=tflops,
            iterations=1,
            note=note,
            avg_power_watts=cpu_power,
            energy_joules=energy_joules,
        )

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
