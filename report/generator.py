import os
from datetime import datetime
from typing import List

from config import CaseResult, RunnerConfig, SystemInfo, TestTarget


class ReportGenerator:
    def __init__(self, results: List[CaseResult], config: RunnerConfig,
                 system_info: SystemInfo, total_time: float):
        self.results = results
        self.config = config
        self.system_info = system_info
        self.total_time = total_time

    def generate(self) -> str:
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".md"
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        filepath = os.path.join(report_dir, filename)

        content = self._build_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return filepath

    def _build_report(self) -> str:
        lines = []

        lines.append("# QuickPerf 测试报告")
        lines.append("")
        lines.append(f"**测试时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**总耗时**：{self.total_time:.2f}s")
        lines.append("")

        lines.extend(self._build_system_info())
        lines.extend(self._build_config())
        lines.extend(self._build_cpu_results())
        lines.extend(self._build_gpu_results())
        lines.extend(self._build_comparison())

        return "\n".join(lines)

    def _build_system_info(self):
        lines = ["## 系统信息", ""]
        lines.append("| 项目 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| CPU | {self.system_info.cpu_name} |")
        if self.system_info.gpu_name:
            lines.append(f"| GPU | {self.system_info.gpu_name} |")
            lines.append(f"| GPU Backend | {self.system_info.gpu_backend} |")
        if self.system_info.cuda_version:
            lines.append(f"| CUDA Version | {self.system_info.cuda_version} |")
        lines.append(f"| OS | {self.system_info.os_info} |")
        lines.append("")
        return lines

    def _build_config(self):
        lines = ["## 测试配置", ""]
        lines.append(f"- **精度**：{', '.join(p.value for p in self.config.precisions)}")
        lines.append(f"- **时长模式**：{'Normal' if self.config.duration_mode.value == 'normal' else 'Quick'}")
        targets = []
        if TestTarget.CPU in self.config.targets:
            targets.append("CPU")
        if TestTarget.GPU in self.config.targets:
            targets.append("GPU")
        if TestTarget.BOTH in self.config.targets:
            targets = ["CPU", "GPU"]
        lines.append(f"- **测试目标**：[{', '.join(targets)}]")
        lines.append(f"- **测试用例**：{', '.join(self.config.cases)}")
        lines.append("")
        return lines

    def _build_cpu_results(self):
        cpu_results = [r for r in self.results if r.target == "CPU"]
        if not cpu_results:
            return []

        lines = ["## CPU 测试结果", ""]
        lines.append("| 用例 | 精度 | 问题规模 | 耗时 (s) | GFLOPS | 功耗 (W) | 能量 (J) | 能效 (GFLOPS/W) | 备注 |")
        lines.append("|------|------|---------|----------|--------|----------|----------|------------------|------|")
        for r in cpu_results:
            note = r.note if r.note else "-"
            lines.append(
                f"| {r.case_name} | {r.precision} | {r.problem_size} | "
                f"{r.time_seconds:.4f} | {r.gflops:.2f} | {r.avg_power_watts:.2f} | "
                f"{r.energy_joules:.2f} | {r.efficiency_gflops_per_watt:.2f} | {note} |"
            )
        lines.append("")
        return lines

    def _build_gpu_results(self):
        gpu_results = [r for r in self.results if r.target == "GPU"]
        if not gpu_results:
            return []

        lines = ["## GPU 测试结果", ""]
        lines.append("| 用例 | 精度 | 问题规模 | 耗时 (s) | GFLOPS | 功耗 (W) | 能量 (J) | 能效 (GFLOPS/W) | 备注 |")
        lines.append("|------|------|---------|----------|--------|----------|----------|------------------|------|")
        for r in gpu_results:
            note = r.note if r.note else "-"
            lines.append(
                f"| {r.case_name} | {r.precision} | {r.problem_size} | "
                f"{r.time_seconds:.4f} | {r.gflops:.2f} | {r.avg_power_watts:.2f} | "
                f"{r.energy_joules:.2f} | {r.efficiency_gflops_per_watt:.2f} | {note} |"
            )
        lines.append("")
        return lines

    def _build_comparison(self):
        cpu_results = {f"{r.case_name}_{r.precision}": r for r in self.results if r.target == "CPU"}
        gpu_results = {f"{r.case_name}_{r.precision}": r for r in self.results if r.target == "GPU"}

        if not cpu_results or not gpu_results:
            return []

        lines = ["## CPU vs GPU 对比", ""]
        lines.append("| 用例 | 精度 | CPU TFLOPS | GPU TFLOPS | 加速比 | CPU功耗 (W) | GPU功耗 (W) | CPU能效 | GPU能效 | 能效提升 |")
        lines.append("|------|------|-----------|-----------|--------|------------|------------|--------|--------|----------|")
        for key in sorted(cpu_results.keys()):
            cpu = cpu_results[key]
            gpu = gpu_results.get(key)
            if gpu and cpu.tflops > 0:
                speedup = gpu.tflops / cpu.tflops
                efficiency_improvement = gpu.efficiency_gflops_per_watt / cpu.efficiency_gflops_per_watt if cpu.efficiency_gflops_per_watt > 0 else 0
                lines.append(
                    f"| {cpu.case_name} | {cpu.precision} | "
                    f"{cpu.tflops:.4f} | {gpu.tflops:.4f} | {speedup:.1f}x | "
                    f"{cpu.avg_power_watts:.2f} | {gpu.avg_power_watts:.2f} | "
                    f"{cpu.efficiency_gflops_per_watt:.2f} | {gpu.efficiency_gflops_per_watt:.2f} | "
                    f"{efficiency_improvement:.1f}x |"
                )
        lines.append("")
        return lines
