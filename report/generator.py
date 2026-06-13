import os
from datetime import datetime
from typing import List

from config import CaseResult, RunnerConfig, SystemInfo, TestTarget, precision_sort_key


def _fmt_power(watts: float) -> str:
    if watts <= 0:
        return "N/A"
    return f"{watts:.2f}"


def _fmt_efficiency(gflops_per_watt: float) -> str:
    if gflops_per_watt <= 0:
        return "N/A"
    return f"{gflops_per_watt:.2f}"


def _fmt_energy(joules: float, watts: float) -> str:
    if watts <= 0 or joules <= 0:
        return "N/A"
    return f"{joules:.2f}"


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
        lines.extend(self._build_cpu_precision_comparison())
        lines.extend(self._build_gpu_precision_comparison())
        lines.extend(self._build_comparison())
        lines.extend(self._build_precision_comparison())

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
        lines.append("| 用例 | 精度 | 问题规模 | 耗时 (s) | GFLOPS | TFLOPS | 功耗 (W) | 能量 (J) | 能效 (GFLOPS/W) | 备注 |")
        lines.append("|------|------|---------|----------|--------|--------|----------|----------|------------------|------|")
        for r in cpu_results:
            note = r.note if r.note else "-"
            lines.append(
                f"| {r.case_name} | {r.precision} | {r.problem_size} | "
                f"{r.time_seconds:.4f} | {r.gflops:.2f} | {r.tflops:.4f} | "
                f"{_fmt_power(r.avg_power_watts)} | {_fmt_energy(r.energy_joules, r.avg_power_watts)} | "
                f"{_fmt_efficiency(r.efficiency_gflops_per_watt)} | {note} |"
            )
        lines.append("")
        return lines

    def _build_gpu_results(self):
        gpu_results = [r for r in self.results if r.target == "GPU"]
        if not gpu_results:
            return []

        lines = ["## GPU 测试结果", ""]
        lines.append("| 用例 | 精度 | 问题规模 | 耗时 (s) | GFLOPS | TFLOPS | 功耗 (W) | 能量 (J) | 能效 (GFLOPS/W) | 备注 |")
        lines.append("|------|------|---------|----------|--------|--------|----------|----------|------------------|------|")
        for r in gpu_results:
            note = r.note if r.note else "-"
            lines.append(
                f"| {r.case_name} | {r.precision} | {r.problem_size} | "
                f"{r.time_seconds:.4f} | {r.gflops:.2f} | {r.tflops:.4f} | "
                f"{_fmt_power(r.avg_power_watts)} | {_fmt_energy(r.energy_joules, r.avg_power_watts)} | "
                f"{_fmt_efficiency(r.efficiency_gflops_per_watt)} | {note} |"
            )
        lines.append("")
        return lines

    def _build_cpu_precision_comparison(self):
        cpu_results = [r for r in self.results if r.target == "CPU"]
        return self._build_single_precision_comparison(cpu_results, "CPU")

    def _build_gpu_precision_comparison(self):
        gpu_results = [r for r in self.results if r.target == "GPU"]
        return self._build_single_precision_comparison(gpu_results, "GPU")

    @staticmethod
    def _build_single_precision_comparison(results, target_label):
        if not results:
            return []
        precisions = sorted(set(r.precision for r in results), key=precision_sort_key)
        if len(precisions) < 2:
            return []

        cases = sorted(set(r.case_name for r in results))

        lines = [f"## {target_label} 跨精度对比 (TFLOPS)", ""]
        header = "| 用例 |"
        sep = "|------|"
        for prec in precisions:
            header += f" {prec} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        for case in cases:
            row = f"| {case} |"
            for prec in precisions:
                r = next((x for x in results if x.case_name == case and x.precision == prec), None)
                if r and r.tflops > 0:
                    row += f" {r.tflops:.4f} |"
                else:
                    row += " N/A |"
            lines.append(row)

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
                if cpu.efficiency_gflops_per_watt > 0 and gpu.efficiency_gflops_per_watt > 0:
                    eff_str = f"{gpu.efficiency_gflops_per_watt / cpu.efficiency_gflops_per_watt:.1f}x"
                else:
                    eff_str = "N/A"
                lines.append(
                    f"| {cpu.case_name} | {cpu.precision} | "
                    f"{cpu.tflops:.4f} | {gpu.tflops:.4f} | {speedup:.1f}x | "
                    f"{_fmt_power(cpu.avg_power_watts)} | {_fmt_power(gpu.avg_power_watts)} | "
                    f"{_fmt_efficiency(cpu.efficiency_gflops_per_watt)} | {_fmt_efficiency(gpu.efficiency_gflops_per_watt)} | "
                    f"{eff_str} |"
                )
        lines.append("")
        return lines

    def _build_precision_comparison(self):
        cpu_results = [r for r in self.results if r.target == "CPU"]
        gpu_results = [r for r in self.results if r.target == "GPU"]
        if not cpu_results or not gpu_results:
            return []

        precisions = sorted(set(r.precision for r in cpu_results), key=precision_sort_key)
        if len(precisions) < 2:
            return []

        cases = sorted(set(r.case_name for r in cpu_results))

        lines = ["## 跨精度对比 (GPU vs CPU 加速比)", ""]
        header = "| 用例 |"
        sep = "|------|"
        for prec in precisions:
            header += f" {prec} |"
            sep += "---------|"
        lines.append(header)
        lines.append(sep)

        for case in cases:
            row = f"| {case} |"
            for prec in precisions:
                cpu_r = next((r for r in cpu_results if r.case_name == case and r.precision == prec), None)
                gpu_r = next((r for r in gpu_results if r.case_name == case and r.precision == prec), None)
                if cpu_r and gpu_r and cpu_r.tflops > 0:
                    speedup = gpu_r.tflops / cpu_r.tflops
                    row += f" {speedup:.1f}x |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")
        return lines
