import argparse
import sys
import time
import platform
from typing import List, Optional

from config import Precision, TestTarget, DurationMode, RunnerConfig, SystemInfo, PRECISION_DISPLAY, precision_sort_key
from utils.gpu_detect import detect_all_gpus, GpuInfo, get_cpu_info, get_cuda_version
from utils.hardware_info import get_supported_precisions, format_precision_list
from benchmark.runner import Runner
from benchmark.gpu_backend import create_backend, GpuBackend

ALL_CASES = ["matmul", "saxpy", "reduction"]
ALL_PRECISIONS = list(Precision)

TARGET_MAP = {"cpu": TestTarget.CPU, "gpu": TestTarget.GPU, "both": TestTarget.BOTH}
PREC_MAP = {
    "fp64": Precision.FP64, "fp32": Precision.FP32, "fp16": Precision.FP16, "bf16": Precision.BF16,
    "int64": Precision.INT64, "int32": Precision.INT32, "int16": Precision.INT16, "int8": Precision.INT8,
}
MODE_MAP = {"quick": DurationMode.QUICK, "normal": DurationMode.NORMAL}


def build_parser(all_gpus: List[GpuInfo]):
    parser = argparse.ArgumentParser(
        prog="quickperf",
        description="QuickPerf - CPU/GPU 性能测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py -t gpu -c all -p fp32 -m quick\n"
            "  python main.py -t both -c matmul -p fp32,fp16 -m normal\n"
            "  python main.py --list\n"
        ),
    )
    parser.add_argument("-t", "--target", choices=["cpu", "gpu", "both"],
                        help="测试目标: cpu, gpu, both")
    parser.add_argument("-c", "--cases",
                        help="测试用例 (逗号分隔): matmul, saxpy, reduction, all")
    parser.add_argument("-p", "--precision",
                        help="测试精度 (逗号分隔): fp64, fp32, fp16, bf16, int64, int32, int16, int8, all")
    parser.add_argument("-m", "--mode", choices=["quick", "normal"],
                        help="测试时长: quick, normal")

    gpu_help = "选择GPU (逗号分隔索引, all=全部)"
    if all_gpus:
        gpu_names = "; ".join(f"{g.index}={g.name}" for g in all_gpus)
        gpu_help += f"  可用: {gpu_names}"
    parser.add_argument("-g", "--gpus", help=gpu_help)
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用选项并退出")
    return parser


def print_options(all_gpus: List[GpuInfo]):
    print("""
可用选项:
  测试目标:  cpu, gpu, both
  测试用例:  matmul, saxpy, reduction, all
  测试精度:  fp64, fp32, fp16, bf16, int64, int32, int16, int8, all
  测试时长:  quick, normal""")
    if all_gpus:
        print("  可用GPU:")
        for g in all_gpus:
            print(f"    {g.index}: {g.name} ({g.backend_type.value})")
    print("""
快捷示例:
  python main.py -t cpu -c all -p fp32 -m quick
  python main.py -t gpu -g 0 -c all -p fp32 -m quick
""")


def parse_gpu_indices(arg: Optional[str], all_gpus: List[GpuInfo]) -> Optional[List[int]]:
    if arg is None:
        return None
    if arg.lower() == "all":
        return [g.index for g in all_gpus]
    parts = [x.strip() for x in arg.split(",")]
    result = []
    for p in parts:
        try:
            idx = int(p)
            if any(g.index == idx for g in all_gpus):
                result.append(idx)
            else:
                print(f"  无效GPU索引: {idx}, 可用: {[g.index for g in all_gpus]}")
                sys.exit(1)
        except ValueError:
            print(f"  无效GPU参数: {p}, 请使用索引号或 'all'")
            sys.exit(1)
    return result if result else None


def parse_target(arg: Optional[str]) -> Optional[TestTarget]:
    if arg is None:
        return None
    target = TARGET_MAP.get(arg.lower())
    if target is None:
        print(f"  无效目标: {arg}，可选: cpu, gpu, both")
        sys.exit(1)
    return target


def parse_cases(arg: Optional[str]) -> Optional[List[str]]:
    if arg is None:
        return None
    parts = [p.strip().lower() for p in arg.split(",")]
    if "all" in parts:
        return list(ALL_CASES)
    for p in parts:
        if p not in ALL_CASES:
            print(f"  无效用例: {p}，可选: {', '.join(ALL_CASES)}, all")
            sys.exit(1)
    return parts


def parse_precisions(arg: Optional[str]) -> Optional[List[Precision]]:
    if arg is None:
        return None
    parts = [p.strip().lower() for p in arg.split(",")]
    if "all" in parts:
        return list(ALL_PRECISIONS)
    precs = []
    for p in parts:
        if p not in PREC_MAP:
            print(f"  无效精度: {p}，可选: {', '.join(list(PREC_MAP.keys()) + ['all'])}")
            sys.exit(1)
        precs.append(PREC_MAP[p])
    return precs


def parse_mode(arg: Optional[str]) -> Optional[DurationMode]:
    if arg is None:
        return None
    mode = MODE_MAP.get(arg.lower())
    if mode is None:
        print(f"  无效模式: {arg}，可选: quick, normal")
        sys.exit(1)
    return mode


def has_any_arg(args) -> bool:
    return any([args.target, args.cases, args.precision, args.mode, args.gpus])


def main():
    all_gpus = detect_all_gpus()
    parser = build_parser(all_gpus)
    args = parser.parse_args()

    if args.list:
        print_options(all_gpus)
        return

    print_header()
    print_gpu_status(all_gpus)

    supported = get_supported_precisions(all_gpus[0].backend_type if all_gpus else None, "")
    print(f"  支持精度: {format_precision_list(supported)}\n")

    config = build_config(args, all_gpus, supported)
    if config is None:
        return

    if TestTarget.GPU in config.targets or TestTarget.BOTH in config.targets:
        selected_gpus = [g for g in all_gpus if g.index in config.gpu_indices]
    else:
        selected_gpus = []

    backends = []
    for gpu in selected_gpus:
        try:
            backends.append(create_backend(gpu.backend_type, gpu.name))
        except Exception as e:
            print(f"\n  GPU [{gpu.index}] {gpu.name} 后端初始化失败: {e}")

    for i, (gpu, backend) in enumerate(zip(selected_gpus, backends)):
        if hasattr(backend, 'supports_fp64') and not backend.supports_fp64:
            original = config.precisions.copy()
            config.precisions = [p for p in config.precisions if p != Precision.FP64]
            if len(config.precisions) < len(original):
                removed = [p.value for p in original if p not in config.precisions]
                print(f"\n  GPU [{gpu.index}] {gpu.name} 不支持双精度, 已移除: {', '.join(removed)}")
            if not config.precisions:
                print("  错误: 没有可用的测试精度，退出。")
                sys.exit(1)

    runner = Runner(config, backends)

    print(f"\n{'=' * 50}")
    print("  开始执行测试...")
    print(f"{'=' * 50}\n")

    def progress(current, total, msg):
        pct = (current / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  [{current}/{total}] [{bar}] {pct:5.1f}%  {msg}")

    start_time = time.perf_counter()
    results = runner.run(progress_callback=progress)
    total_time = time.perf_counter() - start_time

    gpu_name_str = ", ".join(g.name for g in selected_gpus) if selected_gpus else None
    gpu_list = [(g.name, g.backend_type.value) for g in selected_gpus] if selected_gpus else []
    system_info = SystemInfo(
        cpu_name=get_cpu_info(),
        gpu_name=gpu_name_str or (all_gpus[0].name if all_gpus else None),
        gpu_backend=selected_gpus[0].backend_type.value if selected_gpus else (all_gpus[0].backend_type.value if all_gpus else None),
        os_info=platform.platform(),
        cuda_version=get_cuda_version() if any(g.backend_type.value == "CUDA" for g in selected_gpus) else None,
        gpu_list=gpu_list,
    )

    from report.generator import ReportGenerator
    gen = ReportGenerator(results, config, system_info, total_time)
    filepath = gen.generate()

    print(f"\n{'=' * 50}")
    print("  测试完成!")
    print(f"  报告已保存至: {filepath}")
    print(f"{'=' * 50}")

    print_results_summary(results)


def build_config(args, all_gpus: List[GpuInfo], supported_precisions):
    if not has_any_arg(args):
        return interactive_config(all_gpus, supported_precisions)

    target = parse_target(args.target)
    cases = parse_cases(args.cases)
    precisions = parse_precisions(args.precision)
    mode = parse_mode(args.mode)
    gpu_indices = parse_gpu_indices(args.gpus, all_gpus)

    if target is None:
        target = select_target(all_gpus)
        if target is None:
            return None
    if target in (TestTarget.GPU, TestTarget.BOTH) and not all_gpus:
        print("  未检测到 GPU，请使用 CPU 测试目标")
        sys.exit(1)
    if target in (TestTarget.GPU, TestTarget.BOTH) and gpu_indices is None:
        gpu_indices = select_gpu_indices(all_gpus)
        if gpu_indices is None:
            return None
    elif target == TestTarget.CPU:
        gpu_indices = []
    if cases is None:
        cases = select_cases()
        if not cases:
            return None
    if precisions is None:
        precisions = select_precisions(supported_precisions)
        if not precisions:
            return None

    filtered = [p for p in precisions if p in supported_precisions]
    removed = [p.value for p in precisions if p not in supported_precisions]
    if removed:
        print(f"\n  GPU 不支持以下精度，已自动移除: {', '.join(removed)}")
        precisions = filtered
    if not precisions:
        print("  错误: 没有可用的测试精度，退出。")
        sys.exit(1)

    if mode is None:
        mode = select_duration_mode()
        if mode is None:
            return None

    print_config_summary(target, cases, precisions, mode, all_gpus, gpu_indices)

    targets = [TestTarget.CPU] if target == TestTarget.CPU else (
        [TestTarget.GPU] if target == TestTarget.GPU else [TestTarget.BOTH]
    )

    return RunnerConfig(
        targets=targets,
        cases=cases,
        precisions=precisions,
        duration_mode=mode,
        gpu_indices=gpu_indices,
    )


def print_config_summary(target, cases, precisions, mode, all_gpus, gpu_indices):
    target_label = {TestTarget.CPU: "CPU", TestTarget.GPU: "GPU", TestTarget.BOTH: "CPU + GPU"}
    mode_label = {"quick": "Quick (~5-10s)", "normal": "Normal (~30-60s)"}
    print()
    print("  测试配置确认:")
    print(f"    目标:   {target_label.get(target, target)}")
    if gpu_indices and all_gpus:
        labels = [f"[{i}]{all_gpus[i].name}" for i in gpu_indices if i < len(all_gpus)]
        print(f"    选用GPU: {', '.join(labels)}")
    print(f"    用例:   {', '.join(cases)}")
    print(f"    精度:   {', '.join(p.value for p in precisions)}")
    print(f"    时长:   {mode_label.get(mode.value, mode.value)}")


def print_header():
    print()
    print("=" * 50)
    print("   QuickPerf - CPU/GPU 性能测试工具")
    print("=" * 50)


def print_gpu_status(all_gpus: List[GpuInfo]):
    if all_gpus:
        print(f"\n  检测到 {len(all_gpus)} 块 GPU:")
        for g in all_gpus:
            print(f"    [{g.index}] {g.name} ({g.backend_type.value})")
    else:
        print("\n  未检测到 GPU，仅可进行 CPU 测试")


def interactive_config(all_gpus: List[GpuInfo], supported_precisions):
    target = select_target(all_gpus)
    if target is None:
        return None

    gpu_indices = []
    if target in (TestTarget.GPU, TestTarget.BOTH):
        if not all_gpus:
            print("  未检测到 GPU，请重新选择")
            return None
        gpu_indices = select_gpu_indices(all_gpus)
        if gpu_indices is None:
            return None

    cases = select_cases()
    if not cases:
        return None

    precisions = select_precisions(supported_precisions)
    if not precisions:
        return None

    mode = select_duration_mode()
    if mode is None:
        return None

    targets = [TestTarget.CPU] if target == TestTarget.CPU else (
        [TestTarget.GPU] if target == TestTarget.GPU else [TestTarget.BOTH]
    )

    return RunnerConfig(
        targets=targets,
        cases=cases,
        precisions=precisions,
        duration_mode=mode,
        gpu_indices=gpu_indices,
    )


def select_target(all_gpus: List[GpuInfo]):
    has_gpu = len(all_gpus) > 0
    print("\n  请选择测试目标:")
    print("    1. CPU only")
    if has_gpu:
        print("    2. GPU only")
        print("    3. CPU + GPU (对比测试)")
    print("    q. 退出")

    while True:
        try:
            choice = input("\n  请输入选择: ").strip().lower()
            if choice == "q":
                return None
            choice = int(choice)
            if choice == 1:
                return TestTarget.CPU
            elif choice == 2 and has_gpu:
                return TestTarget.GPU
            elif choice == 3 and has_gpu:
                return TestTarget.BOTH
            else:
                print("  无效选择，请重新输入")
        except (ValueError, IndexError):
            print("  无效输入，请输入数字")


def select_gpu_indices(all_gpus: List[GpuInfo]) -> Optional[List[int]]:
    if len(all_gpus) == 1:
        print(f"\n  使用唯一 GPU: [{all_gpus[0].index}] {all_gpus[0].name}")
        return [all_gpus[0].index]

    print("\n  请选择 GPU (逗号分隔索引, 或输入 'all' 选全部):")
    for g in all_gpus:
        print(f"    [{g.index}] {g.name} ({g.backend_type.value})")
    print("    all - 全部GPU")
    print("    q. 返回")

    while True:
        try:
            raw = input("\n  请输入选择: ").strip().lower()
            if raw == "q":
                return None
            if raw == "all":
                return [g.index for g in all_gpus]
            parts = [x.strip() for x in raw.split(",")]
            indices = []
            for p in parts:
                idx = int(p)
                if any(g.index == idx for g in all_gpus):
                    indices.append(idx)
                else:
                    print(f"  无效GPU索引: {idx}")
                    break
            else:
                if indices:
                    return indices
        except ValueError:
            print("  无效输入")


def select_cases() -> List[str]:
    case_map = {
        "1": "matmul",
        "2": "saxpy",
        "3": "reduction",
        "4": "all",
    }

    print("\n  请选择测试用例 (用逗号分隔，如 1,2,3):")
    print("    1. MatMul    (矩阵乘法)")
    print("    2. SAXPY     (向量乘加)")
    print("    3. Reduction (归约求和)")
    print("    4. All       (全部用例)")

    while True:
        try:
            raw = input("\n  请输入选择: ").strip()
            if not raw:
                print("  请至少选择一个用例")
                continue
            indices = [x.strip() for x in raw.split(",")]
            cases = []
            for idx in indices:
                if idx in case_map:
                    if case_map[idx] == "all":
                        return list(ALL_CASES)
                    cases.append(case_map[idx])
                else:
                    print(f"  无效选项: {idx}")
                    break
            else:
                if cases:
                    return cases
        except Exception:
            print("  无效输入")


def select_precisions(supported_precisions: List[Precision] = None) -> List[Precision]:
    if supported_precisions is None:
        supported_precisions = list(ALL_PRECISIONS)

    print("\n  请选择测试精度 (用逗号分隔，如 1,2 或输入 0 选全部):")
    idx = 1
    prec_map = {}
    for prec in supported_precisions:
        label = PRECISION_DISPLAY.get(prec, prec.value)
        print(f"    {idx}. {label}")
        prec_map[str(idx)] = prec
        idx += 1
    print(f"    0. All  (全部 {len(supported_precisions)} 种精度)")

    while True:
        try:
            raw = input("\n  请输入选择: ").strip()
            if not raw:
                print("  请至少选择一个精度")
                continue
            if raw == "0":
                return list(supported_precisions)
            indices = [x.strip() for x in raw.split(",")]
            precs = []
            for i in indices:
                if i == "0":
                    return list(supported_precisions)
                if i in prec_map:
                    precs.append(prec_map[i])
                else:
                    print(f"  无效选项: {i}")
                    break
            else:
                if precs:
                    return precs
        except Exception:
            print("  无效输入")


def select_duration_mode() -> DurationMode:
    print("\n  请选择测试时长模式:")
    print("    1. Quick  (~5-10s)")
    print("    2. Normal (~30-60s)")

    while True:
        try:
            choice = input("\n  请输入选择: ").strip()
            if choice == "1":
                return DurationMode.QUICK
            elif choice == "2":
                return DurationMode.NORMAL
            else:
                print("  无效选择，请输入 1 或 2")
        except Exception:
            print("  无效输入")


def print_results_summary(results):
    print()
    cpu_results = [r for r in results if r.target == "CPU"]
    gpu_results = [r for r in results if r.target == "GPU"]

    if cpu_results:
        print("  === CPU 结果汇总 ===")
        for r in cpu_results:
            power_str = f"{r.avg_power_watts:.2f}W" if r.avg_power_watts > 0 else "N/A"
            eff_str = f"{r.efficiency_gflops_per_watt:.2f} GFLOPS/W" if r.efficiency_gflops_per_watt > 0 else "N/A"
            print(f"  {r.case_name:12s} ({r.precision:4s})  {r.time_seconds:.4f}s  {r.tflops:.4f} TFLOPS  {power_str}  {eff_str}")
        _print_single_precision_comparison(cpu_results, "CPU")

    if gpu_results:
        print("\n  === GPU 结果汇总 ===")
        for r in gpu_results:
            gpu_tag = f"[{r.gpu_name}]" if r.gpu_name else ""
            power_str = f"{r.avg_power_watts:.2f}W" if r.avg_power_watts > 0 else "N/A"
            eff_str = f"{r.efficiency_gflops_per_watt:.2f} GFLOPS/W" if r.efficiency_gflops_per_watt > 0 else "N/A"
            print(f"  {r.case_name:12s} ({r.precision:4s}) {gpu_tag} {r.time_seconds:.4f}s  {r.tflops:.4f} TFLOPS  {power_str}  {eff_str}")
        _print_single_precision_comparison(gpu_results, "GPU")
        _print_gpu_vs_gpu_comparison(gpu_results)

    if cpu_results and gpu_results:
        gpu_names = sorted(set(r.gpu_name for r in gpu_results if r.gpu_name))

        if len(gpu_names) >= 2:
            _print_three_way_comparison(cpu_results, gpu_results, gpu_names)
        else:
            gpu_label = gpu_names[0] if gpu_names else "GPU"
            print(f"\n  === CPU vs {gpu_label} 加速比 ===")
            cpu_map = {f"{r.case_name}_{r.precision}": r for r in cpu_results}
            gpu_map = {f"{r.case_name}_{r.precision}": r for r in gpu_results}
            for key in sorted(cpu_map.keys()):
                cpu_r = cpu_map[key]
                gpu_r = gpu_map.get(key)
                if gpu_r and cpu_r.tflops > 0:
                    speedup = gpu_r.tflops / cpu_r.tflops
                    if cpu_r.efficiency_gflops_per_watt > 0 and gpu_r.efficiency_gflops_per_watt > 0:
                        efficiency_up = gpu_r.efficiency_gflops_per_watt / cpu_r.efficiency_gflops_per_watt
                        eff_str = f"能效:{efficiency_up:.1f}x"
                    else:
                        eff_str = "能效:N/A"
                    print(f"  {cpu_r.case_name:12s} ({cpu_r.precision:4s})  性能:{speedup:.1f}x  {eff_str}")

        print_precision_comparison(cpu_results, gpu_results)


def _print_single_precision_comparison(results, target_label):
    precisions = sorted(set(r.precision for r in results), key=precision_sort_key)
    if len(precisions) < 2:
        return
    cases = sorted(set(r.case_name for r in results))

    col_w = max(max(len(p) for p in precisions), 8)

    print(f"\n  === {target_label} 跨精度对比 (TFLOPS) ===")
    row = f"  {'用例':12s}"
    for prec in precisions:
        row += f"  {prec:>{col_w}s}"
    print(row)

    for case in cases:
        row = f"  {case:12s}"
        for prec in precisions:
            r = next((x for x in results if x.case_name == case and x.precision == prec), None)
            if r and r.tflops > 0:
                row += f"  {r.tflops:{col_w}.4f}"
            else:
                row += f"  {'N/A':>{col_w}s}"
        print(row)


def _print_gpu_vs_gpu_comparison(gpu_results):
    gpu_names = sorted(set(r.gpu_name for r in gpu_results if r.gpu_name))
    if len(gpu_names) < 2:
        return

    cases = sorted(set(r.case_name for r in gpu_results))
    precisions = sorted(set(r.precision for r in gpu_results), key=precision_sort_key)

    gpu_label_width = max(10, max(len(n) for n in gpu_names))

    print(f"\n  === 多GPU对比 (TFLOPS) ===")
    for prec in precisions:
        print(f"\n  精度: {prec}")
        header = f"  {'用例':12s}"
        for name in gpu_names:
            header += f"  {name:>{gpu_label_width}s}"
        print(header)

        for case in cases:
            row = f"  {case:12s}"
            for name in gpu_names:
                r = next((x for x in gpu_results if x.case_name == case and x.precision == prec and x.gpu_name == name), None)
                if r and r.tflops > 0:
                    row += f"  {r.tflops:{gpu_label_width}.4f}"
                else:
                    row += f"  {'N/A':>{gpu_label_width}s}"
            print(row)


def _print_three_way_comparison(cpu_results, gpu_results, gpu_names):
    cases = sorted(set(r.case_name for r in cpu_results))
    precisions = sorted(set(r.precision for r in cpu_results), key=precision_sort_key)
    name_w = max(10, max(len(n) for n in gpu_names))

    for prec in precisions:
        print(f"\n  === CPU + 多GPU 对比 (TFLOPS) — {prec} ===")
        header = f"  {'用例':12s}  {'CPU':>8s}"
        for name in gpu_names:
            header += f"  {name:>{name_w}s}"
        header += f"  {'加速比':>8s}"
        print(header)

        for case in cases:
            cpu_r = next((r for r in cpu_results if r.case_name == case and r.precision == prec), None)
            if not cpu_r or cpu_r.tflops <= 0:
                continue
            cpu_val = cpu_r.tflops
            row = f"  {case:12s}  {cpu_val:8.4f}"
            speedups = []
            for name in gpu_names:
                gpu_r = next((r for r in gpu_results if r.case_name == case and r.precision == prec and r.gpu_name == name), None)
                if gpu_r and gpu_r.tflops > 0:
                    row += f"  {gpu_r.tflops:{name_w}.4f}"
                    speedups.append(f"{gpu_r.tflops / cpu_val:.1f}x")
                else:
                    row += f"  {'N/A':>{name_w}s}"
                    speedups.append("N/A")
            row += f"  {', '.join(speedups):>8s}"
            print(row)


def print_precision_comparison(cpu_results, gpu_results):
    if not cpu_results or not gpu_results:
        return
    precisions = sorted(set(r.precision for r in cpu_results), key=precision_sort_key)
    if len(precisions) < 2:
        return

    cases = sorted(set(r.case_name for r in cpu_results))

    col_w = max(max(len(p) for p in precisions), 8)

    print("\n  === 跨精度对比 (GPU vs CPU 加速比) ===")
    header = f"  {'用例':12s}"
    for prec in precisions:
        header += f"  {prec:>{col_w}s}"
    print(header)

    for case in cases:
        row = f"  {case:12s}"
        for prec in precisions:
            cpu_r = next((r for r in cpu_results if r.case_name == case and r.precision == prec), None)
            gpu_r = next((r for r in gpu_results if r.case_name == case and r.precision == prec), None)
            if cpu_r and gpu_r and cpu_r.tflops > 0:
                speedup = gpu_r.tflops / cpu_r.tflops
                row += f"  {speedup:{col_w - 1}.1f}x"
            else:
                row += f"  {'N/A':>{col_w}s}"
        print(row)


if __name__ == "__main__":
    main()
