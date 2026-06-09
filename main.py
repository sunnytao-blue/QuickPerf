import argparse
import sys
import time
import platform
from typing import List, Optional

from config import Precision, TestTarget, DurationMode, RunnerConfig, SystemInfo
from utils.gpu_detect import detect_gpu, get_cpu_info, get_cuda_version
from benchmark.runner import Runner
from benchmark.gpu_backend import create_backend, GpuBackend

ALL_CASES = ["matmul", "saxpy", "reduction"]
ALL_PRECISIONS = [Precision.FP64, Precision.FP32, Precision.FP16, Precision.BF16]

TARGET_MAP = {"cpu": TestTarget.CPU, "gpu": TestTarget.GPU, "both": TestTarget.BOTH}
PREC_MAP = {"fp64": Precision.FP64, "fp32": Precision.FP32, "fp16": Precision.FP16, "bf16": Precision.BF16}
MODE_MAP = {"quick": DurationMode.QUICK, "normal": DurationMode.NORMAL}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="quickperf",
        description="QuickPerf - CPU/GPU 性能测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py                                              # 交互模式\n"
            "  python main.py -t cpu -c all -p fp32 -m quick               # CPU 全用例快速测试\n"
            "  python main.py -t both -c matmul,reduction -p fp32,fp16     # CPU+GPU MatMul/Reduction\n"
            "  python main.py --list                                        # 列出所有选项\n"
        ),
    )
    parser.add_argument("-t", "--target", choices=["cpu", "gpu", "both"],
                        help="测试目标: cpu, gpu, both")
    parser.add_argument("-c", "--cases",
                        help="测试用例 (英文逗号分隔): matmul, saxpy, reduction, all")
    parser.add_argument("-p", "--precision",
                        help="测试精度 (英文逗号分隔): fp64, fp32, fp16, bf16")
    parser.add_argument("-m", "--mode", choices=["quick", "normal"],
                        help="测试时长: quick, normal")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用选项并退出")
    return parser


def print_options():
    print("""
可用选项:
  测试目标:  cpu, gpu, both
  测试用例:  matmul, saxpy, reduction, all
  测试精度:  fp64, fp32, fp16, bf16
  测试时长:  quick, normal

快捷示例:
  python main.py -t cpu -c all -p fp32 -m quick
  python main.py -t both -c matmul,saxpy -p fp32,fp16 -m normal
""")


def parse_target(arg: Optional[str], gpu_name: Optional[str]) -> Optional[TestTarget]:
    if arg is None:
        return None
    target = TARGET_MAP.get(arg.lower())
    if target is None:
        print(f"  无效目标: {arg}，可选: cpu, gpu, both")
        sys.exit(1)
    if target in (TestTarget.GPU, TestTarget.BOTH) and gpu_name is None:
        print("  未检测到 GPU，请使用 CPU 测试目标")
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
    precs = []
    for p in parts:
        if p not in PREC_MAP:
            print(f"  无效精度: {p}，可选: {', '.join(PREC_MAP.keys())}")
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
    return any([args.target, args.cases, args.precision, args.mode])


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print_options()
        return

    print_header()

    gpu_type, gpu_name = detect_gpu()
    print_gpu_status(gpu_type, gpu_name)

    config = build_config(args, gpu_type, gpu_name)
    if config is None:
        return

    backend = None
    if TestTarget.GPU in config.targets or TestTarget.BOTH in config.targets:
        try:
            backend = create_backend(gpu_type, gpu_name)
        except Exception as e:
            print(f"\n  GPU 后端初始化失败: {e}")
            print("  请确保已安装对应的 GPU 驱动和库。")
            sys.exit(1)

        if hasattr(backend, 'supports_fp64') and not backend.supports_fp64:
            original = config.precisions.copy()
            config.precisions = [p for p in config.precisions if p != Precision.FP64]
            if len(config.precisions) < len(original):
                removed = [p.value for p in original if p not in config.precisions]
                print(f"\n  GPU '{gpu_name}' 不支持双精度浮点 (cl_khr_fp64)")
                print(f"  已自动移除精度: {', '.join(removed)}")
            if not config.precisions:
                print("  错误: 没有可用的测试精度，退出。")
                sys.exit(1)

    runner = Runner(config, backend)

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

    system_info = SystemInfo(
        cpu_name=get_cpu_info(),
        gpu_name=gpu_name,
        gpu_backend=gpu_type.value if gpu_name else None,
        os_info=platform.platform(),
        cuda_version=get_cuda_version() if gpu_type.value == "CUDA" else None,
    )

    from report.generator import ReportGenerator
    gen = ReportGenerator(results, config, system_info, total_time)
    filepath = gen.generate()

    print(f"\n{'=' * 50}")
    print("  测试完成!")
    print(f"  报告已保存至: {filepath}")
    print(f"{'=' * 50}")

    print_results_summary(results)


def build_config(args, gpu_type, gpu_name):
    if not has_any_arg(args):
        return interactive_config(gpu_type, gpu_name)

    target = parse_target(args.target, gpu_name)
    cases = parse_cases(args.cases)
    precisions = parse_precisions(args.precision)
    mode = parse_mode(args.mode)

    if target is None:
        target = select_target(gpu_type, gpu_name)
        if target is None:
            return None
    if cases is None:
        cases = select_cases()
        if not cases:
            return None
    if precisions is None:
        precisions = select_precisions()
        if not precisions:
            return None
    if mode is None:
        mode = select_duration_mode()
        if mode is None:
            return None

    print_config_summary(target, cases, precisions, mode, gpu_name)

    targets = [TestTarget.CPU] if target == TestTarget.CPU else (
        [TestTarget.GPU] if target == TestTarget.GPU else [TestTarget.BOTH]
    )

    return RunnerConfig(
        targets=targets,
        cases=cases,
        precisions=precisions,
        duration_mode=mode,
    )


def print_config_summary(target, cases, precisions, mode, gpu_name):
    target_label = {TestTarget.CPU: "CPU", TestTarget.GPU: "GPU", TestTarget.BOTH: "CPU + GPU"}
    mode_label = {"quick": "Quick (~5-10s)", "normal": "Normal (~30-60s)"}
    print()
    print("  测试配置确认:")
    print(f"    目标:   {target_label.get(target, target)}")
    print(f"    用例:   {', '.join(cases)}")
    print(f"    精度:   {', '.join(p.value for p in precisions)}")
    print(f"    时长:   {mode_label.get(mode.value, mode.value)}")


def print_header():
    print()
    print("=" * 50)
    print("   QuickPerf - CPU/GPU 性能测试工具")
    print("=" * 50)


def print_gpu_status(gpu_type, gpu_name):
    if gpu_name:
        print(f"\n  检测到 GPU: {gpu_name} ({gpu_type.value})")
    else:
        print("\n  未检测到 GPU，仅可进行 CPU 测试")


def interactive_config(gpu_type, gpu_name):
    target = select_target(gpu_type, gpu_name)
    if target is None:
        return None

    cases = select_cases()
    if not cases:
        return None

    precisions = select_precisions()
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
    )


def select_target(gpu_type, gpu_name):
    has_gpu = gpu_name is not None
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


def select_precisions() -> List[Precision]:
    prec_map = {
        "1": Precision.FP64,
        "2": Precision.FP32,
        "3": Precision.FP16,
        "4": Precision.BF16,
    }

    print("\n  请选择测试精度 (用逗号分隔，如 1,2):")
    print("    1. FP64 (双精度)")
    print("    2. FP32 (单精度)")
    print("    3. FP16 (半精度 - CPU 用 float32 模拟)")
    print("    4. BF16 (BFloat16 - CPU 用 float32 模拟)")

    while True:
        try:
            raw = input("\n  请输入选择: ").strip()
            if not raw:
                print("  请至少选择一个精度")
                continue
            indices = [x.strip() for x in raw.split(",")]
            precs = []
            for idx in indices:
                if idx in prec_map:
                    precs.append(prec_map[idx])
                else:
                    print(f"  无效选项: {idx}")
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
            print(f"  {r.case_name:12s} ({r.precision:4s})  {r.time_seconds:.4f}s  {r.tflops:.4f} TFLOPS")

    if gpu_results:
        print("\n  === GPU 结果汇总 ===")
        for r in gpu_results:
            print(f"  {r.case_name:12s} ({r.precision:4s})  {r.time_seconds:.4f}s  {r.tflops:.4f} TFLOPS")

    if cpu_results and gpu_results:
        print("\n  === CPU vs GPU 加速比 ===")
        cpu_map = {f"{r.case_name}_{r.precision}": r for r in cpu_results}
        gpu_map = {f"{r.case_name}_{r.precision}": r for r in gpu_results}
        for key in sorted(cpu_map.keys()):
            cpu_r = cpu_map[key]
            gpu_r = gpu_map.get(key)
            if gpu_r and cpu_r.tflops > 0:
                speedup = gpu_r.tflops / cpu_r.tflops
                print(f"  {cpu_r.case_name:12s} ({cpu_r.precision:4s})  {speedup:.1f}x")


if __name__ == "__main__":
    main()
