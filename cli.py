"""CLI argument parsing and interactive menu functions."""

import argparse
import sys
from typing import List, Optional

from config import (
    Precision, TestTarget, DurationMode, RunnerConfig,
    PRECISION_DISPLAY,
)
from utils.gpu_detect import GpuInfo

ALL_CASES = ["matmul", "saxpy", "reduction"]
ALL_PRECISIONS = list(Precision)

TARGET_MAP = {"cpu": TestTarget.CPU, "gpu": TestTarget.GPU, "both": TestTarget.BOTH, "gpu-gpu": TestTarget.GPU_VS_GPU}
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
    parser.add_argument("-t", "--target", choices=["cpu", "gpu", "both", "gpu-gpu"],
                        help="测试目标: cpu, gpu, both, gpu-gpu")
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
    elif target == TestTarget.GPU_VS_GPU:
        if len(all_gpus) < 2:
            print("  需要至少 2 块 GPU 才能进行 GPU 对比测试")
            sys.exit(1)
        if gpu_indices is None:
            print("\n  请选择要对比的 2 块 GPU:")
            gpu_indices = select_gpu_indices(all_gpus)
            if gpu_indices is None:
                return None
        if len(gpu_indices) < 2:
            print("  GPU 对比测试需要选择 2 块 GPU")
            sys.exit(1)
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
        [TestTarget.GPU] if target == TestTarget.GPU else (
            [TestTarget.BOTH] if target == TestTarget.BOTH else [TestTarget.GPU_VS_GPU]
        )
    )

    return RunnerConfig(
        targets=targets,
        cases=cases,
        precisions=precisions,
        duration_mode=mode,
        gpu_indices=gpu_indices,
    )


def print_config_summary(target, cases, precisions, mode, all_gpus, gpu_indices):
    target_label = {TestTarget.CPU: "CPU", TestTarget.GPU: "GPU", TestTarget.BOTH: "CPU + GPU", TestTarget.GPU_VS_GPU: "GPU vs GPU"}
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
    elif target == TestTarget.GPU_VS_GPU:
        if len(all_gpus) < 2:
            print("  需要至少 2 块 GPU 才能进行 GPU 对比测试")
            return None
        print("\n  请选择要对比的 2 块 GPU:")
        gpu_indices = select_gpu_indices(all_gpus)
        if gpu_indices is None or len(gpu_indices) < 2:
            print("  GPU 对比测试需要选择 2 块 GPU")
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
        [TestTarget.GPU] if target == TestTarget.GPU else (
            [TestTarget.BOTH] if target == TestTarget.BOTH else [TestTarget.GPU_VS_GPU]
        )
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
    has_multi_gpu = len(all_gpus) >= 2
    print("\n  请选择测试目标:")
    print("    1. CPU only")
    if has_gpu:
        print("    2. GPU only")
    if has_multi_gpu:
        print("    3. GPU vs GPU (对比两块GPU)")
    if has_gpu:
        print("    4. CPU + GPU (CPU与GPU对比)")
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
            elif choice == 3 and has_multi_gpu:
                return TestTarget.GPU_VS_GPU
            elif choice == 4 and has_gpu:
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
    case_map = {"1": "matmul", "2": "saxpy", "3": "reduction", "4": "all"}

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
