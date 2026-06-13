"""QuickPerf - CPU/GPU 性能测试工具 入口"""

import sys
import time
import platform

from config import Precision, TestTarget, SystemInfo
from cli import (
    build_parser, print_options, build_config, print_header, print_gpu_status,
)
from output import print_results_summary
from utils.gpu_detect import detect_all_gpus, get_cpu_info, get_cuda_version
from utils.hardware_info import get_supported_precisions, format_precision_list
from benchmark.runner import Runner
from benchmark.gpu_backend import create_backend


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


if __name__ == "__main__":
    main()
