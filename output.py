"""Result display and comparison table output functions."""

from config import precision_sort_key


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

    elif not cpu_results and len(set(r.gpu_name for r in gpu_results if r.gpu_name)) >= 2:
        _print_gpu_vs_gpu_speedup(gpu_results)


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


def _print_gpu_vs_gpu_speedup(gpu_results):
    gpu_names = sorted(set(r.gpu_name for r in gpu_results if r.gpu_name))
    if len(gpu_names) < 2:
        return

    cases = sorted(set(r.case_name for r in gpu_results))
    precisions = sorted(set(r.precision for r in gpu_results), key=precision_sort_key)
    base_name = gpu_names[0]
    other_names = gpu_names[1:]

    print(f"\n  === GPU vs GPU 加速比 (基准: {base_name}) ===")
    name_w = max(len(n) for n in gpu_names)

    for prec in precisions:
        print(f"\n  精度: {prec}")
        header = f"  {'用例':12s}"
        for name in gpu_names:
            header += f"  {name:>{name_w}s}"
        header += "  加速比"
        print(header)

        for case in cases:
            base = next((r for r in gpu_results if r.case_name == case and r.precision == prec and r.gpu_name == base_name), None)
            if not base or base.tflops <= 0:
                continue
            row = f"  {case:12s}"
            speedups = []
            for name in gpu_names:
                r = next((x for x in gpu_results if x.case_name == case and x.precision == prec and x.gpu_name == name), None)
                if r and r.tflops > 0:
                    row += f"  {r.tflops:{name_w}.4f}"
                else:
                    row += f"  {'N/A':>{name_w}s}"

            for name in other_names:
                r = next((x for x in gpu_results if x.case_name == case and x.precision == prec and x.gpu_name == name), None)
                if r and r.tflops > 0:
                    speedups.append(f"{r.tflops / base.tflops:.1f}x")
                else:
                    speedups.append("N/A")
            row += f"  {', '.join(speedups)}"
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
