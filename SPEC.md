# QuickPerf — CPU/GPU 性能测试工具 规格说明

## 1. 项目概述

QuickPerf 是一个基于 Python CLI 的跨后端 CPU/GPU 性能测试工具，支持多种计算密集型算子，自动探测 GPU 类型并选择最优加速后端（CUDA / OpenCL），支持命令行参数与交互式双模式，最终输出 Markdown 格式的详细测试报告。

## 2. 功能需求

### 2.1 双模式 CLI

- **命令行参数模式**：一次性传入所有配置，直接执行，无需交互
- **混合模式**：部分参数通过 CLI 传入，其余交互补齐
- **纯交互模式**：无参数启动，逐步引导选择

### 2.2 测试目标

| 选项 | 行为 |
|------|------|
| CPU only | 仅运行 CPU 后端 (numpy) |
| GPU only | 仅运行 GPU 后端 |
| CPU + GPU 对比 | 串行执行，输出两份独立报告 + 一份对比报告 |

### 2.3 GPU 后端自动探测

程序启动时自动检测系统 GPU 类型：

| 检测结果 | 后端 | 库 |
|----------|------|-----|
| NVIDIA GPU | CUDA | `cupy` + `nvidia-cublas-cu12` 等 pip 包 |
| Intel / AMD / 其他 GPU | OpenCL | `pyopencl` |
| 无 GPU | — | 提示用户仅可选 CPU |

CUDA 后端支持两种方式获取 CUDA 运行时库：
1. pip 安装 `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12` 等包
2. 安装完整 CUDA Toolkit 并设置 `CUDA_PATH` 环境变量

OpenCL 后端自动检测设备是否支持 `cl_khr_fp64`，不支持时 FP64 自动跳过。

### 2.4 测试用例

每个用例是一个独立模块，继承自 `cases/base.py` 中的 `BenchmarkCase` 基类。

| 用例 | 模块 | 浮点运算量 | 问题规模 (CPU) | 问题规模 (GPU) |
|------|------|-----------|---------------|---------------|
| MatMul | `cases/matmul.py` | 2 × N³ | Quick: 1024², Normal: 2048² | Quick: 512², Normal: 1024² |
| SAXPY | `cases/saxpy.py` | 2 × N | Quick: 10M, Normal: 50M | Quick: 100M, Normal: 500M |
| Reduction | `cases/reduction.py` | N - 1 | Quick: 10M, Normal: 50M | Quick: 100M, Normal: 500M |

### 2.5 精度支持

| 精度 | CPU | GPU (CUDA) | GPU (OpenCL) | 备注 |
|------|-----|-----------|-------------|------|
| FP64 | float64 | float64 | double | OpenCL 需 cl_khr_fp64，不支持则自动跳过 |
| FP32 | float32 | float32 | float | — |
| FP16 | float32 模拟 | float16 | float32 模拟 | 报告标注 "(float32 simulate)" |
| BF16 | float32 模拟 | bfloat16 | float32 模拟 | 报告标注 "(float32 simulate)" |

### 2.6 测试时长模式

| 模式 | 目标耗时 | 迭代策略 |
|------|---------|---------|
| Quick | ~5-10s | 较小问题尺寸，3 次取最小 |
| Normal | ~30-60s | 较大问题尺寸，3-5 次取最小 |

### 2.7 实时进度显示

```
[CPU] MatMul (FP32) ... running
[1/6] [#####-----------] 16.7%  [CPU] MatMul (FP32) ... 0.0052s  0.4140 TFLOPS
```

### 2.8 性能指标

TFLOPS = 浮点运算次数 / 测试时间(秒) / 10¹²

- GPU 计时使用 CUDA Stream 同步 / OpenCL queue.finish() 后取 wall-clock time
- CPU 计时使用 `time.perf_counter_ns()`
- 结果取多次迭代的最小值（排除系统抖动）

### 2.9 测试报告

- **文件名**：`YYYYMMDD_HHMMSS.md`
- **输出目录**：`./reports/`（运行时创建，已加入 .gitignore）
- **内容**：系统信息、测试配置、各用例耗时/TFLOPS、CPU vs GPU 对比及加速比

## 3. 技术架构

### 3.1 项目结构

```
quickperf/
├── main.py                 # CLI 入口 + argparse 参数解析 + 流程调度
├── config.py               # 枚举、数据类、常量定义
├── requirements.txt        # 依赖清单
├── .gitignore
├── benchmark/
│   ├── runner.py           # 统一执行引擎
│   ├── cpu_backend.py      # CPU 后端
│   ├── gpu_backend.py      # GPU 门面（工厂模式，自动分发）
│   ├── cuda_kernels.py     # NVIDIA CUDA 后端 (cupy)
│   └── opencl_kernels.py   # OpenCL 后端 (pyopencl + raw kernel)
├── cases/
│   ├── base.py             # 用例抽象基类
│   ├── matmul.py
│   ├── saxpy.py
│   └── reduction.py
├── report/
│   └── generator.py        # Markdown 报告生成
├── utils/
│   ├── gpu_detect.py       # GPU 类型自动探测
│   └── timer.py            # 高精度计时工具
└── reports/                # 报告输出目录（运行时创建）
```

### 3.2 模块职责

| 模块 | 职责 |
|------|------|
| `main.py` | CLI 入口，argparse 解析，引导交互，编排流程，FP64 兼容性检查 |
| `config.py` | Precision/TestTarget/DurationMode 枚举，CaseResult/SystemInfo/RunnerConfig 数据类 |
| `benchmark/runner.py` | 接收 RunnerConfig，调度 CPU/GPU 后端，收集 CaseResult，进度回调 |
| `benchmark/cpu_backend.py` | CPU 执行封装，numpy 数据准备+计时 |
| `benchmark/gpu_backend.py` | GpuBackend 抽象基类 + create_backend 工厂方法 |
| `benchmark/cuda_kernels.py` | CudaBackend：to_device/from_device/synchronize/matmul/saxpy/sum |
| `benchmark/opencl_kernels.py` | OpenCLBackend：raw kernel 源码（SAXPY/MatMul/Reduction 各含 float+double 版本），FP64 能力检测，grid-stride 大数组支持 |
| `cases/base.py` | BenchmarkCase 抽象基类（get_flops/get_size/run_cpu/run_gpu） |
| `cases/matmul.py` / `saxpy.py` / `reduction.py` | 具体算子实现 |
| `report/generator.py` | 组装 Markdown 报告，写入文件 |
| `utils/gpu_detect.py` | GPU 探测（尝试 cupy → pyopencl → 返回 NONE） |
| `utils/timer.py` | Timer 上下文管理器 + time_it 工具函数 |

### 3.3 执行流程

```
main()
  ├─ argparse.parse_args()
  ├─ detect_gpu() → GpuBackendType + GPU 名称
  ├─ build_config()
  │   ├─ 若 CLI 全部参数 → 直接组装 RunnerConfig
  │   ├─ 若部分参数 → 补齐交互
  │   └─ 若无参数 → 纯交互模式
  ├─ create_backend() → CudaBackend / OpenCLBackend / None
  ├─ FP64 兼容性检查（OpenCL 不支持则自动移除）
  ├─ Runner(config, backend)
  ├─ runner.run(progress_callback) → List[CaseResult]
  │   ├─ CPU: 每个 case × precision → case.run_cpu() → 计时
  │   └─ GPU: 每个 case × precision → case.run_gpu(backend) → 计时
  │       ├─ try/except 异常捕获 → SKIP with 错误信息
  └─ ReportGenerator.generate() → reports/YYYYMMDD_HHMMSS.md
```

## 4. CLI 参数设计

| 参数 | 简写 | 可选值 | 说明 |
|------|------|--------|------|
| `--target` | `-t` | `cpu`, `gpu`, `both` | 测试目标 |
| `--cases` | `-c` | `matmul`, `saxpy`, `reduction`, `all` | 逗号分隔，`all` = 全选 |
| `--precision` | `-p` | `fp64`, `fp32`, `fp16`, `bf16` | 逗号分隔 |
| `--mode` | `-m` | `quick`, `normal` | 测试时长 |
| `--list` | — | — | 列出可用选项 |

## 5. OpenCL Kernel 设计

### SAXPY（grid-stride）
```c
__kernel void saxpy(__global const float *x, __global float *y, float alpha, int N) {
    int i = get_global_id(0);
    for (int stride = get_global_size(0); i < N; i += stride)
        y[i] = alpha * x[i] + y[i];
}
```

### MatMul（naive, 2D global）
```c
__kernel void matmul_naive(__global const float *A, __global const float *B,
                           __global float *C, int N) {
    int row = get_global_id(0), col = get_global_id(1);
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}
```

### Reduction（grid-stride + tree）
```c
__kernel void reduce_sum(__global const float *input, __global float *output,
                          __local float *local_sum, int N) {
    int gid = get_global_id(0), lid = get_local_id(0);
    float acc = 0.0f;
    for (int i = gid; i < N; i += get_global_size(0)) acc += input[i];
    local_sum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0)/2; s > 0; s >>= 1) {
        if (lid < s) local_sum[lid] += local_sum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) output[get_group_id(0)] = local_sum[0];
}
```

所有 kernel 均有 float / double 独立版本，double 版本使用 `#pragma OPENCL EXTENSION cl_khr_fp64 : enable`。

## 6. 依赖

```
numpy>=1.24
cupy-cuda12x              # NVIDIA GPU
nvidia-cublas-cu12        # cuBLAS 12.x
nvidia-cuda-runtime-cu12  # CUDA Runtime 12.x
nvidia-cuda-nvrtc-cu12    # NVRTC 12.x
nvidia-curand-cu12        # cuRAND 12.x
pyopencl                  # 非 NVIDIA GPU
py-cpuinfo                # CPU 型号获取
rich>=13.0                # 终端美化（可选）
```

## 7. 错误处理

| 场景 | 行为 |
|------|------|
| 无 GPU 但选了 GPU 测试 | 提示并退出 |
| OpenCL 设备不支持 FP64 | 自动从精度列表移除 FP64，继续执行 |
| GPU 用例执行异常 | 捕获异常，输出 `SKIP: <错误信息>`，继续下一用例 |
| CUDA 后端缺 DLL | 报告 SKIP，不中断其他用例 |
| 未知 CLI 参数 | argparse 报错退出 |

## 8. 扩展性设计

添加新测试用例：
1. 在 `cases/` 下新建文件，继承 `BenchmarkCase`
2. 实现 `get_flops()`, `get_size()`, `run_cpu()`, `run_gpu()`
3. 在 `benchmark/runner.py` 的 `_get_cases()` 注册
4. CLI 自动发现（通过 `-c` 参数名匹配）

添加新 GPU 后端：
1. 在 `benchmark/` 下新建后端文件，继承 `GpuBackend`
2. 在 `utils/gpu_detect.py` 添加探测逻辑
3. 在 `benchmark/gpu_backend.py` 的 `create_backend()` 注册

---

**版本**：0.1  
**日期**：2026-06-09  
**作者**：Sunnytao
