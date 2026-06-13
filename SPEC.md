# QuickPerf — CPU/GPU 性能测试工具 规格说明

## 1. 项目概述

QuickPerf 是一个基于 Python CLI 的跨后端 CPU/GPU 性能测试工具，支持多种计算和访存密集型算子，自动探测 GPU 类型并选择最优加速后端（CUDA / OpenCL），支持命令行参数与交互式双模式，支持多 GPU 选择与对比，最终输出 Markdown 格式的详细测试报告。

## 2. 功能需求

### 2.1 双模式 CLI

- **命令行参数模式**：一次性传入所有配置，直接执行，无需交互
- **混合模式**：部分参数通过 CLI 传入，其余交互补齐
- **纯交互模式**：无参数启动，逐步引导选择

### 2.2 测试目标

| 选项 | 行为 |
|------|------|
| CPU only | 仅运行 CPU 后端 (numpy) |
| GPU only | 仅运行指定 GPU 后端 |
| CPU + GPU 对比 | 串行执行 CPU + n 块 GPU，输出对比报告 |
| GPU vs GPU 对比 | 仅对比两块 GPU，不测 CPU |

### 2.3 GPU 后端自动探测

程序启动时自动检测所有 GPU 并列出，支持多 GPU 选择（`-g 0,1` 或 `-g all`）。

| 检测结果 | 后端 | 库 |
|----------|------|-----|
| NVIDIA GPU | CUDA | `cupy` + `nvidia-cublas-cu12` 等 pip 包 |
| Intel / AMD / 其他 GPU | OpenCL | `pyopencl` |
| 无 GPU | — | 提示用户仅可选 CPU |

CUDA 后端支持两种方式获取运行时库：
1. pip 安装 `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12` 等包
2. 安装完整 CUDA Toolkit 并设置 `CUDA_PATH` 环境变量

OpenCL 后端自动检测设备是否支持 `cl_khr_fp64`，不支持时 FP64 自动跳过。

### 2.4 硬件精度能力检测

程序启动时自动检测当前硬件支持的精度格式，显示可用精度列表。交互菜单和 CLI 参数仅展示实际支持的精度选项，不支持的精度（如 BF16 需 CC≥8.0）自动过滤。

### 2.5 测试用例

每个用例继承自 `cases/base.py` 中的 `BenchmarkCase` 基类。

| 用例 | 模块 | 浮点运算量 | 问题规模 (CPU) | 问题规模 (GPU) |
|------|------|-----------|---------------|---------------|
| MatMul | `cases/matmul.py` | 2 × N³ | Quick: 1024², Normal: 2048² | Quick: 512², Normal: 1024² |
| SAXPY | `cases/saxpy.py` | 2 × N | Quick: 10M, Normal: 50M | Quick: 100M, Normal: 200M |
| Reduction | `cases/reduction.py` | N - 1 | Quick: 10M, Normal: 50M | Quick: 100M, Normal: 200M |
| FlashAttention | `cases/flashattention.py` | 4 × N² × d + 5 × N² | Quick: N=256, Normal: N=512 | Quick: N=512, Normal: N=1024 |

FlashAttention 实现 attention(Q, K, V, d=64) = softmax(Q·K^T/√d) · V，包含 matmul + softmax + matmul 三步计算。

### 2.6 精度支持

| 精度 | CPU | GPU (CUDA) | GPU (OpenCL) | 备注 |
|------|-----|-----------|-------------|------|
| FP64 | float64 | float64 | double | OpenCL 需 cl_khr_fp64 |
| FP32 | float32 | float32 | float | — |
| FP16 | float32 模拟 | float16 | float32 模拟 | 报告标注 "(float32 simulate)" |
| BF16 | float32 模拟 | bfloat16 | float32 模拟 | 报告标注, 需 CC≥8.0 |
| INT64 | int64 | int64 | int64 | — |
| INT32 | int32 | int32 | int32 | — |
| INT16 | int16 | int16 | int16 | — |
| INT8 | int8 | int8 | int8 | — |

### 2.7 测试时长模式

| 模式 | 目标耗时 | 迭代策略 |
|------|---------|---------|
| Quick | ~5-10s | 较小问题尺寸，3 次取最小 |
| Normal | ~30-60s | 较大问题尺寸，3-5 次取最小 |

### 2.8 实时进度显示

```
[CPU] MatMul (FP32) ... running
[1/6] [#####-----------] 16.7%  [CPU] MatMul (FP32) ... 0.0052s  0.4140 TFLOPS
```

### 2.9 性能指标

TFLOPS = 浮点运算次数 / 测试时间(秒) / 10¹²

- GPU 计时使用 CUDA Stream 同步 / OpenCL queue.finish() 后取 wall-clock time
- 功耗采集仅 NVIDIA GPU（nvidia-smi），其他 GPU 显示 N/A
- CPU 计时使用 `time.perf_counter_ns()`
- 结果取多次迭代的最小值（排除系统抖动）

### 2.10 测试报告

- **文件名**：`YYYYMMDD_HHMMSS.md`
- **输出目录**：`./reports/`（运行时创建，已加入 .gitignore）
- **内容**：系统信息（含多 GPU 列表）、测试配置、各用例耗时/TFLOPS、跨精度对比、多 GPU 对比、CPU+多GPU 三路对比、GPU vs GPU 加速比

## 3. 技术架构

### 3.1 项目结构

```
quickperf/
├── main.py                 # 流程编排 (80 行)
├── cli.py                  # argparse + 交互式菜单
├── output.py               # 终端结果输出 + 对比表打印
├── config.py               # 枚举、数据类、常量定义
├── requirements.txt        # 依赖清单
├── .gitignore
├── benchmark/
│   ├── runner.py           # 统一执行引擎
│   ├── cpu_backend.py      # CPU 后端
│   ├── gpu_backend.py      # GPU 门面（工厂模式，自动分发）
│   ├── cuda_kernels.py     # NVIDIA CUDA 后端 (cupy)
│   ├── opencl_kernels.py   # OpenCL 后端
│   └── opencl/
│       ├── saxpy.cl / saxpy_double.cl
│       ├── matmul.cl / matmul_double.cl
│       ├── reduce.cl / reduce_double.cl
│       └── softmax.cl / softmax_double.cl
├── cases/
│   ├── base.py             # 用例抽象基类
│   ├── utils.py            # 共享工具 (get_dtype/create_array)
│   ├── matmul.py
│   ├── saxpy.py
│   ├── reduction.py
│   └── flashattention.py
├── report/
│   └── generator.py        # Markdown 报告生成
└── utils/
    ├── gpu_detect.py       # 多 GPU 自动探测
    ├── hardware_info.py    # 硬件精度能力检测
    ├── power_monitor.py    # 功耗监控 (nvidia-smi)
    └── timer.py            # 高精度计时工具
```

### 3.2 模块职责

| 模块 | 职责 |
|------|------|
| `main.py` | 流程编排入口，GPU 后端初始化，结果汇总触发 |
| `cli.py` | argparse 解析、交互式菜单引导、config 构建、配置摘要打印 |
| `output.py` | 终端结果打印、跨精度对比表、多 GPU 对比表、三路对比表、GPU vs GPU 加速比 |
| `config.py` | Precision/TestTarget/DurationMode 枚举，CaseResult/SystemInfo/RunnerConfig 数据类，精度排序 key |
| `benchmark/runner.py` | 接收 RunnerConfig，按 targets 顺序调度 CPU + 多 GPU 后端，CaseResult 收集，进度回调 |
| `benchmark/cpu_backend.py` | CPU 执行封装，numpy 用例执行 + 计时 |
| `benchmark/gpu_backend.py` | GpuBackend 抽象基类（to_device/from_device/matmul/saxpy/sum/softmax），工厂方法 |
| `benchmark/cuda_kernels.py` | CudaBackend：cupy 原生操作 + ElementwiseKernel/ReductionKernel 双层回退 |
| `benchmark/opencl_kernels.py` | OpenCLBackend：从 .cl 文件加载 kernel，float/double 独立编译，grid-stride 大数组，矩形 matmul |
| `cases/base.py` | BenchmarkCase 抽象基类（get_flops/get_size/run_cpu/run_gpu） |
| `cases/utils.py` | 共享 get_dtype() / create_array()（支持浮点/整数） |
| `cases/*.py` | 4 种算子具体实现 |
| `report/generator.py` | 组装 Markdown 报告：系统信息、结果表、跨精度对比、多 GPU 对比、三路对比、GPU vs GPU 加速比 |
| `utils/gpu_detect.py` | 多 GPU 探测（cupy getDeviceCount → pyopencl 枚举 → 去重），GpuInfo 数据类 |
| `utils/hardware_info.py` | 按 GPU 计算能力过滤精度（BF16 需 CC≥8.0，FP16 需 CC≥5.3） |
| `utils/power_monitor.py` | 后台线程 nvidia-smi 采样，PowerMonitorContext 上下文管理器 |
| `utils/timer.py` | Timer 类 + time_it 函数 |

### 3.3 执行流程

```
main()
  ├─ argparse.parse_args()
  ├─ detect_all_gpus() → List[GpuInfo]
  ├─ get_supported_precisions() → 硬件可用精度
  ├─ build_config()
  │   ├─ 若 CLI 全部参数 → 直接组装 RunnerConfig
  │   ├─ 若部分参数 → 补齐交互
  │   └─ 若无参数 → 纯交互模式（含 GPU 选择步骤）
  ├─ 创建 1~n 个 GPU backend
  ├─ FP64 兼容性检查（不支持则自动移除）
  ├─ Runner(config, backends)
  └─ runner.run(progress_callback) → List[CaseResult]
      ├─ CPU: 每个 case × precision
      └─ GPU: 每个 backend × case × precision
          └─ 仅 NVIDIA 采功耗
  └─ ReportGenerator.generate() → reports/YYYYMMDD_HHMMSS.md
```

## 4. CLI 参数设计

| 参数 | 简写 | 可选值 | 说明 |
|------|------|--------|------|
| `--target` | `-t` | `cpu`, `gpu`, `both`, `gpu-gpu` | 测试目标 |
| `--cases` | `-c` | `matmul,saxpy,reduction,flashattention,all` | 逗号分隔 |
| `--precision` | `-p` | `fp64,fp32,fp16,bf16,int64,int32,int16,int8,all` | 逗号分隔 |
| `--mode` | `-m` | `quick`, `normal` | 测试时长 |
| `--gpus` | `-g` | GPU 索引逗号分隔或 `all` | 选择 GPU |
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

### MatMul（支持矩形矩阵 M×K @ K×N）
```c
__kernel void matmul_naive(__global const float *A, __global const float *B,
                           __global float *C, int K, int N) {
    int row = get_global_id(0), col = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
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

### Softmax（row-wise, 用于 FlashAttention）
```c
__kernel void softmax_row(__global float *input, __global float *output, int N) {
    int row = get_global_id(0), col = get_global_id(1);
    float max_val = -1e30f;
    for (int j = 0; j < N; j++) max_val = fmax(max_val, input[row * N + j]);
    float sum = 0.0f;
    for (int j = 0; j < N; j++) sum += exp(input[row * N + j] - max_val);
    output[row * N + col] = exp(input[row * N + col] - max_val) / sum;
}
```

所有 kernel 均有 float / double 独立文件和编译单元，double 版本使用 `#pragma OPENCL EXTENSION cl_khr_fp64 : enable`。

## 6. 依赖

```
numpy>=1.24
py-cpuinfo
cupy-cuda12x
nvidia-cublas-cu12
nvidia-cuda-runtime-cu12
nvidia-cuda-nvrtc-cu12
nvidia-curand-cu12
pyopencl
rich>=13.0
```

## 7. 错误处理

| 场景 | 行为 |
|------|------|
| 无 GPU 但选了 GPU 测试 | 提示并退出 |
| OpenCL 设备不支持 FP64 | 自动从精度列表移除 FP64，继续执行 |
| GPU 用例执行异常 | 捕获异常，输出 `SKIP: <错误信息>`，继续下一用例 |
| CUDA 后端缺 DLL | 非 FP32 操作原生 API 失败后自动回退 ElementwiseKernel/ReductionKernel |
| 未知 CLI 参数 | argparse 报错退出 |

## 8. 扩展性设计

添加新测试用例：
1. 在 `cases/` 下新建文件，继承 `BenchmarkCase`
2. 实现 `get_flops()`, `get_size()`, `run_cpu()`, `run_gpu()`
3. 在 `benchmark/runner.py` 的 `_get_cases()` 注册
4. 在 `cli.py` 的 `ALL_CASES` 和交互菜单注册

添加新 GPU 后端：
1. 在 `benchmark/` 下新建后端文件，继承 `GpuBackend`
2. 在 `utils/gpu_detect.py` 添加探测逻辑
3. 在 `benchmark/gpu_backend.py` 的 `create_backend()` 注册

---

**版本**：1.1  
**日期**：2026-06-13  
**作者**：Sunnytao
