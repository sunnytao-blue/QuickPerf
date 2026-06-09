# QuickPerf

CPU/GPU 性能测试工具 — 基于 Python CLI，支持跨后端（CUDA / OpenCL）的自动 GPU 探测与基准测试。

## 功能特性

- 交互式 CLI + 命令行参数双模式
- 自动探测 GPU 类型（NVIDIA → CUDA，Intel/AMD → OpenCL）
- 3 种测试算子：MatMul、SAXPY、Reduction
- 4 种精度：FP64 / FP32 / FP16 / BF16
- Quick / Normal 两种时长模式
- CPU + GPU 对比模式，输出加速比
- 实时进度条
- Markdown 报告自动生成（`reports/YYYYMMDD_HHMMSS.md`）

## 环境要求

```
Python >= 3.9
```

### CPU 测试（必选）
```powershell
pip install numpy py-cpuinfo rich
```

### NVIDIA GPU 测试
```powershell
# 基础（cupy + CUDA 运行时库）
pip install cupy-cuda12x nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12
```

> 如果你已安装 CUDA Toolkit 12.x 并设置了 `CUDA_PATH`，只需 `pip install cupy-cuda12x` 即可。

### Intel / AMD GPU 测试
```powershell
pip install pyopencl
```

## 使用方法

### 方式一：命令行参数（最快）

```powershell
# CPU 全部用例，FP32，快速模式
python main.py -t cpu -c all -p fp32 -m quick

# GPU 全部用例，FP32 + FP16，快速模式
python main.py -t gpu -c all -p fp32,fp16 -m quick

# CPU + GPU 对比，全部用例，FP32，快速模式
python main.py -t both -c all -p fp32 -m quick

# 指定用例组合
python main.py -t both -c matmul,reduction -p fp32,fp64 -m normal

# 查看帮助和可用选项
python main.py --list
```

### 方式二：部分参数 + 交互补齐

```powershell
# 指定目标和用例，精度和时长交互选择
python main.py -t cpu -c all

# 指定目标和精度，用例和时长交互选择
python main.py -t gpu -p fp32,fp16
```

### 方式三：纯交互模式

```powershell
python main.py
```

按提示依次选择：测试目标 → 测试用例 → 测试精度 → 测试时长。

交互菜单中选 `4. All` 可一键选择全部用例。

### 参数一览

| 参数 | 简写 | 可选值 | 说明 |
|------|------|--------|------|
| `--target` | `-t` | `cpu`, `gpu`, `both` | 测试目标 |
| `--cases` | `-c` | `matmul`, `saxpy`, `reduction`, `all` | 测试用例（逗号分隔） |
| `--precision` | `-p` | `fp64`, `fp32`, `fp16`, `bf16` | 精度（逗号分隔） |
| `--mode` | `-m` | `quick`, `normal` | Quick ~5-10s / Normal ~30-60s |
| `--list` | — | — | 打印选项并退出 |

## 测试用例

| 用例 | 公式 | 浮点运算量 | 类型 | 问题规模 (Quick/Normal) |
|------|------|-----------|------|------------------------|
| **MatMul** | C = A × B | 2 × N³ | 计算密集型 | CPU: 1024/2048, GPU: 512/1024 |
| **SAXPY** | y = a·x + y | 2 × N | 访存密集型 | CPU: 10M/50M, GPU: 100M/500M |
| **Reduction** | Σ arr | N - 1 | 归约操作 | CPU: 10M/50M, GPU: 100M/500M |

## 精度说明

| 精度 | CPU 实现 | GPU (CUDA) | GPU (OpenCL) | 备注 |
|------|----------|-----------|-------------|------|
| FP64 | float64 | float64 | double | 需设备支持 cl_khr_fp64，不持支时自动跳过 |
| FP32 | float32 | float32 | float | — |
| FP16 | float32 模拟 | float16 | float32 模拟 | 报告标注 "(float32 simulate)" |
| BF16 | float32 模拟 | bfloat16 | float32 模拟 | 报告标注 "(float32 simulate)" |

## 参考测试数据

以下数据在真实硬件上测得，供对比参考。

### 测试环境 A：NVIDIA CMP 40HX（台式机）

| 项目 | 配置 |
|------|------|
| CPU | Intel Core i7-1165G7 |
| GPU | NVIDIA CMP 40HX (TU102, CUDA 12.9, 8GB) |
| 驱动 | 576.88 |
| OS | Windows 11 |

**Quick 模式结果**（`python main.py -t both -c all -p fp32 -m quick`）：

| 用例 | CPU 耗时 | CPU TFLOPS | GPU 耗时 | GPU TFLOPS | 加速比 |
|------|---------|-----------|---------|-----------|--------|
| MatMul (512²) | 5.2ms | 0.414 | 0.1ms | 2.871 | **6.9x** |
| SAXPY (100M) | 16.9ms | 0.001 | 5.0ms | 0.040 | **33.9x** |
| Reduction (100M) | 3.8ms | 0.003 | 0.9ms | 0.106 | **40.0x** |

**Normal 模式结果**（`python main.py -t both -c all -p fp32 -m normal`）：

| 用例 | CPU 耗时 | CPU TFLOPS | GPU 耗时 | GPU TFLOPS | 加速比 |
|------|---------|-----------|---------|-----------|--------|
| MatMul (2048² / 1024²) | 39.1ms | 0.440 | 0.4ms | 10.23 | **23.3x** |
| SAXPY (50M/500M) | 82.5ms | 0.001 | 25.3ms | 0.040 | **32.5x** |
| Reduction (50M/500M) | 19.1ms | 0.003 | 4.5ms | 0.111 | **43.3x** |

### 测试环境 B：Intel Iris Xe（笔记本）

| 项目 | 配置 |
|------|------|
| CPU | Intel Core i7-1165G7 |
| GPU | Intel Iris Xe Graphics (OpenCL) |
| OS | Windows 11 |

**Quick 模式结果**：

| 用例 | CPU 耗时 | CPU TFLOPS | GPU 耗时 | GPU TFLOPS | 加速比 |
|------|---------|-----------|---------|-----------|--------|
| MatMul (1024²/512²) | 5.8ms | 0.372 | 19.1ms | 0.014 | 0.04x |
| SAXPY (10M/100M) | 17.5ms | 0.001 | 23.1ms | 0.009 | 7.7x |
| Reduction (10M/100M) | 3.9ms | 0.003 | 8.3ms | 0.012 | 4.8x |

> Iris Xe 的 MatMul 慢于 CPU，因为 OpenCL naive kernel 未做 shared memory 优化，而 CPU 端 numpy 使用了高度优化的 BLAS。SAXPY/Reduction 受益于 GPU 更高的内存带宽。

### 性能解读

- **MatMul**：计算密集。CPU 用 numpy BLAS（单核 ~0.4 TFLOPS），GPU 用 cuBLAS（CMP 40HX ~10 TFLOPS Normal）。矩阵越大 GPU 优势越明显。
- **SAXPY**：内存密集，3 次访存 / 2 FLOP。GPU 带宽远高于 CPU，加速 30-40x。
- **Reduction**：纯读取带宽。GPU 可达理论带宽 90%，加速 40x+。

## 模块结构

```
quickperf/
├── main.py                 # CLI 入口 + 流程调度
├── config.py               # 枚举 / 数据类 / 常量
├── requirements.txt
├── benchmark/
│   ├── runner.py           # 统一执行引擎
│   ├── cpu_backend.py      # CPU 后端
│   ├── gpu_backend.py      # GPU 门面（工厂，自动分发 CUDA/OpenCL）
│   ├── cuda_kernels.py     # NVIDIA CUDA 后端 (cupy)
│   └── opencl_kernels.py   # OpenCL 后端 (pyopencl + raw kernel)
├── cases/
│   ├── base.py             # 用例基类
│   ├── matmul.py
│   ├── saxpy.py
│   └── reduction.py
├── report/
│   └── generator.py        # Markdown 报告生成
└── utils/
    ├── gpu_detect.py       # GPU 自动探测
    └── timer.py            # 高精度计时
```

添加新用例只需继承 `cases/base.py` 的 `BenchmarkCase` 类并实现 4 个抽象方法。

## 输出

测试完成后在 `reports/` 目录生成 Markdown 报告，文件名 `YYYYMMDD_HHMMSS.md`。

报告包含系统信息、测试配置、各用例耗时及 TFLOPS、CPU vs GPU 对比及加速比。

---

**版本**：0.1  
**日期**：2026-06-09  
**作者**：Sunnytao
