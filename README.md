# QuickPerf

CPU/GPU 性能测试工具 — 基于 Python CLI，支持跨后端（CUDA / OpenCL），自动 GPU 探测，多 GPU 选择与对比，覆盖计算密集型和访存密集型算子。

## 功能特性

- 交互式 CLI + 命令行参数双模式
- 自动探测所有 GPU 类型（NVIDIA → CUDA，Intel/AMD → OpenCL），支持多 GPU 选择
- 4 种测试算子：MatMul、SAXPY、Reduction、FlashAttention
- 8 种精度：FP64 / FP32 / FP16 / BF16 / INT64 / INT32 / INT16 / INT8
- 按硬件计算能力自动过滤不支持的精度
- Quick / Normal 两种时长模式
- CPU + GPU 对比、多 GPU 对比、GPU vs GPU 对比模式
- 跨精度对比表、三路对比表（CPU + iGPU + dGPU）
- GPU 功耗采集（NVIDIA nvidia-smi）
- 实时进度条
- Markdown 报告自动生成

## 使用前准备

### 1. 检查 Python 版本

```powershell
python --version
```

需要 **Python >= 3.9**。

### 2. 安装依赖库

```powershell
cd QuickPerf

# 一键安装全部依赖（如果全部都需要）
pip install -r requirements.txt

# 或根据你的需要选择性安装：
# 仅 CPU 测试
pip install numpy py-cpuinfo rich

# NVIDIA GPU 测试
pip install cupy-cuda12x nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-curand-cu12

# Intel / AMD GPU 测试
pip install pyopencl
```

### 3. 检查 GPU 驱动和库（如使用 GPU 测试）

```powershell
# 检查是否有 GPU 可用
python -c "from utils.gpu_detect import detect_all_gpus; gpus=detect_all_gpus(); print(f'检测到 {len(gpus)} 块 GPU') if gpus else print('未检测到 GPU')"
```

**NVIDIA GPU 用户**：
- 确保已安装 NVIDIA 显卡驱动（`nvidia-smi` 可执行）
- 如果没有安装完整的 CUDA Toolkit，需额外安装 pip 包：`nvidia-cublas-cu12`、`nvidia-cuda-runtime-cu12`、`nvidia-cuda-nvrtc-cu12`、`nvidia-curand-cu12`（已包含在 `requirements.txt` 中）
- 如果已安装 CUDA Toolkit 12.x 并设置了 `CUDA_PATH` 环境变量，只需 `pip install cupy-cuda12x`

**Intel / AMD GPU 用户**：
- Windows/Linux 下需要安装对应驱动（通常已随系统安装）
- 无需额外配置，`pyopencl` 会自动探测设备

### 4. 验证安装

```powershell
# 验证程序可正常启动
python main.py --list
```

启动后应该看到类似输出：
```
检测到 2 块 GPU:
  [0] NVIDIA CMP 40HX (CUDA)
  [1] Intel(R) Iris(R) Xe Graphics (OpenCL)
支持精度: FP64, FP32, FP16, INT64, INT32, INT16, INT8
```

## 使用方法

### 方式一：命令行参数（推荐）

```powershell
# CPU 全部用例，FP32，快速模式
python main.py -t cpu -c all -p fp32 -m quick

# GPU 全部用例，FP32 + FP16，指定 GPU 0
python main.py -t gpu -g 0 -c all -p fp32,fp16 -m quick

# 测试所有 GPU
python main.py -t gpu -g all -c all -p fp32 -m quick

# CPU + 所有 GPU 对比，全部用例
python main.py -t both -g all -c all -p fp32 -m quick

# GPU vs GPU 对比（不测 CPU）
python main.py -t gpu-gpu -g all -c all -p fp32 -m quick

# 部分参数 + 交互补齐
python main.py -t cpu -c all

# 查看帮助
python main.py --list
```

### 方式二：纯交互模式

```powershell
python main.py
```

### 参数一览

| 参数 | 简写 | 可选值 | 说明 |
|------|------|--------|------|
| `--target` | `-t` | `cpu`, `gpu`, `both`, `gpu-gpu` | 测试目标 |
| `--cases` | `-c` | `matmul,saxpy,reduction,flashattention,all` | 测试用例（逗号分隔） |
| `--precision` | `-p` | `fp64,fp32,fp16,bf16,int64,int32,int16,int8,all` | 精度（逗号分隔） |
| `--mode` | `-m` | `quick`, `normal` | Quick ~5-10s / Normal ~30-60s |
| `--gpus` | `-g` | `0`, `1`, `0,1`, `all` | 选择 GPU |
| `--list` | — | — | 打印选项并退出 |

## 测试用例详解

### 1. MatMul — 矩阵乘法

```
C = A × B
浮点运算: 2 × N³
类型: 计算密集型
测什么: GPU 浮点计算峰值
场景: 深度学习全连接层、卷积 GEMM 实现
```

| 模式 | CPU 规模 | GPU 规模 |
|------|---------|---------|
| Quick | 1024×1024 | 512×512 |
| Normal | 2048×2048 | 1024×1024 |

### 2. SAXPY — 向量乘加

```
y = α·x + y
浮点运算: 2 × N
类型: 访存密集型 (0.33 FLOP/Byte)
测什么: GPU 内存带宽
场景: Layer Normalization、优化器权重更新
```

| 模式 | CPU 规模 | GPU 规模 |
|------|---------|---------|
| Quick | 10M 元素 | 100M 元素 |
| Normal | 50M 元素 | 200M 元素 |

### 3. Reduction — 归约求和

```
result = Σ arr[i]
浮点运算: N - 1
类型: 访存密集型 (0.25 FLOP/Byte)，树形归约
测什么: GPU 内存带宽极限
场景: Softmax 分母、Loss 计算、Normalization
```

| 模式 | CPU 规模 | GPU 规模 |
|------|---------|---------|
| Quick | 10M 元素 | 100M 元素 |
| Normal | 50M 元素 | 200M 元素 |

### 4. FlashAttention — 注意力机制

```
S = Q · K^T / √d
P = softmax(S)
O = P · V
浮点运算: 4 × N² × d + 5 × N²  (d=64)
类型: 计算 + 访存混合
测什么: GPU 综合性能，模拟 Transformer 核心计算
场景: LLM 推理/训练，GPT/BERT/Llama 等架构的核心算子
```

| 模式 | CPU 规模 | GPU 规模 |
|------|---------|---------|
| Quick | N=256, d=64 | N=512, d=64 |
| Normal | N=512, d=64 | N=1024, d=64 |

### 算子对比

| 算子 | FLOP/Byte | 瓶颈 | 代表场景 |
|------|-----------|------|---------|
| MatMul | 高 (~N/3) | 计算能力 | 矩阵运算、GEMM |
| SAXPY | 极低 (0.33) | 内存带宽 | Layer Norm、优化器 |
| Reduction | 极低 (0.25) | 内存带宽 | Softmax、Loss |
| FlashAttention | 中 | 计算 + 带宽 | Transformer/Llama |

## 精度说明

| 精度 | CPU 实现 | GPU (CUDA) | GPU (OpenCL) | 备注 |
|------|----------|-----------|-------------|------|
| FP64 | float64 | float64 | double | 需设备支持 cl_khr_fp64 |
| FP32 | float32 | float32 | float | — |
| FP16 | float32 模拟 | float16 | float32 模拟 | 报告标注 "(float32 simulate)" |
| BF16 | float32 模拟 | bfloat16 | float32 模拟 | 报告标注, 需 CC≥8.0 |
| INT64 | int64 | int64 | int64 | — |
| INT32 | int32 | int32 | int32 | — |
| INT16 | int16 | int16 | int16 | — |
| INT8 | int8 | int8 | int8 | — |

## 参考测试数据

以下数据在真实硬件上测得，供对比参考。

### 测试环境 A：NVIDIA CMP 40HX + Intel Iris Xe（台式机）

| 项目 | 配置 |
|------|------|
| CPU | Intel Core i7-1165G7 |
| GPU 0 | NVIDIA CMP 40HX (TU102, CUDA 12.9, 8GB) |
| GPU 1 | Intel Iris Xe Graphics (OpenCL) |
| OS | Windows 11 |

**Quick 模式**（`python main.py -t both -g all -c all -p fp32 -m quick`）：

| 算子 | CPU TFLOPS | NVIDIA TFLOPS | Iris Xe TFLOPS | NVIDIA/CPU |
|------|-----------|-------------|---------------|-----------|
| MatMul | 0.327 | 2.239 | 0.013 | 6.8x |
| SAXPY | 0.001 | 0.067 | 0.008 | 63.5x |
| Reduction | 0.002 | 0.103 | 0.012 | 45.4x |
| FlashAttention | 0.026 | 0.072 | 0.002 | 2.8x |

**Normal 模式**（`python main.py -t both -g all -c all -p fp32 -m normal`）：

| 算子 | CPU TFLOPS | NVIDIA TFLOPS | Iris Xe TFLOPS | NVIDIA/CPU |
|------|-----------|-------------|---------------|-----------|
| MatMul | 0.304 | 4.883 | 0.013 | 16.1x |
| SAXPY | 0.001 | 0.007 | 0.008 | 6.1x |
| Reduction | 0.002 | 0.107 | 0.012 | 45.4x |
| FlashAttention | — | — | — | — |

### 性能解读

- **MatMul**：矩阵越大 GPU 优势越明显（Normal 模式 1024² 比 Quick 512² 加速比 2.4x）
- **SAXPY**：访存密集，GPU 带宽远高于 CPU，加速 30-60x
- **Reduction**：纯读取带宽，GPU 可达理论带宽 90%+
- **FlashAttention**：计算+访存混合，NVIDIA 相比 Iris Xe 快 40x+

## 模块结构

```
quickperf/
├── main.py                 # 流程编排入口
├── cli.py                  # argparse + 交互式菜单
├── output.py               # 终端结果对比表输出
├── config.py               # 枚举 / 数据类 / 常量
├── requirements.txt
├── benchmark/
│   ├── runner.py           # 统一执行引擎
│   ├── cpu_backend.py      # CPU 后端
│   ├── gpu_backend.py      # GPU 门面（工厂）
│   ├── cuda_kernels.py     # NVIDIA CUDA 后端 (cupy)
│   ├── opencl_kernels.py   # OpenCL 后端
│   └── opencl/             # OpenCL kernel 源码 (.cl 文件)
├── cases/
│   ├── base.py             # 用例基类
│   ├── utils.py            # 共享工具函数
│   ├── matmul.py
│   ├── saxpy.py
│   ├── reduction.py
│   └── flashattention.py
├── report/
│   └── generator.py        # Markdown 报告生成
└── utils/
    ├── gpu_detect.py       # 多 GPU 自动探测
    ├── hardware_info.py    # 硬件精度能力检测
    ├── power_monitor.py    # 功耗监控
    └── timer.py            # 高精度计时
```

添加新用例只需继承 `cases/base.py` 的 `BenchmarkCase` 类并实现 4 个抽象方法，然后在 `runner.py` 和 `cli.py` 注册。

## 输出

测试完成后在 `reports/` 目录生成 Markdown 报告，文件名 `YYYYMMDD_HHMMSS.md`。

报告包含系统信息（含多 GPU 列表）、测试配置、各用例耗时及 TFLOPS、跨精度对比表、多 GPU 对比表、CPU+多GPU 三路对比表、GPU vs GPU 加速比。

---

**版本**：1.1  
**日期**：2026-06-13  
**作者**：Sunnytao
