# QuickPerf 版本发布记录

## v1.1 (2026-06-13)

### 新增功能

- **FlashAttention 算子**：新增第 4 个测试用例，实现 Q@K^T + softmax + P@V，支持 CPU/CUDA/OpenCL
- **INT 精度扩展**：新增 INT64 / INT32 / INT16 / INT8 四种整数精度，CPU/GPU 双端支持
- **多 GPU 选择**：启动时自动列出所有 GPU，支持交互式和 CLI（`-g 0,1` / `-g all`）选择
- **GPU vs GPU 对比模式**：新增 `TestTarget.GPU_VS_GPU`，交互菜单选项 3，CLI `-t gpu-gpu`
- **硬件能力检测**：按 GPU 计算能力自动过滤不支持的精度（如 BF16 需 CC ≥ 8.0）
- **跨精度对比表**：单目标多精度时自动输出 TFLOPS 矩阵；多精度 CPU+GPU 输出加速比矩阵
- **三路对比表**：CPU + iGPU + dGPU 同表对比 TFLOPS 和加速比
- **GPU 功耗采集**：仅 NVIDIA GPU（nvidia-smi），iGPU/OpenCL 跳过

### 架构优化

- **提取重复代码**：`cases/utils.py`（get_dtype / create_array）
- **拆分 OpenCL kernel**：独立 `.cl` 文件（saxpy / matmul / reduce / softmax，float + double 独立文件）
- **拆分 main.py**：`cli.py`（argparse + 交互菜单）+ `output.py`（结果对比表输出），main.py 从 570 行精简至 80 行
- **OpenCL matmul 支持矩形矩阵**：适配 FlashAttention 的 M×K @ K×N 计算

### Bug 修复

- `cudaErrorAlreadyMapped`：SAXPY/Reduction 双层回退（原生 API → ElementwiseKernel/ReductionKernel）
- OpenCL double kernel 编译失败：float/double 文件独立编译，按 FP64 能力分别加载
- BF16 不支持仍显示在菜单：hardware_info.py 按 CC 过滤
- 精度列排序混乱：固定 FP64→FP32→FP16→BF16→INT64→INT32→INT16→INT8
- 对比表列对齐错位：自适应列宽
- GPU 报告缺少 GPU 名称标注

---

## v0.1 (2026-06-09)

### 首发功能

- **交互式 CLI + 命令行参数双模式**：支持 `-t` / `-c` / `-p` / `-m` 参数
- **自动 GPU 探测**：cupy（NVIDIA CUDA）/ pyopencl（Intel/AMD OpenCL），自动选择后端
- **3 种测试算子**：MatMul（计算密集）、SAXPY（访存密集）、Reduction（归约）
- **4 种浮点精度**：FP64 / FP32 / FP16 / BF16
- **Quick / Normal 时长模式**：Quick ~5-10s，Normal ~30-60s
- **CPU + GPU 对比模式**：输出加速比
- **实时进度条**
- **Markdown 报告**：自动生成 `reports/YYYYMMDD_HHMMSS.md`
- **OpenCL raw kernel 实现**：SAXPY / MatMul / Reduction，含 grid-stride 大数组支持

---

**作者**：Sunnytao
