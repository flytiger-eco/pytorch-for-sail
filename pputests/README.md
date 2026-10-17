# PPU 算子精度对比测试框架

配置驱动、可插拔的算子精度对比测试框架。对指定算子，在 **FP32 / BF16（预留 FP16）** 下，
与 PyTorch **CPU fp64 高精度参考实现**逐元素对比，输出量化误差指标与 PASS/FAIL 判定，
覆盖 **前向 + 反向（梯度）** 两条路径。

> 新增算子只需实现一个 `OpSpec`（输入构造 + 参考函数 + 阈值），**无需改动框架代码**。

---

## 1. 环境准备

框架运行在 PyTorch-for-SAIL 的 PPU 容器内。真武 PPU 通过 CUDA 设备层暴露，因此
**PPU 设备串即 `cuda`**（可用 `--device` 覆写）。

```bash
# 进入容器
docker run -it pytorch-for-sail:ubuntu-ppu

# 自检：确认 PPU 可见
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
ppu-smi || echo "warn: 未分配 PPU 设备"

# 安装框架依赖（torch 由容器内的 wheel 提供）
pip install -r pputests/requirements.txt
```

无 PPU 的开发机上也可运行：加 `--allow-cpu-fallback`，框架会把 `ppufn` 放到 CPU 执行，
变成 **golden-vs-golden 自检**，用于验证框架本身与阈值逻辑（不代表 PPU 精度）。

---

## 2. 运行方式

所有命令均在仓库根目录下、以 `pputests/` 为工作根执行。

### 2.1 CLI（推荐，聚合报告最完整）

```bash
cd pputests

# 一键全量冒烟（DoD #1）
python run.py --all --level smoke

# 指定算子 + dtype + 级别 + 种子
python run.py --ops relu,softmax --dtypes bf16,fp32 --level full --seed 0

# 按类别，仅反向
python run.py --category activation --backward-only

# 全量 + 指定报告格式与输出目录
python run.py --all --report markdown,junit --out reports/

# full 模式并行（DoD #9），8 线程
python run.py --all --level full -n 8

# FP16 opt-in（DoD #6）
python run.py --ops relu,sigmoid,tanh,add,mul --dtypes fp16 --level smoke

# 列出已注册算子
python run.py --list

# 无 PPU 自检
python run.py --all --level smoke --allow-cpu-fallback
```

`--ops` 接受逗号分隔的**具体、完整算子名**，不接受需求编号或模糊前缀。例如 ReLU 传
`--ops relu`，Softmax 传 `--ops softmax`；后续新增 Sum 后直接传 `--ops sum`。
可用 `python run.py --list` 查看当前名称。

退出码：出现任一 **FAIL** 返回 `1`（SKIP 不算失败），可直接接入 CI。

> CI 集成脚本：`.ci/ppu/op_accuracy_test.sh`（`LEVEL=smoke|full`、`DTYPES`、`JOBS`、`OPS` 环境变量控制），
> 与 `.ci/ppu/accuracy_test.sh` 同风格：装 wheel → 环境自检 → 跑框架 → 产出 `pputests/reports/`。
> CI 环境必须至少存在一个可用 PPU；设备不可用会直接失败，避免全量 SKIP 后误报成功。

### 2.2 pytest

```bash
# 单个/一组用例调试
pytest pputests/tests/test_ops.py -k "softmax and bf16" -v

# 通过 --ppu-* 选项控制矩阵
pytest pputests/tests/test_ops.py --ppu-ops softmax --ppu-level full --ppu-dtypes bf16,fp32

# full 模式并行（pytest-xdist）；每个 worker 写到 reports/workers/<id>/
pytest pputests/tests/test_ops.py --ppu-level full -n 8

# 无 PPU 自检
pytest pputests/tests/test_ops.py --ppu-allow-cpu-fallback
```

pytest 常用选项：`--ppu-config/--ppu-level/--ppu-dtypes/--ppu-ops/--ppu-category/`
`--ppu-direction/--ppu-backward-only/--ppu-device/--ppu-seed/--ppu-out/--ppu-report/`
`--ppu-allow-cpu-fallback/--ppu-no-dump`。

### 2.3 级别与规模

| level | 内容 | 用途 |
|---|---|---|
| `smoke` | 每算子每 dtype/方向最多 5 个基础组合，不含 large；反向再覆盖各 `grad_pattern` | CI 快速门禁（目标 < 2 min） |
| `full` | 全矩阵 Shape × Pattern × dtype × fwd/bwd | 夜间回归 |

---

## 3. 目录结构

```
pputests/
├── README.md                本文档
├── requirements.txt
├── pytest.ini               独立运行时的 pytest 配置
├── conftest.py              pytest 选项 / fixture / 报告钩子 / 动态参数化
├── run.py                   CLI 入口（非 pytest 也可跑）
├── core/
│   ├── tolerance.py         §4.3 阈值表 + dtype 解析 + 覆写逻辑
│   ├── inputgen.py          §5 Shape / Pattern 生成器
│   ├── opspec.py            OpSpec 数据类 + 注册表 OP_REGISTRY + @register
│   ├── metrics.py           §4.2 全部指标 + NaN/Inf 硬性判据
│   ├── reference.py         §4.1 golden（CPU fp64→cast）+ 次基准交叉校验
│   ├── runner.py            配置加载 / 设备解析 / 用例枚举 / 执行引擎
│   ├── report.py            CSV / JSON / JUnit / Markdown + 环境快照
│   └── dump.py              失败现场保存 + 复现脚本生成
├── ops/                     算子定义（__init__ 自动发现并注册本目录所有模块）
│   ├── activation.py        ReLU, ReLU6, Sigmoid, Softplus, Tanh
│   ├── reduction.py         Softmax
│   ├── unarymath.py         Exp
│   ├── binaryeltwise.py     Add, Sub, Mul
│   └── demo_abs.py          demo 算子（可扩展性验收，DoD #5）
├── configs/
│   ├── default.yaml         全局默认
│   ├── smoke.yaml
│   └── full.yaml
├── tests/
│   └── test_ops.py          参数化驱动
└── reports/                 产物（已 gitignore）
```

---

## 4. 如何新增算子（唯一改动点）

在 `ops/` 下**任意模块**（新建或已有）里写一个 `@register` 工厂，返回 `OpSpec` 即可。
`ops/__init__.py` 会自动发现并导入本目录下所有 `*.py`，因此**新增算子 = 0 行框架改动**。

```python
# ops/activation.py（或新建 ops/myop.py）
@register
def relu6():
    return OpSpec(
        name="relu6", category="activation",
        dtypes=[BF16, FP16, FP32],
        shapes=preset_shapes("activation"),
        patterns=[NORMAL, ZEROS, ONES, UNIFORM10,
                  NEARBOUNDARY(values=[0.0, 6.0], ulp_span=2),
                  EXTREME_VALUES(0.01)],
        params=[{}],
        reffn=lambda x: F.relu6(x),                 # golden：CPU fp64
        ppufn=lambda x: F.relu6(x),                 # PPU：在 PPU 设备上执行
        has_backward=BackwardMode.REQUIRED,
        notes="x=0/6 拐点梯度取 torch hardtanh 语义",
    )
```

`OpSpec` 关键字段：

| 字段 | 说明 |
|---|---|
| `name` / `category` | 通用小写算子名（CLI 精确匹配，如 `relu6`、`softmax`、`sum`）；类别过滤 |
| `dtypes` | 该算子**能力**支持的 dtype；实际跑哪些由 `--dtypes`/config 决定（FP16 opt-in） |
| `shapes` | 复用 `preset_shapes(kind)` 或自定义 `ShapeCase` 列表 |
| `patterns` | 复用 `inputgen` 预设常量或 `NEARBOUNDARY()/MASKFILL()/EXTREME_VALUES()` 等工厂 |
| `params` | 算子属性组合列表，如 `{"dim": -1}`、`{"beta":1.0,"threshold":20}` |
| `reffn` / `ppufn` | 参考实现（吃 CPU fp64）/ PPU 调用封装（吃 PPU 设备张量） |
| `has_backward` | `REQUIRED` / `OPTIONAL` / `NONE` |
| `tolerance` | `ToleranceOverride(...)`，见下节 |
| `n_inputs` | 一元=1，二元=2 |
| `grad_patterns` | 反向 `grad_output` 使用的 pattern；每个 pattern 独立生成用例（默认 `[NORMAL]`） |
| `case_filter` | 可选的 `(shape, pattern, params) -> bool` 谓词，用于过滤无效笛卡尔积 |
| `notes` | 已知偏差 / 阈值放宽理由（会出现在报告中） |

**二元算子的 broadcast**：用保留参数键描述第二个操作数（调用前会被自动剥离，不传给算子）：

```python
params=[
    {},                          # 同 shape
    {"_shape_b": ()},            # 标量
    {"_shape_b": (1,)},          # 1-元素广播
    {"_shape_b": (1, 4, 1, 1)},  # per-channel NCHW
]
```
其它保留键：`_view_b`（第二操作数布局视图）、`_pattern_b`（第二操作数分布）、
`_shapes`/`_views`（完全自定义）。`CANCELLATION` pattern 下第二操作数自动取 `a + δ`，
且要求两个操作数的有效 shape 相同。当前二元逐元素算子通过 `case_filter` 自动剔除不可广播组合。

---

## 5. 如何调阈值

默认阈值表（§4.3）在 [`core/tolerance.py`](core/tolerance.py) 的 `DEFAULT_TOLERANCE`：

| dtype | max_abs_err | max_rel_err | cosine_sim | mismatch_rate |
|---|---|---|---|---|
| FP32 | ≤ 1e-5（归约类 ≤ 1e-4） | ≤ 1e-5 | ≥ 1−1e-6 | ≤ 0.01% |
| BF16 | ≤ 8e-3 | ≤ 1.6e-2 | ≥ 1−1e-3 | ≤ 1% |
| FP16 | ≤ 1e-3 | ≤ 2e-3 | ≥ 1−1e-4 | ≤ 0.1% |

**硬性规则**（不受阈值放宽影响，违反直接 FAIL）：

- NaN / Inf 的**位置与符号**必须与参考完全一致；
- 输出 **shape / dtype / stride 语义**必须一致（`stride` 需 `OpSpec.check_stride=True` 开启）；
- 反向梯度的 NaN 检查同前向。

**按算子覆写**：在 `OpSpec.tolerance` 里给 `ToleranceOverride`，并**必须**填 `reason`：

```python
tolerance=ToleranceOverride(
    reduction=True,          # 归约类：FP32 abs 放宽到 1e-4
    max_abs_err=1e-2,        # 直接指定某项（None = 保持默认）
    reason="Softmax 在 BF16 下归约累加误差较大",
)
```

> 说明：`exp`、`add/sub/mul` 等输出量级可远超 1.0 的算子，其**绝对**误差会随量级自然增大
> （例如 BF16 在 ~2000 处 1 ULP ≈ 16）。这些算子已在各自 `OpSpec` 中显式放宽 `max_abs_err`
> 并附理由，主判据落到**相对误差 + cosine + NaN/Inf 一致性**上。详见各 `ops/*.py` 的 `notes`。

---

## 6. 如何看报告

产物默认写入 `pputests/reports/`（由 `--out` / config `out_dir` 控制）：

| 文件 | 内容 |
|---|---|
| `summary.md` | 人读汇总：**算子 × dtype × 方向**透视表（用例数 / PASS / FAIL / SKIP + 各指标最值）、失败明细（含 Top-K 离群位置）、算子备注与放宽理由 |
| `summary.csv` | 每用例一行的扁平表，便于入库/趋势分析 |
| `results.json` | 完整结构化结果（含 `sub` 逐张量对比、`tolerance`、环境快照） |
| `junit.xml` | 供 CI 消费；xdist 并行时在 `reports/workers/<id>/junit.xml` |
| `dumps/<case_id>/` | 失败现场：`case.pt`（inputs / ppu_out / golden_out / diff / grads）、`meta.json`、`repro.sh` |

**环境快照**（写入 JSON 与 Markdown）：PyTorch 版本、commit hash、`ppu-smi` 摘要、
`PPU_SDK`、时间戳、CUDA 可用性。

指标含义见 §4.2：`max_abs_err`、`max_rel_err`（主判据），`mean_abs_err`、`mse/rmse`、
`cosine_sim`、`mismatch_rate`、可选 `ulp_err_max`（仅 fp32/fp16）。

---

## 7. 失败复现（DoD #7）

任一失败用例都会 dump 现场。凭 dump 产物 + 复现命令行可独立重跑：

```bash
# 1) 直接跑复现脚本
bash reports/dumps/<case_id>/repro.sh

# 2) 或用 meta.json 中的完整 case_id 与派生 seed 精确重跑一个用例
python run.py --ops softmax --dtypes bf16 --level full \
  --case-id '<case_id>' --case-seed <case_seed>

# 3) 载入现场张量离线分析
python3 - <<'PY'
import torch
a = torch.load("reports/dumps/<case_id>/case.pt")
print(a.keys())
print("max|diff| =", a["diff"].abs().max())
PY
```

`case.pt` 内包含：`inputs`、`ppu_out`、`golden_out`（cast 后主基准）、`golden_out_fp64`、
`golden_out_secondary_cpu`（次基准）、`diff`；反向用例另含 `grad_output`、`ppu_grads`、
`golden_grads`。

---

## 8. 精度对比方法（§4.1）

- **主基准（golden）**：PyTorch CPU **fp64** 计算后 cast 回目标 dtype。PPU 输入以目标 dtype
  生成，golden 消费**同一批输入**上采到 fp64（对 bf16/fp16/fp32 无损），保证对比同源。
- **次基准（交叉校验）**：PyTorch CPU **同 dtype** 原生实现，用于区分“PPU 实现问题”与
  “低精度固有误差”。结果记录在 `results.json` 的 `sub` 中（`out_vs_secondary_cpu`），**不参与**门禁。

---

## 9. 里程碑与验收对照

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | 框架骨架 core/ + 注册机制 + metrics + 报告 | ✅ |
| M2 | inputgen 全量 Shape/Pattern + dump/复现 | ✅ |
| M3 | 接入 O20–O29 前向 | ✅ |
| M4 | 反向链路 + 阈值调优 + 已知偏差归档 | ✅ |
| M5 | FP16 扩展 + CI 集成 + README | ✅ |

验收项（DoD）落点：
1. `python run.py --all --level smoke` 一键跑通 → §2.1
2. O20–O29 全部注册、前向产出报告 → `ops/*.py`
3. 必测反向算子梯度对比 → `has_backward=REQUIRED`
4. Shape 6 类 / Pattern 11 类均在 `inputgen.py` 实现且被使用 → §4/§5
5. demo 算子（`ops/demo_abs.py`）≤ 20 行、0 框架改动 → §4
6. `--dtypes fp16` 可启用、阈值表含 FP16 列 → §2.1/§5
7. 失败用例凭 dump + 复现命令行独立重跑 → §7
8. README 覆盖环境/运行/新增/阈值/报告 → 本文
9. smoke CI < 5 min；full 支持并行 `-n` → §2.1/§2.2
