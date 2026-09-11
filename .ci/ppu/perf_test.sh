#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA 性能（perf）测试入口。
# 由 .github/workflows/ppu-perf.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 由 PPU pytorch 发布镜像预装，本脚本不再 pip install torch。
#
# 测试范围（为什么不是 inductor-perf-test-nightly-h100.yml 的那套模型套件）：
#   - h100 那条线跑的是 benchmarks/dynamo/{huggingface,timm_models,torchbench}.py，
#     依赖 transformers / timm / torchbench 仓库和 HF hub 上的模型配置。PPU pod 只有
#     内网出口、镜像里也没预装这些包，torchbench 更是要 clone 外部仓库，整套拿不到。
#   - 因此这里只保留其**性能语义**，换成零外部模型依赖的两段：
#       1) inductor 性能单测（test/inductor/test_perf.py 等）：断言 inductor 生成代码的
#          读写字节数 / 融合决策 / kernel benchmark 路径，全部跑在 GPU_TYPE 上，是能在
#          自建集群里稳定复现的性能回归门禁。
#       2) benchmarks/gpt_fast 的 micro benchmark：MLP/LayerNorm/GEMV 这类算子级
#          flops / 带宽利用率，模型权重全部本地随机初始化，不下载任何东西。
#   - 设备覆盖：只跑 CUDA 一条线（PPU 设备以 CUDA 形态可见），不做纯 CPU 对照——这几个
#     文件里有 CPU 侧的同名副本（BenchmarkFusionCpuTest / test_benchmark_cpu_smoke /
#     device 参数化出的 *_cpu 变体），一并用 -k 摘掉。
#   - FP8：真武 PPU 当前不支持 FP8，用 -k 表达式把 fp8/float8/e4m3/e5m2 相关用例整体
#     摘掉（与 smoke_test.sh / accuracy_test.sh 的裁剪保持一致）。当前这几个文件里本来
#     就没有 FP8 用例，这条过滤是防止上游后续加进来时把门禁跑红。
#
# 依赖环境变量：
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PIP_INDEX        - 内部 pip 源（可选；不设则用镜像自带的 pip 默认源）
#   PR_NUMBER        - 仅用于日志溯源（可选）
#   RUN_GPT_FAST     - 是否跑 gpt_fast micro benchmark（默认 1；置 0 可只跑单测）
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[perf] 源码目录: $(pwd)"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的测试进程。
source .ci/ppu/sdk_env.sh

echo "=== 环境自检 (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
# 自检必须在源码树外执行：`python -c` 会把 cwd（此时是源码根）放进 sys.path[0]，
# 未编译的源码目录 torch/ 会遮蔽镜像预装的 torch，而 torch/version.py 是构建产物、
# 源码树里不存在，于是报 ModuleNotFoundError: No module named 'torch.version'。
# 打印 torch.__file__ 以便一眼确认导入的是 site-packages 里的那份。
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

# 测试框架依赖（pytest 及其插件、expecttest、hypothesis）镜像里没预装对版本，必须在这里补：
# run_test.py 的 check_pip_packages() 硬校验 pytest-rerunfailures / pytest-flakefinder /
# pytest-xdist，缺任意一个直接 exit 1。
# 版本一律对齐 .ci/docker/requirements-ci.txt（官方 CI 跑 test case 的同一份约束），
# 安装逻辑（含 pip 源诊断与多源回退）抽到共享脚本，smoke_test.sh / accuracy_test.sh 用同一份。
bash .ci/ppu/install_test_deps.sh

# boto3 故意不装：它只被 tools/stats/upload_metrics.py 用于往官方 S3 上报指标，缺失时
# EMIT_METRICS=False 静默降级（日志里那条 "Unable to import boto3" 只是提示，不影响退出码），
# 自建集群也没有对应凭证。

TEST_REPORTS_DIR="$(pwd)/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"

# 用例过滤：用 pytest -k 的否定表达式而不是从 --include 里删文件，因为下面这几个文件是
# 混装的（GPU 性能断言为主，夹着少量 CPU 副本），只需摘掉个别用例而不是整份跳过。
#
# FP8：真武 PPU 不支持 FP8。
# 非 CUDA：CPU 侧用例的名字里一定带 cpu——
#   test_perf.py 的 test_fusion_choice4_cpu、test_benchmarking.py 的
#   test_benchmark_cpu_smoke 与 @parametrize("device", (GPU_TYPE, "cpu")) 生成的 *_cpu 变体、
#   test_benchmark_fusion.py 里 HAS_CPU 为真时才建的 BenchmarkFusionCpuTest。
#   -k 匹配的是用例及其父节点（类名/模块名）的名字且大小写不敏感，所以一条 cpu
#   能同时盖住 BenchmarkFusionCpuTest；而这四个模块名和 GPU 用例名里都不含 cpu，不会误伤。
TEST_FILTER="not fp8 and not float8 and not e4m3 and not e5m2 and not cpu"

echo "=== CUDA inductor 性能单测（run_test.py --include 白名单 + -k 过滤 FP8/CPU） ==="
# 这批文件对应上游 inductor 性能回归的核心断言，主体跑在 GPU_TYPE 上（无 GPU 时
# __main__ 里的 HAS_GPU_AND_TRITON 守卫会直接退出）：
#   inductor/test_perf              - 生成 kernel 的读写字节数（带宽）断言
#   inductor/test_benchmarking      - benchmarker（GPU 计时器）本身的正确性
#   inductor/test_benchmark_fusion  - 基于 benchmark 的融合决策
#   inductor/test_kernel_benchmark  - kernel 级 benchmark 产物与 ncu/roofline 路径
# 白名单是"可调项"：请按 PPU 实际支持情况增删。
# 不使用 --upload-artifacts-while-running：那是官方 S3 上传路径，自建集群上没有。
python test/run_test.py \
    --include \
        inductor/test_perf \
        inductor/test_benchmarking \
        inductor/test_benchmark_fusion \
        inductor/test_kernel_benchmark \
    -k "${TEST_FILTER}" \
    --verbose

# -----------------------------------------------------------------------------
# gpt_fast micro benchmark（算子级 flops / 带宽利用率）
#
# 只跑 micro benchmark，不跑同文件注册的 llama2_7b_* / mixtral_8x7b_int8：那三个是
# 7B/8x7B 级别的 OSS 模型，显存和编译耗时都远超 PR 门禁的预算。
#
# benchmark.py -> generate.py 顶层 `import torchao`，而官方 install_torchao 是从
# github 源码编译（pip_build_and_install git+https://github.com/pytorch/ao.git@<pin>），
# pod 内没有 github 出口。所以这里先探测/尽力从内网 pip 源装 torchao，装不上就跳过这段，
# 不让它把整条门禁带红——上面的 inductor 性能单测才是必过项。
# -----------------------------------------------------------------------------
if [[ "${RUN_GPT_FAST:-1}" != "1" ]]; then
    echo "=== gpt_fast micro benchmark：RUN_GPT_FAST=${RUN_GPT_FAST:-1}，跳过 ==="
    echo "[perf] 完成"
    exit 0
fi

echo "=== gpt_fast micro benchmark 依赖自检（torchao） ==="
if (cd /tmp && python -c "import torchao" >/dev/null 2>&1); then
    echo "[perf] torchao 已可用"
    HAS_TORCHAO=1
else
    echo "[perf] torchao 缺失，尝试从 pip 源安装（失败则跳过 gpt_fast）"
    pip_args=(--disable-pip-version-check --retries 1 --timeout 20)
    if [[ -n "${PIP_INDEX:-}" ]]; then
        pip_args+=(-i "${PIP_INDEX}")
    fi
    if python -m pip install "${pip_args[@]}" torchao; then
        HAS_TORCHAO=1
    else
        HAS_TORCHAO=0
    fi
fi

if [[ "${HAS_TORCHAO}" != "1" ]]; then
    cat <<'EOF'
[perf][warn] torchao 装不上，跳过 gpt_fast micro benchmark（不影响本 job 结论）。
             要打开这段覆盖，需要内网 pip 源提供与镜像里 torch 版本匹配的 torchao，
             或让 PPU 发布镜像预装 torchao。
EOF
    echo "[perf] 完成（gpt_fast 已跳过）"
    exit 0
fi

echo "=== gpt_fast micro benchmark（仅 micro，device 由 torch.cuda.is_available() 决定为 cuda） ==="
# 逐个 --only 跑而不是一次全量：全量会把 llama2/mixtral 一起带进来。
# benchmark.py 是追加写 csv/json，多次调用会累积到同一份产物里。
for experiment in mlp_layer_norm_gelu layer_norm gather_gemv gemv; do
    echo "[perf] --only ${experiment}"
    python benchmarks/gpt_fast/benchmark.py \
        --only "${experiment}" \
        --output "${TEST_REPORTS_DIR}/gpt_fast_benchmark.csv"
done

echo "=== gpt_fast 结果 ==="
cat "${TEST_REPORTS_DIR}/gpt_fast_benchmark.csv"

echo "[perf] 完成"
