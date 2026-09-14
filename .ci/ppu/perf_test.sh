#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA inductor 性能测试入口。
# 由 .github/workflows/ppu_perf.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 来自 ppu-linux-build 编译产出的 whl（workflow 侧已下载到 WHEEL_DIR，随源码一起
# 送进 pod），由 install_wheel.sh 安装：跑的是 PPU 基础镜像，镜像里没有 torch，
# 门禁必须测本 PR 编出来的那一份。
#
# 为什么不照搬 inductor-perf-test-nightly-h100.yml 的跑批内容（重要）：
#   那条流水线的 perf 语义是 .ci/pytorch/test.sh 的 test_perf_for_dashboard，即用
#   DASHBOARD_TAG 驱动 benchmarks/dynamo/{huggingface,timm_models,torchbench}.py 带
#   --performance 跑模型套件。这三个套件全部要联网拉模型：
#     - huggingface.py   走 AutoConfig.from_pretrained(...)，要访问 huggingface.co
#     - timm_models.py   list_models(pretrained=True) / create_model(pretrained=True)，要拉预训练权重
#     - torchbench.py    还要额外 clone pytorch/benchmark 仓库并装模型依赖
#   而 PPU pod 没有外网出口（参见 install_test_deps.sh 末尾的诊断说明：「公网源不可达
#   -> pod 无外网出口，只能走内网源」），照搬必然卡死在拉模型阶段。
#   因此这里保留 perf 的**测试语义**（inductor 生成代码的性能/访存特征 + 端到端微基准
#   计时），但只用**零下载、自包含**的用例，全部构造随机张量在设备上现算。
#   等运维把 HF 缓存 / torchbench 仓库预置到已挂载的 NAS（HOST_VOLUMES 里的 /nas_aisw、
#   /wl_nas）之后，再考虑把 dashboard 套件那一层加回来。
#
# FP8：真武 PPU 当前不支持 FP8。下面选中的用例文件经 grep 确认零 FP8 命中
# （fp8|float8|e4m3|e5m2 在这 5 个文件里一个都没有），仍显式挂上 -k 排除表达式作为
# 防回归护栏 —— rebase 上游后若有人往这些文件里加 FP8 用例，不必再改这份脚本。
#
# 依赖环境变量：
#   WHEEL_DIR        - torch whl 所在目录（默认 <repo>/.ci/ppu/wheelhouse，供 install_wheel.sh 使用）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PIP_INDEX        - 内部 pip 源（可选；不设则用 pip_sources.sh 的内置候选源）
#   TRITON_INDEX     - 装 triton 的**唯一**源（默认取 PIP_INDEX，供 install_triton.sh 使用）
#   TRITON_VERSION   - 钉住的 triton 版本（默认 3.6.0，供 install_triton.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
#   RUN_GPT_FAST     - 是否跑 gpt_fast 微基准（默认 1；设 0 可只跑 inductor perf 单测）
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[perf] 源码目录: $(pwd)"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的 pip / 测试进程；
# 也必须放在装 torch 之前 —— import torch 要能找到 SDK 里的运行时库。
source .ci/ppu/sdk_env.sh

# 安装被测的 torch：本 PR 由 ppu_linux_build.yml 编出来的 whl
bash .ci/ppu/install_wheel.sh

# inductor 用例的 codegen 后端：钉版本、且只从内部源装（理由见 install_triton.sh 头注释）。
# 必须在 install_wheel.sh 之后；四条测试门禁用同一份脚本，改行为只改那一处。
# 对本门禁尤其关键：这里全是 benchmark / kernel 计时类用例，triton 版本错位会直接改变
# 生成的 kernel，测出来的数与基线不可比。
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
# 自检必须在源码树外执行：`python -c` 会把 cwd（此时是源码根）放进 sys.path[0]，
# 未编译的源码目录 torch/ 会遮蔽刚装上的 torch，而 torch/version.py 是构建产物、
# 源码树里不存在，于是报 ModuleNotFoundError: No module named 'torch.version'。
# 打印 torch.__file__ 以便一眼确认导入的是 site-packages 里的那份。
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

# 测试框架依赖（pytest 及其插件、expecttest、hypothesis）镜像里没预装对版本，必须在这里补
# （注意顺序：先装 torch 再装这些，torch 的 whl 不会反过来动这些测试期包的版本）：
# run_test.py 的 check_pip_packages() 硬校验 pytest-rerunfailures / pytest-flakefinder /
# pytest-xdist，缺任意一个直接 exit 1。
# 版本一律对齐 .ci/docker/requirements-ci.txt（官方 CI 跑 test case 的同一份约束），
# 安装逻辑（含 pip 源诊断与多源回退）抽到共享脚本，smoke_test.sh / accuracy_test.sh 用同一份。
bash .ci/ppu/install_test_deps.sh

# boto3 故意不装：它只被 tools/stats/upload_metrics.py 用于往官方 S3 上报指标，缺失时
# EMIT_METRICS=False 静默降级（日志里那条 "Unable to import boto3" 只是提示，不影响退出码），
# 自建集群也没有对应凭证。

TEST_REPORTS_DIR="$REPO_ROOT/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"

# -----------------------------------------------------------------------------
# 第 1 段（硬门禁）：inductor 性能相关单测，仅 CUDA 一条线。
#
# 这 5 个文件是 test/inductor 下真正测「性能」而不是「数值正确性」的那一批，且全部
# 零下载、自包含（随机张量现构造）：
#   - inductor/test_perf              统计生成代码的访存量（memory traffic），
#                                     融合/内联退化会让计数变化 -> 直接失败
#   - inductor/test_benchmark_fusion  benchmark 驱动的融合决策（config.benchmark_fusion）
#   - inductor/test_kernel_benchmark  生成的 triton kernel 能否独立 benchmark、
#                                     以及 -DTORCHINDUCTOR_BENCHMARK_KERNEL 路径
#   - inductor/test_benchmarking      benchmarker / TritonBenchmarker 计时基座本身
#   - inductor/test_analysis          profile_analysis（kernel 耗时/带宽/FLOPS 归因）
#
# 只跑 CUDA：这几个文件统一走 torch.testing._internal.inductor_utils 的 GPU_TYPE，
# 在 PPU 上 GPU_TYPE == "cuda"（PPU 通过 torch.cuda 接口暴露），纯 CPU 分支由
# HAS_GPU / requires_gpu_and_triton 等装饰器自行处理，不需要额外传设备参数。
#
# H100/B200 专属用例不必手工剔除：它们由 IS_BIG_GPU / is_big_gpu() / SM80OrLater 等
# 运行期能力探测装饰，在 PPU 上是 skip 而不是 fail（与 ppu_accuracy.yml 里手工剔
# inductor/test_max_autotune 那类整文件 H100 用例的处理方式不同：这里是文件内少数用例）。
#
# 用例名必须是 run_test.py 发现得到的测试名（即 tools/testing/discover_tests.py 的 TESTS，
# 大体对应 test/ 下去掉 .py 后缀的相对路径）：-i/--include 的 choices 就是这份清单，
# 写错一个名字 argparse 会直接 exit 2（invalid choice），一个用例也不会跑。
# 因此 rebase 上游后若有测试文件被重命名/删除，这里要跟着改。可在仓库根目录用
#   python -c "import sys;sys.path.insert(0,'.');from tools.testing.discover_tests import TESTS;print('inductor/test_perf' in TESTS)"
# 校验（不需要装 torch）。
#
# -k 是 run_test.py 的 --pytest-k-expr，会原样透传给 pytest 作为 -k：用来兜住 FP8。
# 不使用 --upload-artifacts-while-running：那是官方 S3 上传路径，自建集群上没有。
# -----------------------------------------------------------------------------
echo "=== [1/2] CUDA inductor 性能单测（run_test.py --include 白名单 + -k 排除 FP8） ==="
python test/run_test.py \
    --include \
        inductor/test_perf \
        inductor/test_benchmark_fusion \
        inductor/test_kernel_benchmark \
        inductor/test_benchmarking \
        inductor/test_analysis \
    -k "not fp8 and not float8 and not e4m3 and not e5m2" \
    --verbose

# -----------------------------------------------------------------------------
# 第 2 段（尽力而为）：gpt_fast 微基准，对齐 .ci/pytorch/test.sh 的
# test_inductor_micro_benchmark —— 这是官方 CI 里唯一一条零下载的 perf 跑批，
# 真正吐出 torch.compile 后的 flops 利用率 / 访存带宽数字。
#
# 为什么是「尽力而为」而不是硬门禁：
#   benchmarks/gpt_fast/benchmark.py 顶部 `from generate import get_arch_name`，
#   而 generate.py 顶部 `import torchao` —— 也就是说这个入口**必须**有 torchao 才能
#   import。官方的装法是 common_utils.sh 的 install_torchao()，走
#   `pip install git+https://github.com/pytorch/ao.git@<pin>` 从源码编译，pod 内没有
#   github.com 出口，这条路走不通。
#   退一步从内网 pip 源装 torchao 的 wheel 是可行的，但必须 --no-deps：torchao 的
#   metadata 声明依赖 torch，不加 --no-deps 会让 pip 从公网/内网源拉一个与本 PR 无关的
#   torch 覆盖掉上面刚装好的被测 whl —— 那等于门禁测了个别的包，比不测更糟。
#   --no-deps 装出来的 torchao 其 C 扩展未必与本 whl 的 ABI 匹配，所以装完还要真的
#   `import torchao` 验一次；验不过就跳过本段，只在日志里说明原因，不把 job 判红：
#   第 1 段才是这条流水线的门禁面，torchao 装不上是环境问题、不是被测代码的问题。
# 想彻底稳掉这一段：让 PPU 基础镜像按 .github/ci_commit_pins/torchao.txt 预装 torchao，
# 本段检测到 import 成功就会直接跑。
# -----------------------------------------------------------------------------
if [[ "${RUN_GPT_FAST:-1}" != "1" ]]; then
    echo "=== [2/2] gpt_fast 微基准：已由 RUN_GPT_FAST=0 显式关闭，跳过 ==="
    echo "[perf] 完成"
    exit 0
fi

echo "=== [2/2] gpt_fast 微基准（torch.compile 端到端计时） ==="

# 探测 torchao 是否可用；不可用则尝试从内网源补装（--no-deps，绝不能动 torch）
if ! (cd /tmp && python -c "import torchao" >/dev/null 2>&1); then
    echo "[perf] torchao 不可用，尝试从候选 pip 源补装（--no-deps）"
    # shellcheck source=.ci/ppu/pip_sources.sh
    source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
    ppu_build_pip_candidates
    for index in "${PIP_CANDIDATES[@]}" __pip_default__; do
        pip_args=(--disable-pip-version-check --retries 1 --timeout 20 --no-deps)
        if [[ "${index}" == "__pip_default__" ]]; then
            label="pip 默认源"
        else
            label="${index}"
            pip_args+=(-i "${index}")
        fi
        echo "[perf] 尝试源: ${label}"
        if python -m pip install "${pip_args[@]}" torchao; then
            echo "[perf] torchao 安装成功（源: ${label}）"
            break
        fi
        echo "[perf][warn] 源 ${label} 失败，换下一个"
    done
fi

# 装完再验一次真能 import：--no-deps 的 wheel 可能与本 whl 的 torch ABI 不匹配，
# 那种情况下 import 会炸在 torchao._C，必须在跑 benchmark 之前拦下来。
# 同样要在源码树外执行，否则源码目录 torch/ 会遮蔽装好的 torch。
if ! (cd /tmp && python -c "import torchao; print('torchao', torchao.__version__)"); then
    cat <<'EOF'
[perf][warn] torchao 仍不可用，跳过 gpt_fast 微基准（不影响本 job 的结论）。
  原因：benchmarks/gpt_fast/benchmark.py -> generate.py 顶部 `import torchao`，
        没有 torchao 连 import 都过不去；而官方装法是从 github.com 源码编译，
        pod 无外网出口。
  处理：让 PPU 基础镜像预装 torchao（版本参考 .github/ci_commit_pins/torchao.txt），
        或确认内网 pip 源里有与本 torch 版本 ABI 匹配的 torchao wheel。
EOF
    echo "[perf] 完成（仅第 1 段）"
    exit 0
fi

# 只跑 4 个微基准，不跑 llama2_7b / mixtral_8x7b：
#   - 那 3 个实验虽然也是随机权重（generate.py 的 _load_model 用 torch.randn 填
#     state_dict，不下载 checkpoint），但 7B / 8x7B 显存与耗时都远超单卡门禁的预算，
#     且 int8 那两个还要走 torchao 的量化 handler。
#   - benchmark.py 的 --only 一次只接受一个实验名（main() 里是
#     `experiments = [all_experiments[only_model]]`），所以这里逐个跑；
#     输出走 output_csv，同一个文件是追加语义，4 次跑完汇总在一张 csv 里。
#   - 实验名写错会命中 all_experiments 的 KeyError 直接失败（不是静默跳过），
#     rebase 上游后若实验被改名，这里会立刻暴露。
GPT_FAST_CSV="$TEST_REPORTS_DIR/gpt_fast_benchmark.csv"
rm -f "$GPT_FAST_CSV" "${GPT_FAST_CSV%.csv}.json"
for experiment in mlp_layer_norm_gelu layer_norm gather_gemv gemv; do
    echo "--- gpt_fast: ${experiment} ---"
    python benchmarks/gpt_fast/benchmark.py \
        --only "${experiment}" \
        --output "$GPT_FAST_CSV"
done

# 自建集群没有官方那套 S3 / benchmark database 上传通道，perf 数字只能落到日志里，
# 否则 pod 销毁后就查不到了（csv 本身在 test/test-reports 下，action 不会带回来）。
# 这里不能直接 cat：万一 benchmark.py 没吐出 csv（例如上游改了输出路径语义），
# set -e 下 cat 失败会让日志最后一行只剩一句 "No such file"，把真正的现场盖掉。
echo "=== gpt_fast 微基准结果 ==="
if [[ -s "$GPT_FAST_CSV" ]]; then
    cat "$GPT_FAST_CSV"
else
    echo "[perf][warn] 没有在 ${GPT_FAST_CSV} 拿到结果：上面 4 次调用均以 0 退出，但没写出 csv。"
    echo "[perf][warn] 先看 benchmarks/gpt_fast/benchmark.py 的 --output / output_csv 语义是否被上游改过。"
fi

echo "[perf] 完成"
