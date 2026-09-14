#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA 全量单元测试（full UT）入口。
# 由 .github/workflows/ppu_full_ut.yml 经 flytiger-eco/ppu-distributed-action
# 在单卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 来自 ppu-linux-build 编译产出的 whl（workflow 侧已下载到 WHEEL_DIR，随源码一起
# 送进 pod），由 install_wheel.sh 安装：跑的是 PPU 基础镜像，镜像里没有 torch，
# 门禁必须测本 PR 编出来的那一份。
#
# -----------------------------------------------------------------------------
# 测试语义从哪来（对齐 periodic.yml + slow.yml）
# -----------------------------------------------------------------------------
# periodic.yml / slow.yml 本身只是"薄编排层"：它们把 test-matrix 交给 _linux-test.yml，
# 真正跑什么由 .ci/pytorch/test.sh 按 TEST_CONFIG 决定。两条流水线的 config 落到脚本上是：
#   - periodic.yml 的 config: "default"（5 分片） -> test_python_shard，即
#       run_test.py --exclude-jit-executor --exclude-distributed-tests
#                   --exclude-quantization-tests --shard <i> <n>
#     并且因为 BUILD_ENVIRONMENT 带 cuda，test.sh 会 export
#       PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda
#     ——这就是官方"只跑 CUDA、不在 GPU 机器上白跑 CPU 变体"的做法。
#   - slow.yml 的 config: "slow"（3 分片） -> 同一个 test_python_shard，只是额外
#       export PYTORCH_TEST_WITH_SLOW=1 PYTORCH_TEST_SKIP_FAST=1
#     即"只跑被 @slowTest 标记的慢用例"。
# 本脚本用 UT_CONFIG 复刻这两种语义（default / slow），分片编号由 workflow 的 matrix 给。
# 其余 config 不做：
#   - distributed / multigpu 要多卡，本 pod 是单卡（nproc_per_node: 1）；
#   - nogpu_AVX512 / nogpu_NO_AVX2 是纯 CPU 线，与"仅 CUDA"的要求直接冲突；
#   - jit_legacy 走 legacy jit executor，恰好是 --exclude-jit-executor 排掉的那批。
#
# -----------------------------------------------------------------------------
# 仅 CUDA：两层过滤
# -----------------------------------------------------------------------------
# 第 1 层（用例级）PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda：
#   common_device_type.py 的 instantiate_device_type_tests 读这个变量，只实例化 cuda
#   变体。test_ops / test_nn / test_linalg 这类设备无关的大文件因此只跑 cuda 那一半，
#   不在 PPU 卡上白跑 CPU 变体。这是官方 CUDA CI 的同一个开关。
# 第 2 层（文件级）--exclude 清单：
#   上面那个开关管不到"整个文件都与 CUDA 无关"的情况（如 torch_np/ 的 NumPy 兼容层、
#   test_mkldnn 的 oneDNN 图融合、xpu/ 的 XPU 专属用例）。这些文件即使能在 CPU 上跑过，
#   也只是占着 PPU 卡跑与被测设备无关的东西，一律在文件级剔掉。清单见下面的
#   EXCLUDE_PREFIXES / EXCLUDE_NAMES，按组注明了剔除理由。
#   注意 run_test.py 自己已经默认排掉了 C++ 测试、test_mps / test_metal、test_xpu、
#   test_openreg 和 onnx/*（见 get_selected_tests），这里不重复列。
#
# -----------------------------------------------------------------------------
# FP8：真武 PPU 当前不支持 FP8，必须过滤（硬性要求，不是可选优化）
# -----------------------------------------------------------------------------
# 同样两层，缺一不可：
#   - 文件级：test_scaled_matmul_cuda（整个文件只有 TestFP8Matmul）与 inductor/test_fp8
#     （全量 FP8 语义）整文件剔掉，与 ppu_smoke.yml / ppu_accuracy.yml 的取舍一致。
#   - 用例级：-k 排除表达式，兜住散落在其它文件里的 FP8 用例（例如 dtype 参数化出来的
#     ..._float8_e4m3fn、test_sparse_semi_structured 的 test_sparse_fp8fp8_mm）。
#     pytest 的 -k 是**大小写敏感**的子串匹配，所以 fp8/FP8/Fp8、float8/Float8、
#     e4m3/E4M3、e5m2/E5M2 都要各写一份 —— 只写小写会漏掉 TestFP8Matmul /
#     TestFloat8Dtype 这类类名。
# PPU 支持 FP8 后：删掉 FP8 那一组文件级排除 + 下面的 K_EXPR 即可。
#
# 依赖环境变量：
#   UT_CONFIG        - default | slow（默认 default），语义见上
#   SHARD_NUMBER     - 当前分片编号，从 1 开始（必填）
#   NUM_TEST_SHARDS  - 总分片数（必填）
#   WHEEL_DIR        - torch whl 所在目录（默认 <repo>/.ci/ppu/wheelhouse，供 install_wheel.sh 使用）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PIP_INDEX        - 内部 pip 源（可选；不设则用 pip_sources.sh 的内置候选源）
#   TRITON_INDEX     - 装 triton 的**唯一**源（默认取 PIP_INDEX，供 install_triton.sh 使用）
#   TRITON_VERSION   - 钉住的 triton 版本（默认 3.6.0，供 install_triton.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
#   PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL - 可选逃生阀，见文末「显存不够怎么办」
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

UT_CONFIG="${UT_CONFIG:-default}"
case "$UT_CONFIG" in
    default | slow) ;;
    *)
        echo "[full-ut][error] UT_CONFIG 只支持 default / slow，实际: ${UT_CONFIG}" >&2
        exit 1
        ;;
esac

# 分片参数必填：漏传会让 run_test.py 跑全量，一个分片顶 5 个分片的时长，直接撞 pod 超时，
# 而日志上只表现为"超时"，看不出是漏了参数，所以在这里就拦下来。
if [[ -z "${SHARD_NUMBER:-}" || -z "${NUM_TEST_SHARDS:-}" ]]; then
    echo "[full-ut][error] 必须传 SHARD_NUMBER 与 NUM_TEST_SHARDS（由 workflow 的 matrix 给）" >&2
    exit 1
fi
if (( SHARD_NUMBER < 1 || SHARD_NUMBER > NUM_TEST_SHARDS )); then
    echo "[full-ut][error] SHARD_NUMBER=${SHARD_NUMBER} 不在 1..${NUM_TEST_SHARDS} 内" >&2
    exit 1
fi

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[full-ut] 源码目录: $(pwd)"
echo "[full-ut] 配置: UT_CONFIG=${UT_CONFIG} 分片=${SHARD_NUMBER}/${NUM_TEST_SHARDS}"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的 pip / 测试进程；
# 也必须放在装 torch 之前 —— import torch 要能找到 SDK 里的运行时库。
source .ci/ppu/sdk_env.sh

# 安装被测的 torch：本 PR 由 ppu_linux_build.yml 编出来的 whl
bash .ci/ppu/install_wheel.sh

# inductor 用例的 codegen 后端：钉版本、且只从内部源装（理由见 install_triton.sh 头注释）。
# 必须在 install_wheel.sh 之后；四条测试门禁用同一份脚本，改行为只改那一处。
# 注：每个分片都是独立的 pod，所以这一步每个分片各自跑一次，不能只在某一个分片里装。
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
# 安装逻辑（含 pip 源诊断与多源回退）抽到共享脚本，smoke/accuracy/perf 用同一份。
#
# 注意这份依赖只覆盖"跑得起来"的最小集，不含 scipy / numba / torchvision 等可选依赖：
# 全量 UT 里依赖它们的用例会走 @skipIfNoSciPy 这类装饰器 skip 掉（不是 fail），
# 想提高覆盖率应该让 PPU 基础镜像预装这些包，而不是在这里临时 pip install
# —— pod 无外网出口，且任何会牵出 torch 依赖的安装都有覆盖掉被测 whl 的风险。
bash .ci/ppu/install_test_deps.sh

# boto3 故意不装：它只被 tools/stats/upload_metrics.py 用于往官方 S3 上报指标，缺失时
# EMIT_METRICS=False 静默降级（日志里那条 "Unable to import boto3" 只是提示，不影响退出码），
# 自建集群也没有对应凭证。

# -----------------------------------------------------------------------------
# 第 1 层过滤：仅实例化 CUDA 变体（对齐 .ci/pytorch/test.sh 对 cuda BUILD_ENVIRONMENT 的处理）
# -----------------------------------------------------------------------------
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"

# slow 语义（对齐 .ci/pytorch/test.sh 里 TEST_CONFIG == 'slow' 的分支）
# 说明：pod 内没有 CI 环境变量，run_test.py 的 IS_CI 为 False，因此不会带
# --import-slow-tests 去下载 test-infra 的 slow-tests.json（pod 也没有外网出口）。
# 结果是 slow 这一档只跑源码里被 @slowTest 静态标记的用例，不含官方按历史耗时动态判定
# 的那批。这是 pod 网络条件决定的既定范围，不是漏配。
if [[ "$UT_CONFIG" == "slow" ]]; then
    export PYTORCH_TEST_WITH_SLOW=1
    export PYTORCH_TEST_SKIP_FAST=1
fi

# -----------------------------------------------------------------------------
# 第 2 层过滤：文件级 --exclude 清单
# -----------------------------------------------------------------------------
# 以 '/' 结尾的条目是"前缀组"，会在下面按 discover_tests.py 的 TESTS 展开成具体文件名；
# 其余条目必须是 TESTS 里的精确成员。
#
# 为什么要展开而不是直接把前缀丢给 run_test.py：--exclude 的 choices 就是 TESTS，
# 传一个不在清单里的字符串（比如 'torch_np/'）argparse 直接 exit 2（invalid choice），
# 一个用例都不会跑。而 torch_np/ 下有 36 个文件，逐个列出来既啰嗦又容易在 rebase 后漏改。
EXCLUDE_PREFIXES=(
    # NumPy 兼容层（torch._numpy）：验的是 numpy 语义等价性，纯 CPU，与 CUDA 无关
    'torch_np/'
    # Lazy Tensor 的 TorchScript 后端：本仓库不做 lazy 后端验证，默认也跑在 CPU 上
    'lazy/'
    # XPU（Intel GPU）专属用例。run_test.py 只默认排掉了顶层的 test_xpu，
    # xpu/ 目录下这几个不在它的 XPU_TEST 里，要手工剔
    'xpu/'
)

# 精确文件名。分组即剔除理由；每组都能独立回滚（例如 PPU 支持 FP8 后只删第 1 组）。
EXCLUDE_NAMES=(
    # --- 1) FP8 专属整文件：真武 PPU 不支持 FP8 ---------------------------------
    # 与 ppu_smoke.yml / ppu_accuracy.yml 的裁剪保持一致；PPU 支持 FP8 后加回来。
    # 这两个文件里**所有**用例都是 FP8 语义，留在门禁里只会整体失败。
    test_scaled_matmul_cuda
    inductor/test_fp8

    # --- 2) 其它后端 / 平台专属：在 PPU 上跑不出任何 CUDA 侧信息 ----------------
    profiler/test_xpu_profiler
    inductor/test_xpu_basic
    inductor/test_mps_basic
    inductor/test_halide            # Halide 后端，非 triton/CUDA 代码生成路径
    inductor/test_pallas            # Pallas(JAX) 后端
    inductor/test_triton_cpu_backend
    backends/xeon/test_launch       # Intel Xeon CPU 启动器
    test_vulkan
    test_mobile_optimizer
    test_set_default_mobile_cpu_allocator
    test_model_exports_to_core_aten
    test_numa_binding               # CPU NUMA 绑核
    test_cpp_extensions_mtia_backend
    test_privateuseone_python_backend
    test_rename_privateuse1_to_existing_device

    # --- 3) CPU 专属计算路径 ----------------------------------------------------
    inductor/test_cpu_cpp_wrapper
    inductor/test_cpu_repro
    inductor/test_cpu_select_algorithm
    inductor/test_mkldnn_pattern_matcher
    test_mkldnn
    test_mkldnn_fusion
    test_mkldnn_verbose
    test_mkl_verbose
    test_jit_llga_fuser             # oneDNN Graph fuser，CPU only
    test_openmp
    test_xnnpack_integration

    # --- 4) 与设备无关的 meta / 工具链类，且依赖 PPU pod 里没有的东西 -----------
    # 这些用例的结论与被测 torch 在 PPU 上的行为无关：要么校验打包/类型标注，
    # 要么需要 mypy / xdoctest / tensorboard / numba，要么要访问公网。
    doctests                        # 需 xdoctest
    test_typing                     # 需 mypy
    test_type_hints                 # 需 mypy
    typing/test_python_operators    # 需 mypy
    test_license                    # 校验 LICENSE 打包
    test_import_stats               # 统计 import 耗时，非功能验证
    test_tensorboard                # 需 tensorboard
    test_numba_integration          # 需 numba
    test_hub                        # 需访问 github.com 拉模型

    # --- 5) H100/B200(SM90+ / TMA / CUTLASS) 整文件专属 -------------------------
    # 与 ppu_accuracy.yml 同一个取舍：**整文件**都是大卡专属的才在这里剔；
    # 文件内少数大卡用例（如 inductor/test_flex_attention 里那个 TMA 用例）不动 ——
    # 它们由 IS_BIG_GPU / is_big_gpu() / SM90OrLater 等运行期能力探测装饰，
    # 在 PPU 上是 skip 而不是 fail。
    inductor/test_max_autotune
    inductor/test_cutedsl_grouped_mm
    inductor/test_cutlass_backend
    inductor/test_cutlass_evt
    inductor/test_flex_flash
    inductor/test_nv_universal_gemm
    nn/attention/test_fa3           # FlashAttention-3，Hopper 专属
    nn/attention/test_fa4           # FlashAttention-4，Blackwell 专属

    # --- 6) 多卡：本 pod 只有 1 张 PPU ------------------------------------------
    # distributed/* 由 --exclude-distributed-tests 统一排掉，这个是顶层的多卡文件。
    test_cuda_multigpu
)

# 展开前缀组 + 校验精确名。
# 这里主动校验而不是等 argparse 报 invalid choice：rebase 上游后测试文件被改名/删除时，
# argparse 只会吐一大段 choices 列表，看不出到底哪个名字过期了；而"清单里有过期项"必须
# 硬失败而不是静默跳过 —— 静默跳过会让排除范围悄悄变宽（最坏情况是 FP8 用例又跑起来）。
echo "=== 组装 --exclude 清单 ==="
EXCLUDE_RAW="$(python - "${EXCLUDE_PREFIXES[@]}" "${EXCLUDE_NAMES[@]}" <<'PY'
import sys

sys.path.insert(0, ".")
from tools.testing.discover_tests import TESTS  # noqa: E402

known = set(TESTS)
selected: set[str] = set()
empty_prefixes: list[str] = []
stale_names: list[str] = []

for arg in sys.argv[1:]:
    if arg.endswith("/"):
        matched = [t for t in TESTS if t.startswith(arg)]
        if not matched:
            empty_prefixes.append(arg)
        selected.update(matched)
    elif arg in known:
        selected.add(arg)
    else:
        stale_names.append(arg)

problems = []
if empty_prefixes:
    problems.append("前缀组没匹配到任何测试文件: " + " ".join(empty_prefixes))
if stale_names:
    problems.append("以下名字不在 discover_tests.py 的 TESTS 里（被上游改名或删除了）: " + " ".join(stale_names))
if problems:
    sys.exit(
        "[full-ut] --exclude 清单已过期，请修正 .ci/ppu/full_ut_test.sh：\n  "
        + "\n  ".join(problems)
    )

print(" ".join(sorted(selected)))
PY
)"
read -r -a EXCLUDE_TESTS <<<"$EXCLUDE_RAW"
echo "[full-ut] 文件级排除 ${#EXCLUDE_TESTS[@]} 个测试文件"

# 用例级 FP8 排除表达式（-k 会被 run_test.py 原样透传给 pytest 的 -k）。
# 大小写各写一份的原因见文件头。
K_EXPR="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2"

# -----------------------------------------------------------------------------
# 跑批
# -----------------------------------------------------------------------------
# --exclude-jit-executor / --exclude-distributed-tests / --exclude-quantization-tests
#   三个开关逐字对齐 .ci/pytorch/test.sh 的 test_python_shard（periodic / slow 两条流水线
#   的 default / slow config 走的就是它）：legacy jit executor 单独成线、distributed 要多卡、
#   quantization 单独成线。
# --continue-through-error
#   全量 UT 的价值在于"一次跑完拿到完整失败清单"。默认行为是某个测试文件失败就中止，
#   那样每修一个问题都要重跑几小时，PPU 适配期完全不可接受。退出码仍然反映失败。
# 不使用 --upload-artifacts-while-running：那是官方 S3 上传路径，自建集群上没有
#   （pod 内 IS_CI 为 False，这个开关的默认值本来也是关的）。
# 显存不够怎么办：run_test.py 默认起 2~3 个 pytest 进程并行跑（tools/testing/
#   test_selections.py 的 NUM_PROCS），共用同一张 PPU。若出现并发导致的显存不足，
#   给本脚本传 PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL=1 退回串行（会显著变慢，
#   要同步调大 workflow 里的 timeout_minutes）。
echo "=== CUDA 全量 UT（config=${UT_CONFIG} shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS}） ==="
python test/run_test.py \
    --exclude-jit-executor \
    --exclude-distributed-tests \
    --exclude-quantization-tests \
    --exclude "${EXCLUDE_TESTS[@]}" \
    --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" \
    -k "$K_EXPR" \
    --continue-through-error \
    --verbose

echo "[full-ut] 完成 (config=${UT_CONFIG} shard=${SHARD_NUMBER}/${NUM_TEST_SHARDS})"
