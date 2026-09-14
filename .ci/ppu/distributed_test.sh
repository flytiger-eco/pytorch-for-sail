#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：CUDA 分布式测试入口（2 卡）。
# 由 .github/workflows/ppu_distribute.yml 经 flytiger-eco/ppu-distributed-action
# 在**单个** 2 卡 PPU pod 内执行（源码已由 action 解压到 pod 的 source_dir）。
#
# 单独抽成脚本而不是内联到 yaml 的 command：command 由 pod 的默认 shell 执行，
# 未必是 bash，而 sdk_env.sh 依赖 bash 语法（[[ ]] / BASH_SOURCE）且必须被 source。
#
# torch 来自 ppu-linux-build 编译产出的 whl（workflow 侧已下载到 WHEEL_DIR，随源码一起
# 送进 pod），由 install_wheel.sh 安装：跑的是 PPU 基础镜像，镜像里没有 torch，
# 门禁必须测本 PR 编出来的那一份。镜像也没有 run_test.py 需要的 pytest 插件，仍需补装。
#
# -----------------------------------------------------------------------------
# 「2 卡」是怎么来的：本脚本只跑一次，不是每个 rank 跑一次
# -----------------------------------------------------------------------------
# action 的 nnodes: 0 是 SINGLE_MODE（NNODES=1，只起 1 个 worker pod、不建 PodGroup），
# nproc_per_node 是**每 Pod 请求的 alibabacloud.com/ppu 卡数**（见 action.yml 里该 input 的
# 描述），不是进程数：nproc_per_node: 2 得到的是「1 个 pod，容器里可见 2 张 PPU」，
# command 只被执行一次。
# 所以这里不需要任何 rank 守卫，直接按单机多卡的方式跑：进程由 run_test.py 下面的用例
# 自己 spawn（torch.testing._internal 的 MultiProcessTestCase / MultiProcContinuousTest
# 按各自的 world_size 起子进程），本脚本只是那个「单机启动器」。
#
# 必须 unset action 注入的分布式环境变量：submit.sh 给 pod 注了一整套
#   RANK=0 LOCAL_RANK=0 WORLD_SIZE=1(=NNODES) NODE_RANK=0 MASTER_PORT=29500
#   MASTER_ADDR=<job>-worker-0.<job>.<ns>.svc.cluster.local
# 这套值是给「用户自己写的 torchrun/train.py」用的，对 PyTorch 单机多卡测试是纯污染：
#   - WORLD_SIZE=1 与用例自己声明的 world_size(=2) 冲突；
#   - SINGLE_MODE 下 action 不建 headless Service，那个 MASTER_ADDR 在 pod 内根本解析不出来，
#     init_process_group 会卡到 rendezvous 超时。
# 已核对本文件下面用到的测试文件都不依赖这些变量（没有用 skip_if_no_gpu 这类直接读
# os.environ["WORLD_SIZE"] 的装饰器），unset 是安全的。
#
# -----------------------------------------------------------------------------
# 用例清单从哪来
# -----------------------------------------------------------------------------
# 以 .ci/pytorch/test.sh 的 test_inductor_distributed()（官方 inductor_distributed 配置）
# 为主，叠加 test_h100_distributed()（h100-distributed.yml 的 case），再按「PPU pod 只有
# 2 卡」和「PPU 不支持 FP8」两条约束裁剪。裁剪掉的每一条都在下面就地注释了原因。
#
# 裁剪原则：只在「整条 entry 的**所有**用例都需要 >2 卡」时才删。单条 entry 里既有 2 卡
# 又有 4/8 卡用例的（例如 test_dtensor_compile、test_fully_shard_* 的多数文件），保留整条 ——
# 需要更多卡的那些用例带 @skip_if_lt_x_gpu(N) / with_comms，2 卡上是 skip 而不是 fail。
#
# -----------------------------------------------------------------------------
# 两层「只跑 CUDA」过滤
# -----------------------------------------------------------------------------
#   1. PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda：用例级。instantiate_device_type_tests 只实例化
#      cuda 变体，不会生成 cpu/meta 那一份。
#   2. --include 白名单：文件级。本脚本逐条列出要跑的测试文件，天然不含非 CUDA 文件。
#
# FP8 过滤同样两层（真武 PPU 当前不支持 FP8）：
#   1. 文件级：本清单里没有纯 FP8 文件（test_scaled_matmul_cuda / inductor/test_fp8 不在其中）。
#   2. 用例级：下面的 K_FP8 表达式，透传给 pytest 的 -k。pytest 的 -k 大小写敏感，
#      所以 fp8/FP8/Fp8、float8/Float8、e4m3/E4M3、e5m2/E5M2 各写一份 —— 只写小写会漏掉
#      TestFP8Matmul 这类类名。另外补两个「名字里没有 fp8 字样但实质是 FP8」的：
#        - scaled_matmul：distributed/tensor/parallel/test_micro_pipeline_tp 里的
#          test_fuse_all_gather_scaled_matmul / test_fuse_scaled_matmul_reduce_scatter[_*]
#          走的是 torch._scaled_mm 的 e4m3 路径。
#          （注意别误伤 test_find_all_gather_patterns：它里面的 _fp8_all_gather 只是 float32 上
#           的 dtype-view 技巧，不是真 FP8，且名字里带 fp8 已被上面的通用式排掉。）
#        - test_fixed_striding：distributed/test_c10d_functional_native 里唯一的 FP8 用例
#          （e4m3_type + torch._scaled_mm，只由 PLATFORM_SUPPORTS_FP8 守卫）。
# PPU 支持 FP8 后：删掉 K_FP8 即可（清单本身不用动）。
#
# 依赖环境变量：
#   WHEEL_DIR        - torch whl 所在目录（默认 <repo>/.ci/ppu/wheelhouse，供 install_wheel.sh 使用）
#   SDK_INSTALL_DIR  - PPU SDK 安装目录（默认 /usr/local，供 sdk_env.sh 使用）
#   PIP_INDEX        - 内部 pip 源（可选；不设则用 install_test_deps.sh 的内置候选源）
#   TRITON_INDEX     - 装 triton 的**唯一**源（默认取 PIP_INDEX，供 install_triton.sh 使用）
#   TRITON_VERSION   - 钉住的 triton 版本（默认 3.6.0，供 install_triton.sh 使用）
#   PR_NUMBER        - 仅用于日志溯源（可选）
# =============================================================================
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

# 切到源码根目录：action 的 source_dir 可配置，这里按脚本自身位置反推，不写死路径
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[distributed] 源码目录: $(pwd)"

# sdk_env.sh 会 source PPU SDK 的 envsetup.sh（设置 LD_LIBRARY_PATH / PATH）并校验 nvcc。
# 必须 source（而非执行），环境变量才能作用于后续的 pip / 测试进程；
# 也必须放在装 torch 之前 —— import torch 要能找到 SDK 里的运行时库。
source .ci/ppu/sdk_env.sh

# 安装被测的 torch：本 PR 由 ppu_linux_build.yml 编出来的 whl
bash .ci/ppu/install_wheel.sh

# inductor 用例的 codegen 后端：钉版本、且只从内部源装（理由见 install_triton.sh 头注释）。
# 必须在 install_wheel.sh 之后；四条测试门禁用同一份脚本，改行为只改那一处。
# 本门禁里的 distributed/test_inductor_collectives 等用例走 inductor codegen，同样依赖它。
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 (CUDA, 2 卡) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
# 先原样打印 action 注入的那套值再清掉，出问题时能一眼看出 pod 侧给了什么
echo "action 注入: NNODES=${NNODES:-} NPROC_PER_NODE=${NPROC_PER_NODE:-} RANK=${RANK:-}" \
     "LOCAL_RANK=${LOCAL_RANK:-} WORLD_SIZE=${WORLD_SIZE:-} NODE_RANK=${NODE_RANK:-}" \
     "MASTER_ADDR=${MASTER_ADDR:-} MASTER_PORT=${MASTER_PORT:-}"
# 理由见文件头「必须 unset ...」：留着会让用例的 init_process_group 用错 world_size、
# 或连到一个在 SINGLE_MODE 下不存在的 MASTER_ADDR 上卡死。
unset RANK LOCAL_RANK WORLD_SIZE NODE_RANK MASTER_ADDR MASTER_PORT
echo "已清理上述 rank / rendezvous 变量，交由各用例自行 spawn 子进程"

python --version
# 自检必须在源码树外执行：`python -c` 会把 cwd（此时是源码根）放进 sys.path[0]，
# 未编译的源码目录 torch/ 会遮蔽刚装上的 torch，而 torch/version.py 是构建产物、
# 源码树里不存在，于是报 ModuleNotFoundError: No module named 'torch.version'。
# 打印 torch.__file__ 以便一眼确认导入的是 site-packages 里的那份。
# 同时硬校验卡数 >= 2：本门禁全部用例都是多卡语义，只有 1 张卡的话它们会集体 skip、
# job 却是绿的（假通过）。宁可在这里快速失败，把「pod 没按 nproc_per_node: 2 拿到卡」
# 这类环境问题直接暴露出来。
(cd /tmp && python - <<'PY'
import sys

import torch

print("torch", torch.__version__, torch.__file__)
print("cuda_available", torch.cuda.is_available())
count = torch.cuda.device_count()
print("device_count", count)
if count < 2:
    sys.exit(
        f"[distributed] 需要至少 2 张 PPU，实际只看到 {count} 张。"
        "本门禁全部用例都是多卡语义，卡数不足时它们只会被 skip（假通过），故直接失败。"
        "请检查 workflow 的 nproc_per_node 是否为 2、以及 pod 是否真的拿到了 2 张卡。"
    )
PY
)
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"

# 测试框架依赖（pytest 及其插件、expecttest、hypothesis）镜像里没预装对版本，必须在这里补
# （注意顺序：先装 torch 再装这些，torch 的 whl 不会反过来动这些测试期包的版本）：
# run_test.py 的 check_pip_packages() 硬校验 pytest-rerunfailures / pytest-flakefinder /
# pytest-xdist，缺任意一个直接 exit 1。
# 版本一律对齐 .ci/docker/requirements-ci.txt（官方 CI 跑 test case 的同一份约束），
# 安装逻辑（含 pip 源诊断与多源回退）抽到共享脚本，smoke_test.sh / full_ut_test.sh 用同一份。
bash .ci/ppu/install_test_deps.sh

# boto3 故意不装：它只被 tools/stats/upload_metrics.py 用于往官方 S3 上报指标，缺失时
# EMIT_METRICS=False 静默降级（日志里那条 "Unable to import boto3" 只是提示，不影响退出码），
# 自建集群也没有对应凭证。

# -----------------------------------------------------------------------------
# 过滤条件
# -----------------------------------------------------------------------------
# 第 1 层：用例级「只跑 CUDA」。见文件头。
export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"

# 用例级 FP8 排除表达式（-k 会被 run_test.py 原样透传给 pytest 的 -k）。
# 大小写各写一份、以及最后两条「名字里没有 fp8 字样」的补充，原因见文件头。
K_FP8="not fp8 and not FP8 and not Fp8 \
and not float8 and not Float8 \
and not e4m3 and not E4M3 \
and not e5m2 and not E5M2 \
and not scaled_matmul \
and not test_fixed_striding"

# -----------------------------------------------------------------------------
# 跑批
# -----------------------------------------------------------------------------
# 与 .ci/pytorch/test.sh 里「一条命令一个 python test/run_test.py」的写法保持一致（分布式
# 测试必须一个文件一个进程地串起来：run_test.py 的 must_serial() 对所有 distributed/* 强制
# 串行，2 张卡也容不下并发），但额外做两件事：
#   1. 统一叠加 -k 过滤（把该 entry 自带的 -k 与 K_FP8 用 and 组合）；
#   2. 失败不中止，记进 FAILED_CASES 末尾汇总后统一非零退出。
# 第 2 点等价于 full_ut 用的 --continue-through-error（那个开关只在单次 run_test.py 调用内
# 生效，管不了这里的多次调用）：PPU 适配期需要「一次跑完拿到完整失败清单」，默认的
# set -e 首条失败即中止会让每修一个问题都要重跑整条流水线。
FAILED_CASES=()

# run_case "<该 entry 自带的 -k 过滤，留空表示整文件>" <run_test.py 的 --include 测试名...>
#
# 测试名必须是 run_test.py 发现得到的名字（tools/testing/discover_tests.py 的 TESTS，
# 大体对应 test/ 下去掉 .py 后缀的相对路径）：-i/--include 的 choices 就是这份清单，
# 写错一个名字 argparse 直接 exit 2（invalid choice），一个用例也不会跑。
# 因此 rebase 上游后若有测试文件被重命名/删除，这里要跟着改。可在仓库根目录用
#   python -c "import sys;sys.path.insert(0,'.');from tools.testing.discover_tests import TESTS;print('distributed/test_c10d_functional_native' in TESTS)"
# 校验（不需要装 torch）。
run_case() {
    local own_k="$1"
    shift
    local k_expr label
    if [ -n "$own_k" ]; then
        # 加括号：K_FP8 是一串 and 连接的 not，不加括号时若 own_k 里出现 or 会被错误结合
        k_expr="( ${own_k} ) and ( ${K_FP8} )"
    else
        k_expr="${K_FP8}"
    fi
    label="$*${own_k:+ -k ${own_k}}"

    echo "=== [case] ${label} ==="
    # 放在 if 条件里，errexit 在此不生效，失败可继续
    if python test/run_test.py --include "$@" -k "$k_expr" --verbose; then
        echo "=== [pass] ${label} ==="
    else
        echo "::error::[fail] ${label}"
        FAILED_CASES+=("${label}")
    fi
}

echo "=== CUDA 分布式用例（2 卡；仅 CUDA、已排除 FP8） ==="

# --- AOTInductor 的多 device 路径（对齐 test_inductor_distributed 的前 4 条）---
# 这 4 个用例都带 @requires_multigpu()（>=2 卡），整文件跑太重，仍按官方那样逐个 -k。
run_case "test_replicate_on_devices"      inductor/test_aot_inductor
run_case "test_on_gpu_device1"            inductor/test_aot_inductor
run_case "test_non_default_gpu_device"    inductor/test_aot_inductor
run_case "test_load_package_multiple_gpus" inductor/test_aot_inductor

# --- 通信算子 / DTensor / 编译协同 ---
# test_c10d_functional_native: world_size=2。唯一的 FP8 用例 test_fixed_striding 由 K_FP8 排掉。
run_case "" distributed/test_c10d_functional_native
# test_dtensor_compile: TestDTensorCompile world_size=2；同文件的 E2E 类要 4 卡，
# 由 with_comms 在卡数不足时自动 skip，不影响本条。
run_case "" distributed/tensor/test_dtensor_compile
# test_micro_pipeline_tp: 主体用 FakeStore 单卡即可；3 个真 FP8 用例由 K_FP8 的
# not scaled_matmul 排掉。
run_case "" distributed/tensor/parallel/test_micro_pipeline_tp
# test_replicate_with_compiler: world_size=min(2, device_count)；同文件的 DDP_TP_Test
# 在上游已被 @unittest.skip 标掉。
run_case "" distributed/_composable/test_replicate_with_compiler

# --- FSDP2（test_fully_shard_*）---
# 整文件跑：world_size=2，且它同时覆盖了 h100-distributed.yml 的第二条
# （-k TestFullyShardAllocFromPG）—— 那个类带 @requires_multicast_support()，
# PPU 上没有 multicast 支持时会自动 skip，所以不必单独再列一条。
run_case "" distributed/_composable/fsdp/test_fully_shard_comm
run_case "test_train_parity_multi_group"                   distributed/_composable/fsdp/test_fully_shard_training
run_case "test_train_parity_with_activation_checkpointing" distributed/_composable/fsdp/test_fully_shard_training
# test_train_parity_hsdp: world_size=min(4, device_count)，2 卡时退化成 shard_size=1，仍可跑
run_case "test_train_parity_hsdp"                          distributed/_composable/fsdp/test_fully_shard_training
run_case "test_gradient_accumulation"                      distributed/_composable/fsdp/test_fully_shard_training
# 官方清单里还有一条 -k test_train_parity_2d_transformer_checkpoint_resume：本版本上游已把
# 该用例从 test_fully_shard_training.py 移到
# distributed/_composable/test_composability/test_2d_composability.py，留在这里 pytest 收集
# 不到任何用例（run_test.py 会把 pytest 的 exit code 5 归一化成 0，于是静默空跑）；
# 而新位置那个类 world_size 硬编码 4、MultiProcContinuousTest 不按 device_count 裁剪，
# 2 卡上会真起 4 个进程。两头都不合适，故整条去掉。
run_case "test_dp_state_dict_save_load" distributed/_composable/fsdp/test_fully_shard_state_dict
run_case "" distributed/_composable/fsdp/test_fully_shard_frozen
run_case "test_compute_dtype" distributed/_composable/fsdp/test_fully_shard_mixed_precision
run_case "test_reduce_dtype"  distributed/_composable/fsdp/test_fully_shard_mixed_precision
# 官方写的是 -k test_clip_grad_norm_2d，但那个用例在 TestClipGradNormWorldSize4 里，2 卡必 skip
# （整条 entry 零覆盖）。放宽成前缀 test_clip_grad_norm，让同文件的
# TestClipGradNormWorldSize2::test_clip_grad_norm_1d 真正跑起来；4 卡那个仍自动 skip。
run_case "test_clip_grad_norm" distributed/_composable/fsdp/test_fully_shard_clip_grad_norm_
run_case "" distributed/_composable/fsdp/test_fully_shard_compile

# 官方清单里的 distributed/fsdp/test_fsdp_tp_integration -k test_fsdp_tp_integration 去掉：
# 该用例带 @skip_if_lt_x_gpu(4)，2 卡上必定 skip，是纯粹的空跑。
# h100-distributed.yml 的 distributed/_composable/test_composability/test_pp_composability
# 同理去掉：world_size=8，文件内 4 个用例全部 @skip_if_lt_x_gpu(8)。
# 集群拿到 4 卡/8 卡 pod 后（nproc_per_node 调大）再把这两条加回来。

# --- dynamo / inductor 的分布式路径（均为 world_size=2）---
# 这 4 个文件官方是一次 --include 全带上的，保持一致：它们之间没有 -k 差异。
run_case "" \
    distributed/test_dynamo_distributed \
    distributed/test_inductor_collectives \
    distributed/test_aten_comm_compute_reordering \
    distributed/test_compute_comm_reordering

# 不使用 --upload-artifacts-while-running（h100-distributed.yml 里有）：那是官方 S3 上传路径，
# 自建集群上没有对应的 bucket 与凭证。

# -----------------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------------
if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== 失败清单（${#FAILED_CASES[@]} 条）==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[distributed] ${#FAILED_CASES[@]} 条用例失败，详见上方各 [case] 段落的日志"
    exit 1
fi

echo "[distributed] 完成，全部通过"
