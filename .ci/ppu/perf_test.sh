#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[perf] 源码目录: $(pwd)"
source .ci/ppu/sdk_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_triton.sh

echo "=== 环境自检 (CUDA) ==="
echo "pr=${PR_NUMBER:-none} hostname=$(hostname) node=${NODE_NAME:-unknown}"
echo "rank=${RANK:-0} nproc_per_node=${NPROC_PER_NODE:-1}"
python --version
(cd /tmp && python -c "import torch; print('torch', torch.__version__, torch.__file__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())")
ppu-smi || echo "[warn] ppu-smi 不可用，请确认 pod 已分配 PPU 设备"
bash .ci/ppu/install_test_deps.sh
source .ci/ppu/cuda_only_filter.sh

TEST_REPORTS_DIR="$REPO_ROOT/test/test-reports"
mkdir -p "$TEST_REPORTS_DIR"
echo "=== [1/2] CUDA inductor 性能单测（run_test.py --include 白名单 + -k 排除 FP8/CPU） ==="
K_FP8="not fp8 and not float8 and not e4m3 and not e5m2"
K_SKIP_CASES="not test_fusion_choice4_cpu \
and not test_cat \
and not test_cat_pointwise_many_complex_inputs \
and not test_cat_pointwise_many_simple_inputs \
and not test_cat_pointwise_config_option \
and not test_cat_pointwise \
and not test_partitioning_with_view \
and not test_equivalent_template_code \
and not test_split_scan"
python test/run_test.py \
    --include \
        inductor/test_perf \
        inductor/test_benchmark_fusion \
        inductor/test_benchmarking \
        inductor/test_analysis \
    -k "$(ppu_cuda_only_k_expr "$K_FP8" "$K_SKIP_CASES")" \
    --verbose

if [[ "${RUN_GPT_FAST:-1}" != "1" ]]; then
    echo "=== [2/2] gpt_fast 微基准：已由 RUN_GPT_FAST=0 显式关闭，跳过 ==="
    echo "[perf] 完成"
    exit 0
fi

echo "=== [2/2] gpt_fast 微基准（torch.compile 端到端计时） ==="
if ! (cd /tmp && python -c "import torchao" >/dev/null 2>&1); then
    echo "[perf] torchao 不可用，尝试从候选 pip 源补装（--no-deps）"
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

GPT_FAST_CSV="$TEST_REPORTS_DIR/gpt_fast_benchmark.csv"
rm -f "$GPT_FAST_CSV" "${GPT_FAST_CSV%.csv}.json"
for experiment in mlp_layer_norm_gelu layer_norm gather_gemv gemv; do
    echo "--- gpt_fast: ${experiment} ---"
    python benchmarks/gpt_fast/benchmark.py \
        --only "${experiment}" \
        --output "$GPT_FAST_CSV"
done

echo "=== gpt_fast 微基准结果 ==="
if [[ -s "$GPT_FAST_CSV" ]]; then
    cat "$GPT_FAST_CSV"
else
    echo "[perf][warn] 没有在 ${GPT_FAST_CSV} 拿到结果：上面 4 次调用均以 0 退出，但没写出 csv。"
    echo "[perf][warn] 先看 benchmarks/gpt_fast/benchmark.py 的 --output / output_csv 语义是否被上游改过。"
fi

echo "[perf] 完成"
