#!/usr/bin/env bash
set -euo pipefail

export SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[full_ut] 源码目录: $(pwd)"

UT_FILES="${UT_FILES:-.ci/ppu/badcase_list/ut_files.txt}"
NAS_SKIP_FILE="${NAS_SKIP_FILE:-/mnt/wl_nas/ut_test/pytorch/full_ut_skip.txt}"
LOCAL_SKIP_FILE=".ci/ppu/badcase_list/full_ut_skip.txt"
CRASH_LIST="${CRASH_LIST:-$LOCAL_SKIP_FILE}"
SHARD_NUMBER="${SHARD_NUMBER:-1}"
NUM_TEST_SHARDS="${NUM_TEST_SHARDS:-1}"

if [[ ! -f "$NAS_SKIP_FILE" || ! -r "$NAS_SKIP_FILE" ]]; then
    echo "::error::[full_ut] 网盘 skip 清单不存在或不可读: ${NAS_SKIP_FILE}" >&2
    exit 1
fi
if ! cp -- "$NAS_SKIP_FILE" "$LOCAL_SKIP_FILE"; then
    echo "::error::[full_ut] 复制网盘 skip 清单失败: ${NAS_SKIP_FILE} -> ${LOCAL_SKIP_FILE}" >&2
    exit 1
fi
echo "[full_ut] 已复制网盘 skip 清单: ${NAS_SKIP_FILE} -> ${LOCAL_SKIP_FILE}"

TEST_FILES_RAW="$(python - "$UT_FILES" "$CRASH_LIST" "$SHARD_NUMBER" "$NUM_TEST_SHARDS" <<'PY'
import re
import sys

ut_files, skip_file, shard_number, num_shards = sys.argv[1:]
if not all(re.fullmatch(r"[1-9][0-9]*", value) for value in (shard_number, num_shards)):
    sys.exit("[full_ut] SHARD_NUMBER 和 NUM_TEST_SHARDS 必须为正整数")
shard_number, num_shards = int(shard_number), int(num_shards)
if shard_number > num_shards:
    sys.exit("[full_ut] SHARD_NUMBER 不能大于 NUM_TEST_SHARDS")

with open(ut_files) as f:
    test_files = [text for line in f if (text := line.split("#", 1)[0].strip())]
if not test_files:
    sys.exit("[full_ut] 测试文件清单为空")

with open(skip_file) as f:
    # 整文件跳过应发生在收集前，避免导入错误或零用例被执行器判为失败。
    skipped_files = {
        text for line in f
        if (text := line.partition("\t")[0].strip())
        and not text.startswith("#") and "::" not in text and text.endswith(".py")
    }
selected_files = []
for test_file in test_files:
    if test_file in skipped_files:
        print(f"=== [skip] {test_file}（整文件）===", file=sys.stderr)
    else:
        selected_files.append(test_file)

for index, test_file in enumerate(selected_files):
    if index % num_shards == shard_number - 1:
        print(test_file)
PY
)"

source .ci/ppu/sdk_env.sh
source .ci/ppu/nccl_env.sh
bash .ci/ppu/install_wheel.sh
bash .ci/ppu/install_test_deps.sh

FAILED_CASES=()
run_pytest() {
    local test_file="$1"
    echo "=== [case] ${test_file}  -k cuda ==="
    if python .ci/ppu/multi_run_case.py "$test_file" --crash-list "$CRASH_LIST" --allow-empty -k "cuda"; then
        echo "=== [done] ${test_file}（通过或正常跳过）==="
    else
        echo "::error::[fail] ${test_file}"
        FAILED_CASES+=("${test_file}")
    fi
}

echo "=== 全量 UT 用例（分片 ${SHARD_NUMBER}/${NUM_TEST_SHARDS}）==="
TEST_FILE_COUNT=0
if [[ -n "$TEST_FILES_RAW" ]]; then
    while IFS= read -r test_file; do
        TEST_FILE_COUNT=$((TEST_FILE_COUNT + 1))
        run_pytest "$test_file"
    done <<<"$TEST_FILES_RAW"
fi

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    echo "=== 失败清单（${#FAILED_CASES[@]} 个文件）==="
    for case_label in "${FAILED_CASES[@]}"; do
        echo "  - ${case_label}"
    done
    echo "::error::[full_ut] ${#FAILED_CASES[@]} 个测试文件失败，详见上方各 [case] 段落的日志"
    exit 1
fi

echo "[full_ut] 完成，分片 ${SHARD_NUMBER}/${NUM_TEST_SHARDS} 的 ${TEST_FILE_COUNT} 个测试文件全部通过或正常跳过"