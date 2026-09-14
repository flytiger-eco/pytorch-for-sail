#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：把测试框架依赖对齐到 PyTorch 官方 CI 跑 test case 的版本。
#
# 为什么单独抽一个脚本：smoke_test.sh / accuracy_test.sh 都需要这段逻辑，而
# "pip 源在 PPU pod 内到底通不通" 是这套自建集群里最容易出问题的一环，需要带
# 诊断信息和多源回退，内联进两个脚本会重复且难维护。
#
# 版本以 .ci/docker/requirements-ci.txt 为唯一来源（官方 CI 镜像用的就是它），
# 由本脚本从该文件里解析出来，不在这里另写一份版本号，避免两处漂移。
# 但只取下面 WANTED 里的测试框架包，不执行 run_test.py 报错提示里的整份
# `pip install -r .ci/docker/requirements-ci.txt`：那个文件还钉了 numpy/sympy/onnx
# 等几十个版本，会重装、降级镜像里预装 torch 所依赖的包，有把 PPU torch 环境搞坏的风险。
#
# pytest 自身也钉（官方 CI 是 7.3.2，py3.12 配置同样跑这个版本）：镜像自带的是
# pytest 8.x，与上游测试代码预设的环境不一致（例如 test/conftest.py 的
# pytest_pycollect_makemodule 曾因 pytest 8 从 hookspec 移除 path 参数而在收集阶段
# PluginValidationError）。钉到官方版本后，测试行为与上游 CI 保持一致。
#
# 依赖环境变量：
#   PIP_INDEX           - 首选 pip 源（可选；不设则从下面的候选列表开始试）
#   PIP_INDEX_FALLBACKS - 空格分隔的备用源（可选；覆盖内置候选列表）
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$REPO_ROOT/.ci/docker/requirements-ci.txt"

# 需要对齐的包。pytest 及其后三个插件是 run_test.py 跑得起来的前提（其中
# pytest-xdist / pytest-flakefinder / pytest-rerunfailures 被 check_pip_packages() 硬校验，
# 缺一个直接 exit 1）；pytest-subtests / expecttest / hypothesis 是
# torch.testing._internal 与大量用例真正 import 的测试工具，钉住可避免行为与官方 CI
# 不一致（hypothesis 的 pin 在官方就是为了压 flakiness）。
# 这些包都只在测试期使用，不是 torch 的运行时依赖，因此降级 pytest 不会影响镜像里的 torch。
#
# pyyaml 也必须在这里补，尽管 requirements-ci.txt 把它归在「To build PyTorch itself」下：
# test/test_custom_ops.py、test/test_meta.py、test/test_namedtuple_return_api.py 都在模块
# 顶层 import yaml（读 native_functions.yaml / derivatives.yaml 校验签名与 meta 注册），缺了
# 是 collect 阶段 ModuleNotFoundError —— 整个文件 fail，而不是像 scipy / numba 那类可选依赖
# 走 @skipIfNoXxx 静默 skip。pod 里的 torch 是 whl 装上的、不跑 torchgen，所以 PPU 基础镜像
# 并不保证带 pyyaml，不能指望「能编 torch 的环境一定有」。
#
# 注意这里放的是包名，版本由下面从 requirements-ci.txt 解析（写成 pyyaml==6.0.3 会让 norm()
# 拿整个字符串去查表、匹配不到而直接报「找不到版本约束」退出）。
WANTED=(pytest pytest-xdist pytest-flakefinder pytest-rerunfailures pytest-subtests expecttest hypothesis pyyaml)

# -----------------------------------------------------------------------------
# 1) 从 requirements-ci.txt 解析版本约束
# -----------------------------------------------------------------------------
SPECS_RAW="$(python - "$REQ_FILE" "${WANTED[@]}" <<'PY'
import re
import sys

req_file, wanted = sys.argv[1], sys.argv[2:]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# 行形如 "pytest==7.3.2" / "pytest-rerunfailures>=10.3"；注释行以 # 开头，直接跳过。
# 带环境标记（如 numpy 的 ; python_version == "3.12"）的包不在 WANTED 里，无需处理。
found: dict[str, str] = {}
with open(req_file) as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*((?:==|>=|<=|~=|!=|<|>).+)$", line)
        if m and norm(m.group(1)) not in found:
            found[norm(m.group(1))] = m.group(1) + m.group(2).replace(" ", "")

missing = [p for p in wanted if norm(p) not in found]
if missing:
    sys.exit(f"[deps] requirements-ci.txt 里找不到版本约束: {' '.join(missing)}")
print(" ".join(found[norm(p)] for p in wanted))
PY
)"
read -r -a PKGS <<<"$SPECS_RAW"
echo "=== 目标版本（来自 .ci/docker/requirements-ci.txt） ==="
printf '  %s\n' "${PKGS[@]}"

# -----------------------------------------------------------------------------
# 2) 已满足则直接跳过：镜像可能已预装对版本，跳过能省掉一次外网往返。
# 注意判据是「版本满足约束」而不是「包存在」：镜像自带 pytest 8.x 时必须走安装
# 把它降到 7.3.2，只判存在会把这种情况漏掉。
# -----------------------------------------------------------------------------
echo "=== 测试框架依赖自检 ==="
if python - "${PKGS[@]}" <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version


def key(v: str) -> tuple[int, ...]:
    # 只比数字段（7.3.2 -> (7,3,2)）：这几个包的约束都是 == / >=，够用且不依赖 packaging
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


unsatisfied = []
for spec in sys.argv[1:]:
    m = re.match(r"^([A-Za-z0-9._-]+)\s*(==|>=|<=|~=|<|>)\s*(.+)$", spec)
    if not m:
        unsatisfied.append(spec)
        continue
    name, op, want = m.groups()
    try:
        got = version(name)
    except PackageNotFoundError:
        print(f"  {name} 未安装 (需要 {op}{want})")
        unsatisfied.append(spec)
        continue
    a, b = pad(key(got), key(want))
    ok = {
        "==": a == b,
        ">=": a >= b,
        "<=": a <= b,
        ">": a > b,
        "<": a < b,
        "~=": a >= b and a[: len(b) - 1] == b[: len(b) - 1],
    }[op]
    print(f"  {name} {got} (需要 {op}{want}) {'OK' if ok else '不满足'}")
    if not ok:
        unsatisfied.append(spec)
raise SystemExit(1 if unsatisfied else 0)
PY
then
    echo "[deps] 版本均已满足，跳过安装"
    exit 0
fi

# -----------------------------------------------------------------------------
# 3) 候选源列表
# 候选源列表与去重逻辑抽到 .ci/ppu/pip_sources.sh，install_wheel.sh 共用同一份，
# 避免两处漂移（历史教训：两边候选表不一致，install_wheel.sh 在首选源偶发失败时
# 没有回退，先它失败、轮不到这里）。仓库为什么要逐个试、VIRTUAL/REMOTE 的背景
# 都记在 pip_sources.sh 的头注释里。
# -----------------------------------------------------------------------------
# shellcheck source=.ci/ppu/pip_sources.sh
source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates
CANDIDATES=("${PIP_CANDIDATES[@]}")

# -----------------------------------------------------------------------------
# 4) 连通性诊断：pip 只会吐一句 "from versions: none"，分不清「仓库不存在(404)」、
# 「DNS 不通」和「被代理拦截」，所以先把这三类信息各自打出来。
# 用 python 而不是 curl：curl 未必装在镜像里，python 一定有。
# -----------------------------------------------------------------------------
echo "=== pip 源连通性诊断 ==="
echo "[probe] proxy 环境变量: $(env | grep -iE '^(http_proxy|https_proxy|no_proxy|HTTP_PROXY|HTTPS_PROXY|NO_PROXY)=' | tr '\n' ' ' || echo '<无>')"
echo "[probe] pip 配置:"
python -m pip config list 2>&1 | sed 's/^/        /' || true
python - "${CANDIDATES[@]}" <<'PY' || true
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

dns_cache: dict[str, str] = {}
for index in sys.argv[1:]:
    host = urlsplit(index).hostname or "?"
    if host not in dns_cache:
        try:
            dns_cache[host] = ", ".join(socket.gethostbyname_ex(host)[2])
        except OSError as exc:
            dns_cache[host] = f"解析失败 ({exc})"
    print(f"[probe] {index}")
    print(f"        dns({host}) = {dns_cache[host]}")
    url = index.rstrip("/") + "/pytest-rerunfailures/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            print(f"        GET .../pytest-rerunfailures/ = {resp.status}")
    except urllib.error.HTTPError as exc:
        # 404 = 该仓库/包在这个 Artifactory 视图里不存在；401/403 = 需要认证
        print(f"        GET .../pytest-rerunfailures/ = {exc.code} {exc.reason}")
    except Exception as exc:
        print(f"        GET .../pytest-rerunfailures/ = 不可达 ({type(exc).__name__}: {exc})")
PY

# -----------------------------------------------------------------------------
# 5) 逐个源尝试安装，第一个成功即返回。
# --retries 1 --timeout 20：源不通时快速失败，否则 6 个候选会把 job 拖到超时。
# -----------------------------------------------------------------------------
echo "=== 安装测试框架依赖 ==="
for index in "${CANDIDATES[@]}" __pip_default__; do
    pip_args=(--disable-pip-version-check --retries 1 --timeout 20)
    if [[ "${index}" == "__pip_default__" ]]; then
        label="pip 默认源"
    else
        label="${index}"
        pip_args+=(-i "${index}")
    fi
    echo "[deps] 尝试源: ${label}"
    if python -m pip install "${pip_args[@]}" "${PKGS[@]}"; then
        echo "[deps] 安装成功（源: ${label}）"
        exit 0
    fi
    echo "[deps][warn] 源 ${label} 失败，换下一个"
done

cat >&2 <<'EOF'
[deps][error] 所有候选 pip 源都装不上上面列出的测试框架依赖。
请对照上面「pip 源连通性诊断」的输出判断：
  - 全是 404      -> pod 内的 Artifactory 视图没有这些 pypi 仓库，需要运维确认 pod 侧该用哪个仓库名，
                     确认后把 workflow 里的 PIP_INDEX 改掉（或用 PIP_INDEX_FALLBACKS 追加候选）。
  - 401/403       -> 该源需要认证，需要在 pod 内提供凭证（pip.conf / .netrc）。
  - DNS 解析失败  -> pod 未挂到能解析该域名的 DNS。
  - 公网源不可达  -> pod 无外网出口，只能走内网源。
兜底办法：让镜像按 .ci/docker/requirements-ci.txt 的版本预装这些包，本脚本检测到版本已满足会直接跳过。
EOF
exit 1