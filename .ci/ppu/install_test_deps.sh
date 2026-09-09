#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：安装 test/run_test.py 硬校验的 pytest 插件。
#
# 为什么单独抽一个脚本：smoke_test.sh / accuracy_test.sh 都需要这段逻辑，而
# "pip 源在 PPU pod 内到底通不通" 是这套自建集群里最容易出问题的一环，需要带
# 诊断信息和多源回退，内联进两个脚本会重复且难维护。
#
# 只装 run_test.py 的 check_pip_packages() 真正要求的三个插件，不执行报错提示里的
# `pip install -r .ci/docker/requirements-ci.txt`：那个文件钉了 numpy/sympy/onnx 等
# 几十个版本，会重装、降级镜像里预装 torch 所依赖的包，有把 PPU torch 环境搞坏的风险。
# 三个插件的版本与 .ci/docker/requirements-ci.txt 保持一致；不钉 pytest 自身版本，
# 避免降级镜像自带的 pytest。
#
# 依赖环境变量：
#   PIP_INDEX           - 首选 pip 源（可选；不设则从下面的候选列表开始试）
#   PIP_INDEX_FALLBACKS - 空格分隔的备用源（可选；覆盖内置候选列表）
# =============================================================================
set -euo pipefail

PKGS=("pytest-rerunfailures>=10.3" "pytest-flakefinder==1.1.0" "pytest-xdist==3.3.1")

# -----------------------------------------------------------------------------
# 1) 已装则直接跳过：镜像有可能预装，跳过能省掉一次外网往返
# -----------------------------------------------------------------------------
echo "=== 测试框架依赖自检 ==="
if python -c '
from importlib.metadata import PackageNotFoundError, version

missing = []
for p in ("pytest", "pytest-rerunfailures", "pytest-flakefinder", "pytest-xdist"):
    try:
        print(f"  {p} {version(p)}")
    except PackageNotFoundError:
        print(f"  {p} 未安装")
        if p != "pytest":  # pytest 只作参考信息，缺插件才需要装
            missing.append(p)
raise SystemExit(1 if missing else 0)
'; then
    echo "[deps] 三个 pytest 插件均已存在，跳过安装"
    exit 0
fi

# -----------------------------------------------------------------------------
# 2) 候选源列表
# 背景：曾把仓库名写成 pypiindex（正确是 pypi_index），Artifactory 对不存在的仓库
# 一律返回 404，而 `-i` 又替换掉了默认源，pip 只会报 "(from versions: none)"，
# 看起来像版本不匹配、实际是整个索引取不到。改对仓库名后 pod 内仍是同样报错，
# 说明 pod 内看到的 Artifactory 视图与公网侧不一致，因此这里逐个试而不是钉死一个。
# pypi_index 是 VIRTUAL 仓，聚合了 pypi_formal / pypi_aliyun / pypi_huawei /
# pypi_tsinghua 这几个 REMOTE 代理，单独指某个 REMOTE 仓可以绕开虚拟仓的解析问题。
# -----------------------------------------------------------------------------
ARTIFACTORY=https://pkg.flytiger-eco.com/artifactory/api/pypi
DEFAULT_FALLBACKS=(
    "${ARTIFACTORY}/pypi_index/simple"
    "${ARTIFACTORY}/pypi_formal/simple"
    "${ARTIFACTORY}/pypi_aliyun/simple"
    "${ARTIFACTORY}/pypi_tsinghua/simple"
    https://pypi.tuna.tsinghua.edu.cn/simple
    https://mirrors.aliyun.com/pypi/simple
)

CANDIDATES=()
if [[ -n "${PIP_INDEX:-}" ]]; then
    CANDIDATES+=("${PIP_INDEX}")
fi
if [[ -n "${PIP_INDEX_FALLBACKS:-}" ]]; then
    read -r -a _extra <<<"${PIP_INDEX_FALLBACKS}"
    CANDIDATES+=("${_extra[@]}")
else
    CANDIDATES+=("${DEFAULT_FALLBACKS[@]}")
fi
# PIP_INDEX 常常就等于候选表里的第一项，去重避免白试一遍
_seen=""
_uniq=()
for idx in "${CANDIDATES[@]}"; do
    if [[ "${_seen}" != *"|${idx}|"* ]]; then
        _uniq+=("${idx}")
        _seen="${_seen}|${idx}|"
    fi
done
CANDIDATES=("${_uniq[@]}")

# -----------------------------------------------------------------------------
# 3) 连通性诊断：pip 只会吐一句 "from versions: none"，分不清「仓库不存在(404)」、
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
# 4) 逐个源尝试安装，第一个成功即返回。
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
[deps][error] 所有候选 pip 源都装不上 pytest-rerunfailures / pytest-flakefinder / pytest-xdist。
请对照上面「pip 源连通性诊断」的输出判断：
  - 全是 404      -> pod 内的 Artifactory 视图没有这些 pypi 仓库，需要运维确认 pod 侧该用哪个仓库名，
                     确认后把 workflow 里的 PIP_INDEX 改掉（或用 PIP_INDEX_FALLBACKS 追加候选）。
  - 401/403       -> 该源需要认证，需要在 pod 内提供凭证（pip.conf / .netrc）。
  - DNS 解析失败  -> pod 未挂到能解析该域名的 DNS。
  - 公网源不可达  -> pod 无外网出口，只能走内网源。
兜底办法：让镜像预装这三个插件，本脚本检测到已存在会直接跳过。
EOF
exit 1
