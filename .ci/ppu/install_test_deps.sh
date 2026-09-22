#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$REPO_ROOT/.ci/docker/requirements-ci.txt"
WANTED=(pytest pytest-xdist pytest-flakefinder pytest-rerunfailures pytest-subtests expecttest hypothesis pyyaml)
SPECS_RAW="$(python - "$REQ_FILE" "${WANTED[@]}" <<'PY'
import re
import sys

req_file, wanted = sys.argv[1], sys.argv[2:]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

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

source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates
CANDIDATES=("${PIP_CANDIDATES[@]}")

echo "=== pip 源连通性诊断 ==="
echo "[probe] 已设置的 proxy 环境变量（仅列名称，不打印取值，避免泄漏内嵌凭证）: $(
    _set_proxy_names=""
    for _pv in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
        [[ -n "${!_pv:-}" ]] && _set_proxy_names="${_set_proxy_names}${_pv} "
    done
    printf '%s' "${_set_proxy_names:-<无>}"
)"
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