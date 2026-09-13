#!/usr/bin/env bash
# =============================================================================
# PPU pod 内：pip 候选源列表的唯一来源。install_wheel.sh 与 install_test_deps.sh
# 都 source 本文件，避免两边各存一份、日后漂移。
#
# 为什么必须两边共用：install_wheel.sh（装 torch 及其运行时依赖）跑在
# install_test_deps.sh（装测试框架依赖）之前。若两边候选源不一致，install_wheel.sh
# 在首选源偶发失败时又没有回退，就会先它失败、根本轮不到后面那个带回退的脚本救场
# —— 曾表现为 smoke 侥幸命中、accuracy 没命中，"一个过一个挂"。
#
# 背景（候选源为什么要逐个试而不是钉死一个）：
#   曾把仓库名写成 pypiindex（正确是 pypi_index），Artifactory 对不存在的仓库一律
#   返回 404，而 `-i` 又替换掉了默认源，pip 只会报 "(from versions: none)"，看起来
#   像版本不匹配、实际是整个索引取不到。改对仓库名后 pod 内仍是同样报错，说明 pod
#   内看到的 Artifactory 视图与公网侧不一致。pypi_index 是 VIRTUAL 仓，聚合了
#   pypi_formal / pypi_aliyun / pypi_huawei / pypi_tsinghua 这几个 REMOTE 代理，
#   VIRTUAL 仓在 pod 内偶发把某个包解析成 "from versions: none"，单独指某个 REMOTE
#   仓可以绕开虚拟仓的解析问题，故这里从 VIRTUAL 退到各 REMOTE、再退到公网镜像。
#
# 用法：source 本文件后调用 ppu_build_pip_candidates，结果写入全局数组 PIP_CANDIDATES。
#   source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
#   ppu_build_pip_candidates
#   for index in "${PIP_CANDIDATES[@]}" __pip_default__; do ... done
#
# 依赖环境变量：
#   PIP_INDEX           - 首选 pip 源（可选；排在候选表最前）
#   PIP_INDEX_FALLBACKS - 空格分隔的备用源（可选；一旦设置则覆盖内置 DEFAULT_FALLBACKS）
# =============================================================================

# 把候选 pip 源写入全局数组 PIP_CANDIDATES（已按出现顺序去重）。
ppu_build_pip_candidates() {
    local artifactory=https://pkg.flytiger-eco.com/artifactory/api/pypi
    local default_fallbacks=(
        "${artifactory}/pypi_index/simple"
        "${artifactory}/pypi_formal/simple"
        "${artifactory}/pypi_aliyun/simple"
        "${artifactory}/pypi_tsinghua/simple"
        https://pypi.tuna.tsinghua.edu.cn/simple
        https://mirrors.aliyun.com/pypi/simple
    )

    local candidates=()
    if [[ -n "${PIP_INDEX:-}" ]]; then
        candidates+=("${PIP_INDEX}")
    fi
    if [[ -n "${PIP_INDEX_FALLBACKS:-}" ]]; then
        local _extra
        read -r -a _extra <<<"${PIP_INDEX_FALLBACKS}"
        candidates+=("${_extra[@]}")
    else
        candidates+=("${default_fallbacks[@]}")
    fi

    # PIP_INDEX 常常就等于候选表里的第一项，去重避免白试一遍
    local _seen="" idx
    PIP_CANDIDATES=()
    for idx in "${candidates[@]}"; do
        if [[ "${_seen}" != *"|${idx}|"* ]]; then
            PIP_CANDIDATES+=("${idx}")
            _seen="${_seen}|${idx}|"
        fi
    done
}
