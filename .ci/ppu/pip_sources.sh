#!/usr/bin/env bash
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

    local _seen="" idx
    PIP_CANDIDATES=()
    for idx in "${candidates[@]}"; do
        if [[ "${_seen}" != *"|${idx}|"* ]]; then
            PIP_CANDIDATES+=("${idx}")
            _seen="${_seen}|${idx}|"
        fi
    done
}
