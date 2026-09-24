#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


EXTRA_CPU_ONLY_CLASSES = (
    "SDPAPatternRewriterCpuDynamicTests",
    "TestSDPACpuOnly",
    "TestAutocastCPU",
)

COPY_TESTS_RE = re.compile(
    r"""copy_tests\(\s*[\w.]+\s*,\s*(\w+)\s*,\s*['"]cpu['"]""", re.S
)
CONVENTION_RE = re.compile(
    r"^\s*class\s+(\w*(?:(?:Cpu|CPU)Test|Test(?:Cpu|CPU))\w*)\s*[(:]", re.M
)
ANY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)\s*[(:]", re.M)


def collect(test_dir: Path) -> tuple[set[str], set[str]]:
    cpu_only: set[str] = set()
    declared: set[str] = set()
    for path in sorted(test_dir.rglob("*.py")):
        # errors="ignore"：test/ 下有少量刻意构造的非 UTF-8 / 二进制样例文件
        src = path.read_text(encoding="utf-8", errors="ignore")
        cpu_only.update(COPY_TESTS_RE.findall(src))
        cpu_only.update(CONVENTION_RE.findall(src))
        declared.update(ANY_CLASS_RE.findall(src))
    return cpu_only, declared


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"用法: {argv[0]} <test 目录>", file=sys.stderr)
        return 2

    test_dir = Path(argv[1])
    if not test_dir.is_dir():
        print(f"[cuda-only] 目录不存在: {test_dir}", file=sys.stderr)
        return 1

    cpu_only, declared = collect(test_dir)

    stale = sorted(name for name in EXTRA_CPU_ONLY_CLASSES if name not in declared)
    if stale:
        print(
            "[cuda-only] EXTRA_CPU_ONLY_CLASSES 已过期，请修正 "
            ".ci/ppu/list_cpu_only_test_classes.py：\n"
            "  以下类名在 test/ 下已不存在（被上游改名或删除了）: " + " ".join(stale),
            file=sys.stderr,
        )
        return 1
    cpu_only.update(EXTRA_CPU_ONLY_CLASSES)

    if not cpu_only:
        print(
            '[cuda-only] 没有从 test/ 扫到任何 CPU 专属测试类：copy_tests(..., "cpu") '
            "与 *CpuTest* 命名约定可能都被上游改掉了，\n"
            "  请修正 .ci/ppu/list_cpu_only_test_classes.py",
            file=sys.stderr,
        )
        return 1

    minimal = sorted(
        name
        for name in cpu_only
        if not any(other != name and other in name for other in cpu_only)
    )
    print(" and ".join(f"not {name}" for name in minimal))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
