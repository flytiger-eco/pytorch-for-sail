#!/usr/bin/env python3
"""读取 FFA 独立库构建的既有清单和参数，不导入 torch、不执行编译。"""
import importlib.util
import json
import os
from pathlib import Path
import sys


def metadata(module_dir):
    # 清单来自子模块，禁止在父仓复制一份会随版本漂移的 TU/参数表。
    module_dir = Path(module_dir).resolve()
    os.environ.pop('PPU_SDK', None)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        'ffa_library_build', module_dir / 'build_lib.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    sources = [str(Path(source).resolve()) for source in build.sources()]
    if not sources or any(not Path(source).is_file() for source in sources):
        raise RuntimeError('FFA 源文件清单为空或包含不存在的文件')
    return {
        'sources': sources,
        'includes': build.kernel_include_dirs(),
        'cuda_flags': build.NVCC_FLAGS + build.GENCODE,
        'cxx_flags': build.CXX_FLAGS,
    }


if __name__ == '__main__':
    print(json.dumps(metadata(sys.argv[1])))
