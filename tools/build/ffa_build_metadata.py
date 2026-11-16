#!/usr/bin/env python3
"""Read the existing FFA library manifest and flags without importing torch or compiling."""
import importlib.util
import json
import os
from pathlib import Path
import sys


def metadata(module_dir):
    # Read the submodule manifest instead of maintaining a duplicate TU/flag table in the parent repository.
    module_dir = Path(module_dir).resolve()
    os.environ.pop('PPU_SDK', None)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        'ffa_library_build', module_dir / 'build_lib.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    sources = [str(Path(source).resolve()) for source in build.sources()]
    if not sources or any(not Path(source).is_file() for source in sources):
        raise RuntimeError('FFA source manifest is empty or contains nonexistent files')
    return {
        'sources': sources,
        'includes': build.kernel_include_dirs(),
        'cuda_flags': build.NVCC_FLAGS + build.GENCODE,
        'cxx_flags': build.CXX_FLAGS,
    }


if __name__ == '__main__':
    print(json.dumps(metadata(sys.argv[1])))
