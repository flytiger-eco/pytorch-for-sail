#!/usr/bin/env python3
"""在 PPU 双架构编译中裁剪 sm80a 特化，并保留 sm89 的完整 host 注册。"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


PRUNE_DEFINE = '-DPPU_ELEMENTWISE_SM89_ONLY=1'


def expand_response(args):
    result = []
    for arg in args:
        if arg.startswith('@'):
            result.extend(expand_response(shlex.split(Path(arg[1:]).read_text())))
        else:
            result.append(arg)
    return result


def requested_arches(args):
    return set(re.findall(r'arch=compute_(\w+)', ' '.join(args)))


def is_compile(args):
    return ('-c' in args or '--compile' in args) and not any(
        flag in args for flag in ('-E', '-M', '-MM', '--preprocess', '--dryrun',
                                  '-dlink', '--device-link'))


def option(args, name):
    return args[args.index(name) + 1] if name in args else None


def parse_plan(text):
    plan = []
    for line in text.splitlines():
        if not line.startswith('#$ '):
            continue
        payload = line[3:]
        if payload.startswith('-- Merge Dependencies -- '):
            inputs, output = payload[len('-- Merge Dependencies -- '):].split(' --> ')
            plan.append(('dependencies', (shlex.split(inputs), shlex.split(output)[0])))
            continue
        # dryrun 赋值不是命令；任何未知的实际命令都拒绝执行。
        if re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', payload):
            continue
        args = shlex.split(payload)
        if not args or not (Path(args[0]).is_absolute() or Path(args[0]).name in ('g++', 'gcc')):
            raise RuntimeError(f'未知的 PPU dryrun 阶段: {payload}')
        plan.append(('command', args))
    devices = [args for kind, args in plan if kind == 'command' and '-fcuda-is-device' in args]
    if [option(args, '-target-cpu') for args in devices] != ['ppu001', 'ppu0015']:
        raise RuntimeError('PPU SDK 设备编译计划已变化，拒绝生成缺少注册的对象')
    if not any(kind == 'command' and '-split-host-code' in args for kind, args in plan):
        raise RuntimeError('缺少 PPU host splitter')
    if not any(kind == 'command' and Path(args[0]).name == 'hgfatbinary' for kind, args in plan):
        raise RuntimeError('缺少 PPU fatbinary 阶段')
    sm89 = devices[1]
    if '-ppu-arch-list' not in sm89:
        raise RuntimeError('PPU SDK 缺少独立注册上下文参数')
    sm89[sm89.index('-ppu-arch-list') + 1] = 'ppu0015'
    # SDK 两种设备预处理默认共用一个 .d；必须在覆盖前保留每份依赖。
    return plan


def run_staged(command):
    # 每个 TU 使用独立临时目录，兼容同名源文件并行编译。
    with tempfile.TemporaryDirectory(prefix='ppu_elementwise_') as work:
        args = list(command)
        if not any(arg == '--keep-dir' or arg.startswith('--keep-dir=') for arg in args):
            args += ['--keep', '--keep-dir', work]
        dry = subprocess.run(args + ['--dryrun'], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        if dry.returncode:
            sys.stderr.write(dry.stderr)
            return dry.returncode
        plan = parse_plan(dry.stdout + '\n' + dry.stderr)
        dependencies = {}
        for kind, stage in plan:
            if kind == 'dependencies':
                sources, destination = stage
                contents = []
                for source in dict.fromkeys(sources):
                    contents.extend(dependencies[source] if source in dependencies
                                    else [Path(source).read_text()])
                # 多个相同 target 的 make 规则可由 Ninja/CMake 正常合并。
                Path(destination).write_text('\n'.join(dict.fromkeys(contents)))
                continue
            status = subprocess.run(stage)
            if status.returncode:
                return status.returncode
            dependency = option(stage, '-MF')
            if dependency:
                dependencies.setdefault(dependency, []).append(Path(dependency).read_text())
        return 0


def launch(args, inner=False):
    args = expand_response(args)
    if not args:
        raise RuntimeError('未提供 CUDA 编译命令')
    if inner:
        if requested_arches(args) == {'80a', '89'} and is_compile(args):
            return run_staged(args)
        return subprocess.call(args)
    cached = Path(args[0]).name == 'ccache'
    compiler_index = 1 if cached else 0
    if compiler_index >= len(args) or Path(args[compiler_index]).name != 'nvcc':
        # 自定义 launcher 无法可靠插入时保留原双架构生成，不启用头文件裁剪。
        return subprocess.call(args)
    flags = args[compiler_index + 1:]
    arches = requested_arches(flags)
    if not is_compile(flags) or not arches or not arches <= {'80a', '89'}:
        return subprocess.call(args)
    if cached and os.environ.get('CCACHE_PREFIX') and arches == {'80a', '89'}:
        # 不覆盖用户已有 distcc 等 prefix；该路径不启用裁剪。
        return subprocess.call(args)
    args.append(PRUNE_DEFINE)
    if arches != {'80a', '89'}:
        return subprocess.call(args)
    if cached and not any(char.isspace() for char in str(Path(__file__).resolve())) and any(
            flag in flags for flag in ('-MD', '-MMD', '--generate-dependencies-with-compile')):
        env = dict(os.environ)
        # host 预处理不能代表条件设备头；依赖模式以实际三阶段 depfile 为缓存键。
        env['CCACHE_DEPEND'] = '1'
        script = str(Path(__file__).resolve())
        env['CCACHE_PREFIX'] = script
        env['PPU_ELEMENTWISE_CACHE_MISS'] = '1'
        extras = env.get('CCACHE_EXTRAFILES', '')
        env['CCACHE_EXTRAFILES'] = script + (os.pathsep + extras if extras else '')
        # nvcc 仍是 ccache 的 compiler；prefix 只在缓存未命中时执行分阶段编译。
        return subprocess.call(args, env=env)
    return run_staged(args[compiler_index:])


def main():
    inner = os.environ.get('PPU_ELEMENTWISE_CACHE_MISS') == '1' and '--' not in sys.argv
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    try:
        return launch(args, inner)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'PPU elementwise launcher: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
