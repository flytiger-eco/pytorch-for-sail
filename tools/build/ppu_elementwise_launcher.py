#!/usr/bin/env python3
"""Prune sm80a PPU specializations while preserving complete sm89 host registration."""
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
        # Dry-run assignments are not commands; reject unknown executable stages.
        if re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', payload):
            continue
        args = shlex.split(payload)
        if not args or not (Path(args[0]).is_absolute() or Path(args[0]).name in ('g++', 'gcc')):
            raise RuntimeError(f'Unknown PPU dry-run stage: {payload}')
        plan.append(('command', args))
    devices = [args for kind, args in plan if kind == 'command' and '-fcuda-is-device' in args]
    if [option(args, '-target-cpu') for args in devices] != ['ppu001', 'ppu0015']:
        raise RuntimeError('PPU SDK device compilation plan changed; refusing to generate objects with missing registrations')
    if not any(kind == 'command' and '-split-host-code' in args for kind, args in plan):
        raise RuntimeError('Missing PPU host splitter')
    if not any(kind == 'command' and Path(args[0]).name == 'hgfatbinary' for kind, args in plan):
        raise RuntimeError('Missing PPU fatbinary stage')
    sm89 = devices[1]
    if '-ppu-arch-list' not in sm89:
        raise RuntimeError('PPU SDK is missing the independent registration-context option')
    sm89[sm89.index('-ppu-arch-list') + 1] = 'ppu0015'
    # Both device preprocessing passes share one .d file; preserve each dependency set before it is overwritten.
    return plan


def run_staged(command):
    # Use a separate temporary directory per TU for parallel compilation of identically named sources.
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
                # Ninja/CMake can merge multiple make rules for the same target.
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
        raise RuntimeError('No CUDA compilation command provided')
    if inner:
        if requested_arches(args) == {'80a', '89'} and is_compile(args):
            return run_staged(args)
        return subprocess.call(args)
    cached = Path(args[0]).name == 'ccache'
    compiler_index = 1 if cached else 0
    if compiler_index >= len(args) or Path(args[compiler_index]).name != 'nvcc':
        # Preserve both architectures without header pruning when a custom launcher cannot be wrapped reliably.
        return subprocess.call(args)
    flags = args[compiler_index + 1:]
    arches = requested_arches(flags)
    if not is_compile(flags) or not arches or not arches <= {'80a', '89'}:
        return subprocess.call(args)
    if cached and os.environ.get('CCACHE_PREFIX') and arches == {'80a', '89'}:
        # Preserve existing prefixes such as distcc; do not enable pruning on this path.
        return subprocess.call(args)
    args.append(PRUNE_DEFINE)
    if arches != {'80a', '89'}:
        return subprocess.call(args)
    if cached and not any(char.isspace() for char in str(Path(__file__).resolve())) and any(
            flag in flags for flag in ('-MD', '-MMD', '--generate-dependencies-with-compile')):
        env = dict(os.environ)
        # Host preprocessing misses conditional device headers; use all three actual depfiles for the cache key.
        env['CCACHE_DEPEND'] = '1'
        script = str(Path(__file__).resolve())
        env['CCACHE_PREFIX'] = script
        env['PPU_ELEMENTWISE_CACHE_MISS'] = '1'
        extras = env.get('CCACHE_EXTRAFILES', '')
        env['CCACHE_EXTRAFILES'] = script + (os.pathsep + extras if extras else '')
        # Keep nvcc as the ccache compiler; the prefix executes staged compilation only on cache misses.
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
