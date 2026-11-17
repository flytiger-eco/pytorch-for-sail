#!/usr/bin/env python3
"""CUDA compiler launcher for PPU elementwise architecture pruning.

dispatch.cuh instantiates the specialized kernels only for sm89, but the PPU
SDK splitter derives host registrations from the first device pass (ppu001).
For dual-target builds this launcher replays the nvcc stages with the sm89
registration context repaired, so pruning never drops sm89 host registrations.

Routing (see launch):
  - non-compilation phases and single-target builds pass through unchanged;
  - dual-target builds with ccache re-enter this script on cache misses;
  - every other dual-target compile runs the staged replay (run_staged).
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


SUPPORTED_ARCHES = frozenset(('80a', '89'))
NON_COMPILE_FLAGS = ('-E', '-M', '-MM', '--preprocess', '--dryrun', '-dlink', '--device-link')
DEPENDENCY_FLAGS = ('-MD', '-MMD', '--generate-dependencies-with-compile')


# ---------------------------------------------------------------------------
# Command inspection
# ---------------------------------------------------------------------------

def expand_response(args):
    """Inline @response files so every flag is visible to the checks below."""
    result = []
    for arg in args:
        if arg.startswith('@'):
            result.extend(expand_response(shlex.split(Path(arg[1:]).read_text())))
        else:
            result.append(arg)
    return result


def is_compile(args):
    return ('-c' in args or '--compile' in args) and not any(
        flag in args for flag in NON_COMPILE_FLAGS)


def requested_arches(args):
    arches = set(re.findall(r'arch=compute_(\w+)', ' '.join(args)))
    # PPU compute_80 expands to both targets; leave the compiler arguments unchanged.
    if '80' in arches:
        arches.remove('80')
        arches.update(SUPPORTED_ARCHES)
    return arches


def option(args, name):
    return args[args.index(name) + 1] if name in args else None


def ccache_supported(flags):
    """ccache can wrap the replay only with depfile-based keys and a space-free script path."""
    return (not any(char.isspace() for char in str(Path(__file__).resolve()))
            and any(flag in flags for flag in DEPENDENCY_FLAGS))


# ---------------------------------------------------------------------------
# SDK dry-run plan: validate the stage layout, repair the sm89 context
# ---------------------------------------------------------------------------

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
    if not any('-split-host-code' in args for kind, args in plan if kind == 'command'):
        raise RuntimeError('Missing PPU host splitter')
    if not any(Path(args[0]).name == 'hgfatbinary' for kind, args in plan if kind == 'command'):
        raise RuntimeError('Missing PPU fatbinary stage')

    # Registrations follow the first device pass; point the sm89 pass at its
    # own architecture so the pruned ppu001 pass cannot drop them.
    sm89 = devices[1]
    if '-ppu-arch-list' not in sm89:
        raise RuntimeError('PPU SDK is missing the independent registration-context option')
    sm89[sm89.index('-ppu-arch-list') + 1] = 'ppu0015'
    return plan


# ---------------------------------------------------------------------------
# Staged replay
# ---------------------------------------------------------------------------

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

        # Both device passes share one .d file; snapshot each dependency set
        # before the next pass overwrites it, then merge them per destination.
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


def _run_with_ccache(args):
    # Keep nvcc as the cache compiler; cache misses re-enter this script (see
    # main) and take the staged path. Host preprocessing misses device-only
    # headers, so the cache key must come from the actual depfiles.
    env = dict(os.environ)
    script = str(Path(__file__).resolve())
    env['CCACHE_DEPEND'] = '1'
    env['CCACHE_PREFIX'] = script
    env['PPU_ELEMENTWISE_CACHE_MISS'] = '1'
    extras = env.get('CCACHE_EXTRAFILES', '')
    env['CCACHE_EXTRAFILES'] = script + (os.pathsep + extras if extras else '')
    return subprocess.call(args, env=env)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def launch(args, inner=False):
    args = expand_response(args)
    if not args:
        raise RuntimeError('No CUDA compilation command provided')

    # 1. Only real compilations need the registration fix.
    if not is_compile(args):
        return subprocess.call(args)

    # 2. The compiler must be nvcc, optionally wrapped by ccache. `inner` is
    #    the ccache-miss re-entry, which always starts at nvcc directly.
    cached = not inner and Path(args[0]).name == 'ccache'
    compiler_index = 1 if cached else 0
    if compiler_index >= len(args) or Path(args[compiler_index]).name != 'nvcc':
        raise RuntimeError('PPU elementwise compilation requires nvcc, optionally preceded by ccache')
    flags = args[compiler_index + 1:]

    # 3. Pruning is unconditional in dispatch.cuh, so unsupported targets must
    #    fail loudly instead of silently producing objects without registrations.
    arches = requested_arches(flags)
    if not arches or not arches <= SUPPORTED_ARCHES:
        raise RuntimeError('PPU elementwise compilation requires explicit compute_80a, compute_89, or compute_80 targets')

    # 4. Single-target builds need no fix: the only device pass generates the
    #    registrations (sm89), or there are no specializations at all (80a).
    if arches != SUPPORTED_ARCHES:
        return subprocess.call(args)

    # 5. Dual-target builds must repair the sm89 registration context.
    if inner:
        return run_staged(args)
    if cached and os.environ.get('CCACHE_PREFIX'):
        raise RuntimeError('PPU elementwise dual-target compilation requires its own CCACHE_PREFIX; unset the existing prefix')
    if cached and ccache_supported(flags):
        return _run_with_ccache(args)
    return run_staged(args[compiler_index:])


def main():
    # CMake invokes: python launcher.py -- <original compiler command>.
    # ccache invokes the prefix as: launcher.py <nvcc command> (no '--').
    inner = os.environ.get('PPU_ELEMENTWISE_CACHE_MISS') == '1' and '--' not in sys.argv
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    try:
        return launch(args, inner)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'PPU elementwise launcher: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
