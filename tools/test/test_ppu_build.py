import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.build import ffa_build_metadata as ffa
from tools.build import ppu_elementwise_launcher as launcher


class TestPPUElementwiseLauncher(unittest.TestCase):
    @staticmethod
    def plan():
        return '\n'.join([
            '#$ PATH=/sdk/bin',
            '#$ g++ -E source.cu -o host.hggci',
            '#$ /sdk/ppu_clang -cc1 -split-host-code host.hggci',
            '#$ /sdk/ppu_clang -cc1 -fcuda-is-device -target-cpu ppu001 -ppu-arch-list ppu0010,ppu001',
            '#$ /sdk/ppu_clang -cc1 -fcuda-is-device -target-cpu ppu0015 -ppu-arch-list ppu0010,ppu0015',
            '#$ /sdk/hgfatbinary -image3=file=80.out,mcpu=ppu001 -image3=file=89.out,mcpu=ppu0015',
            '#$ g++ host.cpp -c -o object.o',
        ])

    def test_only_registration_context_changes(self):
        plan = launcher.parse_plan(self.plan())
        self.assertEqual(plan[0][1], ['g++', '-E', 'source.cu', '-o', 'host.hggci'])
        self.assertEqual(launcher.option(plan[2][1], '-ppu-arch-list'), 'ppu0010,ppu001')
        self.assertEqual(launcher.option(plan[3][1], '-ppu-arch-list'), 'ppu0015')
        self.assertEqual(launcher.option(plan[3][1], '-target-cpu'), 'ppu0015')
        self.assertIn('-image3=file=80.out,mcpu=ppu001', plan[4][1])
        self.assertIn('-image3=file=89.out,mcpu=ppu0015', plan[4][1])

    def test_sdk_drift_fails_closed(self):
        for text in (self.plan().replace('-target-cpu ppu0015', '-target-cpu ppu002'),
                     self.plan().replace('-split-host-code', '-unrecognized'),
                     self.plan().replace('hgfatbinary', 'other_packager'),
                     self.plan() + '\n#$ unknown-shell-operation'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                launcher.parse_plan(text)

    def test_arch_profiles_and_ptx(self):
        self.assertEqual(launcher.requested_arches(['-gencode', 'arch=compute_80a,code=sm_80a']), {'80a'})
        self.assertEqual(launcher.requested_arches(['--generate-code=arch=compute_89,code=[sm_89,compute_89]']), {'89'})
        self.assertEqual(launcher.requested_arches(['arch=compute_89,code=sm_89',
                                                   'arch=compute_80a,code=sm_80a']), {'80a', '89'})

    def test_single_arch_preserves_requested_targets(self):
        for arch in ('80a', '89'):
            command = ['/sdk/nvcc', '-c', 'input.cu', '-gencode', f'arch=compute_{arch},code=sm_{arch}']
            with self.subTest(arch=arch), patch.object(launcher.subprocess, 'call', return_value=0) as call:
                self.assertEqual(launcher.launch(command), 0)
                actual = call.call_args.args[0]
                self.assertIn(launcher.PRUNE_DEFINE, actual)
                self.assertEqual(launcher.requested_arches(actual), {arch})

    def test_preprocess_and_custom_launcher_passthrough(self):
        for command in (['/sdk/nvcc', '-E', 'input.cu'],
                        ['sccache', '/sdk/nvcc', '-c', 'input.cu'],
                        ['/sdk/nvcc', '-c', 'input.cu', 'arch=compute_90,code=sm_90']):
            with self.subTest(command=command), patch.object(launcher.subprocess, 'call', return_value=0) as call:
                launcher.launch(command)
                self.assertEqual(call.call_args.args[0], command)

    def test_cache_prefix_and_actual_dependency_mode(self):
        command = ['ccache', '/sdk/nvcc', '-c', 'input.cu', '-MD',
                   'arch=compute_80a,code=sm_80a', 'arch=compute_89,code=sm_89']
        with patch.dict(os.environ, {}, clear=True), patch.object(launcher.subprocess, 'call', return_value=0) as call:
            launcher.launch(command)
            env = call.call_args.kwargs['env']
            self.assertEqual(env['CCACHE_DEPEND'], '1')
            self.assertEqual(env['PPU_ELEMENTWISE_CACHE_MISS'], '1')
            self.assertEqual(env['CCACHE_PREFIX'], str(Path(launcher.__file__).resolve()))
            self.assertEqual(env['CCACHE_EXTRAFILES'], str(Path(launcher.__file__).resolve()))
        with patch.dict(os.environ, {'CCACHE_PREFIX': 'distcc'}), patch.object(launcher.subprocess, 'call', return_value=0) as call:
            launcher.launch(command)
            self.assertEqual(call.call_args.args[0], command)

    def test_response_file_with_spaces(self):
        with tempfile.TemporaryDirectory() as work:
            response = Path(work) / 'compile arguments.rsp'
            response.write_text('-c "source with spaces.cu" -gencode arch=compute_89,code=sm_89')
            self.assertEqual(launcher.expand_response(['@' + str(response)]),
                             ['-c', 'source with spaces.cu', '-gencode', 'arch=compute_89,code=sm_89'])

    def test_dependency_snapshots_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as work:
            dep = Path(work) / 'device.d'
            output = Path(work) / 'merged.d'
            plan = [('command', ['g++', '-E', '-MF', str(dep)]) for _ in range(3)]
            plan += [('dependencies', ([str(dep), str(dep)], str(output)))]
            count = 0

            def execute(args, **kwargs):
                nonlocal count
                if '--dryrun' in args:
                    return subprocess.CompletedProcess(args, 0, stdout='', stderr='')
                count += 1
                dep.write_text(f'object.o: dependency_{count}.h\n')
                return subprocess.CompletedProcess(args, 0)

            with patch.object(launcher, 'parse_plan', return_value=plan), patch.object(launcher.subprocess, 'run', side_effect=execute):
                self.assertEqual(launcher.run_staged(['nvcc', '-c', 'input.cu']), 0)
            for index in range(1, 4):
                self.assertIn(f'dependency_{index}.h', output.read_text())
            with patch.object(launcher, 'parse_plan', return_value=[('command', ['compiler']), ('command', ['must-not-run'])]), patch.object(
                    launcher.subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 0, stdout='', stderr=''),
                                                          subprocess.CompletedProcess([], 7)]) as run:
                self.assertEqual(launcher.run_staged(['nvcc', '-c', 'input.cu']), 7)
                self.assertEqual(run.call_count, 2)


class TestFFABuildMetadata(unittest.TestCase):
    def test_sources_and_flags_have_one_truth_source(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / 'kernel.cu').write_text('// 测试清单\n')
            (root / 'build_lib.py').write_text('''from pathlib import Path
NVCC_FLAGS = ['-mllvm', 'first', '-mllvm', 'second']
GENCODE = ['-gencode', 'arch=compute_89,code=sm_89']
CXX_FLAGS = ['-DFLEX_FLASH_ATTENTION_LIBRARY_BUILD']
def sources():
    return [Path(__file__).parent / 'kernel.cu']
def kernel_include_dirs():
    import os
    assert 'PPU_SDK' not in os.environ
    return [str(Path(__file__).parent)]
''')
            with patch.dict(os.environ, {'PPU_SDK': '/unused'}):
                actual = ffa.metadata(root)
            self.assertEqual(actual['sources'], [str(root / 'kernel.cu')])
            self.assertEqual(actual['cuda_flags'], ['-mllvm', 'first', '-mllvm', 'second',
                                                   '-gencode', 'arch=compute_89,code=sm_89'])
            self.assertEqual(actual['includes'], [str(root)])
            self.assertEqual(actual['cxx_flags'], ['-DFLEX_FLASH_ATTENTION_LIBRARY_BUILD'])

    def test_missing_manifest_source_rejected(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / 'build_lib.py').write_text("def sources():\n    return ['missing-source.cu']\n")
            with self.assertRaises(RuntimeError):
                ffa.metadata(root)


if __name__ == '__main__':
    unittest.main()
