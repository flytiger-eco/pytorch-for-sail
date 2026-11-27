"""Test native PPU cudafy names and build wiring without importing or building Torch."""

import ast
import importlib.util
import io
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CUDAFY = ROOT / "third_party/cudafy-for-sail"
KERNELS = ROOT / "aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernels"
FFA_FILES = [
    "csrc/flex_flash_attention_fwd_kernel_sm80.h",
    "csrc/flex_flash_attention_fwd_kernel_sm80_f32.h",
    "csrc/flex_flash_attention_bwd_kernel_sm80.h",
    "csrc/flex_flash_attention_bwd_kernel_sm80_f32.h",
    "hopper/utils.h",
]
FFA_COPY_FILES = [
    "hopper/pack_gqa.h", "hopper/paged_kv.h",
    "hopper/mainloop_fwd_sm80.hpp", "hopper/mainloop_bwd_sm80.hpp",
    "hopper/mainloop_fwd_sm80_f32.hpp", "hopper/mainloop_bwd_sm80_f32.hpp",
]
FFA_COPY_TYPES = {
    "SM80_CP_ASYNC_CACHEGLOBAL_ZFILL": "PPU_CP_ASYNC_CACHEGLOBAL_ZFILL",
    "SM75_U32x4_LDSM_N": "PPU_U32x4_LDSM_N",
    "SM75_U16x8_LDSM_T": "PPU_U16x8_LDSM_T",
    "SM75_U32x2_LDSM_N": "PPU_U32x2_LDSM_N",
    "SM75_U16x4_LDSM_T": "PPU_U16x4_LDSM_T",
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestCudafyNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(CUDAFY))
        cls.general = load_module("native_cudafy_general", CUDAFY / "general.py")
        cls.map_generator = load_module(
            "native_sdk_generator", CUDAFY / "tools/generate_sdk_map.py"
        )
        cls.kernels = load_module("native_fmha_generator", KERNELS / "generate_kernels.py")

    def preprocess(self, source, defines):
        """Check active branches and include paths without compiling C++."""
        cpp = shutil.which("cpp")
        if cpp is None:
            self.skipTest("cpp is required to check active branches")
        source = re.sub(
            r'(?m)^\s*#\s*include\s*[<"]([^>"\n]+)[>"][^\n]*',
            lambda match: f'NATIVE_INCLUDE("{match[1]}")',
            source,
        )
        result = subprocess.run(
            [cpp, "-E", "-P", "-dD", "-x", "c++"]
            + [f"-D{define}" for define in defines] + ["-"],
            input=source, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_architecture_and_library_names_preserved(self):
        source = """#if __HGGC_ARCH__ >= 100 && __HGGC_ARCH__ < 150
#if COMPATIBLE_ARCH >= 800
int cu_count, device_cus_, cu_occupancy_, MaxCuCount;
PPU0010 tag; PPU_CP_ASYNC_CACHEGLOBAL<int> copy;
CUTE_ARCH_CP_ASYNC_PPU_ENABLED;
CuId();
#endif
#endif
"""
        self.assertEqual(self.general.convert_text(source), source)
        for token in ("COMPATIBLE_ARCH", "__HGGC_ARCH__", "__CUDA_ARCH__"):
            self.assertIn(token, self.map_generator.NEVER_MAP)
            self.assertNotIn(token, self.general.SDK_TOKEN_MAP)
        self.assertNotIn("COMPATIBLE_ARCH", self.map_generator.SAILIFY_TOKEN_MAP)

    def test_map_regeneration_blocks_architecture_rules(self):
        generator = self.map_generator
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory)
            ppu, cuda = target / "ppu", target / "cuda"
            ppu.mkdir()
            cuda.mkdir()
            (ppu / "hggc_runtime.h").write_text(
                "void hggcMalloc();\n#define __HGGC_ARCH__ 100\n"
                "#define COMPATIBLE_ARCH 800\n"
            )
            (cuda / "cuda_runtime.h").write_text(
                "void cudaMalloc();\n#define __CUDA_ARCH__ 800\n"
            )
            output = target / "sdk_map.py"
            with mock.patch.multiple(
                generator,
                PPU_INCLUDE_DIRS=[str(ppu)],
                CUDA_INCLUDE_DIR=str(cuda),
                MAP_OUT=output,
                REPORT_OUT=target / "report.txt",
                HISTORICAL_TOKENS=list(generator.HISTORICAL_TOKENS) + [
                    ("COMPATIBLE_ARCH", "__CUDA_ARCH__"),
                    ("__HGGC_ARCH__", "__CUDA_ARCH__"),
                ],
            ), mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(generator.main(), 0)
            namespace = {}
            exec(compile(output.read_text(), str(output), "exec"), namespace)
            mappings = {}
            for value in namespace.values():
                if isinstance(value, dict):
                    mappings.update(value)
            for token in ("COMPATIBLE_ARCH", "__HGGC_ARCH__", "__CUDA_ARCH__"):
                self.assertNotIn(token, mappings)
            self.assertEqual(mappings["hggcMalloc"], "cudaMalloc")
            self.assertEqual(mappings["hggc_runtime.h"], "cuda_runtime.h")

    def test_python_syntax(self):
        for path in (
            ROOT / "setup.py", Path(__file__), KERNELS / "generate_kernels.py",
            CUDAFY / "sdk_map.py", CUDAFY / "tools/generate_sdk_map.py",
        ):
            with self.subTest(path=path.relative_to(ROOT)):
                compile(path.read_text(), str(path), "exec")

    def test_sdk_conversion_and_ppu_options(self):
        source = "#include <hggc_runtime.h>\nhggcError_t error;\nhggcMalloc(&p, 16);\n"
        converted = self.general.convert_text(source)
        self.assertIn("cuda_runtime.h", converted)
        self.assertIn("cudaError_t", converted)
        self.assertIn("cudaMalloc", converted)
        self.assertEqual(self.general.convert_text(converted), converted)
        options = "#pragma hggc mmatiestrictly\nhgcc -DUSE_PPU --no-hggc-embed-bc"
        converted = self.general.convert_text(options)
        self.assertIn("#pragma hggc mmatiestrictly", converted)
        self.assertIn("nvcc -DUSE_PPU --no-hggc-embed-bc", converted)

    def test_general_cli_dry_run_and_idempotence(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory) / "include"
            target.mkdir()
            source = target / "hggc_runtime.h"
            source.write_text("hggcError_t error; int cu_count;\n")
            cmd = [sys.executable, str(CUDAFY / "cudafy.py"), "general", str(target)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            subprocess.run(cmd + ["--dry-run"], check=True, capture_output=True, env=env)
            self.assertTrue(source.exists())
            self.assertIn("hggcError_t", source.read_text())
            subprocess.run(cmd, check=True, capture_output=True, env=env)
            converted = target / "cuda_runtime.h"
            self.assertTrue(converted.exists())
            self.assertEqual(converted.read_text(), "cudaError_t error; int cu_count;\n")
            self.assertEqual(list(target.iterdir()), [converted])
            subprocess.run(cmd, check=True, capture_output=True, env=env)
            self.assertEqual(converted.read_text(), "cudaError_t error; int cu_count;\n")

    def setup_functions(self):
        tree = ast.parse((ROOT / "setup.py").read_text())
        names = {"_missing_submodule_error", "_cudafy_restore_roots", "run_cudafy_once"}
        selected = ast.Module(
            body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names],
            type_ignores=[],
        )
        namespace = {
            "os": os, "sys": sys, "subprocess": subprocess,
            "_restore_pristine": mock.Mock(),
        }
        exec(compile(selected, str(ROOT / "setup.py"), "exec"), namespace)
        return namespace

    def test_setup_conversion_and_patch_order_with_mock(self):
        tree = ast.parse((ROOT / "setup.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        ppu_block = next(node for node in main.body if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Name) and node.test.id == "on_ppu")
        isolated = ast.Module(body=ppu_block.body, type_ignores=[])
        code = compile(isolated, str(ROOT / "setup.py"), "exec")
        for enabled in (False, True):
            namespace = self.setup_functions()
            namespace.update({
                "__file__": str(ROOT / "setup.py"),
                "Init_ppu_build_env": mock.Mock(),
                "run_cudafy_once": mock.Mock(), "apply_patch": mock.Mock(),
            })
            events = mock.Mock()
            events.attach_mock(namespace["run_cudafy_once"], "convert")
            events.attach_mock(namespace["apply_patch"], "patch")
            flags = {
                "USE_FLASH_ATTENTION": str(enabled),
                "USE_FLEX_FLASH_ATTENTION": str(enabled),
            }
            with self.subTest(enabled=enabled), mock.patch.dict(os.environ, flags):
                exec(code, namespace)
            third_party = ROOT / "third_party"
            expected = [
                mock.call.convert(str(ROOT), "actlize", "v0.8.0", str(third_party / "cutlass/include")),
                mock.call.patch(str(third_party / "cutlass_platform_numeric_limits.patch"), str(third_party / "cutlass")),
            ]
            if enabled:
                expected += [
                    mock.call.convert(str(ROOT), "flash-attention", "2.7.2", str(third_party / "flash-attention")),
                    mock.call.patch(str(third_party / "flash_attention_namespace_config.patch"), str(third_party / "flash-attention")),
                    mock.call.convert(str(ROOT), "flex-flash-attention", "2.8.2", str(third_party / "flex-flash-attention")),
                    mock.call.patch(str(third_party / "flex_flash_attention_native_names.patch"), str(third_party / "flex-flash-attention")),
                ]
            self.assertEqual(events.mock_calls, expected)
            namespace["_restore_pristine"].assert_not_called()

    def test_setup_uses_general_and_restores_nested_headers(self):
        namespace = self.setup_functions()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory)
            include = target / "csrc/actlize/include"
            util = target / "csrc/actlize/tools/util/include"
            include.mkdir(parents=True)
            util.mkdir(parents=True)
            for command, version in (("flash-attention", "2.7.2"), ("flex-flash-attention", "2.8.2")):
                with self.subTest(command=command), mock.patch.object(subprocess, "run") as run:
                    namespace["run_cudafy_once"](str(ROOT), command, version, str(target))
                    self.assertEqual(run.call_args.args[0], [
                        sys.executable, "third_party/cudafy-for-sail/cudafy.py", "general",
                        os.path.relpath(target, ROOT),
                    ])
                    roots = namespace["_cudafy_restore_roots"](command, str(target))
                    self.assertEqual(roots, [str(target), str(include), str(util)])
                    self.assertEqual(
                        namespace["_restore_pristine"].call_args_list[-3:],
                        [mock.call(root) for root in roots],
                    )
            self.assertEqual(
                namespace["_cudafy_restore_roots"]("actlize", str(include)),
                [str(include), str(util)],
            )
        self.assertNotIn("--compat", (ROOT / "setup.py").read_text())

    def test_setup_safety_checks_before_restore(self):
        namespace = self.setup_functions()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            target = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Missing repo-local ACTLIZE"):
                namespace["run_cudafy_once"](str(ROOT), "flash-attention", "2.7.2", str(target))
            link = target / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "through symlink"):
                namespace["run_cudafy_once"](str(ROOT), "actlize", "0.8", str(link))
            namespace["_restore_pristine"].assert_not_called()

    def prepare_flex_patch(self, target):
        patch = ROOT / "third_party/flex_flash_attention_native_names.patch"
        self.assertTrue(patch.is_file())
        for relative in FFA_FILES + FFA_COPY_FILES:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "third_party/flex-flash-attention" / relative, destination)
        self.general.convert_path(str(target))
        result = subprocess.run(
            ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)], cwd=target,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_flex_patch_native_copy_atoms(self):
        ffa = ROOT / "third_party/flex-flash-attention"
        definitions = (ffa / "csrc/actlize/include/cute/arch/copy_ppu.hpp").read_text()
        for native in FFA_COPY_TYPES.values():
            self.assertRegex(definitions, r"\bstruct\s+" + native + r"\b")
            self.assertNotIn(native, self.general.SDK_TOKEN_MAP)
        # Apply the patch to two pristine copies covering regular and fp32 kernels in both directions.
        for _ in range(2):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                target = Path(directory)
                self.prepare_flex_patch(target)
                for relative in FFA_COPY_FILES:
                    before = self.general.convert_text((ffa / relative).read_text())
                    after = (target / relative).read_text()
                    expected = before
                    for old, native in FFA_COPY_TYPES.items():
                        expected = re.sub(r"\b" + old + r"\b", native, expected)
                    self.assertEqual(after, expected, relative)
                    self.assertEqual(self.general.convert_text(after), after)
                    for arch in (100, 150):
                        for aiu in (0, 1):
                            with self.subTest(path=relative, arch=arch, aiu=aiu):
                                active = self.preprocess(after, [
                                    "USE_PPU=1", f"USE_AIU={aiu}", f"__HGGC_ARCH__={arch}",
                                    "__CUDACC__=1", "__CUDACC_VER_MAJOR__=13",
                                ])
                                self.assertNotRegex(active, r"\bSM\d+_\w*(?:LDSM|CP_ASYNC)\w*\b")
                                self.assertIn("PPU_CP_ASYNC_CACHEGLOBAL_ZFILL", active)

    def test_flex_patch_matches_native_fields_and_wait(self):
        for _ in range(2):
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                target = Path(directory)
                self.prepare_flex_patch(target)
                for relative in FFA_FILES[:4]:
                    content = (target / relative).read_text()
                    self.assertNotIn("hw_info.sm_count", content)
                    self.assertIn("hw_info.cu_count", content)
                content = (target / "hopper/utils.h").read_text()
                self.assertIn("defined(CUTE_ARCH_CP_ASYNC_PPU_ENABLED)", content)
                self.assertIn("ppu.cp.async.wait_group", content)
                active = self.preprocess(content, ["CUTE_ARCH_CP_ASYNC_PPU_ENABLED=1"])
                self.assertIn("ppu.cp.async.wait_group", active)
        tree = ast.parse((ROOT / "setup.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        convert = next(node for node in calls if isinstance(node.func, ast.Name)
                       and node.func.id == "run_cudafy_once"
                       and any(isinstance(arg, ast.Constant) and arg.value == "flex-flash-attention"
                               for arg in node.args))
        patch_call = next(node for node in calls if isinstance(node.func, ast.Name)
                          and node.func.id == "apply_patch"
                          and "flex_flash_attention_native_names.patch" in ast.unparse(node))
        self.assertLess(convert.lineno, patch_call.lineno)

    def test_ppu_kernel_generation(self):
        fwd = self.kernels.FwdKernel.get_all(ppu=True)
        bwd = self.kernels.BwdKernel.get_all(ppu=True)
        self.assertTrue(fwd and bwd)
        for kernel in fwd + bwd:
            self.assertIn("cutlass::arch::PPU0010", kernel.cpp_class)
            self.assertNotIn("cutlass::arch::Sm", kernel.cpp_class)
            self.assertIn("__HGGC_ARCH__", kernel.cpp_impl)
            self.assertNotIn("__CUDA_ARCH__", kernel.cpp_impl)
        self.assertEqual(len({kernel.name for kernel in fwd + bwd}), len(fwd + bwd))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.kernels.main(directory)
            for path in Path(directory).iterdir():
                content = path.read_text()
                self.assertEqual(content, (KERNELS / path.name).read_text(), path.name)
                self.assertTrue(all(line == line.rstrip() for line in content.splitlines()))
                ppu = content.split("#else\n", 1)[0]
                self.assertNotIn("cutlass::arch::Sm", ppu)

    def test_cuda_recipes_and_registry_preserved(self):
        for version, rocm, tile in ((12040, False, 32), (12030, False, 64), (12040, True, 64)):
            defines = [f"CUDA_VERSION={version}", "__CUDA_ARCH__=500"]
            if rocm:
                defines.append("USE_ROCM=1")
            with self.subTest(version=version, rocm=rocm):
                registry = self.preprocess((KERNELS / "cutlassB.h").read_text(), defines)
                for max_k in (32, 64):
                    stem = f"fmha_cutlassB_f32_aligned_{{}}x{{}}_k{max_k}_dropout_sm50"
                    chosen = stem.format(tile, tile)
                    other = stem.format(64 if tile == 32 else 32, 64 if tile == 32 else 32)
                    source = self.preprocess(
                        (KERNELS / f"cutlassB_f32_aligned_k{max_k}_dropout.cu").read_text(), defines,
                    )
                    self.assertIn(chosen, source)
                    self.assertIn(chosen, registry)
                    self.assertNotIn(other, source)
                    self.assertNotIn(other, registry)
        for family, generator in (
            ("cutlassF", self.kernels.FwdKernel), ("cutlassB", self.kernels.BwdKernel),
        ):
            registry = self.preprocess((KERNELS / f"{family}.h").read_text(), [])
            self.assertIn("cc <= 120", registry)
            for kernel in generator.get_all():
                if kernel.sm_range[0] == 80:
                    self.assertIn("__CUDA_ARCH__ <= 1200", kernel.cpp_impl)

    def test_ppu_active_consumers_and_includes(self):
        native = ROOT / "aten/src/ATen/native"
        paths = list((native / "cuda/cutlass_extensions").rglob("*.h"))
        paths += list(KERNELS.parent.rglob("*.h")) + list(KERNELS.glob("*.cu"))
        paths += [native / name for name in (
            "cuda/MixedDtypesLinear.cu", "nested/cuda/NestedTensorMatmul.cu",
            "sparse/cuda/SparseSemiStructuredLinear.cu", "sparse/cuda/SparseSemiStructuredOps.cu",
        )]
        self.assertGreater(len(paths), 50)
        base = [
            "USE_PPU=1", "__CUDACC__=1", "__CUDACC_VER_MAJOR__=12", "CUDA_VERSION=12040",
        ]
        # Define the CUDA architecture too, to ensure PPU excludes legacy CUDA branches.
        for arch in (None, 100, 150):
            defines = base + (
                [] if arch is None else [f"__HGGC_ARCH__={arch}", "__CUDA_ARCH__=800"]
            )
            for path in paths:
                with self.subTest(arch=arch, path=path.relative_to(ROOT)):
                    active = self.preprocess(path.read_text(), defines)
                    legacy = re.search(r"\b(?:cutlass::)?arch::Sm\d+|\bSm\d+EVT\b", active)
                    self.assertIsNone(legacy, str(path))
                    headers = re.findall(
                        r'NATIVE_INCLUDE\("((?:cutlass|cute)/[^"\n]+)"\)', active
                    )
                    for header in headers:
                        self.assertTrue(
                            (ROOT / "third_party/cutlass/include" / header).is_file(), header
                        )
                    names = re.findall(r"(?m)^\s*(fmha_cutlass[FB]_\w+)\(typename", active)
                    self.assertEqual(len(names), len(set(names)), path.name)
        for arch in (100, 150):
            defines = base + [f"__HGGC_ARCH__={arch}"]
            backward = self.preprocess((KERNELS.parent / "kernel_backward.h").read_text(), defines)
            self.assertIn("ppu.st.global.release.gpu.b32", backward)
            self.assertIn("__nanosleep(40)", backward)
            self.assertIn("gridDim.x", backward)
            runner = self.preprocess(
                (native / "cuda/cutlass_extensions/gemm/kernel/fpA_intB_gemm.h").read_text(),
                defines,
            )
            self.assertIn("KernelRunner<compile_needed>::run_kernel", runner)
            generated = self.preprocess((KERNELS / "cutlassF_f16_aligned.cu").read_text(), defines)
            self.assertIn("::attention_kernel(p)", generated)

    def test_ppu_ldsm_build_configuration(self):
        """Run the CMake block to verify Torch, FA2, and standalone FFA share one setting."""
        if shutil.which("cmake") is None:
            self.skipTest("cmake is required to check the build configuration")
        cmake = (ROOT / "CMakeLists.txt").read_text()
        block = re.search(r"(?ms)^if\(USE_PPU\)\n.*?^endif\(\)", cmake)
        self.assertIsNotNone(block)
        ffa = (ROOT / "cmake/flex_flash_attention.cmake").read_text()
        defines = re.search(r'(?ms)^set\(_fa_quoted_defines ""\)\n.*?^endif\(\)', ffa)
        self.assertIsNotNone(defines)
        native = "CUTE_ARCH_LDSM_PPU_ENABLED=1"
        self.assertEqual(cmake.count(native), 1)
        command = next(line for line in ffa.splitlines() if "COMMAND bash -c" in line)
        self.assertIn("${_fa_quoted_defines}", command)
        self.assertNotIn(native, ffa)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                target = Path(directory)
                script = "\n".join([
                    "cmake_minimum_required(VERSION 3.18)",
                    "project(PR165LDSM NONE)", block.group(), defines.group(),
                    "get_directory_property(_actual COMPILE_DEFINITIONS)",
                    'file(WRITE "${CMAKE_BINARY_DIR}/definitions.txt" "${_actual}")',
                    'file(WRITE "${CMAKE_BINARY_DIR}/ffa_flags.txt" "${_fa_quoted_defines}")',
                ])
                (target / "CMakeLists.txt").write_text(script + "\n")
                result = subprocess.run([
                    "cmake", "-S", str(target), "-B", str(target / "build"),
                    "-DUSE_PPU=" + ("ON" if enabled else "OFF"),
                ], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                actual = (target / "build/definitions.txt").read_text().split(";")
                flags = shlex.split((target / "build/ffa_flags.txt").read_text())
                self.assertEqual(actual.count(native), int(enabled))
                self.assertEqual(flags, ["--define", native] if enabled else [])
                self.assertFalse(any("__HGGCCC_VER_" in item or "__CUDA_ARCH__" in item
                                     for item in actual))

    def test_ppu_ldsm_native_gate_preserves_sdk_version(self):
        """Check the native feature switch under SDK 2 and CUDA 13 without spoofing versions."""
        for library in ("cutlass", "flash-attention/csrc/actlize",
                        "flex-flash-attention/csrc/actlize"):
            path = ROOT / "third_party" / library / "include/cute/arch/copy_ppu.hpp"
            config = self.general.convert_text(path.read_text()).split("namespace cute", 1)[0]
            config += "\n#if CUTE_ARCH_LDSM_SUPPORTED\nint native_supported;\n#endif\n"
            config += "#if defined(CUTE_ARCH_LDSM_PPU_ACTIVATED)\nint native_activated;\n#endif\n"
            base = ["__CUDACC__=1", "__CUDACC_VER_MAJOR__=13", "__HGGCCC_VER_MAJOR__=2"]
            # FFA's ACTLIZE v1 already enables LDSM; only the older Torch and FA2 copies gate it on versions.
            always_enabled = library == "flex-flash-attention/csrc/actlize"
            if always_enabled:
                self.assertIn("#define CUTE_ARCH_LDSM_PPU_ACTIVATED 1", config)
            for arch in (None, 100, 150):
                for enabled in ((None, 1) if always_enabled else (None, 0, 1)):
                    with self.subTest(library=library, arch=arch, enabled=enabled):
                        defines = base + ([] if arch is None else [f"__HGGC_ARCH__={arch}"])
                        if enabled is not None:
                            defines.append(f"CUTE_ARCH_LDSM_PPU_ENABLED={enabled}")
                        active = self.preprocess(config, defines)
                        self.assertNotIn("int native_supported;", active)
                        self.assertEqual("int native_activated;" in active,
                                         always_enabled or (arch is not None and enabled == 1))
                        self.assertIn("#define __HGGCCC_VER_MAJOR__ 2", active)
                        self.assertIn("#define __CUDACC_VER_MAJOR__ 13", active)

    def test_real_actlize_and_fa2_samples(self):
        relative_headers = (
            "cutlass/arch/arch.h", "cutlass/kernel_hardware_info.h",
            "cutlass/arch/memory_ppu.h", "cutlass/arch/mma_ppu0010.h", "cute/arch/copy_ppu.hpp",
        )
        libraries = ("cutlass", "flash-attention/csrc/actlize", "flex-flash-attention/csrc/actlize")
        protected = re.compile(
            r"\b(?:__HGGC_ARCH__|PPU\w*|CUTE_ARCH_\w*PPU\w*|cu_count|CuId)\b"
        )
        for library in libraries:
            original = ROOT / "third_party" / library
            with self.subTest(library=library), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                target = Path(directory)
                expected = {}
                headers = list(relative_headers)
                if library == "flex-flash-attention/csrc/actlize":
                    headers[3] = "cutlass/arch/mma_ppu.h"
                for header in headers:
                    source = original / "include" / header
                    self.assertTrue(source.is_file(), str(source))
                    relative = "include/" + header
                    expected[relative] = source.read_text()
                util = "tools/util/include/cutlass/util/device_memory.h"
                expected[util] = (original / util).read_text()
                for relative, content in expected.items():
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(content)
                self.general.convert_path(str(target / "include"))
                self.assertEqual(
                    {str(path.relative_to(target)) for path in target.rglob("*") if path.is_file()},
                    set(expected),
                )
                for relative, before in expected.items():
                    after = (target / relative).read_text()
                    self.assertEqual(protected.findall(before), protected.findall(after), relative)
                    before_arch = [line for line in before.splitlines() if "__HGGC_ARCH__" in line]
                    self.assertEqual(
                        before_arch,
                        [line for line in after.splitlines() if "__HGGC_ARCH__" in line],
                    )
                    self.assertNotIn("__CUDA_ARCH__", after)
                    self.assertEqual(self.general.convert_text(before), after)
                self.assertEqual(self.general.convert_path(str(target / "include"))[1:3], (0, 0))
        fa2 = ROOT / "third_party/flash-attention/csrc/flash_attn/src"
        for filename in ("utils.h", "flash_fwd_kernel.h", "flash_bwd_kernel.h"):
            with self.subTest(fa2=filename):
                before = (fa2 / filename).read_text()
                after = self.general.convert_text(before)
                self.assertEqual(protected.findall(before), protected.findall(after))
                self.assertEqual(self.general.convert_text(after), after)
        wait = self.preprocess(
            self.general.convert_text((fa2 / "utils.h").read_text()),
            ["CUTE_ARCH_CP_ASYNC_PPU_ENABLED=1"],
        )
        self.assertIn("ppu.cp.async.wait_group", wait)


if __name__ == "__main__":
    unittest.main()
