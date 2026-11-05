"""Preserved-reduction layout contract: reads compiled results only, never mutates compiler IR."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import threading
from types import SimpleNamespace
from typing import Any

import torch


log = logging.getLogger(__name__)


CONTRACT_VERSION = 2


def _recoverable_triton_errors() -> tuple[type[BaseException], ...]:
    # Recoverable compilation/resource failures of the optional Gluon candidate must fall
    # back to the standard sequence instead of failing the model invocation. They are not
    # OSError subclasses, so collect them explicitly; missing Triton means no extra types.
    errors: list[type[BaseException]] = []
    try:
        from triton.compiler.errors import CompilationError

        errors.append(CompilationError)
    except ImportError:
        pass
    try:
        from triton.runtime.errors import OutOfResources, PTXASError

        errors.extend((OutOfResources, PTXASError))
    except ImportError:
        pass
    return tuple(errors)


class UnsupportedReductionContract(ValueError):
    """A reduction contract that cannot be proven supported from the actual compiled result."""


def _power_of_two(value: int) -> bool:
    return type(value) is int and value > 0 and value & (value - 1) == 0


@dataclasses.dataclass(frozen=True)
class BlockedReductionLayout:
    size_per_thread: tuple[int, int]
    threads_per_warp: tuple[int, int]
    warps_per_cta: tuple[int, int]
    order: tuple[int, int]

    def __post_init__(self) -> None:
        for field in (self.size_per_thread, self.threads_per_warp, self.warps_per_cta):
            if len(field) != 2 or not all(_power_of_two(v) for v in field):
                raise UnsupportedReductionContract("blocked layout must be a 2D positive power of two")
        if len(self.order) != 2 or set(self.order) != {0, 1}:
            raise UnsupportedReductionContract("blocked order must be a 2D permutation")

    def validate_launch(self, num_warps: int, warp_size: int, num_ctas: int) -> None:
        if (
            not _power_of_two(num_warps)
            or not _power_of_two(warp_size)
            or math.prod(self.threads_per_warp) != warp_size
            or math.prod(self.warps_per_cta) != num_warps
            or num_ctas != 1
        ):
            raise UnsupportedReductionContract("blocked layout does not match the actual launch config")

    def to_gluon(self) -> Any:
        # Deferred import so CPU or Gluon-less platforms can still parse and reject candidates.
        from triton.experimental.gluon.language import BlockedLayout

        return BlockedLayout(
            list(self.size_per_thread),
            list(self.threads_per_warp),
            list(self.warps_per_cta),
            list(self.order),
        )


@dataclasses.dataclass(frozen=True)
class ReductionLaunchConfig:
    kwargs: tuple[tuple[str, int], ...]
    num_warps: int
    num_stages: int
    num_ctas: int

    @classmethod
    def from_config(cls, config: Any) -> ReductionLaunchConfig:
        result = cls(
            tuple(sorted(config.kwargs.items())),
            config.num_warps,
            config.num_stages,
            getattr(config, "num_ctas", 1),
        )
        if (
            set(dict(result.kwargs)) != {"XBLOCK", "R0_BLOCK"}
            or not all(_power_of_two(v) for _, v in result.kwargs)
            or not _power_of_two(result.num_warps)
            or type(result.num_stages) is not int
            or result.num_stages < 1
            or result.num_ctas != 1
            or getattr(config, "pre_hook", None) is not None
        ):
            raise UnsupportedReductionContract("only hook-free 2D single-loop reduction configs are supported")
        return result


@dataclasses.dataclass(frozen=True)
class ReductionEncoding:
    layout: BlockedReductionLayout
    shape: tuple[int, int]
    dtype: str
    axis: int
    num_warps: int
    warp_size: int
    num_ctas: int

    def validate_config(self, config: ReductionLaunchConfig) -> None:
        self.layout.validate_launch(config.num_warps, self.warp_size, config.num_ctas)
        blocks = dict(config.kwargs)
        if (
            self.num_warps != config.num_warps
            or self.num_ctas != config.num_ctas
            or self.shape != (blocks["XBLOCK"], blocks["R0_BLOCK"])
        ):
            raise UnsupportedReductionContract("actual reduction tensor does not match the selected config")


def _module_integer(ttgir: str, name: str) -> int:
    matches = re.findall(rf'"{re.escape(name)}" = ([0-9]+) : i32', ttgir)
    if len(matches) != 1:
        raise UnsupportedReductionContract(f"missing or duplicated module field: {name}")
    return int(matches[0])


def extract_reduction_encoding(ttgir: str) -> ReductionEncoding:
    """Strictly parse the real operand encoding of a single sum; reject unknown formats, never guess a default layout."""
    if len(re.findall(r'"tt.reduce"\(', ttgir)) != 1:
        raise UnsupportedReductionContract("exactly one actual tt.reduce is required")
    reduction = re.search(
        r'"tt.reduce"\(%[\w.]+\) <\{axis = ([0-9]+) : i32\}> '
        r'\(\{\s*\^bb[0-9]+\(%([\w.]+): (f32|f64)[^\n]*\):\s*'
        r'%([\w.]+) = arith.addf %([\w.]+), %([\w.]+) : (f32|f64)[^\n]*\n'
        r'\s*tt.reduce.return %([\w.]+) : (f32|f64)[^\n]*\n'
        r'\s*\}\) : \(tensor<([0-9]+)x([0-9]+)x(f32|f64), (#[\w.]+)>\)',
        ttgir,
    )
    if reduction is None:
        raise UnsupportedReductionContract("unknown tt.reduce format or non-floating-point add combiner")
    (
        axis, _, block_dtype, result, _, _, add_dtype, returned, return_dtype,
        x, r, dtype, alias,
    ) = reduction.groups()
    if (
        axis != "1"
        or result != returned
        or len({block_dtype, add_dtype, return_dtype, dtype}) != 1
    ):
        raise UnsupportedReductionContract("inconsistent reduction axis, dtype or combiner")
    # Verify the two block arguments really are the addf operands; no extra expressions allowed.
    region = reduction[0]
    block_args = re.findall(r"%([\w.]+): f(?:32|64)", region)
    if len(block_args) != 2 or (reduction[5], reduction[6]) != tuple(block_args):
        raise UnsupportedReductionContract("sum combiner must add the two arguments in their original order")
    arrays = r"\[([0-9]+), ([0-9]+)\]"
    layout_pattern = (
        rf"^{re.escape(alias)} = #ttg.blocked<\{{sizePerThread = {arrays}, "
        rf"threadsPerWarp = {arrays}, warpsPerCTA = {arrays}, order = {arrays}\}}>\s*$"
    )
    layouts = list(re.finditer(layout_pattern, ttgir, re.MULTILINE))
    if len(layouts) != 1:
        raise UnsupportedReductionContract("unknown, duplicated or extended blocked layout")
    values = tuple(map(int, layouts[0].groups()))
    layout = BlockedReductionLayout(
        values[0:2], values[2:4], values[4:6], values[6:8]
    )
    num_warps = _module_integer(ttgir, "ttg.num-warps")
    warp_size = _module_integer(ttgir, "ttg.threads-per-warp")
    num_ctas = _module_integer(ttgir, "ttg.num-ctas")
    layout.validate_launch(num_warps, warp_size, num_ctas)
    return ReductionEncoding(
        layout, (int(x), int(r)), dtype, int(axis), num_warps, warp_size, num_ctas
    )


@dataclasses.dataclass(frozen=True)
class ReductionCacheIdentity:
    source_hash: str
    config: ReductionLaunchConfig
    encoding: ReductionEncoding
    target: tuple[str, int | str, int]
    compiler_version: str
    backend_hash: str
    compile_options_json: str
    gluon_source_hash: str = ""
    candidate_arg_map: tuple[tuple[int, int], ...] = ()
    contract_version: int = CONTRACT_VERSION

    @classmethod
    def create(
        cls,
        *,
        source: str,
        config: ReductionLaunchConfig,
        encoding: ReductionEncoding,
        target: Any,
        compiler_version: str,
        backend_hash: str,
        compile_options: dict[str, Any],
        gluon_source: str = "",
        candidate_arg_map: tuple[tuple[int, int], ...] = (),
    ) -> ReductionCacheIdentity:
        encoding.validate_config(config)
        return cls(
            hashlib.sha256(source.encode()).hexdigest(),
            config,
            encoding,
            (target.backend, target.arch, target.warp_size),
            compiler_version,
            backend_hash,
            json.dumps(compile_options, sort_keys=True, separators=(",", ":")),
            hashlib.sha256(gluon_source.encode()).hexdigest(),
            candidate_arg_map,
        )

    def cache_key(self) -> str:
        serialized = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode()).hexdigest()


class _SequenceEndpoint:
    def __init__(self, owner: PreservedReductionCall, pointwise: bool) -> None:
        self._owner = owner
        self._pointwise = pointwise

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if self._pointwise:
            return self._owner._run_pointwise(*args, **kwargs)
        return self._owner._run_reduction(*args, **kwargs)


class PreservedReductionCall:
    """One-time selection between the full standard sequence and the explicit-layout Gluon kernel."""

    def __init__(
        self,
        standard_kernel: Any,
        *,
        gluon_source: str,
        gluon_arg_names: tuple[str, ...],
        gluon_layout_arg: str,
        pointwise_kernel: Any | None = None,
        gluon_function_name: str = "",
        pointwise_arg_count: int = 0,
        reduction_arg_count: int = 0,
        candidate_arg_map: tuple[tuple[int, int], ...] = (),
        candidate_output_indices: tuple[int, ...] = (),
        candidate_xnumel_index: int = -1,
        device_str: str = "cuda",
    ) -> None:
        self._standard_or_future = standard_kernel
        self._pointwise_or_future = pointwise_kernel
        self._gluon_source = gluon_source
        self._gluon_function_name = gluon_function_name
        self._gluon_arg_names = gluon_arg_names
        self._gluon_layout_arg = gluon_layout_arg
        self._pointwise_arg_count = pointwise_arg_count
        self._reduction_arg_count = reduction_arg_count
        self._candidate_arg_map = candidate_arg_map
        self._candidate_output_indices = candidate_output_indices
        self._candidate_xnumel_index = candidate_xnumel_index
        self._device_str = device_str
        self._selected: Any | None = None
        self._selected_mode: str | None = None
        self._identity: ReductionCacheIdentity | None = None
        self._candidate: Any | None = None
        self._candidate_config: ReductionLaunchConfig | None = None
        self._candidate_layout: Any | None = None
        self._candidate_fp_fusion: bool | None = None
        # Steady-state fast path: precompute launch arguments after selection to avoid rebuilding a dict per call.
        # The grid cache is thread-local so concurrent sequences never overwrite each other's grid.
        self._launch_kwargs: dict[str, Any] | None = None
        self._pending = threading.local()
        # Serializes the one-time selection so concurrent sequences never race between
        # stashing a pointwise call and observing the selected mode.
        self._selection_lock = threading.Lock()
        self.pointwise = _SequenceEndpoint(self, True)
        self.reduction = _SequenceEndpoint(self, False)

    @staticmethod
    def _resolve(value: Any) -> Any:
        if hasattr(value, "result") and callable(value.result):
            return value.result()
        return value

    @property
    def _standard(self) -> Any:
        self._standard_or_future = self._resolve(self._standard_or_future)
        return self._standard_or_future

    @property
    def _pointwise_kernel(self) -> Any:
        self._pointwise_or_future = self._resolve(self._pointwise_or_future)
        return self._pointwise_or_future

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Legacy single-reduction wrapper; this mode only validates the contract and uses the standard kernel."""
        kernel = self._standard
        result = kernel.run(*args, **kwargs)
        if self._selected is None:
            with self._selection_lock:
                if self._selected is None:
                    try:
                        self._contract(kernel)
                    except (ImportError, OSError, UnsupportedReductionContract, torch.OutOfMemoryError):
                        pass
                    self._selected = kernel
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._standard, name)

    def _run_pointwise(self, *args: Any, **kwargs: Any) -> Any:
        mode = self._selected_mode
        if mode == "standard":
            return self._pointwise_kernel.run(*args, **kwargs)
        if mode == "gluon":
            # In steady state just stash the arguments without extra checks; first selection still takes the full path.
            self._pending.pointwise = (args, kwargs)
            return None
        if getattr(self._pending, "pointwise", None) is not None:
            raise RuntimeError("preserved-reduction pointwise endpoint invoked again before the previous sequence completed")
        self._pending.pointwise = (args, kwargs)
        return None

    def _run_reduction(self, *args: Any, **kwargs: Any) -> Any:
        mode = self._selected_mode
        if mode is None:
            # Serialize the one-time selection; pairs stashed before another thread finished
            # selecting are consumed below once the mode becomes visible.
            with self._selection_lock:
                mode = self._selected_mode
                if mode is None:
                    return self._select_mode(*args, **kwargs)
        pending = getattr(self._pending, "pointwise", None)
        if pending is not None:
            del self._pending.pointwise
        if mode == "standard":
            # A pair deferred before selection must still run its pointwise call; skipping
            # it would leave the pointwise outputs unwritten.
            if pending is not None:
                self._pointwise_kernel.run(*pending[0], **pending[1])
            return self._standard.run(*args, **kwargs)
        if pending is None:
            raise RuntimeError("preserved-reduction reduction endpoint has no matching pointwise call")
        try:
            return self._run_candidate_fast(pending, (args, kwargs))
        except (
            UnsupportedReductionContract,
            torch.OutOfMemoryError,
            *_recoverable_triton_errors(),
        ):
            # A later dynamic-shape call can trigger a fresh JIT specialization of the
            # candidate; compilation/resource failures (or a changed argument contract)
            # must not fail the invocation with both standard kernels skipped. Switch to
            # the standard sequence permanently and run the saved pointwise/reduction pair.
            self._selected_mode = "standard"
            if self._identity is not None:
                self._store_choice(self._identity, "standard")
            self._pointwise_kernel.run(*pending[0], **pending[1])
            return self._standard.run(*args, **kwargs)

    def _select_mode(self, *args: Any, **kwargs: Any) -> Any:
        pending = getattr(self._pending, "pointwise", None)
        if pending is None:
            raise RuntimeError("preserved-reduction reduction endpoint has no matching pointwise call")
        del self._pending.pointwise
        endpoints = (pending, (args, kwargs))

        # The first call runs the untouched full two-kernel sequence. It also completes the standard
        # reduction autotune, and user-visible outputs always come from this path; candidate validation
        # and timing never contribute to that result.
        self._run_standard(endpoints)
        # Candidate validation and benchmarking need host synchronization, which is forbidden
        # during CUDA graph capture. Run the standard sequence for this call but leave the mode
        # unselected so a later non-capturing invocation can still perform the one-time choice.
        if self._is_capturing():
            return None
        try:
            identity = self._contract(self._standard)
            self._identity = identity
            cached = self._load_choice(identity)
            if cached == "standard":
                self._selected_mode = "standard"
                return None
            self._load_candidate()
            try:
                valid = self._validate_candidate(endpoints)
            except _recoverable_triton_errors() as exc:
                raise UnsupportedReductionContract(
                    "Gluon candidate failed to compile"
                ) from exc
            if not valid:
                self._selected_mode = "standard"
                self._store_choice(identity, "standard")
                return None
            if cached == "gluon":
                self._selected_mode = "gluon"
                return None
            try:
                standard_ms, candidate_ms = self._benchmark(endpoints)
            except _recoverable_triton_errors() as exc:
                raise UnsupportedReductionContract(
                    "Gluon candidate failed to compile"
                ) from exc
            self._selected_mode = (
                "gluon" if candidate_ms < standard_ms else "standard"
            )
            self._store_choice(identity, self._selected_mode)
            log.debug(
                "preserved reduction picked %s (standard %.4f ms, Gluon %.4f ms)",
                self._selected_mode,
                standard_ms,
                candidate_ms,
            )
        except (ImportError, OSError, UnsupportedReductionContract, torch.OutOfMemoryError):
            # Validation/benchmark allocations may OOM even when the standard sequence fits;
            # the optional optimization is rejected rather than failing the model invocation.
            self._selected_mode = "standard"
        return None

    def _is_capturing(self) -> bool:
        try:
            device_module = getattr(torch, self._device_str, None)
            checker = getattr(device_module, "is_current_stream_capturing", None)
            return bool(checker()) if checker is not None else False
        except Exception:
            return False

    def _run_standard(
        self,
        endpoints: tuple[
            tuple[tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
    ) -> None:
        pointwise, reduction = endpoints
        self._pointwise_kernel.run(*pointwise[0], **pointwise[1])
        self._standard.run(*reduction[0], **reduction[1])

    def _candidate_arguments(
        self,
        endpoints: tuple[
            tuple[tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
    ) -> list[Any]:
        expected = (self._pointwise_arg_count, self._reduction_arg_count)
        if tuple(len(endpoint[0]) for endpoint in endpoints) != expected:
            raise UnsupportedReductionContract("standard sequence runtime argument count changed")
        try:
            return [endpoints[side][0][position] for side, position in self._candidate_arg_map]
        except (IndexError, TypeError) as exc:
            raise UnsupportedReductionContract("invalid fused candidate argument mapping") from exc

    def _run_candidate(
        self,
        endpoints: tuple[
            tuple[tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
        candidate_args: list[Any] | None = None,
    ) -> Any:
        if self._candidate is None or self._candidate_config is None:
            raise UnsupportedReductionContract("Gluon candidate is not prepared yet")
        if candidate_args is None:
            candidate_args = self._candidate_arguments(endpoints)
        blocks = dict(self._candidate_config.kwargs)
        xnumel = int(candidate_args[self._candidate_xnumel_index])
        xblock = blocks["XBLOCK"]
        grid = ((xnumel + xblock - 1) // xblock, 1, 1)
        return self._candidate[grid](
            *candidate_args,
            **blocks,
            **{self._gluon_layout_arg: self._candidate_layout},
            num_warps=self._candidate_config.num_warps,
            num_stages=self._candidate_config.num_stages,
            num_ctas=self._candidate_config.num_ctas,
            enable_fp_fusion=self._candidate_fp_fusion,
        )

    def _run_candidate_fast(
        self,
        pointwise: tuple[tuple[Any, ...], dict[str, Any]],
        reduction: tuple[tuple[Any, ...], dict[str, Any]],
    ) -> Any:
        """Steady-state path after selection: precomputed launch arguments and grid cache.

        Falls back to the full validation path when argument counts or types change; numerical
        behavior is identical to _run_candidate.
        """
        launch_kwargs = self._launch_kwargs
        if launch_kwargs is None:
            return self._run_candidate((pointwise, reduction))
        pw_args = pointwise[0]
        red_args = reduction[0]
        if len(pw_args) != self._pointwise_arg_count or len(red_args) != self._reduction_arg_count:
            raise UnsupportedReductionContract("standard sequence runtime argument count changed")
        candidate_args = [
            (pw_args if side == 0 else red_args)[position]
            for side, position in self._candidate_arg_map
        ]
        xnumel = candidate_args[self._candidate_xnumel_index]
        if type(xnumel) is not int:
            # Symbolic or non-integer sizes fall back to the full path (including int conversion and validation).
            return self._run_candidate((pointwise, reduction), candidate_args)
        state = self._pending
        if getattr(state, "fast_xnumel", None) != xnumel:
            xblock = launch_kwargs["XBLOCK"]
            state.fast_xnumel = xnumel
            state.fast_grid = ((xnumel + xblock - 1) // xblock, 1, 1)
        return self._candidate[state.fast_grid](*candidate_args, **launch_kwargs)

    @staticmethod
    def _bitwise_equal(reference: torch.Tensor, actual: torch.Tensor) -> bool:
        if (
            reference.shape != actual.shape
            or reference.stride() != actual.stride()
            or reference.dtype != actual.dtype
        ):
            return False
        reference_bytes = reference.detach().contiguous().reshape(-1).view(torch.uint8)
        actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        return bool(torch.equal(reference_bytes, actual_bytes))

    def _validate_candidate(
        self,
        endpoints: tuple[
            tuple[tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
    ) -> bool:
        candidate_args = self._candidate_arguments(endpoints)
        references: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index in self._candidate_output_indices:
            try:
                reference = candidate_args[index]
            except IndexError as exc:
                raise UnsupportedReductionContract("invalid fused candidate output argument index") from exc
            if not isinstance(reference, torch.Tensor):
                raise UnsupportedReductionContract("fused candidate outputs must be Tensors")
            candidate = torch.empty_strided(
                reference.size(),
                reference.stride(),
                dtype=reference.dtype,
                device=reference.device,
            )
            candidate_args[index] = candidate
            references.append((reference, candidate))
        if not references:
            raise UnsupportedReductionContract("fused candidate has no verifiable output")
        self._run_candidate(endpoints, candidate_args)
        return all(self._bitwise_equal(reference, actual) for reference, actual in references)

    @staticmethod
    def _shadow_endpoints(
        endpoints: tuple[
            tuple[tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
    ) -> tuple[
        tuple[tuple[Any, ...], dict[str, Any]],
        tuple[tuple[Any, ...], dict[str, Any]],
    ]:
        # Kernels write their output arguments on every launch. With in-place buffer
        # reuse an output may share storage with a live input, and the codegen mutation
        # checks do not see that aliasing; benchmarking on the live endpoints would
        # overwrite caller-visible data even when standard is ultimately selected.
        # Benchmark private clones instead, keeping the argument structure intact.
        from ..utils import clone_preserve_strides

        def shadow(value: Any) -> Any:
            # clone() compacts non-dense views while the compiled kernels keep their
            # original indexing, so the copy must span the same storage offsets.
            return clone_preserve_strides(value) if isinstance(value, torch.Tensor) else value

        return tuple(
            (tuple(shadow(arg) for arg in args), {key: shadow(value) for key, value in kwargs.items()})
            for args, kwargs in endpoints
        )

    def _benchmark(
        self,
        endpoints: tuple[
            tuple[tuple[Any, ...], dict[str, Any]],
            tuple[tuple[Any, ...], dict[str, Any]],
        ],
    ) -> tuple[float, float]:
        from .benchmarking import benchmarker

        # Each benchmark gets its own private copies so one kernel's writes cannot
        # poison the other's measurement inputs.
        standard_endpoints = self._shadow_endpoints(endpoints)
        candidate_endpoints = self._shadow_endpoints(endpoints)
        standard_ms = benchmarker.benchmark(
            lambda: self._run_standard(standard_endpoints),
            device=self._device_str,
            rep=40,
            is_vetted_benchmarking=True,
        )
        candidate_ms = benchmarker.benchmark(
            lambda: self._run_candidate(candidate_endpoints),
            device=self._device_str,
            rep=40,
            is_vetted_benchmarking=True,
        )
        return standard_ms, candidate_ms

    def _selected_compile_result(self, standard: Any) -> tuple[Any, Any]:
        if not getattr(standard, "launchers", None):
            raise UnsupportedReductionContract("standard reduction has no selected launcher")
        if len(standard.launchers) != 1:
            raise UnsupportedReductionContract("standard reduction has not finished autotune")
        launcher = standard.launchers[0]
        from .runtime_utils import triton_config_to_hashable

        selected = triton_config_to_hashable(launcher.config)
        for result in getattr(standard, "compile_results", ()):
            if triton_config_to_hashable(result.config) == selected:
                return launcher, result
        raise UnsupportedReductionContract("compile result for the selected launcher not found")

    @staticmethod
    def _read_ttgir(standard: Any, result: Any) -> str:
        kernel = result.kernel
        asm = getattr(kernel, "asm", None)
        if asm and isinstance(asm.get("ttgir"), str):
            return asm["ttgir"]
        from .runtime_utils import triton_cache_dir, triton_hash_to_path_key

        kernel_hash = getattr(kernel, "hash", None)
        kernel_name = getattr(kernel, "name", None)
        if not kernel_hash or not kernel_name:
            raise UnsupportedReductionContract("static launcher lacks TTGIR cache identity")
        path = os.path.join(
            triton_cache_dir(standard.triton_meta.get("device", 0)),
            triton_hash_to_path_key(kernel_hash),
            f"{kernel_name}.ttgir",
        )
        try:
            with open(path, encoding="utf-8") as ttgir_file:
                return ttgir_file.read()
        except FileNotFoundError as exc:
            raise UnsupportedReductionContract("selected reduction TTGIR missing from Triton cache") from exc

    @staticmethod
    def _fp_fusion_option(kernel: Any) -> bool:
        # Numerical compiler options must be inherited from the standard kernels: with
        # emulate_precision_casts the standard path disables FMA contraction, and a candidate
        # launched with the default option could produce different bits on later inputs.
        meta = getattr(kernel, "triton_meta", None)
        value = meta.get("enable_fp_fusion") if isinstance(meta, dict) else None
        if type(value) is not bool:
            raise UnsupportedReductionContract(
                "standard kernel does not expose enable_fp_fusion"
            )
        return value

    def _contract(self, standard: Any) -> ReductionCacheIdentity:
        launcher, result = self._selected_compile_result(standard)
        config = ReductionLaunchConfig.from_config(launcher.config)
        encoding = extract_reduction_encoding(self._read_ttgir(standard, result))
        encoding.validate_config(config)
        fp_fusion = self._fp_fusion_option(standard)
        pointwise = self._pointwise_or_future
        if pointwise is not None:
            if self._fp_fusion_option(self._pointwise_kernel) != fp_fusion:
                raise UnsupportedReductionContract(
                    "standard kernels disagree on enable_fp_fusion"
                )
        self._candidate_fp_fusion = fp_fusion
        self._candidate_config = config
        self._candidate_layout = encoding.layout.to_gluon()
        self._launch_kwargs = {
            **dict(config.kwargs),
            self._gluon_layout_arg: self._candidate_layout,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
            "num_ctas": config.num_ctas,
            "enable_fp_fusion": fp_fusion,
        }
        device = standard.device_props
        compile_meta = result.compile_meta
        compile_options = {
            key: compile_meta[key]
            for key in (
                "debug",
                "device_type",
                "cc",
                "enable_fp_fusion",
                "num_warps",
                "num_stages",
            )
            if key in compile_meta
            and isinstance(compile_meta[key], (bool, int, float, str, type(None)))
        }
        source = str(getattr(standard.fn, "src", standard.fn))
        kernel_hash = str(getattr(result.kernel, "hash", ""))
        try:
            import triton

            compiler_version = triton.__version__
        except (AttributeError, ImportError):
            compiler_version = "unknown"
        return ReductionCacheIdentity.create(
            source=source,
            config=config,
            encoding=encoding,
            target=SimpleNamespace(
                backend=device.type,
                arch=device.cc,
                warp_size=device.warp_size,
            ),
            compiler_version=compiler_version,
            backend_hash=kernel_hash,
            compile_options=compile_options,
            gluon_source=self._gluon_source,
            candidate_arg_map=self._candidate_arg_map,
        )

    def _load_candidate(self) -> None:
        if self._candidate is not None:
            return
        if not self._gluon_function_name:
            raise UnsupportedReductionContract("Gluon candidate lacks a function name")
        runtime_names = tuple(
            name
            for name in self._gluon_arg_names
            if name not in {"XBLOCK", "R0_BLOCK"}
        )
        if len(runtime_names) != len(self._candidate_arg_map):
            raise UnsupportedReductionContract("Gluon candidate signature does not match the argument mapping")
        from torch._inductor.codecache import PyCodeCache

        module = PyCodeCache.load(self._gluon_source)
        candidate = getattr(module, self._gluon_function_name, None)
        if candidate is None:
            raise UnsupportedReductionContract("Gluon source does not define the candidate function")
        self._candidate = candidate

    @staticmethod
    def _choice_cache() -> Any:
        from torch._inductor.codecache import LocalCache

        return LocalCache()

    def _load_choice(self, identity: ReductionCacheIdentity) -> str | None:
        try:
            value = self._choice_cache().lookup(
                f"preserved_reduction:{identity.cache_key()}"
            )
        except Exception:
            # The choice cache is best-effort (its system probe may touch the device);
            # an unreadable cache only means the one-time selection runs again.
            log.warning("preserved reduction choice cache unreadable", exc_info=True)
            return None
        return value if value in {"standard", "gluon"} else None

    def _store_choice(
        self, identity: ReductionCacheIdentity, choice: str
    ) -> None:
        try:
            self._choice_cache().set_value(
                f"preserved_reduction:{identity.cache_key()}", value=choice
            )
        except Exception:
            # Persisting the choice is optional; a cache failure must not break the
            # model invocation that already produced its outputs.
            log.warning("preserved reduction choice cache unwritable", exc_info=True)
