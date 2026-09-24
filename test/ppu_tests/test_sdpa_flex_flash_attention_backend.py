# CI gate for the SDPA flex_flash_attention backend (PPU torch fork).
#
# Covers EVERY branch of can_use_flex_flash_attention
# (aten/src/ATen/native/transformers/cuda/sdp_utils.cpp), plus numerical
# correctness vs the MATH backend, routing behavior, and the hardening
# suites for a production deployment:
#   A. can_use accept branches      B. can_use reject branches
#   C. routing behavior             D. dropout semantics
#   E. mask cache semantics (in-place mutation / ABA)
#   F. tile-quantization boundary shapes (fwd 128 / bwd 768 Q-tiles)
#   G. torch.compile path           H. multi-thread concurrency
#   I. is_causal non-square semantics J. dropout corner combinations
#   K. peak-memory envelope         L. determinism / deterministic mode
#   M. deployment integrity (installed package sanity)
#   N. randomized fuzz              O. CUDA graph behavior
#   P. grad-mask combos & layouts   Q. autocast (AMP)
#   R. inference/no_grad modes      S. cache-eviction soak
#   T. training-loop integration    U. double backward
#   V. long-sequence numerics       W. env-var kill switch
#
# Oracle for can_use: under sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION])
# exactly one backend is enabled, so
#   call succeeds        <=> can_use == true
#   RuntimeError raised  <=> can_use == false   (no silent fallback exists)
#
# Run (inside the test container, torch wheel with USE_FLEX_FLASH_ATTENTION):
#   python -m pytest test/ppu_tests/test_sdpa_flex_flash_attention_backend.py -v -p no:warnings
#
# Deployment note: after swapping libtorch_cpu.so / libflex_flash_attention.so
# into an installed torch, clear the Inductor cache (rm -rf
# /tmp/torchinductor_* ~/.cache/torch_inductor) — codegen artifacts pin
# stale meta strides across library updates.

import os
import subprocess
import sys

# Enable this backend before importing torch; env-switch probes override their own subprocess environments.
os.environ["TORCH_FLEX_FLASH_SDPA_ENABLED"] = "1"

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BF16 = torch.bfloat16
FP16 = torch.float16
FP32 = torch.float32

DEVICE = "cuda"

# Tolerances vs the MATH fwd reference / fp64 MATH grad oracle, set to the
# measured worst-case diff across this suite times a ~2.7-5x safety margin.
TOL = {
    BF16: dict(atol=2e-2, rtol=2e-2),
    FP16: dict(atol=2e-3, rtol=2e-3),
    FP32: dict(atol=1e-4, rtol=1e-4),
}

def flex_only():
    return sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION])


def math_only():
    return sdpa_kernel([SDPBackend.MATH])

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _mk(b, hq, hkv, sq, sk, d, dv=None, dtype=BF16, requires_grad=True):
    dv = d if dv is None else dv
    q = torch.randn(b, hq, sq, d, device=DEVICE, dtype=dtype,
                    requires_grad=requires_grad)
    k = torch.randn(b, hkv, sk, d, device=DEVICE, dtype=dtype,
                    requires_grad=requires_grad)
    v = torch.randn(b, hkv, sk, dv, device=DEVICE, dtype=dtype,
                    requires_grad=requires_grad)
    return q, k, v


def _causal_mask(sq, sk):
    # PyTorch is_causal semantics: tril(diagonal=0) — top-left aligned,
    # row i attends to columns j <= i, for non-square shapes too
    # (upstream attention.cpp .tril() and the functional.py pseudocode
    # both use diagonal 0).
    idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kidx = torch.arange(sk, device=DEVICE).unsqueeze(0)
    return kidx <= idx


def _holey_mask(sq, sk, seed=0):
    """Random checkerboard-ish mask: far beyond the layered-interval
    envelope, decomposition must fail."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.rand(sq, sk, device=DEVICE, generator=g) > 0.5


# Scheme A: dtype parametrization for the accept suite.  The fp32
# (TF32 compute) kernel set is a mandatory, always-built part of
# libflex_flash_attention, so no runtime probe / skip mark is needed.
ACCEPT_DTYPES = [
    pytest.param(BF16, id="bf16"),
    pytest.param(FP16, id="fp16"),
    pytest.param(FP32, id="fp32"),
]


def _run_flex(q, k, v, mask=None, is_causal=False, dropout_p=0.0):
    # NOTE: the aten op schema carries no generator argument, so dropout
    # always draws from the default CUDA generator; dropout tests seed it
    # with torch.manual_seed and must never run after a graph-capture
    # attempt (see TestCudaGraph findings).
    with flex_only():
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p,
            is_causal=is_causal)


def _ref_math(q, k, v, mask=None, dropout_p=0.0, detach_inputs=True):
    """Forward oracle.  This fork's MATH backend cannot run GQA directly
    (head-count mismatch at the scores matmul), so for GQA we expand k/v
    to the q head count — output is identical by definition."""
    with math_only():
        if detach_inputs:  # fwd comparison: keep grads off the flex inputs
            q, k, v = q.detach(), k.detach(), v.detach()
        hq, hkv = q.size(-3), k.size(-3)
        if hq != hkv:
            rep = hq // hkv
            k = k.repeat_interleave(rep, dim=-3)
            v = v.repeat_interleave(rep, dim=-3)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p)


def _ref_math_grads(q, k, v, grad_out, mask=None, dropout_p=0.0):
    """Gradient oracle.  MATH cannot backprop through GQA broadcast views,
    so for GQA we expand k/v to the q head count and sum-reduce the grads
    back — the exact definition of the GQA k/v gradient.

    The reference math runs in fp64: an fp32 MATH reference carries up to
    ~1e-1 error itself at headdim 256 with cancelling dV terms (arbitrated
    against fp64: the kernel matched truth within bf16 rounding while the
    fp32 reference did not), which produced false fuzz failures."""
    hq, hkv = q.size(-3), k.size(-3)
    q2 = q.detach().double().clone().requires_grad_(True)
    k2 = k.detach().double().clone().requires_grad_(True)
    v2 = v.detach().double().clone().requires_grad_(True)
    if hq != hkv:
        rep = hq // hkv
        k3 = k2.repeat_interleave(rep, dim=-3)
        v3 = v2.repeat_interleave(rep, dim=-3)
    else:
        k3, v3 = k2, v2
    with math_only():
        ref = F.scaled_dot_product_attention(
            q2, k3, v3, attn_mask=mask, dropout_p=dropout_p)
    ref.backward(grad_out.double())
    return (q2.grad.to(q.dtype), k2.grad.to(k.dtype), v2.grad.to(v.dtype))


def _assert_close_flex(q, k, v, mask=None, dropout_p=0.0, check_grad=True):
    """Run flex flash attention backend + MATH reference, compare fwd and bwd."""
    dtype = q.dtype
    out = _run_flex(q, k, v, mask=mask, dropout_p=dropout_p)
    ref = _ref_math(q, k, v, mask=mask, dropout_p=dropout_p)
    torch.testing.assert_close(out, ref, **TOL[dtype])
    if check_grad:
        g = torch.randn_like(out)
        _assert_close_flex.last_grad = g  # for failure dumps in fuzz
        out.backward(g)
        exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, g, mask=mask,
                                              dropout_p=dropout_p)
        for got, exp in ((q.grad, exp_q), (k.grad, exp_k),
                         (v.grad, exp_v)):
            torch.testing.assert_close(got, exp, **TOL[dtype])
    return out


class TinyAttn(torch.nn.Module):
    """Minimal attention block shared by the compile / training suites:
    per-head projections -> SDPA -> output projection (no LN/FFN)."""

    def __init__(self, nhead=4, dim=64):
        super().__init__()
        self.nhead, self.dim = nhead, dim
        self.wq = torch.nn.Linear(dim, nhead * dim, bias=False)
        self.wk = torch.nn.Linear(dim, nhead * dim, bias=False)
        self.wv = torch.nn.Linear(dim, nhead * dim, bias=False)
        self.wo = torch.nn.Linear(nhead * dim, dim, bias=False)

    def forward(self, x, mask):
        B, S, _ = x.shape
        h, d = self.nhead, self.dim
        q = self.wq(x).view(B, S, h, d).transpose(1, 2)
        kk = self.wk(x).view(B, S, h, d).transpose(1, 2)
        vv = self.wv(x).view(B, S, h, d).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        return self.wo(o.transpose(1, 2).reshape(B, S, h * d))


# --------------------------------------------------------------------------
# A. can_use ACCEPT branches  (call must succeed + be numerically correct)
# --------------------------------------------------------------------------
class TestCanUseAccept:
    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_2d_mask(self, dtype):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        mask = torch.tril(torch.ones(256, 256, device=DEVICE, dtype=torch.bool))
        mask[:64, 64:128] = True  # extra block -> 2 intervals/row, still in env
        _assert_close_flex(q, k, v, mask=mask)

    # ── hole-pattern precision ──────────────────────────────────────────
    # Holed masks decompose into layered, pairwise-disjoint slice sets
    # whose union is EXACTLY the mask (never an approximation), so all of
    # these must match the MATH oracle within the normal tolerances.
    # They close the gap between the single 2-interval case above and
    # the reject/routing tests on non-decomposable random masks.

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_holed_isolated_point_tril(self, dtype):
        # Single point-hole in tril: only row 128 splits into 2 intervals
        # (peel layer 2), every other row stays single-interval.
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        mask = torch.tril(torch.ones(256, 256, device=DEVICE, dtype=torch.bool))
        mask[128, 64] = False
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_holed_isolated_point_dense(self, dtype):
        # Point-hole in an otherwise fully-dense mask (adv_probe's
        # dense_hole shape, which the layered scheme must now accept).
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        mask = torch.ones(256, 256, device=DEVICE, dtype=torch.bool)
        mask[128, 64] = False
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_holed_three_intervals(self, dtype):
        # Longformer-like mix: causal local window + global columns + a
        # fixed far block -> rows >= 304 carry 3 disjoint intervals
        # (3 peel layers), still exact.
        q, k, v = _mk(1, 2, 2, 512, 512, 64, dtype=dtype)
        idx = torch.arange(512, device=DEVICE).unsqueeze(1)
        kidx = torch.arange(512, device=DEVICE).unsqueeze(0)
        window = (kidx >= idx - 32) & (kidx <= idx)      # causal window
        globalc = kidx < 16                              # global columns
        farblock = (kidx >= 256) & (kidx < 272) & (idx >= 300)
        mask = window | globalc | farblock
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_holed_sink_plus_window_gqa(self, dtype):
        # StreamingLLM-style sink columns + trailing causal window under
        # GQA: rows >= 80 carry 2 disjoint intervals and the kv-grad
        # reduction across repeated heads must stay exact.
        q, k, v = _mk(1, 8, 2, 256, 256, 64, dtype=dtype)
        idx = torch.arange(256, device=DEVICE).unsqueeze(1)
        kidx = torch.arange(256, device=DEVICE).unsqueeze(0)
        mask = ((kidx >= idx - 64) & (kidx <= idx)) | (kidx < 16)
        _assert_close_flex(q, k, v, mask=mask)

    # ── deep-envelope hole patterns: push peel-layer depth toward the
    # 16-layer cap with a controlled construction (periodic column
    # windows: every row carries exactly n disjoint intervals, so the
    # decomposition needs exactly n peel layers).

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    @pytest.mark.parametrize("n_intervals", [10, 15])
    def test_holed_periodic_intervals(self, dtype, n_intervals):
        # n=15 sits exactly ON the real accept envelope: _MAX_HOLE_LAYERS
        # is 16 peels, but the residual-empty return only fires BEFORE a
        # peel, so a mask whose residual empties on the 16th peel (i.e.
        # 16 intervals/row) still raises/rejects (Python raise and the
        # device decomp_finalize_kernel mirror each other deliberately).
        # All rows identical: each peel layer emits one FULL slice.
        period, win, sq = 48, 16, 128
        sk = period * n_intervals
        q, k, v = _mk(1, 2, 2, sq, sk, 64, dtype=dtype)
        row = (torch.arange(sk, device=DEVICE) % period) < win
        mask = row.repeat(sq, 1)  # exactly n_intervals intervals/row
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    @pytest.mark.parametrize("n_intervals", [10, 15])
    def test_holed_periodic_intervals_runs_on_flex(self, dtype, n_intervals):
        # Profiler evidence: deep-envelope masks must actually execute on
        # the flex flash attention backend (flex_flash_attention::fwd op + its kernel),
        # not silently pass through another path.
        from torch.profiler import profile, ProfilerActivity
        period, win, sq = 48, 16, 128
        sk = period * n_intervals
        q, k, v = _mk(1, 2, 2, sq, sk, 64, dtype=dtype, requires_grad=False)
        row = (torch.arange(sk, device=DEVICE) % period) < win
        mask = row.repeat(sq, 1)
        with flex_only():
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        assert any("flex_flash_attention::fwd" in n for n in names), names
        assert any("_scaled_dot_product_flex_flash_attention" in n
                   for n in names), names

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_causal_mask(self, dtype):
        # attn_mask carrier (vs the is_causal=True flag tested below)
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        mask = _causal_mask(256, 256)
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_no_mask(self, dtype):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        _assert_close_flex(q, k, v, mask=None)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_is_causal(self, dtype):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        out = _run_flex(q, k, v, is_causal=True)
        ref = _ref_math(q, k, v, mask=_causal_mask(256, 256))
        torch.testing.assert_close(out, ref, **TOL[dtype])

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_4d_singleton_mask(self, dtype):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype)
        m2d = torch.tril(torch.ones(256, 256, device=DEVICE, dtype=torch.bool))
        _assert_close_flex(q, k, v, mask=m2d.unsqueeze(0).unsqueeze(0))

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_gqa_q8_kv2(self, dtype):
        q, k, v = _mk(2, 8, 2, 256, 256, 64, dtype=dtype)
        mask = _causal_mask(256, 256)
        # reference: expand kv heads, MATH does not do GQA expansion itself
        k_exp = k.detach().repeat_interleave(4, dim=1)
        v_exp = v.detach().repeat_interleave(4, dim=1)
        out = _run_flex(q, k, v, mask=mask)
        ref = _ref_math(q, k_exp, v_exp, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[dtype])

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_head_dim_v_neq_head_dim(self, dtype):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dv=80, dtype=dtype)
        mask = _causal_mask(256, 256)
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    @pytest.mark.parametrize("hd", [96, 128, 192, 256])
    def test_headdim_tiers(self, hd, dtype):
        q, k, v = _mk(1, 2, 2, 128, 128, hd, dtype=dtype)
        mask = _causal_mask(128, 128)
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_rectangular_sq_neq_sk(self, dtype):
        q, k, v = _mk(2, 4, 4, 128, 320, 64, dtype=dtype)
        mask = _causal_mask(128, 320)
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_long_seqlen_4096(self, dtype):
        q, k, v = _mk(1, 2, 2, 4096, 4096, 64, dtype=dtype)
        mask = _causal_mask(4096, 4096)
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_seqlen_boundary_65536(self, dtype):
        # exactly at the cap, no mask (mask would cost 65536^2 bools);
        # only assert can_use accepts + output finite
        q, k, v = _mk(1, 1, 1, 65536, 65536, 8, dtype=dtype,
                      requires_grad=False)
        out = _run_flex(q, k, v)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("dtype", ACCEPT_DTYPES)
    def test_dropout_zero_equals_nodropout(self, dtype):
        q, k, v = _mk(2, 4, 4, 256, 256, 64, dtype=dtype, requires_grad=False)
        mask = _causal_mask(256, 256)
        out_p0 = _run_flex(q, k, v, mask=mask, dropout_p=0.0)
        out_none = _run_flex(q, k, v, mask=mask)
        torch.testing.assert_close(out_p0, out_none)  # bit-equal expected


# --------------------------------------------------------------------------
# B. can_use REJECT branches (forced backend must raise, no silent fallback)
# --------------------------------------------------------------------------
class TestCanUseReject:
    def _expect_reject(self, q, k, v, mask=None, dropout_p=0.0,
                       is_causal=False):
        with pytest.raises(RuntimeError):
            with flex_only():
                F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout_p,
                    is_causal=is_causal)

    def test_reject_user_disabled(self):
        enable = getattr(torch.backends.cuda,
                         "enable_flex_flash_attention_sdp", None)
        if enable is None:
            pytest.skip("no user toggle exposed")
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        # the toggle must be flipped INSIDE the context: entering flex_only()
        # re-enables every listed backend via _set_sdp_use_*
        with flex_only():
            enable(False)
            try:
                with pytest.raises(RuntimeError):
                    F.scaled_dot_product_attention(q, k, v)
            finally:
                enable(True)

    def test_reject_nested(self):
        try:
            nt = torch.nested.nested_tensor(
                [torch.randn(2, 128, 64), torch.randn(2, 64, 64)],
                device=DEVICE, dtype=BF16)
        except Exception:
            pytest.skip("nested tensor construction unsupported here")
        with pytest.raises(Exception):
            with flex_only():
                F.scaled_dot_product_attention(nt, nt, nt)

    def test_reject_cpu_tensors(self):
        q = torch.randn(1, 2, 64, 64, dtype=BF16)
        with pytest.raises(Exception):
            with flex_only():
                F.scaled_dot_product_attention(q, q, q)

    def test_reject_fp64(self):
        # fp32 is accepted (isolated fwd_f32/bwd_f32 set); fp64 is not.
        q, k, v = _mk(1, 2, 2, 128, 128, 64, dtype=torch.float64,
                      requires_grad=False)
        self._expect_reject(q, k, v)

    def test_reject_dtype_mismatch(self):
        q = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=FP16)
        v = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, v)

    def test_reject_dropout_one(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        self._expect_reject(q, k, v, dropout_p=1.0)

    def test_reject_dropout_negative(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        self._expect_reject(q, k, v, dropout_p=-0.1)

    def test_reject_kv_head_mismatch(self):
        q = torch.randn(1, 4, 128, 64, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 128, 64, device=DEVICE, dtype=BF16)
        v = torch.randn(1, 3, 128, 64, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, v)

    def test_reject_q_heads_not_divisible(self):
        q, k, v = _mk(1, 5, 2, 128, 128, 64, requires_grad=False)
        self._expect_reject(q, k, v)

    def test_reject_head_dim_gt_256(self):
        q, k, v = _mk(1, 1, 1, 64, 64, 257, requires_grad=False)
        self._expect_reject(q, k, v)

    def test_reject_seqlen_zero(self):
        q = torch.empty(1, 2, 0, 64, device=DEVICE, dtype=BF16)
        k = torch.empty(1, 2, 0, 64, device=DEVICE, dtype=BF16)
        v = torch.empty(1, 2, 0, 64, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, v)

    def test_reject_seqlen_gt_16m_masked(self):
        # The cascaded phase-B scan caps explicit-mask decomposition at
        # 16777216 rows/cols; beyond that the can_use gate rejects before
        # any large allocation (long-row short-col mask keeps memory tiny).
        q, k, v = _mk(1, 1, 1, 16777217, 1, 8, requires_grad=False)
        mask = torch.ones(16777217, 1, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_accept_seqlen_gt_65536_masked(self):
        # Since the cascaded phase-B scan, explicit masks beyond the old
        # 64K cap decompose and run fine (long-row short-col keeps the
        # mask tiny; nb=257 exercises the cascaded path).
        q, k, v = _mk(1, 1, 1, 65537, 128, 8, requires_grad=False)
        mask = torch.ones(65537, 128, device=DEVICE, dtype=torch.bool)
        with flex_only():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.shape == q.shape

    def test_accept_seqlen_gt_65536_unmasked(self):
        # Without a mask the synthetic single-slice descriptors bypass the
        # decomposition entirely, so >64K sequences stay admissible.
        q, k, v = _mk(1, 1, 1, 65537, 65537, 8, requires_grad=False)
        with flex_only():
            out = F.scaled_dot_product_attention(q, k, v)
        assert out.shape == q.shape

    def test_reject_float_mask(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.zeros(128, 128, device=DEVICE, dtype=FP32)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_non_4d(self):
        # The kernels consume strictly 4-D BHSD; a 3-D input must fall
        # back instead of crashing inside the op.
        t = torch.randn(2, 64, 32, device=DEVICE, dtype=BF16)
        self._expect_reject(t, t, t)

    def test_reject_strided_last_dim(self):
        # The kernels load the head dim with unit stride and the glue
        # does not re-layout, so a strided last dim must fall back.
        full = torch.randn(1, 2, 64, 16, device=DEVICE, dtype=BF16)
        t = full[..., ::2]
        assert t.stride(-1) == 2
        self._expect_reject(t, t, t)

    def test_reject_zero_heads(self):
        # A zero head count used to hit a division by zero in the gate's
        # GQA modulo (process-level SIGFPE, uncatchable from Python);
        # it must be rejected cleanly instead.
        t = torch.randn(1, 0, 64, 16, device=DEVICE, dtype=BF16)
        self._expect_reject(t, t, t)

    def test_reject_batch_mismatch(self):
        # The API derives the batch from q and shape-checks k/v against
        # it, so mismatched batches must fall back at the gate.
        q = torch.randn(2, 2, 64, 16, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 64, 16, device=DEVICE, dtype=BF16)
        self._expect_reject(q, k, k)

    def test_reject_mask_3d(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(1, 128, 128, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_batch_gt_1(self):
        q, k, v = _mk(2, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(2, 1, 128, 128, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_head_gt_1(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(1, 2, 128, 128, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_wrong_seqlen(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(64, 64, device=DEVICE, dtype=torch.bool)
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_mask_on_cpu(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64, requires_grad=False)
        mask = torch.ones(128, 128, dtype=torch.bool)  # CPU
        self._expect_reject(q, k, v, mask=mask)

    def test_reject_non_decomposable_mask(self):
        # random ~50% mask: interval layers exceed the 16-layer envelope
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        self._expect_reject(q, k, v, mask=_holey_mask(256, 256))

    def test_reject_16_intervals(self):
        # One step over the REAL accept envelope (companion of
        # TestCanUseAccept::test_holed_periodic_intervals at n=15):
        # 16 intervals/row empty the residual only on the 16th peel,
        # which the loop never checks -> must be rejected (the off-by-one
        # in _MAX_HOLE_LAYERS is mirrored deliberately by the device
        # finalizer; pin the actual behavior until it is ever revisited).
        period, win, n, sq = 48, 16, 16, 128
        sk = period * n
        q, k, v = _mk(1, 2, 2, sq, sk, 64, requires_grad=False)
        row = (torch.arange(sk, device=DEVICE) % period) < win
        mask = row.repeat(sq, 1)
        self._expect_reject(q, k, v, mask=mask)


# --------------------------------------------------------------------------
# C. Routing behavior
# --------------------------------------------------------------------------
class TestRouting:
    def test_holed_mask_default_mode_routes_elsewhere(self):
        # Default (all backends enabled): non-decomposable mask must NOT
        # fail — another backend (math) takes over.  Profiler evidence:
        # the can_use probe may run flex_flash_attention::decompose_mask (it
        # must decompose to learn the mask is outside the envelope), but
        # the COMPUTE ops fwd/bwd must never run (correct fallback vs a
        # can_use envelope bug silently accepting the mask).
        from torch.profiler import profile, ProfilerActivity
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _holey_mask(256, 256)
        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        assert any("flex_flash_attention::decompose_mask" in n for n in names), \
            f"can_use probe did not run: {names}"
        assert not any("flex_flash_attention::fwd" in n or "flex_flash_attention::bwd" in n
                       for n in names), \
            f"flex flash attention backend must not compute for non-decomposable mask: {names}"
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])

    def test_forced_backend_is_actually_used(self):
        # Direct evidence: the profiler must show flex_flash_attention ops and
        # the kernel from libflex_flash_attention.so.
        from torch.profiler import profile, ProfilerActivity
        q = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=BF16)
        mask = _causal_mask(256, 256)
        with flex_only():
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
                torch.cuda.synchronize()
        names = [e.key for e in prof.key_averages()]
        assert any("flex_flash_attention::fwd" in n for n in names), names
        assert any("_scaled_dot_product_flex_flash_attention" in n
                   for n in names), names


# --------------------------------------------------------------------------
# D. Dropout semantics
# --------------------------------------------------------------------------
class TestDropout:
    def test_dropout_same_seed_deterministic(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(123)
        out1 = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        torch.manual_seed(123)
        out2 = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        assert torch.equal(out1, out2), "same seed must reproduce dropout"

    def test_dropout_holed_mask_same_seed_deterministic(self):
        # dropout over a 2-interval (sink + window) mask: the holed slice
        # layout must not perturb the rng-state consumption order.
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        idx = torch.arange(256, device=DEVICE).unsqueeze(1)
        kidx = torch.arange(256, device=DEVICE).unsqueeze(0)
        mask = ((kidx >= idx - 64) & (kidx <= idx)) | (kidx < 16)
        torch.manual_seed(123)
        out1 = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        torch.manual_seed(123)
        out2 = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        assert torch.equal(out1, out2), "same seed must reproduce dropout"

    def test_dropout_different_seed_differs(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(123)
        out1 = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        torch.manual_seed(456)
        out2 = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        assert not torch.equal(out1, out2)

    def test_dropout_fwd_bwd_same_seed_consistent(self):
        # bwd must reuse fwd's rng_state: same seed -> identical grads.
        # Inputs are built ONCE (outside the seeded section) so both runs
        # start from identical RNG state.
        q0, k0, v0 = _mk(1, 2, 2, 256, 256, 64)
        qd, kd, vd = (t.detach() for t in (q0, k0, v0))
        mask = _causal_mask(256, 256)

        def run(seed):
            q = qd.clone().requires_grad_(True)
            k = kd.clone().requires_grad_(True)
            v = vd.clone().requires_grad_(True)
            torch.manual_seed(seed)
            out = _run_flex(q, k, v, mask=mask, dropout_p=0.3)
            out.backward(torch.ones_like(out))
            return out.detach(), q.grad, k.grad, v.grad

        o1, dq1, dk1, dv1 = run(777)
        o2, dq2, dk2, dv2 = run(777)
        assert torch.equal(o1, o2)
        for a, b in ((dq1, dq2), (dk1, dk2), (dv1, dv2)):
            assert torch.equal(a, b), "grads must reproduce under same seed"

    def test_dropout_stats_reasonable(self):
        # averaged over many runs, dropout output magnitude should not blow
        # up; (1-p) scaling is applied internally so means stay comparable.
        q, k, v = _mk(1, 4, 4, 512, 512, 64, requires_grad=False)
        mask = _causal_mask(512, 512)
        out0 = _run_flex(q, k, v, mask=mask, dropout_p=0.0)
        torch.manual_seed(0)
        out_p = _run_flex(q, k, v, mask=mask, dropout_p=0.5)
        r = (out_p.norm() / out0.norm()).item()
        assert 0.4 < r < 1.6, f"dropout output norm ratio {r} implausible"


# --------------------------------------------------------------------------
# E. Mask cache semantics (in-place mutation / ABA)
# --------------------------------------------------------------------------
class TestMaskCacheSemantics:
    def test_inplace_mutation_invalidates_cache(self):
        # Same tensor object, content mutated in place (version bump): the
        # resolve/decompose caches must NOT serve the stale decomposition.
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        out_causal = _run_flex(q, k, v, mask=mask)
        ref_causal = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out_causal, ref_causal, **TOL[BF16])
        mask.fill_(True)  # in-place: now dense, same TensorImpl + data_ptr
        out_dense = _run_flex(q, k, v, mask=mask)
        ref_dense = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out_dense, ref_dense, **TOL[BF16])
        # and switching back again must still be correct
        mask.copy_(_causal_mask(256, 256))
        out_again = _run_flex(q, k, v, mask=mask)
        torch.testing.assert_close(out_again, ref_causal, **TOL[BF16])

    def test_mutation_between_fwd_bwd_raises(self):
        # PyTorch contract: do not mutate inputs between fwd and bwd.
        # The autograd version guard must detect it and raise cleanly
        # (never silently produce stale-mask gradients or crash).
        q, k, v = _mk(1, 2, 2, 256, 256, 64)
        mask = _causal_mask(256, 256)
        out = _run_flex(q, k, v, mask=mask)
        mask.fill_(True)
        with pytest.raises(RuntimeError, match="inplace operation"):
            out.backward(torch.ones_like(out))

    def test_different_tensor_same_shape_distinct_results(self):
        # two distinct mask tensors (different TensorImpl) used alternately
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        m1 = _causal_mask(256, 256)
        m2 = torch.ones(256, 256, device=DEVICE, dtype=torch.bool)
        o1 = _run_flex(q, k, v, mask=m1)
        o2 = _run_flex(q, k, v, mask=m2)
        o1b = _run_flex(q, k, v, mask=m1)
        torch.testing.assert_close(o1, _ref_math(q, k, v, mask=m1), **TOL[BF16])
        torch.testing.assert_close(o2, _ref_math(q, k, v, mask=m2), **TOL[BF16])
        assert torch.equal(o1, o1b)

    def test_cache_hit_skips_decompose_kernels(self):
        # Decomp results are cached INSIDE flex_flash_attention::decompose_mask
        # (the op itself still dispatches every call), keyed by
        # (TensorImpl*, version, kblock_m).  A second call with the same
        # mask object must serve the cache and launch ZERO decompose
        # kernels on the GPU.  Strong assertion on the device pipeline's
        # kernel symbols (mask_decomp_kernels.cu; row stats are fused
        # into the WITH_FLAGS peel variant):
        _DECOMP_KERNELS = ("peel_kernel", "decomp_gate_done_kernel",
                           "decomp_finalize_kernel",
                           "decomp_guard_zero_kernel")
        from torch.profiler import profile, ProfilerActivity
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)   # fresh identity: miss, then hit

        def cuda_kernel_names():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _run_flex(q, k, v, mask=mask)
                torch.cuda.synchronize()
            return [e.key for e in prof.key_averages()
                    if e.device_type == torch.autograd.DeviceType.CUDA]

        miss = cuda_kernel_names()      # 1st call: full decomp pipeline
        hit = cuda_kernel_names()       # 2nd call: cache serves the desc
        assert any("peel_kernel" in n for n in miss), miss
        assert any("decomp_finalize_kernel" in n for n in miss), miss
        assert not any(any(d in n for d in _DECOMP_KERNELS) for n in hit), hit


# --------------------------------------------------------------------------
# F. Tile-quantization boundary shapes
# --------------------------------------------------------------------------
class TestTileBoundaries:
    # fwd Q-tile = 128 rows, bwd Q-tile = 768 rows: exercise the grid
    # boundaries where peel decomposition and kernel tiling meet.
    @pytest.mark.parametrize("s", [127, 128, 129, 255, 256, 257,
                                   767, 768, 769])
    def test_seqlen_boundaries(self, s):
        q, k, v = _mk(1, 2, 2, s, s, 64)
        mask = _causal_mask(s, s)
        _assert_close_flex(q, k, v, mask=mask)

    def test_seqlen_q_one(self):
        q, k, v = _mk(1, 2, 2, 1, 256, 64)
        mask = _causal_mask(1, 256)
        _assert_close_flex(q, k, v, mask=mask)

    def test_seqlen_k_one(self):
        q, k, v = _mk(1, 2, 2, 256, 1, 64)
        mask = torch.ones(256, 1, device=DEVICE, dtype=torch.bool)
        _assert_close_flex(q, k, v, mask=mask)

    @pytest.mark.parametrize("hd", [1, 8, 65, 127])
    def test_odd_head_dims(self, hd):
        # kernel rounds head_dim up to a padded width; odd widths must work
        q, k, v = _mk(1, 2, 2, 128, 128, hd)
        mask = _causal_mask(128, 128)
        _assert_close_flex(q, k, v, mask=mask)


# --------------------------------------------------------------------------
# G. torch.compile path (meta impls + FakeTensor probe bypass)
# --------------------------------------------------------------------------
class TestTorchCompile:
    def test_compile_fwd_bwd_matches_eager(self):
        def fn(q, k, v, mask):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        mask = _causal_mask(256, 256)
        with flex_only():
            eager = fn(q, k, v, mask)
            cfn = torch.compile(fn, fullgraph=True)
            comp = cfn(q, k, v, mask)
        torch.testing.assert_close(comp, eager, **TOL[BF16])
        # backward through the compiled graph
        g = torch.randn_like(eager)
        eager.backward(g)
        grads_eager = (q.grad.clone(), k.grad.clone(), v.grad.clone())
        q.grad = k.grad = v.grad = None
        with flex_only():
            comp2 = cfn(q, k, v, mask)
        comp2.backward(g)
        for ge, gc in zip(grads_eager, (q.grad, k.grad, v.grad)):
            torch.testing.assert_close(gc, ge, **TOL[BF16])

    def test_compile_attention_block_training(self):
        # Module-level compile: real graph structure (projections -> SDPA ->
        # output projection).  fullgraph=True fails on ANY graph break, so
        # this locks compile-safety of the backend inside a real module;
        # the loss-decrease check locks the compiled training loop.
        # autocast(bf16) feeds SDPA bf16 (the backend rejects fp32) while
        # keeping Linear math in fp32, same as the T-suite integration.
        torch.manual_seed(42)
        h, d, s, b = 4, 64, 128, 2
        model = TinyAttn(nhead=h, dim=d).to(DEVICE)
        x = torch.randn(b, s, d, device=DEVICE)
        target = torch.randn(b, s, d, device=DEVICE)
        mask = _causal_mask(s, s)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        cmodel = torch.compile(model, fullgraph=True)
        losses = []
        for _ in range(3):
            opt.zero_grad()
            with flex_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(cmodel(x, mask), target)
            loss.backward()
            assert all(torch.isfinite(p.grad).all()
                       for p in model.parameters())
            opt.step()
            losses.append(loss.item())
            assert torch.isfinite(loss)
        assert losses[-1] < losses[0], \
            f"compiled loss did not decrease: {losses}"


# --------------------------------------------------------------------------
# H. Multi-thread concurrency (global caches under mutex)
# --------------------------------------------------------------------------
class TestConcurrency:
    def test_parallel_calls_consistent(self):
        import concurrent.futures
        specs = [(1, 2, 2, 256, 256, 64), (2, 4, 4, 192, 192, 64),
                 (1, 8, 2, 256, 256, 64), (2, 2, 2, 320, 320, 96),
                 (1, 2, 2, 128, 384, 64), (1, 4, 4, 256, 256, 128),
                 (2, 2, 1, 256, 256, 64), (1, 2, 2, 384, 128, 64)]
        inputs, serial_outs = [], []
        for (b, hq, hkv, sq, sk, d) in specs:
            q, k, v = _mk(b, hq, hkv, sq, sk, d, requires_grad=False)
            mask = _causal_mask(sq, sk)
            inputs.append((q, k, v, mask))
            with flex_only():
                serial_outs.append(F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask))

        def worker(i):
            q, k, v, mask = inputs[i]
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # flex_only sets the global backend flags for the whole process;
        # threads inherit them, so no per-thread flag toggling (which would
        # race on the same global state).
        with flex_only():
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=8) as pool:
                outs = list(pool.map(worker, range(len(specs))))
        for out, ref in zip(outs, serial_outs):
            assert torch.equal(out, ref), "parallel result diverges"


# --------------------------------------------------------------------------
# I. is_causal non-square semantics (torch = top-left aligned, tril d=0)
# --------------------------------------------------------------------------
class TestCausalNonSquare:
    @pytest.mark.parametrize("sq,sk", [(128, 320), (320, 128), (1, 256),
                                       (256, 1)])
    def test_is_causal_topleft_aligned(self, sq, sk):
        # fwd AND bwd through the synthesized is_causal path, referenced
        # against an explicit top-left (tril diagonal 0) bool mask: pins
        # the synthesis in both directions (bwd consumes the same
        # resolved mask but on its own 768-row tile grid).
        q, k, v = _mk(1, 2, 2, sq, sk, 64)
        mask_ref = _causal_mask(sq, sk)
        out = _run_flex(q, k, v, is_causal=True)
        torch.testing.assert_close(out, _ref_math(q, k, v, mask=mask_ref),
                                   **TOL[BF16])
        # cross-arbitrate against this fork's OWN is_causal
        # implementation: guards against a wrong _causal_mask formula
        # passing by self-consistency, and against semantics drift.
        with math_only():
            out_torch = F.scaled_dot_product_attention(
                q.detach(), k.detach(), v.detach(), is_causal=True)
        torch.testing.assert_close(out, out_torch, **TOL[BF16])
        g = torch.randn_like(out)
        out.backward(g)
        exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, g, mask=mask_ref)
        for got, exp in ((q.grad, exp_q), (k.grad, exp_k),
                         (v.grad, exp_v)):
            torch.testing.assert_close(got, exp, **TOL[BF16])


# --------------------------------------------------------------------------
# J. Dropout corner combinations
# --------------------------------------------------------------------------
class TestDropoutCorners:
    def test_dropout_near_one(self):
        # Inverted dropout at p=0.999.  The OUTPUT is a weighted sum over
        # the surviving attention entries, so output sparsity is low by
        # construction; the meaningful check is that the retained attention
        # mass is tiny: with softmax probs summing to 1 per row, keeping
        # ~0.1% of entries leaves a near-zero sum before the 1/(1-p) scale.
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(3)
        out = _run_flex(q, k, v, mask=mask, dropout_p=0.999)
        assert torch.isfinite(out).all()
        # Causal rows average ~128 visible entries; at p=0.999 the expected
        # survivors per row is ~0.13, so the majority of rows drop EVERY
        # entry and must output an all-zero row (lse = -inf handled).
        zero_rows = (out.abs().amax(dim=-1) == 0).float().mean().item()
        assert zero_rows > 0.5, f"expected mostly zero rows, got {zero_rows}"
        # kept entries carry the 1/(1-p)=1000x inverted-dropout scale
        assert out.abs().max().item() <= 1000.0 * v.abs().max().item() * 1.01

    def test_dropout_gqa_deterministic_seed(self):
        q, k, v = _mk(1, 8, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        torch.manual_seed(11)
        o1 = _run_flex(q, k, v, mask=mask, dropout_p=0.2)
        torch.manual_seed(11)
        o2 = _run_flex(q, k, v, mask=mask, dropout_p=0.2)
        assert torch.equal(o1, o2)

    def test_dropout_hd_v_neq_hd(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, dv=80)
        mask = _causal_mask(256, 256)
        torch.manual_seed(5)
        out = _run_flex(q, k, v, mask=mask, dropout_p=0.2)
        assert torch.isfinite(out).all()
        out.backward(torch.ones_like(out))
        for t in (q, k, v):
            assert torch.isfinite(t.grad).all()


# --------------------------------------------------------------------------
# K. Peak-memory envelope (no S x S materialization)
# --------------------------------------------------------------------------
class TestMemoryEnvelope:
    def test_peak_memory_sub_quadratic(self):
        s = 4096
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        q, k, v = _mk(1, 2, 2, s, s, 64, requires_grad=False)
        with flex_only():
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        # scores matrix would cost s*s*4 (fp32) = 64MB at s=4096; the fused
        # path must stay far below that (mask synthesis is s*s*1 = 16MB).
        assert peak < s * s * 3, f"peak {peak} bytes suggests SxS materialization"


# --------------------------------------------------------------------------
# L. Determinism
# --------------------------------------------------------------------------
class TestDeterminism:
    def test_fwd_bitwise_repeatable(self):
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        o1 = _run_flex(q, k, v, mask=mask)
        o2 = _run_flex(q, k, v, mask=mask)
        assert torch.equal(o1, o2), "dropout=0 fwd must be deterministic"

    def test_deterministic_mode_no_crash(self):
        q, k, v = _mk(1, 2, 2, 128, 128, 64)
        mask = _causal_mask(128, 128)
        try:
            torch.use_deterministic_algorithms(True)
            with flex_only():
                out = F.scaled_dot_product_attention(q, k, v,
                                                     attn_mask=mask)
                out.backward(torch.ones_like(out))
            assert torch.isfinite(out).all()
        finally:
            torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# M. Deployment integrity (installed package sanity)
# --------------------------------------------------------------------------
class TestDeploymentIntegrity:
    def test_torch_imported_from_site_packages(self):
        import site
        assert any(sp in torch.__file__ for sp in site.getsitepackages()), \
            f"torch loaded from {torch.__file__} — source-tree shadowing?"

    def test_library_present_and_loadable(self):
        import os
        lib = os.path.join(os.path.dirname(torch.__file__), "lib",
                           "libflex_flash_attention.so")
        assert os.path.exists(lib), f"{lib} missing from installed torch"
        # backend ops registered by the loaded library
        assert hasattr(torch.ops, "flex_flash_attention")
        assert torch.backends.cuda.flex_flash_attention_sdp_enabled()


# --------------------------------------------------------------------------
# N. Randomized fuzz (random decomposable masks x random shapes)
# --------------------------------------------------------------------------
def _rand_stair_mask(sq, sk, seed):
    """Piecewise-constant staircase with <=6 row segments: each segment is
    one rectangle, so peel layers stay far inside the 16-layer envelope."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nseg = min(torch.randint(2, 7, (1,), generator=g).item(), max(sq - 1, 1))
    edges = sorted(set(
        torch.randint(1, max(sq, 2), (nseg - 1,), generator=g).tolist())) \
        if sq > 1 else []
    ks = torch.zeros(sq, dtype=torch.long)
    ke = torch.full((sq,), sk - 1, dtype=torch.long)
    prev = 0
    for seg, end in enumerate(edges + [sq]):
        s = torch.randint(0, sk, (1,), generator=g).item()
        e = torch.randint(s, sk, (1,), generator=g).item()
        ks[prev:end] = s
        ke[prev:end] = e
        prev = end
    j = torch.arange(sk)
    return ((j[None, :] >= ks[:, None]) & (j[None, :] <= ke[:, None])).to(
        DEVICE)


def _rand_band_mask(sq, sk, seed):
    """Random diagonal band: one interval per row."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randint(1, max(sk // 4, 2), (1,), generator=g).item()
    shift = torch.randint(-sk // 4, sk // 4, (1,), generator=g).item()
    i = torch.arange(sq)[:, None]
    j = torch.arange(sk)[None, :]
    center = i * sk // max(sq, 1) + shift
    return ((j >= center - w) & (j <= center + w)).to(DEVICE)


def _cancel_cond_grads(q, k, v, grad_out, mask):
    """Per-cell cancellation condition numbers for the dK/dV gradients.

    Each dK/dV cell is a sum over q rows of signed terms; bf16 rounding
    perturbs each term by ~eps*|term|, so the achievable absolute error
    scales with ||terms||_2 — NOT with the (possibly cancelling) net sum.
    Returns (cond_k, cond_v), each (b, hkv, sk, d): the l2 norm over q
    rows of the term magnitudes."""
    hq, hkv = q.size(-3), k.size(-3)
    rep = hq // hkv
    scale = q.size(-1) ** -0.5
    if mask is not None and mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    qf = q.detach().double()
    kf = k.detach().double().repeat_interleave(rep, dim=-3)
    vf = v.detach().double().repeat_interleave(rep, dim=-3)
    gf = grad_out.detach().double()
    mb = mask.bool() if mask is not None else None
    b_, _, sk_, _ = kf.shape
    cond_k = torch.zeros(b_, hkv, sk_, q.size(-1),
                         device=q.device, dtype=torch.float64)
    cond_v = torch.zeros(b_, hkv, sk_, v.size(-1),
                         device=q.device, dtype=torch.float64)
    for bb in range(b_):
        for hh in range(hq):
            s = (qf[bb, hh] @ kf[bb, hh].T) * scale
            if mb is not None:
                s = s.masked_fill(~mb[bb % mb.size(0), hh % mb.size(1)],
                                  float('-inf'))
            p = torch.softmax(s, dim=-1).nan_to_num(0.0)
            # dV[n, c] = sum_m P[m, n] * dO[m, c]
            cond_v[bb, hh // rep] += (p.unsqueeze(-1) * gf[
                bb, hh].unsqueeze(1)).norm(dim=0) ** 2
            # dK[n, c] = scale * sum_m dS[m, n] * Q[m, c]
            dP = gf[bb, hh] @ vf[bb, hh].T
            delta = (p * dP).sum(-1, keepdim=True)
            dS = p * (dP - delta)
            cond_k[bb, hh // rep] += (dS.unsqueeze(-1) * qf[
                bb, hh].unsqueeze(1)).norm(dim=0) ** 2
    return cond_k.sqrt() * scale, cond_v.sqrt()


def _assert_close_grad_tol(got, exp, dtype, cond=None):
    """assert_close with an extra bf16-rounding allowance: cells fed by
    cancelling q-row sums get tol widened by 4*eps*||terms||_2 (the
    observed random-walk error is ~1*eps*l2, so 4x keeps real O(0.1)
    mapping bugs detectable).  cond=None keeps the plain tolerance."""
    if cond is None or dtype != BF16:
        torch.testing.assert_close(got, exp, **TOL[dtype])
        return
    atol, rtol = TOL[dtype]["atol"], TOL[dtype]["rtol"]
    diff = (got.double() - exp.double()).abs()
    tol = atol + rtol * exp.double().abs() + 4 * (2 ** -8) * cond
    bad = diff > tol
    nbad = int(bad.sum().item())
    assert nbad == 0, (
        f"{nbad}/{diff.numel()} grad cells exceed cancellation-aware "
        f"tolerance; worst diff {diff[bad].max().item():.6g}")


class TestFuzz:
    def test_random_shapes_and_masks(self):
        # Randomized coverage of the accept path: shapes / GQA ratios /
        # headdims / mask geometries a serving stack could plausibly send.
        g = torch.Generator(device="cpu").manual_seed(20260826)
        torch.manual_seed(20260826)
        for i in range(30):
            hkv = torch.randint(1, 3, (1,), generator=g).item()
            hq = hkv * torch.randint(1, 5, (1,), generator=g).item()
            b = torch.randint(1, 3, (1,), generator=g).item()
            sq = torch.randint(16, 1025, (1,), generator=g).item()
            sk = torch.randint(16, 1025, (1,), generator=g).item()
            d = torch.tensor([16, 64, 96, 128, 192, 256])[
                torch.randint(0, 6, (1,), generator=g)].item()
            q, k, v = _mk(b, hq, hkv, sq, sk, d)
            if i % 3 == 0:
                mask = _rand_stair_mask(sq, sk, seed=i)
            elif i % 3 == 1:
                mask = _rand_band_mask(sq, sk, seed=i)
            else:
                mask = _causal_mask(sq, sk)
            print(f"FUZZ_CONFIG i={i} b={b} hq={hq} hkv={hkv} sq={sq} "
                  f"sk={sk} d={d} mask_type={i % 3}")
            try:
                # fwd on the shared helper, grads checked with the
                # cancellation-aware tolerance (bf16 sums over q rows can
                # cancel to ~0 while per-term rounding stays ~eps*l2).
                dtype = q.dtype
                out = _run_flex(q, k, v, mask=mask)
                ref = _ref_math(q, k, v, mask=mask)
                torch.testing.assert_close(out, ref, **TOL[dtype])
                gout = torch.randn_like(out)
                _assert_close_flex.last_grad = gout
                out.backward(gout)
                exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, gout,
                                                      mask=mask)
                cond_k, cond_v = _cancel_cond_grads(q, k, v, gout, mask)
                torch.testing.assert_close(q.grad, exp_q, **TOL[dtype])
                _assert_close_grad_tol(k.grad, exp_k, dtype, cond_k)
                _assert_close_grad_tol(v.grad, exp_v, dtype, cond_v)
            except Exception:
                print(f"FUZZ_FAIL i={i} b={b} hq={hq} hkv={hkv} sq={sq} "
                      f"sk={sk} d={d} mask_type={i % 3}")
                torch.save({"q": q.detach().cpu(), "k": k.detach().cpu(),
                            "v": v.detach().cpu(),
                            "mask": mask.detach().cpu(),
                            "g": getattr(_assert_close_flex, "last_grad",
                                         torch.empty(0)).detach().cpu()},
                           "/tmp/fuzz_fail.pt")
                print("FUZZ_FAIL_DUMP /tmp/fuzz_fail.pt")
                raise


# --------------------------------------------------------------------------
# O. CUDA graph support (warmup-then-capture, in-graph decomposition)
# --------------------------------------------------------------------------
# MODEL (implemented 2026-09; the old "capture unsupported" finding is
# superseded):
#   * capture IS supported for fixed AND per-replay changing masks,
#     fwd-only (inference) and fwd+bwd (training), dropout on/off.
#   * the capture-time gate decides from a HOST-side verdict recorded by
#     the eager warmup probe (no device syncs during capture); capturing
#     a never-warmed mask rejects cleanly.
#   * inside the graph the mask decomposition re-runs on EVERY replay
#     (decompose_mask_nocache), so replays track in-place mask content
#     changes; full-capacity descriptors keep the launch config shape-
#     derived.  Trivial fast path is disabled in graph mode.
#   * PPU swallows device-side faults inside replays (probe-verified:
#     assert AND illegal access both silent), so envelope violations
#     during replay are surfaced via a pinned flag: the in-graph
#     decomposition publishes its `supported` verdict into pinned memory
#     and torch.ops... sdpa_graph_mask_check() raises after the replay.
# LEGACY NOTE: pre-support, a FAILED capture permanently poisoned the
# default CUDA generator.  With gate rejection happening BEFORE capture
# starts, that hazard is gone; probes still run isolated (subprocess)
# because graph experiments are heavy and must not share CUDA context or
# generator state with the rest of the suite.  These tests run by default
# (no opt-in switch).
class TestCudaGraph:
    _PROBE_FWD = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask, p=0.0):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              dropout_p=p)

def ref(q, k, v, mask):
    with sdpa_kernel([SDPBackend.MATH]):
        q64, k64, v64 = (t.detach().to(torch.float64) for t in (q, k, v))
        am = torch.zeros_like(mask, dtype=torch.float64)
        am = am.masked_fill(~mask, float("-inf"))
        return F.scaled_dot_product_attention(q64, k64, v64, attn_mask=am)

DROP = __DROP__
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
device = "cuda"
q = torch.randn(2, 4, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
# 2-interval (sink+window) holed mask: exercises the real decomposition
# inside the graph, not just the trivial dense path.
sq = 128
idx = torch.arange(sq, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)

expected = flex(q, k, v, mask, DROP)  # eager warmup (also gates the probe)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        flex(q, k, v, mask, DROP)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = flex(q, k, v, mask, DROP)
print("CAPTURE:OK")
torch.ops.flex_flash_attention.sdpa_graph_mask_check()

g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
if DROP > 0:
    # Graph-safe RNG (mem_eff convention): PyTorch refreshes the generator's
    # extragraph tensors before every replay, so each replay draws a FRESH
    # dropout stream — replays differ and only statistical checks apply.
    o1 = out.detach().clone()
    g.replay()
    torch.cuda.synchronize()
    torch.ops.flex_flash_attention.sdpa_graph_mask_check()
    print("REPLAY2_FRESH:",
          "OK" if not torch.equal(out.detach(), o1) else "FAIL")
    print("REPLAY_FINITE:",
          "OK" if torch.isfinite(out).all() else "FAIL")
else:
    ok_exp = torch.allclose(out.float(), expected.float(), atol=0, rtol=0)
    print("REPLAY_EQ_EAGER:", "OK" if ok_exp else "FAIL")
    ok_ref = torch.allclose(out.float(), ref(q, k, v, mask).float(),
                            atol=2e-2, rtol=2e-2)
    print("REPLAY_VS_MATH:", "OK" if ok_ref else "FAIL")
"""

    _PROBE_FWD_CHANGING = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def ref(q, k, v, mask):
    with sdpa_kernel([SDPBackend.MATH]):
        q64, k64, v64 = (t.detach().to(torch.float64) for t in (q, k, v))
        am = torch.zeros_like(mask, dtype=torch.float64)
        am = am.masked_fill(~mask, float("-inf"))
        return F.scaled_dot_product_attention(q64, k64, v64, attn_mask=am)

torch.manual_seed(3)
device = "cuda"
q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
sq = 128
idx = torch.arange(sq, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)

flex(q, k, v, mask)  # eager warmup
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    flex(q, k, v, mask)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = flex(q, k, v, mask)
print("CAPTURE:OK")

# MUTATE the mask in place between replays: window wider, sink narrower.
# Replay must track the new content (in-graph decomposition re-runs).
with torch.no_grad():
    mask.copy_(((idx.unsqueeze(1) >= idx.unsqueeze(0) - 48) &
                (idx.unsqueeze(1) <= idx.unsqueeze(0))) |
               (idx.unsqueeze(1) < 4))
g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
ok = torch.allclose(out.float(), ref(q, k, v, mask).float(),
                    atol=2e-2, rtol=2e-2)
print("CHANGED_MASK_TRACKED:", "OK" if ok else "FAIL")

# Second mutation, replay again: decomposition still tracks.
with torch.no_grad():
    mask.copy_(((idx.unsqueeze(1) >= idx.unsqueeze(0) - 16) &
                (idx.unsqueeze(1) <= idx.unsqueeze(0))) |
               (idx.unsqueeze(1) < 16))
g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
ok2 = torch.allclose(out.float(), ref(q, k, v, mask).float(),
                     atol=2e-2, rtol=2e-2)
print("SECOND_MUTATION_TRACKED:", "OK" if ok2 else "FAIL")
"""

    _PROBE_FWDBWD = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def ref_grads(q, k, v, g, mask):
    with sdpa_kernel([SDPBackend.MATH]):
        q64 = q.detach().to(torch.float64).requires_grad_(True)
        k64 = k.detach().to(torch.float64).requires_grad_(True)
        v64 = v.detach().to(torch.float64).requires_grad_(True)
        am = torch.zeros_like(mask, dtype=torch.float64)
        am = am.masked_fill(~mask, float("-inf"))
        o = F.scaled_dot_product_attention(q64, k64, v64, attn_mask=am)
        return torch.autograd.grad(o, (q64, k64, v64), g.to(torch.float64))

DROP = __DROP__
torch.manual_seed(11)
torch.cuda.manual_seed_all(11)
device = "cuda"
q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16,
                requires_grad=True)
k = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16,
                requires_grad=True)
v = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16,
                requires_grad=True)
sq = 128
idx = torch.arange(sq, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)
go = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)

def run():
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              dropout_p=DROP)

# Leaf AccumulateGrad nodes bind the stream of their FIRST backward, and
# graph capture is only legal on that very stream (else the engine's
# stream-sync pulls the legacy stream into the capturing blocking stream).
# So ALL warmup runs on the side stream and capture passes it explicitly —
# the same recipe torch.cuda.make_graphed_callables uses.  fwd+bwd are
# captured in ONE graph (the training-standard pattern).
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        o = run()
        torch.autograd.grad(o, (q, k, v), go)
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    out_c = run()
    dq_c, dk_c, dv_c = torch.autograd.grad(out_c, (q, k, v), go)
print("CAPTURE_FWDBWD:OK")

g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
if DROP > 0:
    # Graph-safe RNG: every replay draws a fresh Philox stream, so both fwd
    # outputs and the dropout-replayed grads change across replays.
    finite = all(torch.isfinite(t).all()
                 for t in (out_c, dq_c, dk_c, dv_c))
    print("REPLAY_FINITE:", "OK" if finite else "FAIL")
    o1 = out_c.detach().clone()
    g1 = [t.detach().clone() for t in (dq_c, dk_c, dv_c)]
    g.replay()
    torch.cuda.synchronize()
    torch.ops.flex_flash_attention.sdpa_graph_mask_check()
    print("REPLAY2_FRESH:",
          "OK" if not torch.equal(out_c.detach(), o1) else "FAIL")
    grads_fresh = any(not torch.equal(a, b.detach())
                      for a, b in zip(g1, (dq_c, dk_c, dv_c)))
    print("GRADS_FRESH:", "OK" if grads_fresh else "FAIL")
else:
    # Deterministic: replay must reproduce an eager re-run bit-exactly and
    # match the fp64 MATH oracle within tolerance.
    out_e = run()
    dq_e, dk_e, dv_e = torch.autograd.grad(out_e, (q, k, v), go)
    torch.cuda.synchronize()
    ok_fwd = torch.equal(out_c.detach(), out_e.detach())
    print("REPLAY_FWD_EQ_EAGER:", "OK" if ok_fwd else "FAIL")
    ok_grads = all(torch.equal(a.detach(), b.detach())
                   for a, b in ((dq_c, dq_e), (dk_c, dk_e), (dv_c, dv_e)))
    print("REPLAY_GRADS_EQ_EAGER:", "OK" if ok_grads else "FAIL")
    ok_oracle = all(torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)
                    for a, b in zip((dq_c, dk_c, dv_c),
                                    ref_grads(q, k, v, go, mask)))
    print("REPLAY_GRADS_VS_MATH:", "OK" if ok_oracle else "FAIL")
"""

    _PROBE_ENVELOPE = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(5)
device = "cuda"
sq, n, period, win = 128, 16, 48, 16
q = torch.randn(1, 2, sq, 64, device=device, dtype=torch.bfloat16)
k = torch.randn(1, 2, n * period, 64, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
row = (torch.arange(n * period, device=device) % period) < win
mask = row.repeat(sq, 1)  # 16 intervals/row: outside the 15-layer envelope

try:
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print("WARMUP:ACCEPTED")
except RuntimeError:
    print("WARMUP:REJECTED")

g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print("CAPTURE:OK")
except RuntimeError:
    print("CAPTURE:REJECTED")
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
print("CHECK:NO_THROW")
"""

    _PROBE_NO_WARMUP = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(9)
device = "cuda"
q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
idx = torch.arange(128, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0)))

g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print("CAPTURE:OK")
except RuntimeError:
    print("CAPTURE:REJECTED")

# After a proper warmup the SAME capture succeeds (verdict path proven).
with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
torch.cuda.synchronize()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
torch.cuda.current_stream().wait_stream(s)
g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
print("CAPTURE_AFTER_WARMUP:OK")
"""

    _PROBE_FOLDED_CHECK = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

torch.manual_seed(21)
device = "cuda"
sq = 512
q = torch.randn(1, 1, sq, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

def stair():
    row = torch.arange(sq, device=device).view(-1, 1)
    col = torch.arange(sq, device=device).view(1, -1)
    return ((col <= row) &
            (col >= ((row // 64 - 3) * 64).clamp(min=0))).view(1, 1, sq, sq)

def checker(nseg):  # nseg separated intervals per row
    w = sq // (2 * nseg)
    col = torch.arange(sq, device=device).view(1, -1)
    keep = torch.zeros(1, 1, sq, sq, dtype=torch.bool, device=device)
    for i in range(nseg):
        keep |= (col >= i * 2 * w) & (col < i * 2 * w + w)
    return keep

mask = stair().contiguous()  # in-envelope warmup + capture mask
for _ in range(3):
    flex(q, k, v, mask)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        flex(q, k, v, mask)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g, stream=s):
    out = flex(q, k, v, mask)
g.replay()
torch.cuda.synchronize()
print("CAPTURE:OK")

# In-envelope content change: replay then eager must stay clean (no false
# alarm from the folded envelope check).
mask.copy_(checker(8))
g.replay()
torch.cuda.synchronize()
flex(q, k, v, stair().contiguous())
print("IN_ENVELOPE_CLEAN")

# Out-of-envelope content (16 intervals/row — eager-rejected).  NO manual
# sdpa_graph_mask_check() is called after this replay: the check folded
# into the next sdpa_fwd entry must raise on its own.
mask.copy_(checker(16))
g.replay()
torch.cuda.synchronize()
try:
    flex(q, k, v, stair().contiguous())
    print("FOLDED_RAISED: FAIL")
except RuntimeError as e:
    print("FOLDED_RAISED:",
          "OK" if "decomposition envelope" in str(e) else "WRONG_MSG")

# The flag was reset after raising: a clean eager call passes again.
flex(q, k, v, stair().contiguous())
print("SELF_HEALED: OK")
"""

    _PROBE_MGC = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(17)
device = "cuda"

class Attn(torch.nn.Module):
    def forward(self, q, k, v, mask):
        with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

q0 = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
k0 = torch.randn_like(q0)
v0 = torch.randn_like(q0)
idx = torch.arange(128, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)
go = torch.randn_like(q0)

# Eager reference (deterministic: dropout=0).
q = q0.clone().requires_grad_(True)
k = k0.clone().requires_grad_(True)
v = v0.clone().requires_grad_(True)
out_e = Attn()(q, k, v, mask)
dq_e, dk_e, dv_e = torch.autograd.grad(out_e, (q, k, v), go)
torch.cuda.synchronize()

# Official API: make_graphed_callables warms up eagerly on its own side
# stream (num_warmup_iters=3), which automatically satisfies the FLEX
# warmup contract — NO manual warmup here, on purpose.
gmod = torch.cuda.make_graphed_callables(
    Attn(), (q0.clone().requires_grad_(True),
             k0.clone().requires_grad_(True),
             v0.clone().requires_grad_(True), mask))
print("MGC_BUILT:OK")
qg, kg, vg = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
out_g = gmod(qg, kg, vg, mask)
dq_g, dk_g, dv_g = torch.autograd.grad(out_g, (qg, kg, vg), go)
torch.cuda.synchronize()
print("MGC_FWD_EQ:",
      "OK" if torch.equal(out_g.detach(), out_e.detach()) else "FAIL")
ok = all(torch.equal(a.detach(), b.detach())
         for a, b in ((dq_g, dq_e), (dk_g, dk_e), (dv_g, dv_e)))
print("MGC_GRADS_EQ:", "OK" if ok else "FAIL")
"""

    _PROBE_F32 = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask, p=0.0):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              dropout_p=p)

def ref(q, k, v, mask):
    with sdpa_kernel([SDPBackend.MATH]):
        q64, k64, v64 = (t.detach().to(torch.float64) for t in (q, k, v))
        am = torch.zeros_like(mask, dtype=torch.float64)
        am = am.masked_fill(~mask, float("-inf"))
        return F.scaled_dot_product_attention(q64, k64, v64, attn_mask=am)

DROP = __DROP__
torch.manual_seed(31)
torch.cuda.manual_seed_all(31)
device = "cuda"
q = torch.randn(2, 4, 128, 64, device=device, dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)
sq = 128
idx = torch.arange(sq, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)

expected = flex(q, k, v, mask, DROP)  # eager warmup (also gates the probe)
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        flex(q, k, v, mask, DROP)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = flex(q, k, v, mask, DROP)
print("CAPTURE:OK")

g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
if DROP > 0:
    o1 = out.detach().clone()
    g.replay()
    torch.cuda.synchronize()
    torch.ops.flex_flash_attention.sdpa_graph_mask_check()
    print("REPLAY2_FRESH:",
          "OK" if not torch.equal(out.detach(), o1) else "FAIL")
    print("REPLAY_FINITE:",
          "OK" if torch.isfinite(out).all() else "FAIL")
else:
    # TF32 tensor-core path: replay stays bit-exact vs eager re-run, and
    # matches the fp64 MATH oracle within the TF32 floor (~4e-3).
    ok_exp = torch.equal(out, expected)
    print("REPLAY_EQ_EAGER:", "OK" if ok_exp else "FAIL")
    ok_ref = torch.allclose(out.float(), ref(q, k, v, mask).float(),
                            atol=2e-2, rtol=2e-2)
    print("REPLAY_VS_MATH:", "OK" if ok_ref else "FAIL")
"""

    _PROBE_CAUSAL_NONE = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MODE = "__MODE__"
CAUSAL = MODE == "causal"

def flex(q, k, v):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                              is_causal=CAUSAL)

torch.manual_seed(19)
device = "cuda"
q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

expected = flex(q, k, v)  # eager warmup (also gates the probe)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        flex(q, k, v)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = flex(q, k, v)
print("CAPTURE:OK")
g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
rerun = flex(q, k, v)
print("REPLAY_EQ_EAGER:",
      "OK" if torch.equal(out.detach(), rerun.detach()) else "FAIL")
print("REPLAY_EQ_WARMUP:",
      "OK" if torch.equal(out.detach(), expected.detach()) else "FAIL")
"""

    _PROBE_REPLAY20 = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask, p):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                              dropout_p=p)

torch.manual_seed(23)
torch.cuda.manual_seed_all(23)
device = "cuda"
q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
idx = torch.arange(128, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)

flex(q, k, v, mask, 0.5)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        flex(q, k, v, mask, 0.5)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = flex(q, k, v, mask, 0.5)

g.replay()
torch.cuda.synchronize()
first = out.detach().clone()
all_finite = torch.isfinite(out).all().item()
differ = 0
for _ in range(19):
    g.replay()
    torch.cuda.synchronize()
    all_finite = all_finite and torch.isfinite(out).all().item()
    differ += int(not torch.equal(out.detach(), first))
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
print("R20_FINITE:", "OK" if all_finite else "FAIL")
# every replay draws a fresh Philox stream -> all 19 differ from the first
print("R20_FRESH:", "OK" if differ == 19 else "FAIL")
print("R20_CHECK:NO_THROW")
"""

    _PROBE_MULTI_GRAPH = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

torch.manual_seed(29)
device = "cuda"
q = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
idx = torch.arange(128, device=device)
maskA = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
         (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)
maskB = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 16) &
         (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 16)

def capture(mask):
    flex(q, k, v, mask)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flex(q, k, v, mask)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flex(q, k, v, mask)
    return g, out

gA, outA = capture(maskA)
gB, outB = capture(maskB)
print("CAPTURE_BOTH:OK")

# Interleaved replays: the envelope flag is a process-global singleton
# refreshed by whichever graph replayed last — alternating replays of two
# SUPPORTED masks must never false-alarm.
for _ in range(5):
    gA.replay()
    torch.cuda.synchronize()
    gB.replay()
    torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
print("INTERLEAVED_NO_FALSE_ALARM")

okA = torch.equal(outA.detach(), flex(q, k, v, maskA).detach())
okB = torch.equal(outB.detach(), flex(q, k, v, maskB).detach())
print("GRAPH_A_EQ_EAGER:", "OK" if okA else "FAIL")
print("GRAPH_B_EQ_EAGER:", "OK" if okB else "FAIL")
"""

    _PROBE_GQA = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def flex(q, k, v, mask):
    with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

def ref(q, k, v, mask):
    with sdpa_kernel([SDPBackend.MATH]):
        q64, k64, v64 = (t.detach().to(torch.float64) for t in (q, k, v))
        # MATH has no GQA broadcast — expand kv heads to match q.
        reps = q64.shape[1] // k64.shape[1]
        k64 = k64.repeat_interleave(reps, dim=1)
        v64 = v64.repeat_interleave(reps, dim=1)
        am = torch.zeros(q64.shape[0], q64.shape[1], mask.shape[-2],
                         mask.shape[-1], dtype=torch.float64,
                         device=mask.device)
        am = am.masked_fill(~mask, float("-inf"))
        return F.scaled_dot_product_attention(q64, k64, v64, attn_mask=am)

torch.manual_seed(37)
device = "cuda"
# 8 query heads over 2 kv heads (4:1 GQA), holed window+sink mask.
q = torch.randn(1, 8, 128, 64, device=device, dtype=torch.bfloat16)
k = torch.randn(1, 2, 128, 64, device=device, dtype=torch.bfloat16)
v = torch.randn_like(k)
idx = torch.arange(128, device=device)
mask = ((idx.unsqueeze(1) >= idx.unsqueeze(0) - 32) &
        (idx.unsqueeze(1) <= idx.unsqueeze(0))) | (idx.unsqueeze(1) < 8)

expected = flex(q, k, v, mask)  # eager warmup
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        flex(q, k, v, mask)
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out = flex(q, k, v, mask)
print("CAPTURE:OK")
g.replay()
torch.cuda.synchronize()
torch.ops.flex_flash_attention.sdpa_graph_mask_check()
print("REPLAY_EQ_EAGER:",
      "OK" if torch.equal(out.detach(), expected.detach()) else "FAIL")
ok_ref = torch.allclose(out.float(), ref(q, k, v, mask).float(),
                        atol=2e-2, rtol=2e-2)
print("REPLAY_VS_MATH:", "OK" if ok_ref else "FAIL")
"""

    @staticmethod
    def _run_isolated(code):
        import os
        import subprocess
        import sys
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=900,
                           env=env)
        assert r.returncode == 0, \
            f"probe crashed (rc={r.returncode}): {r.stderr[-1500:]}"
        return r.stdout

    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    def test_fwd_graph_capture_and_replay(self, dropout):
        out = self._run_isolated(self._PROBE_FWD.replace("__DROP__",
                                                          repr(dropout)))
        assert "CAPTURE:OK" in out, out
        if dropout > 0:
            # mem_eff-style graph RNG: each replay draws a fresh Philox
            # stream (no frozen seed), so replays differ but stay finite.
            assert "REPLAY2_FRESH: OK" in out, out
            assert "REPLAY_FINITE: OK" in out, out
        else:
            assert "REPLAY_EQ_EAGER: OK" in out, out
            assert "REPLAY_VS_MATH: OK" in out, out

    def test_fwd_graph_tracks_mask_content_changes(self):
        out = self._run_isolated(self._PROBE_FWD_CHANGING)
        assert "CAPTURE:OK" in out, out
        assert "CHANGED_MASK_TRACKED: OK" in out, out
        assert "SECOND_MUTATION_TRACKED: OK" in out, out

    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    def test_fwd_bwd_graph_capture(self, dropout):
        out = self._run_isolated(self._PROBE_FWDBWD.replace("__DROP__",
                                                             repr(dropout)))
        assert "CAPTURE_FWDBWD:OK" in out, out
        if dropout > 0:
            # Fresh Philox stream per replay: outputs AND grads change.
            assert "REPLAY_FINITE: OK" in out, out
            assert "REPLAY2_FRESH: OK" in out, out
            assert "GRADS_FRESH: OK" in out, out
        else:
            assert "REPLAY_FWD_EQ_EAGER: OK" in out, out
            assert "REPLAY_GRADS_EQ_EAGER: OK" in out, out
            assert "REPLAY_GRADS_VS_MATH: OK" in out, out

    def test_out_of_envelope_never_captures(self):
        out = self._run_isolated(self._PROBE_ENVELOPE)
        assert "WARMUP:REJECTED" in out, out
        assert "CAPTURE:REJECTED" in out, out
        assert "CHECK:NO_THROW" in out, out

    def test_capture_without_warmup_rejects_cleanly(self):
        out = self._run_isolated(self._PROBE_NO_WARMUP)
        assert "CAPTURE:REJECTED" in out, out
        assert "CAPTURE_AFTER_WARMUP:OK" in out, out

    def test_folded_envelope_check_auto_raises(self):
        # Mem-eff-grade contract: no manual post-replay check needed — an
        # out-of-envelope replay must raise at the NEXT FLEX call on its own,
        # then self-heal (flag reset).
        out = self._run_isolated(self._PROBE_FOLDED_CHECK)
        assert "CAPTURE:OK" in out, out
        assert "IN_ENVELOPE_CLEAN" in out, out
        assert "FOLDED_RAISED: OK" in out, out
        assert "SELF_HEALED: OK" in out, out

    def test_make_graphed_callables_fwd_bwd(self):
        # Official user-facing entry: its built-in eager warmup must satisfy
        # the FLEX warmup contract automatically (no manual warmup here).
        out = self._run_isolated(self._PROBE_MGC)
        assert "MGC_BUILT:OK" in out, out
        assert "MGC_FWD_EQ: OK" in out, out
        assert "MGC_GRADS_EQ: OK" in out, out

    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    def test_f32_graph_capture_and_replay(self, dropout):
        # The TF32 kernel set is a separate instantiation — prove the
        # pointerized RNG / capture machinery works there too.
        out = self._run_isolated(self._PROBE_F32.replace("__DROP__",
                                                          repr(dropout)))
        assert "CAPTURE:OK" in out, out
        if dropout > 0:
            assert "REPLAY2_FRESH: OK" in out, out
            assert "REPLAY_FINITE: OK" in out, out
        else:
            assert "REPLAY_EQ_EAGER: OK" in out, out
            assert "REPLAY_VS_MATH: OK" in out, out

    @pytest.mark.parametrize("mode", ["causal", "none"])
    def test_causal_and_none_mask_capture(self, mode):
        # Synthetic-desc path (host constants) — the capture branch that
        # never materializes or decomposes a mask.
        out = self._run_isolated(self._PROBE_CAUSAL_NONE.replace("__MODE__",
                                                                  mode))
        assert "CAPTURE:OK" in out, out
        assert "REPLAY_EQ_EAGER: OK" in out, out
        assert "REPLAY_EQ_WARMUP: OK" in out, out

    def test_twenty_replays_stable(self):
        # Long-run replay stability: RNG incrementing and descriptor
        # refresh must not drift or corrupt across many replays.
        out = self._run_isolated(self._PROBE_REPLAY20)
        assert "R20_FINITE: OK" in out, out
        assert "R20_FRESH: OK" in out, out
        assert "R20_CHECK:NO_THROW" in out, out

    def test_two_graphs_coexist_no_false_alarm(self):
        # The envelope flag is a process-global singleton: interleaved
        # replays of two supported graphs must stay clean and each graph
        # must keep reproducing its own eager results.
        out = self._run_isolated(self._PROBE_MULTI_GRAPH)
        assert "CAPTURE_BOTH:OK" in out, out
        assert "INTERLEAVED_NO_FALSE_ALARM" in out, out
        assert "GRAPH_A_EQ_EAGER: OK" in out, out
        assert "GRAPH_B_EQ_EAGER: OK" in out, out

    def test_gqa_graph_capture(self):
        out = self._run_isolated(self._PROBE_GQA)
        assert "CAPTURE:OK" in out, out
        assert "REPLAY_EQ_EAGER: OK" in out, out
        assert "REPLAY_VS_MATH: OK" in out, out


# --------------------------------------------------------------------------
# P. grad-mask combinations & unusual tensor layouts
# --------------------------------------------------------------------------
class TestGradAndLayouts:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7])
    def test_grad_input_combinations(self, bits):
        # every non-empty subset of {q, k, v} requiring grad
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        mask = _causal_mask(256, 256)
        q.requires_grad_(bool(bits & 1))
        k.requires_grad_(bool(bits & 2))
        v.requires_grad_(bool(bits & 4))
        out = _run_flex(q, k, v, mask=mask)
        g = torch.randn_like(out)
        out.backward(g)
        exp_q, exp_k, exp_v = _ref_math_grads(q, k, v, g, mask=mask)
        for got, exp, on in ((q.grad, exp_q, bits & 1),
                             (k.grad, exp_k, bits & 2),
                             (v.grad, exp_v, bits & 4)):
            if not on:
                assert got is None
            else:
                torch.testing.assert_close(got, exp, **TOL[BF16])

    def test_packed_qkv_non_contiguous(self):
        # production layout: q/k/v sliced out of one packed buffer.
        # Views are non-leaf, so grads are pulled with autograd.grad.
        b, s, h, d = 1, 256, 2, 64
        packed = torch.randn(b, s, 3 * h * d, device=DEVICE, dtype=BF16,
                             requires_grad=True)
        qs, ks, vs = packed.chunk(3, dim=-1)
        q = qs.view(b, s, h, d).permute(0, 2, 1, 3)
        k = ks.view(b, s, h, d).permute(0, 2, 1, 3)
        v = vs.view(b, s, h, d).permute(0, 2, 1, 3)
        assert not q.is_contiguous()
        mask = _causal_mask(s, s)
        out = _run_flex(q, k, v, mask=mask)
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])
        g = torch.randn_like(out)
        (got,) = torch.autograd.grad(out, packed, g)
        qc = q.detach().contiguous().requires_grad_(True)
        kc = k.detach().contiguous().requires_grad_(True)
        vc = v.detach().contiguous().requires_grad_(True)
        ref2 = _ref_math(qc, kc, vc, mask=mask, detach_inputs=False)
        ref2.backward(g)
        exp = torch.cat([
            qc.grad.permute(0, 2, 1, 3).reshape(b, s, h * d),
            kc.grad.permute(0, 2, 1, 3).reshape(b, s, h * d),
            vc.grad.permute(0, 2, 1, 3).reshape(b, s, h * d)], dim=-1)
        torch.testing.assert_close(got, exp, **TOL[BF16])

    def test_broadcast_grad_out(self):
        # grads arriving as stride-0 expanded views (common downstream of
        # parameter sharing / broadcast losses)
        q, k, v = _mk(2, 4, 4, 256, 256, 64)
        mask = _causal_mask(256, 256)
        out = _run_flex(q, k, v, mask=mask)
        g_narrow = torch.randn(2, 1, 256, 64, device=DEVICE, dtype=BF16)
        out.backward(g_narrow.expand_as(out))
        exp_q, exp_k, exp_v = _ref_math_grads(
            q, k, v, g_narrow.expand_as(out), mask=mask)
        for got, exp in ((q.grad, exp_q), (k.grad, exp_k), (v.grad, exp_v)):
            torch.testing.assert_close(got, exp, **TOL[BF16])

    def test_non_contiguous_mask(self):
        base = _causal_mask(256, 256)
        mask = base.transpose(0, 1)  # transposed view, still bool 2-D
        assert not mask.is_contiguous()
        q, k, v = _mk(1, 2, 2, 256, 256, 64)
        _assert_close_flex(q, k, v, mask=mask)


# --------------------------------------------------------------------------
# Q. autocast (AMP)
# --------------------------------------------------------------------------
class TestAutocast:
    def test_autocast_bf16_routes_and_matches(self):
        # fp32 inputs under autocast: sdpa casts to bf16 before backend
        # selection, so the flex flash attention backend must accept and be correct.
        q = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=FP32)
        k = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=FP32)
        v = torch.randn(2, 4, 256, 64, device=DEVICE, dtype=FP32)
        mask = _causal_mask(256, 256)
        with flex_only(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.dtype == BF16
        ref = _ref_math(q.bfloat16(), k.bfloat16(), v.bfloat16(), mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])


# --------------------------------------------------------------------------
# R. inference_mode / no_grad
# --------------------------------------------------------------------------
class TestInferenceModes:
    def test_inference_mode(self):
        with torch.inference_mode():
            q = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
            k = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
            v = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
            mask = _causal_mask(256, 256)
            with flex_only():
                out = F.scaled_dot_product_attention(q, k, v,
                                                     attn_mask=mask)
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, **TOL[BF16])

    def test_no_grad(self):
        q = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
        k = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
        v = torch.randn(1, 2, 256, 64, device=DEVICE, dtype=BF16)
        mask = _causal_mask(256, 256)
        with torch.no_grad():
            out = _run_flex(q, k, v, mask=mask)
        assert out.grad_fn is None
        torch.testing.assert_close(out, _ref_math(q, k, v, mask=mask),
                                   **TOL[BF16])


# --------------------------------------------------------------------------
# S. Cache-eviction soak (resolved/decompose caches under churn)
# --------------------------------------------------------------------------
class TestCacheSoak:
    def test_many_masks_bounded_memory(self):
        # 48 distinct mask identities churn both caches past their
        # eviction thresholds; results must stay correct and memory
        # bounded.
        torch.manual_seed(7)
        q, k, v = _mk(1, 2, 2, 256, 256, 64, requires_grad=False)
        for i in range(2):  # warm the allocator
            _run_flex(q, k, v, mask=_rand_stair_mask(256, 256, i))
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated()
        for i in range(48):
            s = 128 + (i % 16) * 16
            qi, ki, vi = _mk(1, 2, 2, s, s, 64, requires_grad=False)
            mask = _rand_stair_mask(s, s, seed=1000 + i)
            out = _run_flex(qi, ki, vi, mask=mask)
            assert torch.isfinite(out).all()
            if i in (17, 33, 47):  # post-eviction correctness spot checks
                torch.testing.assert_close(
                    out, _ref_math(qi, ki, vi, mask=mask), **TOL[BF16])
        torch.cuda.empty_cache()
        delta = torch.cuda.memory_allocated() - base_mem
        assert delta < 64 * 1024 * 1024, f"cache leak: {delta} bytes"


# --------------------------------------------------------------------------
# T. Training-loop integration (end-to-end through a real module)
# --------------------------------------------------------------------------
class TestTrainingIntegration:
    def test_loss_decreases_over_steps(self):
        torch.manual_seed(123)
        h, d, s, b = 4, 64, 128, 2

        model = TinyAttn(nhead=h, dim=d).to(DEVICE)
        x = torch.randn(b, s, d, device=DEVICE)
        target = torch.randn(b, s, d, device=DEVICE)
        mask = _causal_mask(s, s)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            with flex_only(), torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.mse_loss(model(x, mask), target)
            loss.backward()
            assert all(torch.isfinite(p.grad).all()
                       for p in model.parameters())
            opt.step()
            losses.append(loss.item())
            assert torch.isfinite(loss)
        assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


# --------------------------------------------------------------------------
# U. Double backward
# --------------------------------------------------------------------------
class TestDoubleBackward:
    def test_double_backward_clean_behavior(self):
        # Double backward is not a shipped capability; the contract is
        # CLEAN failure (RuntimeError), never a crash or wrong grads.
        q, k, v = _mk(1, 2, 2, 128, 128, 64)
        mask = _causal_mask(128, 128)
        with flex_only():
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            try:
                g1, = torch.autograd.grad(out.sum(), q, create_graph=True)
                torch.autograd.grad(g1.sum(), q)
                double_supported = True
            except RuntimeError:
                double_supported = False
        # either way the library must stay usable
        out_after = _run_flex(q.detach(), k.detach(), v.detach(), mask=mask)
        assert torch.isfinite(out_after).all()
        TestDoubleBackward.supported = double_supported  # documented


# --------------------------------------------------------------------------
# V. Long-sequence numerics
# --------------------------------------------------------------------------
class TestLongSequence:
    def test_8k_fwd_bwd(self):
        q, k, v = _mk(1, 2, 2, 8192, 8192, 64)
        mask = _causal_mask(8192, 8192)
        _assert_close_flex(q, k, v, mask=mask)

    def test_16k_fwd(self):
        q, k, v = _mk(1, 1, 1, 16384, 16384, 64, requires_grad=False)
        mask = _causal_mask(16384, 16384)
        out = _run_flex(q, k, v, mask=mask)
        ref = _ref_math(q, k, v, mask=mask)
        torch.testing.assert_close(out, ref, atol=4e-2, rtol=2e-2)


# --------------------------------------------------------------------------
# W. Runtime env-var kill switch (TORCH_FLEX_FLASH_SDPA_ENABLED)
# --------------------------------------------------------------------------
_ENV_KILL_PROBE = r"""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print("DEFAULT_ENABLED:", torch.backends.cuda.flex_flash_attention_sdp_enabled())
q = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 2, 128, 64, device="cuda", dtype=torch.bfloat16)
mask = torch.ones(128, 128, device="cuda", dtype=torch.bool).tril()

out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
assert torch.isfinite(out).all()
print("DEFAULT_OK")

with sdpa_kernel([SDPBackend.FLEX_FLASH_ATTENTION]):
    try:
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        print("FLEX_OK")
    except RuntimeError:
        print("FLEX_REJECT")

# the kill switch targets only this backend; others must stay usable
with sdpa_kernel([SDPBackend.MATH]):
    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    print("MATH_OK")
"""


class TestEnvKillSwitch:
    # TORCH_FLEX_FLASH_SDPA_ENABLED is latched by a static const inside
    # can_use_flex_flash_attention (the first SDPA call wins), so
    # every scenario must run in a fresh interpreter.

    def _probe(self, value):
        env = dict(os.environ)
        if value is None:
            env.pop("TORCH_FLEX_FLASH_SDPA_ENABLED", None)
        else:
            env["TORCH_FLEX_FLASH_SDPA_ENABLED"] = value
        proc = subprocess.run(
            [sys.executable, "-c", _ENV_KILL_PROBE],
            capture_output=True, text=True, env=env, timeout=600)
        assert proc.returncode == 0, proc.stderr
        assert "DEFAULT_OK" in proc.stdout, proc.stdout
        assert "MATH_OK" in proc.stdout, "other backends must be unaffected"
        return proc.stdout

    @pytest.mark.parametrize(
        "value,expect_ok",
        [
            pytest.param(None, False, id="unset-default-off"),
            pytest.param("1", True, id="explicit-1"),
            pytest.param("0", False, id="kill-switch-0"),
            pytest.param("off", False, id="invalid-off"),
            pytest.param("false", False, id="invalid-false"),
            pytest.param("true", False, id="invalid-true"),
            pytest.param("", False, id="invalid-empty"),
            pytest.param("2", False, id="invalid-2"),
        ],
    )
    def test_kill_switch(self, value, expect_ok):
        out = self._probe(value)
        assert f"DEFAULT_ENABLED: {expect_ok}" in out, out
        if expect_ok:
            assert "FLEX_OK" in out, out
        else:
            assert "FLEX_REJECT" in out, out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
