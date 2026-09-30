"""Shared benchmark mask definitions, independent of the flexflash extension and original tests."""

import torch


# Both backends share these predicates, independently of the flexflash extension.
BENCHMARK_MASKS = (
    "dense", "causal", "sliding_window", "sliding_causal", "chunked_causal",
    "blockwise", "prefix_lm", "document", "causal_doc", "stair",
)
BENCHMARK_MASK_VERSION = 1


def benchmark_mask_params(name, sq, sk, overrides=None):
    """Resolve mask parameters; square defaults match SCENARIOS in test_arb_suite."""
    if name not in BENCHMARK_MASKS:
        raise ValueError(f"Unknown mask: {name}")
    defaults = {
        "sliding_window": {"left": sq // 4, "right": sq // 4},
        "sliding_causal": {"window": 512},
        "chunked_causal": {"chunk": max(1, sq // 8)},
        "blockwise": {"chunk": max(1, sq // 8)},
        "prefix_lm": {"prefix": 512},
        "document": {"n_docs": 16},
        "causal_doc": {"n_docs": 16},
        "stair": {"step": 2048, "cache_mult": 5},
    }.get(name, {})
    overrides = overrides or {}
    if overrides.keys() - defaults.keys():
        raise ValueError(f"Unknown mask parameters for {name}: {overrides.keys() - defaults.keys()}")
    params = {**defaults, **overrides}
    for key, value in params.items():
        minimum = 0 if key in ("left", "right", "prefix", "cache_mult") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Mask parameter {key} must be an integer >= {minimum}")
    if "n_docs" in params and min(sq, sk) < params["n_docs"]:
        raise ValueError("Sq and Sk must not be smaller than n_docs")
    return params


def benchmark_mask_mod(name, sq, sk, params=None):
    """Return a tensor-only predicate suitable for FlexAttention.

    document preserves the original suite's integer-division ID rule (the tail
    may create an extra ID). stair uses a frame-history window, unlike
    make_stair_mask in the original test fixture.
    """
    p = benchmark_mask_params(name, sq, sk, params)
    if name == "dense":
        def predicate(q, k):
            return k >= 0
    elif name == "causal":
        def predicate(q, k):
            return k <= q
    elif name == "sliding_window":
        left, right = p["left"], p["right"]
        def predicate(q, k):
            return (k >= q - left) & (k <= q + right)
    elif name == "sliding_causal":
        window = p["window"]
        def predicate(q, k):
            return (k <= q) & (k >= q - window + 1)
    elif name == "chunked_causal":
        chunk = p["chunk"]
        def predicate(q, k):
            return (k <= q) & (k // chunk == q // chunk)
    elif name == "blockwise":
        chunk = p["chunk"]
        def predicate(q, k):
            return k // chunk <= q // chunk
    elif name == "prefix_lm":
        prefix = p["prefix"]
        def predicate(q, k):
            return (k <= q) | (k < prefix)
    elif name in ("document", "causal_doc"):
        qsize, ksize = sq // p["n_docs"], sk // p["n_docs"]
        causal = name == "causal_doc"
        def predicate(q, k):
            same_doc = q // qsize == k // ksize
            return same_doc & (k <= q) if causal else same_doc
    else:
        step, cache_mult = p["step"], p["cache_mult"]
        def predicate(q, k):
            lo = ((q // step - cache_mult) * step).clamp_min(0)
            return (k >= lo) & (k <= q)

    def mask_mod(b, h, q, k):
        return predicate(q, k) & (q >= 0) & (q < sq) & (k >= 0) & (k < sk)
    return mask_mod


def benchmark_dense_mask(name, sq, sk, params=None, device="cpu"):
    mod = benchmark_mask_mod(name, sq, sk, params)
    q = torch.arange(sq, device=device).view(-1, 1)
    k = torch.arange(sk, device=device).view(1, -1)
    return mod(0, 0, q, k).expand(sq, sk).contiguous()
