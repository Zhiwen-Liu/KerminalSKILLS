"""Heterogeneous platform precision validation utility.

Usage:
    from validate_precision import report, summary

    report("matmul", cpu_out, npu_out, atol=1e-5)
    report("softmax", cpu_out, npu_out, atol=1e-5)
    summary()
"""
import numpy as np
from typing import Optional

_results = []


def report(
    label: str,
    cpu_val,
    npu_val,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    """Compare CPU vs NPU outputs with detailed statistics.

    Args:
        label: Name of the operator or module.
        cpu_val: CPU reference tensor (torch.Tensor or np.ndarray).
        npu_val: NPU tensor to validate.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        dict with comparison statistics.
    """
    if hasattr(cpu_val, "detach"):
        cpu_arr = cpu_val.detach().cpu().float().numpy().ravel()
    else:
        cpu_arr = np.asarray(cpu_val, dtype=np.float32).ravel()

    if hasattr(npu_val, "detach"):
        npu_arr = npu_val.detach().cpu().float().numpy().ravel()
    else:
        npu_arr = np.asarray(npu_val, dtype=np.float32).ravel()

    abs_diff = np.abs(cpu_arr - npu_arr)
    ref_abs = np.maximum(np.abs(cpu_arr), 1e-12)
    rel_diff = abs_diff / ref_abs

    nan_count = int(np.isnan(npu_arr).sum())
    inf_count = int(np.isinf(npu_arr).sum())

    has_hard_fail = (
        nan_count > 0
        or inf_count > 0
        or cpu_arr.shape != npu_arr.shape
    )

    allclose_ok = np.allclose(cpu_arr, npu_arr, atol=atol, rtol=rtol)

    cos_sim = float(
        np.dot(cpu_arr, npu_arr)
        / (np.linalg.norm(cpu_arr) * np.linalg.norm(npu_arr) + 1e-12)
    )

    tag = "FAIL" if has_hard_fail else ("PASS" if allclose_ok else "DIFF")

    stats = {
        "label": label,
        "tag": tag,
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "p90_abs": float(np.percentile(abs_diff, 90)),
        "p99_abs": float(np.percentile(abs_diff, 99)),
        "p999_abs": float(np.percentile(abs_diff, 99.9)),
        "max_rel": float(rel_diff.max()),
        "mean_rel": float(rel_diff.mean()),
        "p99_rel": float(np.percentile(rel_diff, 99)),
        "cosine_sim": cos_sim,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "gt_1e3_ratio": float((abs_diff > 1e-3).mean()),
        "gt_1e2_ratio": float((abs_diff > 1e-2).mean()),
        "atol": atol,
        "allclose": allclose_ok,
    }

    print(f"  [{tag}] {label}")
    print(f"    max_abs={stats['max_abs']:.2e}  mean_abs={stats['mean_abs']:.2e}")
    print(f"    max_rel={stats['max_rel']:.2e}  mean_rel={stats['mean_rel']:.2e}")
    print(f"    cosine_sim={cos_sim:.6f}  nan={nan_count}  inf={inf_count}")

    _results.append(stats)
    return stats


def summary() -> None:
    """Print a summary table of all recorded comparisons."""
    if not _results:
        print("No results recorded.")
        return

    print("\n" + "=" * 80)
    print(f"{'Op':<25} {'Tag':<6} {'max_abs':>10} {'mean_abs':>10} {'cos_sim':>10}")
    print("-" * 80)
    for r in _results:
        print(
            f"{r['label']:<25} {r['tag']:<6} {r['max_abs']:>10.2e} "
            f"{r['mean_abs']:>10.2e} {r['cosine_sim']:>10.6f}"
        )
    print("=" * 80)

    passed = sum(1 for r in _results if r["tag"] == "PASS")
    total = len(_results)
    print(f"\nTotal: {passed}/{total} PASS")
