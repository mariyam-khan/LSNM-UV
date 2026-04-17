"""
Diagnostic trace: CAMUV (additive) vs CAMUV_LSNM on a minimal hidden-cause graph.

Graph:  u (hidden) -> x0, u -> x1.  Ground truth: x0 <-> x1 (bidirected).
Data:   LSNM (heteroscedastic), generated with the same functions as the
        full experiment pipeline (lsnm_data_gen.py).

Both methods use the exact same CAM-UV algorithm (Maeda & Shimizu 2021).
The ONLY difference is the residual function:
  - CAMUV (camuv.py):       x_i - GAM(K_i)          (original, location-only)
  - CAMUV_LSNM (camuv_lsnm.py): (x_i - f(K_i)) / g(K_i) (two-step GAMLSS, Eq. 14)

Run:  python trace_debug.py
"""

import numpy as np

from lsnm_data_gen import _gen_lsnm_variable
from camuv import CAMUV
from camuv_lsnm import CAMUV_LSNM


# ── Data generation ──────────────────────────────────────────────────────────

def gen_simple_lsnm(n, seed):
    """
    Graph: u -> x0, u -> x1.  Returns only observed (x0, x1).
    Uses _gen_lsnm_variable from lsnm_data_gen.py for consistency.
    """
    rng = np.random.default_rng(seed)
    u  = _gen_lsnm_variable(obs_parent_vals=[], hid_parent_vals=[], rng=rng, n=n)
    x0 = _gen_lsnm_variable(obs_parent_vals=[], hid_parent_vals=[u], rng=rng, n=n)
    x1 = _gen_lsnm_variable(obs_parent_vals=[], hid_parent_vals=[u], rng=rng, n=n)
    return np.column_stack([x0, x1])


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = 500
    alpha = 0.01
    n_seeds = 10

    print("=" * 70)
    print("DIAGNOSTIC: CAMUV (additive) vs CAMUV_LSNM residuals")
    print(f"Graph: u -> x0, u -> x1  (ground truth: x0 <-> x1)")
    print(f"n={n}, alpha={alpha}, seeds=0..{n_seeds-1}")
    print("=" * 70)

    results = []

    for seed in range(n_seeds):
        X = gen_simple_lsnm(n, seed)

        for label, cls in [("ADDITIVE (camuv.py)", CAMUV),
                           ("LSNM (camuv_lsnm.py)", CAMUV_LSNM)]:
            print(f"\n{'='*70}")
            print(f"  SEED={seed}  METHOD={label}")
            print(f"{'='*70}")

            model = cls(alpha=alpha, num_explanatory_vals=2, verbose=True)
            model.fit(X)

            mat = model.adjacency_matrix_
            has_bidir = bool(np.any(np.isnan(mat)))
            has_dir = bool(np.any(mat == 1))

            print(f"\n  Adjacency matrix:\n    {mat[0]}\n    {mat[1]}")
            if has_bidir and not has_dir:
                print(f"  -> CORRECT: bidirected detected")
            elif has_dir:
                print(f"  -> WRONG: spurious directed edge")
            else:
                print(f"  -> MISSED: empty graph")

            if "ADDITIVE" in label:
                results.append({'seed': seed, 'add_bidir': has_bidir, 'add_dir': has_dir})
            else:
                results[-1]['lsnm_bidir'] = has_bidir
                results[-1]['lsnm_dir'] = has_dir

    # Aggregate
    print(f"\n\n{'='*70}")
    print(f"AGGREGATE ({n_seeds} seeds)")
    print(f"{'='*70}")
    add_ok = sum(r['add_bidir'] and not r['add_dir'] for r in results)
    lsnm_ok = sum(r['lsnm_bidir'] and not r['lsnm_dir'] for r in results)
    add_wrong = sum(r['add_dir'] for r in results)
    lsnm_wrong = sum(r['lsnm_dir'] for r in results)
    print(f"  ADDITIVE: {add_ok}/{n_seeds} correct,  {add_wrong}/{n_seeds} spurious directed")
    print(f"  LSNM:     {lsnm_ok}/{n_seeds} correct,  {lsnm_wrong}/{n_seeds} spurious directed")
