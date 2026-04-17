"""
Diagnostic: trace CAMUV vs CAMUV_LSNM on a minimal graph.

Graph:  u (hidden) -> x0, u -> x1
        No direct edge between x0 and x1.
        Ground truth: x0 <-> x1 (bidirected / invisible pair).

Data model: LSNM (heteroscedastic), so CAM-UV's additive residual
may behave differently from the LSNM residual.

For each of 10 seeds we trace:
  1. Neighbourhood construction (_get_neighborhoods): raw X pairwise HSIC
  2. Parent search (_find_parents): which parents are found
  3. UBP/UCP detection (Algorithm 2): which pairs are marked invisible

We compare the original CAMUV (additive residual) vs CAMUV_LSNM (LSNM residual).
"""

import numpy as np
from pygam import LinearGAM
from lingam import CAMUV
from camuv_lsnm import CAMUV_LSNM
from lingam.hsic import hsic_test_gamma


def gen_simple_lsnm(n, seed):
    """
    Generate data from:  u -> x0,  u -> x1  (LSNM, heteroscedastic).

    u ~ N(0,1)
    x0 = f0(u) + g0(u) * eps0    where eps0 ~ N(0,1)
    x1 = f1(u) + g1(u) * eps1    where eps1 ~ N(0,1)

    f0, f1 are nonlinear location functions of u.
    g0, g1 are nonlinear scale functions of u (heteroscedasticity).
    """
    rng = np.random.default_rng(seed)

    u = rng.standard_normal(n)
    eps0 = rng.standard_normal(n)
    eps1 = rng.standard_normal(n)

    # Location functions
    f0 = (u + rng.uniform(-3, 3)) ** 2 + rng.uniform(-1, 1)
    f1 = (u + rng.uniform(-3, 3)) ** 2 + rng.uniform(-1, 1)

    # Scale functions (heteroscedastic)
    log_g0 = (u + rng.uniform(-3, 3)) ** 2
    log_g0 = log_g0 / (np.std(log_g0) + 1e-8)
    g0 = np.clip(np.exp(log_g0), 0.1, 10.0)

    log_g1 = (u + rng.uniform(-3, 3)) ** 2
    log_g1 = log_g1 / (np.std(log_g1) + 1e-8)
    g1 = np.clip(np.exp(log_g1), 0.1, 10.0)

    # LSNM: x = f(u) + g(u) * eps
    x0 = f0 + g0 * eps0
    x1 = f1 + g1 * eps1

    # Normalise
    x0 = (x0 - np.mean(x0)) / (np.std(x0) + 1e-8)
    x1 = (x1 - np.mean(x1)) / (np.std(x1) + 1e-8)

    X = np.column_stack([x0, x1])
    return X


def hsic_pvalue(a, b):
    """Return HSIC p-value for two 1D arrays."""
    _, pval = hsic_test_gamma(a.reshape(-1, 1), b.reshape(-1, 1))
    return pval


def additive_residual(X, target, explanatory):
    """CAM-UV additive residual: x_i - GAM(K_i)."""
    explanatory = list(explanatory)
    if len(explanatory) == 0:
        return X[:, target]
    gam = LinearGAM().fit(X[:, explanatory], X[:, target])
    return X[:, target] - gam.predict(X[:, explanatory])


def lsnm_residual(X, target, explanatory):
    """LSNM residual: (x_i - f_hat(K_i)) / g_hat(K_i)."""
    explanatory = list(explanatory)
    if len(explanatory) == 0:
        return X[:, target]
    X_expl = X[:, explanatory]
    xi = X[:, target]
    try:
        gam_loc = LinearGAM().fit(X_expl, xi)
        loc_resid = xi - gam_loc.predict(X_expl)
    except Exception:
        loc_resid = xi - xi.mean()
    try:
        log_sq = np.log(loc_resid ** 2 + 1e-8)
        gam_scale = LinearGAM().fit(X_expl, log_sq)
        log_scale = gam_scale.predict(X_expl)
        scale = np.clip(np.exp(0.5 * log_scale), 1e-6, None)
        return loc_resid / scale
    except Exception:
        return loc_resid


def trace_one_seed(n, seed, alpha=0.01):
    """Run both methods on one seed and trace all HSIC tests."""
    X = gen_simple_lsnm(n, seed)
    p = X.shape[1]  # 2

    print(f"\n{'='*70}")
    print(f"SEED {seed}   (n={n}, p={p}, alpha={alpha})")
    print(f"{'='*70}")

    # ── Step 1: Neighbourhood (same for both — uses raw X) ────────────────
    pval_raw = hsic_pvalue(X[:, 0], X[:, 1])
    in_neighbourhood = pval_raw <= alpha
    print(f"\n[Neighbourhood] raw X:  HSIC p-value = {pval_raw:.6f}")
    print(f"  x0 in N[x1] and x1 in N[x0]?  {in_neighbourhood}"
          f"  ({'PASS — correlated' if in_neighbourhood else 'FAIL — appear independent'})")

    # ── Step 2: Parent search (Algorithm 1) ───────────────────────────────
    # With p=2, the only candidate parent set is {x0,x1} with t=2.
    # _get_child tests: if x0 is child, parent={x1}; if x1 is child, parent={x0}.
    # But _check_correlation requires parent in N[child].
    # If neighbourhood failed, no parents will be found.
    print(f"\n[Parent Search]")
    if not in_neighbourhood:
        print("  Skipped — neighbourhood empty, so _check_correlation fails for all pairs.")
        print("  P[x0] = {}, P[x1] = {}")
        P_add = [set(), set()]
        P_lsnm = [set(), set()]
    else:
        # Additive
        r0_add = additive_residual(X, 0, {1})
        pval_add_01 = hsic_pvalue(r0_add, X[:, 1])
        r1_add = additive_residual(X, 1, {0})
        pval_add_10 = hsic_pvalue(r1_add, X[:, 0])
        print(f"  [Additive] residual(x0|x1) vs x1: p={pval_add_01:.6f}  "
              f"{'indep' if pval_add_01 > alpha else 'dep'}")
        print(f"  [Additive] residual(x1|x0) vs x0: p={pval_add_10:.6f}  "
              f"{'indep' if pval_add_10 > alpha else 'dep'}")

        # LSNM
        r0_lsnm = lsnm_residual(X, 0, {1})
        pval_lsnm_01 = hsic_pvalue(r0_lsnm, X[:, 1])
        r1_lsnm = lsnm_residual(X, 1, {0})
        pval_lsnm_10 = hsic_pvalue(r1_lsnm, X[:, 0])
        print(f"  [LSNM]     residual(x0|x1) vs x1: p={pval_lsnm_01:.6f}  "
              f"{'indep' if pval_lsnm_01 > alpha else 'dep'}")
        print(f"  [LSNM]     residual(x1|x0) vs x0: p={pval_lsnm_10:.6f}  "
              f"{'indep' if pval_lsnm_10 > alpha else 'dep'}")

        # Determine parents (simplified — the child is the one whose residual
        # is most independent of the parent)
        P_add = [set(), set()]
        P_lsnm = [set(), set()]
        # For additive: if residual(x0|{x1}) indep of x1, x1->x0 candidate
        if pval_add_01 > alpha:
            print(f"  [Additive] x1 could be parent of x0 (residual independent)")
        if pval_add_10 > alpha:
            print(f"  [Additive] x0 could be parent of x1 (residual independent)")
        if pval_lsnm_01 > alpha:
            print(f"  [LSNM]     x1 could be parent of x0 (residual independent)")
        if pval_lsnm_10 > alpha:
            print(f"  [LSNM]     x0 could be parent of x1 (residual independent)")

    # ── Step 3: Run full algorithm and get results ────��───────────────────
    print(f"\n[Full Algorithm Run]")

    # Additive (CAMUV)
    model_add = CAMUV(alpha=alpha, num_explanatory_vals=2)
    model_add.fit(X)
    mat_add = model_add.adjacency_matrix_
    print(f"  [Additive] adjacency matrix:")
    print(f"    {mat_add[0]}")
    print(f"    {mat_add[1]}")
    has_bidir_add = np.isnan(mat_add[0, 1]) or np.isnan(mat_add[1, 0])
    has_dir_add = (mat_add[0, 1] == 1) or (mat_add[1, 0] == 1)
    print(f"    Bidirected (NaN)?  {has_bidir_add}")
    print(f"    Directed edge?     {has_dir_add}")

    # LSNM (CAMUV_LSNM)
    model_lsnm = CAMUV_LSNM(alpha=alpha, num_explanatory_vals=2)
    model_lsnm.fit(X)
    mat_lsnm = model_lsnm.adjacency_matrix_
    print(f"  [LSNM]     adjacency matrix:")
    print(f"    {mat_lsnm[0]}")
    print(f"    {mat_lsnm[1]}")
    has_bidir_lsnm = np.isnan(mat_lsnm[0, 1]) or np.isnan(mat_lsnm[1, 0])
    has_dir_lsnm = (mat_lsnm[0, 1] == 1) or (mat_lsnm[1, 0] == 1)
    print(f"    Bidirected (NaN)?  {has_bidir_lsnm}")
    print(f"    Directed edge?     {has_dir_lsnm}")

    # ── Step 4: Algorithm 2 residual independence (the UBP/UCP test) ──────
    # This only runs if no directed edge and pair is in neighbourhood
    print(f"\n[Algorithm 2 — UBP/UCP residual test]")
    if has_dir_add or has_dir_lsnm:
        print("  (A directed edge was found by one method — Algorithm 2 skips this pair)")
    if not in_neighbourhood:
        print("  N-gate: pair NOT in neighbourhood — Algorithm 2 SKIPS this pair entirely")
        print("  Result: no NaN, no directed edge => mat[0,1]=0, mat[1,0]=0 (empty graph)")
    else:
        # Additive residuals of x0 given P[x0], x1 given P[x1]
        r0_add_p = additive_residual(X, 0, P_add[0])
        r1_add_p = additive_residual(X, 1, P_add[1])
        pval_alg2_add = hsic_pvalue(r0_add_p, r1_add_p)
        print(f"  [Additive] residual(x0|P[x0]) vs residual(x1|P[x1]): "
              f"p={pval_alg2_add:.6f}  "
              f"{'indep => no UBP' if pval_alg2_add > alpha else 'DEP => UBP detected!'}")

        r0_lsnm_p = lsnm_residual(X, 0, P_lsnm[0])
        r1_lsnm_p = lsnm_residual(X, 1, P_lsnm[1])
        pval_alg2_lsnm = hsic_pvalue(r0_lsnm_p, r1_lsnm_p)
        print(f"  [LSNM]     residual(x0|P[x0]) vs residual(x1|P[x1]): "
              f"p={pval_alg2_lsnm:.6f}  "
              f"{'indep => no UBP' if pval_alg2_lsnm > alpha else 'DEP => UBP detected!'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[SUMMARY seed={seed}]")
    print(f"  Ground truth: x0 <-> x1 (bidirected)")
    if has_bidir_add:
        print(f"  Additive:  CORRECT (bidirected detected)")
    elif has_dir_add:
        print(f"  Additive:  WRONG (spurious directed edge)")
    else:
        print(f"  Additive:  MISSED (empty graph — no edge found)")

    if has_bidir_lsnm:
        print(f"  LSNM:      CORRECT (bidirected detected)")
    elif has_dir_lsnm:
        print(f"  LSNM:      WRONG (spurious directed edge)")
    else:
        print(f"  LSNM:      MISSED (empty graph — no edge found)")

    return {
        'seed': seed,
        'neighbourhood': in_neighbourhood,
        'pval_raw': pval_raw,
        'add_bidir': has_bidir_add,
        'add_dir': has_dir_add,
        'lsnm_bidir': has_bidir_lsnm,
        'lsnm_dir': has_dir_lsnm,
    }


if __name__ == "__main__":
    n = 500
    results = []
    for seed in range(10):
        r = trace_one_seed(n, seed)
        results.append(r)

    print(f"\n\n{'='*70}")
    print("AGGREGATE (10 seeds)")
    print(f"{'='*70}")
    n_neigh   = sum(r['neighbourhood'] for r in results)
    add_ok    = sum(r['add_bidir'] for r in results)
    lsnm_ok   = sum(r['lsnm_bidir'] for r in results)
    add_dir   = sum(r['add_dir'] for r in results)
    lsnm_dir  = sum(r['lsnm_dir'] for r in results)
    add_empty = sum(not r['add_bidir'] and not r['add_dir'] for r in results)
    lsnm_empty = sum(not r['lsnm_bidir'] and not r['lsnm_dir'] for r in results)

    print(f"  Neighbourhood pass:     {n_neigh}/10")
    print(f"  Additive — correct:     {add_ok}/10  spurious dir: {add_dir}/10  empty: {add_empty}/10")
    print(f"  LSNM     — correct:     {lsnm_ok}/10  spurious dir: {lsnm_dir}/10  empty: {lsnm_empty}/10")
