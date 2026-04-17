"""
Step-by-step trace of Algorithm 1 (_find_parents) for seed 0.

Graph: u (hidden) -> x0, u -> x1.  Ground truth: x0 <-> x1 (bidirected).
We trace every decision in the algorithm for both additive and LSNM residuals,
annotated with the exact line in camuv_lsnm.py / camuv-original.py.
"""

import numpy as np
from pygam import LinearGAM
from lingam.hsic import hsic_test_gamma


# ── Data generation (same as trace_comparison.py, seed=0) ────────────────────

def gen_simple_lsnm(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    eps0 = rng.standard_normal(n)
    eps1 = rng.standard_normal(n)
    f0 = (u + rng.uniform(-3, 3)) ** 2 + rng.uniform(-1, 1)
    f1 = (u + rng.uniform(-3, 3)) ** 2 + rng.uniform(-1, 1)
    log_g0 = (u + rng.uniform(-3, 3)) ** 2
    log_g0 = log_g0 / (np.std(log_g0) + 1e-8)
    g0 = np.clip(np.exp(log_g0), 0.1, 10.0)
    log_g1 = (u + rng.uniform(-3, 3)) ** 2
    log_g1 = log_g1 / (np.std(log_g1) + 1e-8)
    g1 = np.clip(np.exp(log_g1), 0.1, 10.0)
    x0 = f0 + g0 * eps0
    x1 = f1 + g1 * eps1
    x0 = (x0 - np.mean(x0)) / (np.std(x0) + 1e-8)
    x1 = (x1 - np.mean(x1)) / (np.std(x1) + 1e-8)
    return np.column_stack([x0, x1])


def hsic_pval(a, b):
    _, p = hsic_test_gamma(a.reshape(-1, 1), b.reshape(-1, 1))
    return p


def additive_resid(X, target, expl):
    expl = list(expl)
    if len(expl) == 0:
        return X[:, target]
    gam = LinearGAM().fit(X[:, expl], X[:, target])
    return X[:, target] - gam.predict(X[:, expl])


def lsnm_resid(X, target, expl):
    expl = list(expl)
    if len(expl) == 0:
        return X[:, target]
    X_e = X[:, expl]
    xi = X[:, target]
    try:
        gam_loc = LinearGAM().fit(X_e, xi)
        loc_r = xi - gam_loc.predict(X_e)
    except Exception:
        loc_r = xi - xi.mean()
    try:
        log_sq = np.log(loc_r ** 2 + 1e-8)
        gam_s = LinearGAM().fit(X_e, log_sq)
        log_s = gam_s.predict(X_e)
        scale = np.clip(np.exp(0.5 * log_s), 1e-6, None)
        return loc_r / scale
    except Exception:
        return loc_r


# ── Trace ────────────────────────────────────────────────────────────────────

X = gen_simple_lsnm(500, seed=0)
n, d = X.shape  # 500, 2
alpha = 0.01

print(f"Data: n={n}, d={d}, alpha={alpha}")
print(f"Graph: u -> x0, u -> x1.  Ground truth: x0 <-> x1\n")

for label, resid_fn in [("ADDITIVE", additive_resid), ("LSNM", lsnm_resid)]:
    print(f"\n{'='*70}")
    print(f"  {label} RESIDUALS")
    print(f"{'='*70}")

    # =====================================================================
    # fit() line 63: N = self._get_neighborhoods(X)
    # _get_neighborhoods (line 165-176): pairwise HSIC on raw X
    # =====================================================================
    print(f"\n--- Step 1: _get_neighborhoods (line 165-176) ---")
    print(f"  Tests HSIC on raw X columns (same for both methods)")
    pval_01 = hsic_pval(X[:, 0], X[:, 1])
    dep = pval_01 <= alpha
    print(f"  HSIC(x0, x1): p-value = {pval_01:.6f}")
    print(f"  Line 173: not self._is_independent => {dep}")
    if dep:
        N = [{1}, {0}]
        print(f"  Line 174-175: N[0].add(1), N[1].add(0)")
    else:
        N = [set(), set()]
        print(f"  N stays empty")
    print(f"  Result: N = {N}")

    # =====================================================================
    # fit() line 64: P = self._find_parents(X, num_explanatory_vals, N)
    # _find_parents (line 178-229)
    # =====================================================================
    print(f"\n--- Step 2: _find_parents (line 178-229) ---")
    P = [set(), set()]
    Y = X.copy()
    t = 2
    maxnum_vals = 2  # num_explanatory_vals

    iteration = 0
    while True:
        iteration += 1
        print(f"\n  === While-loop iteration {iteration}, t={t} ===")

        changed = False
        # Line 187: all combinations of size t
        import itertools
        variables_set_list = list(itertools.combinations(set(range(d)), t))
        print(f"  Line 187: combinations of size {t} = {variables_set_list}")

        for vs in variables_set_list:
            variables_set = set(vs)
            print(f"\n  Processing variables_set = {variables_set}")

            # Line 191: _check_identified_causality
            # For {0,1}: checks if 1 in P[0] or 0 in P[1]
            already_causal = False
            for i in list(variables_set):
                for j in list(variables_set)[list(variables_set).index(i)+1:]:
                    if (j in P[i]) or (i in P[j]):
                        already_causal = True
            print(f"  Line 191: _check_identified_causality = {not already_causal}")
            if already_causal:
                print(f"    -> continue (already have causality between these)")
                continue

            # Line 194-196: _get_child
            print(f"  Line 194: _get_child(X, {variables_set}, P, N, Y)")
            prev_independence = 0.0  # HSIC mode
            max_independence_child = None

            for child in variables_set:
                parents = variables_set - {child}
                print(f"\n    Trying child={child}, parents={parents}")

                # Line 252: _check_prior_knowledge — no prior knowledge
                print(f"    Line 252: _check_prior_knowledge => False (no PK)")

                # Line 255: _check_correlation — parent must be in N[child]
                corr_ok = all(p in N[child] for p in parents)
                print(f"    Line 255: _check_correlation(child={child}, parents={parents}, N)")
                print(f"      N[{child}] = {N[child]}, parents = {parents}")
                print(f"      All parents in N[child]? {corr_ok}")
                if not corr_ok:
                    print(f"      -> skip this child candidate")
                    continue

                # Line 258: residual = _get_residual(X, child, parents | P[child])
                regress_set = parents | P[child]
                print(f"    Line 258: _get_residual(X, child={child}, {regress_set})")
                residual = resid_fn(X, child, regress_set)
                print(f"      Residual computed ({label})")

                # Line 260: in_Y = Y[:, list(parents)]
                # At this point Y = X (no parents found yet)
                in_Y_vals = Y[:, list(parents)]
                print(f"    Line 260: in_Y = Y[:, {list(parents)}]")

                # Line 261: _is_independent_by(residual, in_Y, prev_independence)
                # This tests: is residual(child | parents) independent of Y[parents]?
                _, pval = hsic_test_gamma(
                    residual.reshape(-1, 1), in_Y_vals.reshape(n, len(parents))
                )
                is_ind = pval > prev_independence
                print(f"    Line 261: HSIC(residual(x{child}|{regress_set}), Y[{list(parents)}])")
                print(f"      p-value = {pval:.6f}")
                print(f"      prev_independence = {prev_independence:.6f}")
                print(f"      is_ind = (p > prev) = ({pval:.6f} > {prev_independence:.6f}) = {is_ind}")

                if is_ind:
                    prev_independence = pval
                    max_independence_child = child
                    print(f"      -> Update: max_independence_child = {child}, prev_independence = {pval:.6f}")
                else:
                    print(f"      -> No update")

            # Line 266-267: final check
            is_independent = prev_independence > alpha
            print(f"\n    Line 266-267: prev_independence={prev_independence:.6f} > alpha={alpha}?  {is_independent}")
            print(f"    _get_child returns: child={max_independence_child}, is_independent={is_independent}")

            # Line 197-200
            if max_independence_child is None:
                print(f"  Line 197: child is None -> continue")
                continue
            if not is_independent:
                print(f"  Line 199: not is_independence_with_K -> continue")
                continue

            child = max_independence_child
            parents = variables_set - {child}
            print(f"\n  Line 202: parents = {parents}, child = {child}")

            # Line 203: _check_independence_withou_K
            # Tests: are Y[child] and Y[parent] dependent? (must be dependent)
            print(f"  Line 203: _check_independence_withou_K(parents={parents}, child={child})")
            withou_k_ok = True
            for parent in parents:
                pval_yk = hsic_pval(Y[:, child], Y[:, parent])
                ind = pval_yk > alpha
                print(f"    HSIC(Y[{child}], Y[{parent}]): p={pval_yk:.6f} -> {'indep' if ind else 'dep'}")
                if ind:
                    withou_k_ok = False
                    print(f"    -> independent! Return False (need dependence)")
                    break
            print(f"  _check_independence_withou_K = {withou_k_ok}")

            if not withou_k_ok:
                print(f"  -> continue")
                continue

            # Line 206-209: ADD PARENT!
            for parent in parents:
                P[child].add(parent)
                changed = True
                print(f"\n  *** Line 207: P[{child}].add({parent}) ***")
                print(f"  *** SPURIOUS PARENT ASSIGNED: x{parent} -> x{child} ***")
                # Line 209: update Y
                Y[:, child] = resid_fn(X, child, P[child])
                print(f"  Line 209: Y[:, {child}] = _get_residual(X, {child}, P[{child}])")

        print(f"\n  End of for-loop. changed={changed}")
        if changed:
            t = 2
            print(f"  Line 211-212: changed=True, reset t=2")
        else:
            t += 1
            print(f"  Line 213-214: changed=False, t incremented to {t}")
            if t > maxnum_vals:
                print(f"  Line 215-216: t={t} > maxnum_vals={maxnum_vals} -> BREAK")
                break

    print(f"\n  After while-loop: P = {P}")

    # =====================================================================
    # Lines 218-227: Prune non-parents
    # =====================================================================
    print(f"\n--- Step 3: Prune non-parents (line 218-227) ---")
    for i in range(d):
        non_parents = set()
        for j in list(P[i]):
            resid_i = resid_fn(X, i, P[i] - {j})
            resid_j = resid_fn(X, j, P[j])
            pval_prune = hsic_pval(resid_i, resid_j)
            ind = pval_prune > alpha
            print(f"  Line 221-225: HSIC(resid(x{i}|P[{i}]-{{{j}}}), resid(x{j}|P[{j}]))")
            print(f"    P[{i}]-{{{j}}} = {P[i]-{j}}, P[{j}] = {P[j]}")
            print(f"    p-value = {pval_prune:.6f}  -> {'PRUNE (independent)' if ind else 'KEEP (dependent)'}")
            if ind:
                non_parents.add(j)
        P[i] = P[i] - non_parents

    print(f"\n  Final P = {P}")

    # =====================================================================
    # fit() lines 67-82: Algorithm 2 — UBP/UCP detection
    # =====================================================================
    print(f"\n--- Step 4: Algorithm 2 — UBP/UCP detection (fit lines 67-82) ---")
    U = []
    for i in range(d):
        for j in range(d)[i+1:]:
            print(f"\n  Pair (x{i}, x{j}):")
            # Line 71: gate (a)
            if (i in P[j]) or (j in P[i]):
                print(f"    Line 71: ({i} in P[{j}]) or ({j} in P[{i}]) = True")
                print(f"    -> SKIP (directed edge exists: gate a)")
                continue
            print(f"    Line 71: gate (a) passed (no directed edge)")

            # Line 73: gate (b) — N-gate
            if (i not in N[j]) or (j not in N[i]):
                print(f"    Line 73: ({i} not in N[{j}]) or ({j} not in N[{i}]) = True")
                print(f"    -> SKIP (not in neighbourhood: gate b)")
                continue
            print(f"    Line 73: gate (b) passed (in neighbourhood)")

            # Line 76-80: HSIC test on residuals
            r_i = resid_fn(X, i, P[i])
            r_j = resid_fn(X, j, P[j])
            pval_ubp = hsic_pval(r_i, r_j)
            dep = pval_ubp <= alpha
            print(f"    Line 76-77: residual(x{i}|P[{i}]={P[i]}), residual(x{j}|P[{j}]={P[j]})")
            print(f"    Line 80: HSIC p-value = {pval_ubp:.6f}")
            if dep:
                U.append({i, j})
                print(f"    -> DEPENDENT: UBP detected! U += {{{i},{j}}}")
            else:
                print(f"    -> independent: no UBP")

    print(f"\n  Final U = {U}")

    # =====================================================================
    # Result
    # =====================================================================
    print(f"\n--- FINAL RESULT ({label}) ---")
    print(f"  P (parents):     {P}")
    print(f"  U (bidirected):  {U}")
    if len(U) > 0 and len(P[0]) == 0 and len(P[1]) == 0:
        print(f"  -> CORRECT: no directed edges, bidirected pair detected")
    elif any(len(p) > 0 for p in P):
        parent_edges = [(child, par) for child in range(d) for par in P[child]]
        print(f"  -> WRONG: spurious directed edges {parent_edges}")
        if len(U) == 0:
            print(f"     and bidirected pair MISSED")
    else:
        print(f"  -> MISSED: empty graph")
