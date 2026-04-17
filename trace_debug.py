"""
=============================================================================
DIAGNOSTIC TRACE: CAMUV (additive) vs CAMUV_LSNM on a minimal hidden-cause graph.
=============================================================================

Graph:
    u (HIDDEN) ──> x0
    u (HIDDEN) ──> x1
    No direct edge between x0 and x1.

Ground truth ADMG (after marginalising u):
    x0 <──> x1   (bidirected / invisible pair)

Data model: LSNM (heteroscedastic)
    x0 = f0(u) + g0(u) * eps0       eps0 ~ N(0,1)
    x1 = f1(u) + g1(u) * eps1       eps1 ~ N(0,1)

    f0, f1 : nonlinear location functions of u
    g0, g1 : nonlinear scale functions of u  (THIS is the heteroscedasticity)

We observe only x0 and x1 (not u).  The algorithm must figure out that
x0 and x1 are connected by a hidden cause, NOT by a direct edge.

This script traces EVERY decision point in the CAM-UV algorithm
(Maeda & Shimizu 2021) for both residual types, printing:
  - what test is being run
  - what the p-value is
  - what the algorithm concludes
  - WHERE in the code (camuv_lsnm.py / camuv-original.py) this happens

Run:  python trace_debug.py
=============================================================================
"""

import itertools
import numpy as np
from pygam import LinearGAM
from lingam.hsic import hsic_test_gamma


# =============================================================================
# DATA GENERATION
# Uses exactly the same functions as lsnm_data_gen.py:
#   random_nlfunc  — Maeda & Shimizu (2021) Eq. (8)
#   _gen_lsnm_variable — two-level LSNM (paper Eq. 1/2/7)
# =============================================================================

from lsnm_data_gen import random_nlfunc, _gen_lsnm_variable


def gen_simple_lsnm(n, seed):
    """
    Generate LSNM data from the minimal hidden-cause graph.

    Graph (full, including hidden variable):
        u -> x0
        u -> x1
        No direct edge between x0 and x1.

    Variable layout:
        x0 = observed variable 0   (index 0)
        x1 = observed variable 1   (index 1)
        u  = hidden common cause   (index 2)

    Generation order (topological):
        1. u  — root node, no parents:
               eta_u = eps_u  (no hidden parents, no observed parents)
               v_u = eta_u
        2. x0 — observed parents K_0 = {} (empty), hidden parents Q_0 = {u}:
               Layer 2: eta_0 = f_0^2(u) + g_0^2(u) * eps_0
               Layer 1: v_0 = eta_0  (K_0 empty, so f_0^1=0, g_0^1=1)
        3. x1 — observed parents K_1 = {} (empty), hidden parents Q_1 = {u}:
               Layer 2: eta_1 = f_1^2(u) + g_1^2(u) * eps_1
               Layer 1: v_1 = eta_1  (K_1 empty, so f_1^1=0, g_1^1=1)

    Key structural property:
        x0 and x1 share hidden parent u, so eta_0 and eta_1 are DEPENDENT.
        But neither is a direct parent of the other.
        Ground truth ADMG: x0 <-> x1 (bidirected / invisible pair).

    Uses _gen_lsnm_variable from lsnm_data_gen.py — exactly the same
    data generation as the full experiment pipeline (gen_lsnm_experiment).
    """
    rng = np.random.default_rng(seed)

    # Step 1: Generate u (hidden common cause)
    # u has no parents at all, so _gen_lsnm_variable with empty lists
    # just returns normalised eps_u ~ N(0,1)
    u = _gen_lsnm_variable(
        obs_parent_vals=[],    # K_u = {} (no observed parents)
        hid_parent_vals=[],    # Q_u = {} (no hidden parents — u is a root)
        rng=rng,
        n=n,
    )

    # Step 2: Generate x0 (observed, child of hidden u)
    # K_0 = {} (no observed parents), Q_0 = {u}
    # Layer 2 builds eta_0 from u:  eta_0 = f_0^2(u) + g_0^2(u) * eps_0
    # Layer 1 is trivial (K_0 empty): x0 = eta_0
    x0 = _gen_lsnm_variable(
        obs_parent_vals=[],    # K_0 = {} (no observed parents of x0)
        hid_parent_vals=[u],   # Q_0 = {u} (hidden parent)
        rng=rng,
        n=n,
    )

    # Step 3: Generate x1 (observed, child of hidden u)
    # Same structure as x0 but with independently drawn random functions
    x1 = _gen_lsnm_variable(
        obs_parent_vals=[],    # K_1 = {} (no observed parents of x1)
        hid_parent_vals=[u],   # Q_1 = {u} (hidden parent)
        rng=rng,
        n=n,
    )

    # Return only the observed variables (x0, x1), not u
    return np.column_stack([x0, x1])


# =============================================================================
# RESIDUAL FUNCTIONS
# =============================================================================

def additive_residual(X, target, explanatory_ids):
    """
    Original CAM-UV residual (Maeda & Shimizu 2021).

    residual = x_target - GAM(X_explanatory)

    This is the ADDITIVE residual: it only removes the conditional mean.
    If the true model has heteroscedastic noise (LSNM), the scale
    variation g(K_i) remains in the residual.
    """
    explanatory_ids = list(explanatory_ids)
    if len(explanatory_ids) == 0:
        return X[:, target]
    gam = LinearGAM().fit(X[:, explanatory_ids], X[:, target])
    return X[:, target] - gam.predict(X[:, explanatory_ids])


def lsnm_residual(X, target, explanatory_ids):
    """
    LSNM residual (our method, paper Eq. 14).

    Step 1: loc_resid = x_target - GAM_location(X_explanatory)
            This removes the conditional mean, same as additive.
            But loc_resid still contains g(K_i) * eta_i.

    Step 2: Fit GAM on log(loc_resid^2) to estimate log(g(K_i)^2).
            Then divide: eta_hat = loc_resid / g_hat(K_i).
            This removes the scale, recovering eta_i.

    THE PROBLEM (found by trace_comparison.py):
    When explanatory_ids contains a non-parent that shares a hidden cause,
    the scale GAM in Step 2 can fit the hidden-cause variance pattern
    through the non-parent, making the residual LOOK independent of that
    non-parent -- causing a spurious parent assignment in Algorithm 1.
    """
    explanatory_ids = list(explanatory_ids)
    if len(explanatory_ids) == 0:
        return X[:, target]

    X_expl = X[:, explanatory_ids]
    xi = X[:, target]

    # Step 1: location (same as additive)
    try:
        gam_loc = LinearGAM().fit(X_expl, xi)
        loc_pred = gam_loc.predict(X_expl)
        loc_resid = xi - loc_pred
    except Exception:
        loc_resid = xi - xi.mean()

    # Step 2: scale (LSNM-specific -- THIS is where the problem can arise)
    try:
        log_sq = np.log(loc_resid ** 2 + 1e-8)
        gam_scale = LinearGAM().fit(X_expl, log_sq)
        log_scale = gam_scale.predict(X_expl)
        scale = np.clip(np.exp(0.5 * log_scale), 1e-6, None)
        return loc_resid / scale
    except Exception:
        return loc_resid


# =============================================================================
# HSIC HELPER
# =============================================================================

def hsic_pval(a, b):
    """Compute HSIC p-value between two 1D arrays."""
    _, p = hsic_test_gamma(a.reshape(-1, 1), b.reshape(-1, 1))
    return p


# =============================================================================
# MAIN TRACE
# =============================================================================

def trace_seed(n, seed, alpha=0.01):
    """
    Trace the full CAM-UV algorithm for one seed, comparing additive vs LSNM.
    Prints every decision with the line number in camuv_lsnm.py / camuv-original.py.
    """
    X = gen_simple_lsnm(n, seed)
    n_samples, d = X.shape   # d = 2 (x0, x1)

    print(f"\n{'#'*70}")
    print(f"# SEED = {seed},  n = {n},  d = {d},  alpha = {alpha}")
    print(f"# Ground truth: x0 <-> x1 (bidirected, hidden common cause)")
    print(f"{'#'*70}")

    for label, resid_fn in [("ADDITIVE", additive_residual),
                             ("LSNM", lsnm_residual)]:
        print(f"\n{'='*70}")
        print(f"   METHOD: {label}")
        print(f"{'='*70}")

        # =================================================================
        # STEP 1: _get_neighborhoods  (camuv_lsnm.py lines 165-176)
        #
        # Purpose: find which pairs of variables are correlated in RAW X.
        # This is IDENTICAL for both methods (uses raw X, not residuals).
        #
        # For each pair (i,j), test:  HSIC(X[:,i], X[:,j])
        # If dependent (p <= alpha), add to each other's neighbourhood.
        # =================================================================
        print(f"\n  ── STEP 1: _get_neighborhoods (lines 165-176) ──")
        print(f"  Purpose: pairwise HSIC on RAW X to find correlated pairs.")
        print(f"  (Same for additive and LSNM -- raw X, no residuals involved)")

        N = [set() for _ in range(d)]
        for i in range(d):
            for j in range(i + 1, d):
                pval = hsic_pval(X[:, i], X[:, j])
                dep = pval <= alpha
                print(f"\n    Line 170-173: HSIC(raw x{i}, raw x{j})")
                print(f"      p-value = {pval:.6f}")
                print(f"      Dependent (p <= {alpha})? {dep}")
                if dep:
                    N[i].add(j)
                    N[j].add(i)
                    print(f"      Line 174-175: N[{i}].add({j}), N[{j}].add({i})")
                else:
                    print(f"      NOT added to neighbourhood")

        print(f"\n    RESULT: N = {N}")

        # =================================================================
        # STEP 2: _find_parents  (camuv_lsnm.py lines 178-229)
        #
        # Purpose: find directed edges (parent sets P[i] for each variable).
        #
        # Main loop: for each subset K of size t, find which variable in K
        # is the "child" (its residual given the others is most independent
        # of the others).  If independent enough, assign the others as parents.
        #
        # For our graph (d=2), the only candidate is K = {0, 1} at t=2.
        # =================================================================
        print(f"\n  ── STEP 2: _find_parents (lines 178-229) ──")
        print(f"  Purpose: find directed edges by testing residual independence.")

        P = [set() for _ in range(d)]    # line 181: parent sets, initially empty
        Y = X.copy()                      # line 183: Y starts as copy of X
        t = 2                             # line 182: start with pairs
        maxnum_vals = 2                   # num_explanatory_vals

        iteration = 0
        while True:
            iteration += 1
            print(f"\n    ── While-loop iteration {iteration}, t = {t} ──")

            changed = False

            # Line 187: enumerate all subsets of size t
            variables_set_list = list(itertools.combinations(set(range(d)), t))
            print(f"    Line 187: subsets of size {t} = {variables_set_list}")

            for vs_tuple in variables_set_list:
                variables_set = set(vs_tuple)
                print(f"\n    ── Processing variables_set = {variables_set} ──")

                # ─────────────────────────────────────────────────────────
                # Line 191: _check_identified_causality (lines 282-288)
                #
                # Returns False if any pair in variables_set already has
                # a directed edge.  Prevents re-processing.
                # ─────────────────────────────────────────────────────────
                skip = False
                vlist = list(variables_set)
                for i_idx, i in enumerate(vlist):
                    for j in vlist[i_idx + 1:]:
                        if (j in P[i]) or (i in P[j]):
                            skip = True
                print(f"    Line 191: _check_identified_causality = {not skip}")
                if skip:
                    print(f"      Already have directed edge between these -> continue")
                    continue

                # ─────────────────────────────────────────────────────────
                # Lines 194-196: _get_child  (lines 243-271)
                #
                # For each candidate child in variables_set:
                #   1. Check _check_correlation: are proposed parents in N[child]?
                #   2. Compute residual(child | parents + P[child])
                #   3. Test HSIC(residual, Y[parents])
                #   4. Pick the child with highest independence (p-value)
                #
                # Returns (child, is_independent):
                #   child = which variable is most likely the child
                #   is_independent = True if p-value > alpha
                #
                # *** THIS IS WHERE ADDITIVE AND LSNM DIVERGE ***
                # ─────────────────────────────────────────────────────────
                print(f"\n    Lines 194-196: _get_child(X, {variables_set}, P, N, Y)")
                print(f"    ────────────────────────────────────────────")

                prev_independence = 0.0    # line 246: start with worst p-value
                max_independence_child = None

                for child in variables_set:
                    parents = variables_set - {child}
                    print(f"\n      ── Candidate: child = x{child}, parents = {parents} ──")

                    # Line 252: _check_prior_knowledge (no PK in our case)
                    print(f"      Line 252: _check_prior_knowledge -> False (no prior knowledge)")

                    # Line 255-256: _check_correlation (lines 290-293)
                    # Checks: is each proposed parent in N[child]?
                    # If NOT, skip this candidate.
                    corr_ok = all(p in N[child] for p in parents)
                    print(f"      Line 255: _check_correlation(child={child}, parents={parents})")
                    print(f"        N[{child}] = {N[child]}")
                    print(f"        All parents in N[{child}]? {corr_ok}")
                    if not corr_ok:
                        print(f"        -> SKIP this child candidate (parent not in neighbourhood)")
                        continue

                    # Line 258: compute residual
                    # THIS IS THE KEY LINE: additive vs LSNM residual
                    regress_set = parents | P[child]
                    print(f"\n      Line 258: residual = _get_residual(X, child={child}, {regress_set})")
                    print(f"        Regressing x{child} on variables {regress_set}")
                    residual = resid_fn(X, child, regress_set)
                    print(f"        Residual computed using {label} method")
                    print(f"        Residual std = {np.std(residual):.4f}")

                    # Line 260: get Y columns for proposed parents
                    # At first iteration, Y = X (no parents found yet)
                    parent_list = list(parents)
                    in_Y = Y[:, parent_list].reshape(n_samples, len(parent_list))
                    print(f"      Line 260: in_Y = Y[:, {parent_list}]  (Y = {'X' if np.allclose(Y, X) else 'residual-updated'})")

                    # Line 261: HSIC test
                    # Question: is residual(child | parents) independent of Y[parents]?
                    # If YES (high p-value): parents successfully "explain" the child.
                    # If NO (low p-value): parents don't explain the child.
                    _, pval = hsic_test_gamma(
                        residual.reshape(-1, 1),
                        in_Y
                    )
                    is_ind = pval > prev_independence
                    print(f"\n      Line 261: HSIC(residual(x{child} | {regress_set}), Y[{parent_list}])")
                    print(f"      ┌─────────────────────────────────────────────────┐")
                    print(f"      │  p-value           = {pval:.6f}                 │")
                    print(f"      │  prev_independence  = {prev_independence:.6f}                 │")
                    print(f"      │  p > prev?          = {is_ind}                       │")
                    print(f"      └─────────────────────────────────────────────────┘")

                    if is_ind:
                        prev_independence = pval
                        max_independence_child = child
                        print(f"      -> UPDATE: best child so far = x{child} (p={pval:.6f})")
                    else:
                        print(f"      -> no update (not better than current best)")

                # Line 266-267: final threshold check
                is_independent = prev_independence > alpha
                print(f"\n    Line 266-267: FINAL CHECK")
                print(f"    ┌──────────────────────────────────────────────────────┐")
                print(f"    │  best p-value     = {prev_independence:.6f}                       │")
                print(f"    │  alpha            = {alpha}                              │")
                print(f"    │  p > alpha?       = {is_independent}                            │")
                print(f"    │  selected child   = {'x' + str(max_independence_child) if max_independence_child is not None else 'None'}                              │")
                print(f"    └──────────────────────────────────────────────────────┘")

                # Line 197: if child is None, skip
                if max_independence_child is None:
                    print(f"    Line 197: child is None -> continue")
                    continue

                # Line 199: if not independent enough, skip
                if not is_independent:
                    print(f"    Line 199: is_independent=False -> continue")
                    print(f"    *** Residuals are DEPENDENT on proposed parents.")
                    print(f"    *** This means the proposed parents DON'T explain the child.")
                    print(f"    *** No parent assignment. CORRECT for our graph! ***")
                    continue

                child = max_independence_child
                parents = variables_set - {child}
                print(f"\n    Line 202: Proposed: parents={parents} -> child=x{child}")
                print(f"    *** is_independent=True: residual LOOKS independent of parents.")
                print(f"    *** Algorithm thinks parents explain the child. ***")

                # ─────────────────────────────────────────────────────────
                # Line 203: _check_independence_withou_K  (lines 273-280)
                #
                # Safety check: are Y[child] and Y[parent] dependent?
                # They MUST be dependent (correlated) for the edge to make sense.
                # If independent, the edge is rejected.
                # ─────────────────────────────────────────────────────────
                print(f"\n    Line 203: _check_independence_withou_K(parents={parents}, child={child})")
                print(f"      Purpose: verify Y[child] and Y[parent] are DEPENDENT")
                withou_k_ok = True
                for parent in parents:
                    pval_yk = hsic_pval(Y[:, child], Y[:, parent])
                    dep_yk = pval_yk <= alpha
                    print(f"      HSIC(Y[x{child}], Y[x{parent}]): p={pval_yk:.6f} -> {'DEPENDENT (good)' if dep_yk else 'INDEPENDENT (reject!)'}")
                    if not dep_yk:
                        withou_k_ok = False
                        break
                print(f"    _check_independence_withou_K = {withou_k_ok}")

                if not withou_k_ok:
                    print(f"    -> Edge rejected: Y[child] and Y[parent] are independent")
                    continue

                # ─────────────────────────────────────────────────────────
                # Lines 206-209: ASSIGN PARENT
                #
                # This is where the algorithm COMMITS to a directed edge.
                # For our graph, this should NEVER happen (no true parents).
                # ─────────────────────────────────────────────────────────
                for parent in parents:
                    P[child].add(parent)
                    changed = True
                    Y[:, child] = resid_fn(X, child, P[child])

                    print(f"\n    ╔══════════════════════════════════════════════════╗")
                    print(f"    ║  Line 207: P[{child}].add({parent})                          ║")
                    print(f"    ║  PARENT ASSIGNED: x{parent} ──> x{child}                     ║")
                    if label == "LSNM":
                        print(f"    ║  *** THIS IS WRONG! No true direct edge. ***     ║")
                    print(f"    ╚══════════════════════════════════════════════════╝")
                    print(f"    Line 209: Y[:, {child}] updated with residual(x{child} | P[{child}]={P[child]})")

            # End of for-loop over subsets
            print(f"\n    End of subset loop. changed = {changed}")
            if changed:
                t = 2
                print(f"    Line 211-212: changed=True -> reset t=2, loop again")
            else:
                t += 1
                print(f"    Line 213-214: changed=False -> t incremented to {t}")
                if t > maxnum_vals:
                    print(f"    Line 215-216: t={t} > maxnum_vals={maxnum_vals} -> BREAK out of while")
                    break

        print(f"\n    After while-loop: P = {P}")

        # =================================================================
        # STEP 2b: Prune non-parents  (lines 218-227)
        #
        # For each assigned parent j of variable i:
        #   Compute residual(i | P[i] - {j}) and residual(j | P[j])
        #   If these are INDEPENDENT, j is not a true parent -> remove.
        # =================================================================
        print(f"\n  ── STEP 2b: Prune non-parents (lines 218-227) ──")
        any_pruned = False
        for i in range(d):
            non_parents = set()
            for j in list(P[i]):
                r_i = resid_fn(X, i, P[i] - {j})
                r_j = resid_fn(X, j, P[j])
                pval_prune = hsic_pval(r_i, r_j)
                ind = pval_prune > alpha
                print(f"    Line 221-225: HSIC(resid(x{i} | P[{i}]-{{{j}}}), resid(x{j} | P[{j}]))")
                print(f"      P[{i}] - {{{j}}} = {P[i] - {j}}")
                print(f"      P[{j}] = {P[j]}")
                print(f"      p-value = {pval_prune:.6f}")
                if ind:
                    print(f"      -> INDEPENDENT: x{j} is NOT a real parent of x{i}, PRUNE it")
                    non_parents.add(j)
                    any_pruned = True
                else:
                    print(f"      -> DEPENDENT: x{j} kept as parent of x{i}")
            P[i] = P[i] - non_parents

        if not any_pruned and all(len(p) == 0 for p in P):
            print(f"    (No parents to prune -- P is empty)")

        print(f"\n    Final P after pruning = {P}")

        # =================================================================
        # STEP 3: Algorithm 2 -- UBP/UCP detection  (fit() lines 67-82)
        #
        # For each pair (i,j) with no directed edge AND in neighbourhood:
        #   Compute residual(i | P[i]) and residual(j | P[j])
        #   If DEPENDENT -> mark as invisible pair (NaN = bidirected)
        #
        # Two gates must pass:
        #   Gate (a) line 71: no directed edge (i not in P[j] and j not in P[i])
        #   Gate (b) line 73: in neighbourhood (i in N[j] and j in N[i])
        # =================================================================
        print(f"\n  ── STEP 3: Algorithm 2 -- UBP/UCP detection (lines 67-82) ──")
        print(f"  Purpose: find hidden-variable pairs (bidirected edges)")

        U = []
        for i in range(d):
            for j in range(i + 1, d):
                print(f"\n    Pair (x{i}, x{j}):")

                # Gate (a): skip if directed edge exists
                gate_a = (i in P[j]) or (j in P[i])
                print(f"    Line 71: Gate (a): (x{i} in P[x{j}]) or (x{j} in P[x{i}])?")
                print(f"      P[{j}] = {P[j]}, P[{i}] = {P[i]}")
                print(f"      Result: {gate_a}")
                if gate_a:
                    print(f"      *** BLOCKED: directed edge exists, SKIP this pair ***")
                    print(f"      *** Algorithm 2 will NEVER test if this is a hidden-cause pair ***")
                    continue
                print(f"      PASSED: no directed edge")

                # Gate (b): skip if not in neighbourhood
                gate_b = (i not in N[j]) or (j not in N[i])
                print(f"    Line 73: Gate (b): (x{i} not in N[x{j}]) or (x{j} not in N[x{i}])?")
                print(f"      N[{j}] = {N[j]}, N[{i}] = {N[i]}")
                print(f"      Result: {gate_b}")
                if gate_b:
                    print(f"      *** BLOCKED: not in neighbourhood, SKIP ***")
                    continue
                print(f"      PASSED: in neighbourhood")

                # Residual independence test
                r_i = resid_fn(X, i, P[i])
                r_j = resid_fn(X, j, P[j])
                pval_ubp = hsic_pval(r_i, r_j)
                dep = pval_ubp <= alpha
                print(f"    Lines 76-80: HSIC(resid(x{i} | P[{i}]={P[i]}), resid(x{j} | P[{j}]={P[j]}))")
                print(f"      p-value = {pval_ubp:.6f}")
                if dep:
                    U.append({i, j})
                    print(f"      -> DEPENDENT: hidden-cause pair detected! U += {{{i},{j}}}")
                else:
                    print(f"      -> INDEPENDENT: no hidden cause between x{i} and x{j}")

        print(f"\n    Final U = {U}")

        # =================================================================
        # FINAL RESULT
        # =================================================================
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║  FINAL RESULT ({label:>8s})                               ║")
        print(f"  ║  P (directed edges) = {str(P):<34s} ║")
        print(f"  ║  U (hidden pairs)   = {str(U):<34s} ║")

        if len(U) > 0 and all(len(p) == 0 for p in P):
            print(f"  ║  VERDICT: ✓ CORRECT  (bidirected detected, no dir edge)  ║")
        elif any(len(p) > 0 for p in P):
            edges = [(child, par) for child in range(d) for par in P[child]]
            print(f"  ║  VERDICT: ��� WRONG    (spurious edge {edges})    ║")
            if len(U) == 0:
                print(f"  ║           Bidirected pair MISSED                        ║")
        else:
            print(f"  ║  VERDICT: ✗ MISSED   (empty graph)                       ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    n = 500
    alpha = 0.01

    print("="*70)
    print("DIAGNOSTIC TRACE: CAM-UV additive vs LSNM residuals")
    print(f"Graph: u -> x0, u -> x1  (hidden common cause)")
    print(f"Ground truth: x0 <-> x1  (bidirected)")
    print(f"n = {n}, alpha = {alpha}, 10 seeds")
    print("="*70)

    results = []
    for seed in range(10):
        trace_seed(n, seed, alpha)
        # Also run the actual algorithm to confirm
        from lingam import CAMUV as CAMUV_orig
        from camuv_lsnm import CAMUV_LSNM
        X = gen_simple_lsnm(n, seed)
        m_add = CAMUV_orig(alpha=alpha, num_explanatory_vals=2)
        m_add.fit(X)
        m_lsnm = CAMUV_LSNM(alpha=alpha, num_explanatory_vals=2)
        m_lsnm.fit(X)
        results.append({
            'seed': seed,
            'add_bidir': bool(np.any(np.isnan(m_add.adjacency_matrix_))),
            'lsnm_bidir': bool(np.any(np.isnan(m_lsnm.adjacency_matrix_))),
            'add_dir': bool(np.any(m_add.adjacency_matrix_ == 1)),
            'lsnm_dir': bool(np.any(m_lsnm.adjacency_matrix_ == 1)),
        })

    # Summary
    print(f"\n\n{'='*70}")
    print(f"AGGREGATE SUMMARY (10 seeds)")
    print(f"{'='*70}")
    add_ok = sum(r['add_bidir'] and not r['add_dir'] for r in results)
    lsnm_ok = sum(r['lsnm_bidir'] and not r['lsnm_dir'] for r in results)
    add_wrong = sum(r['add_dir'] for r in results)
    lsnm_wrong = sum(r['lsnm_dir'] for r in results)
    print(f"  ADDITIVE: {add_ok}/10 correct (bidirected),  {add_wrong}/10 spurious directed edge")
    print(f"  LSNM:     {lsnm_ok}/10 correct (bidirected),  {lsnm_wrong}/10 spurious directed edge")
    print()
    print("The problem is in _get_child (line 258-261):")
    print("  LSNM residual's scale-division step makes residual(x_i | {x_j})")
    print("  look independent of x_j even when x_j is NOT a parent of x_i.")
    print("  The algorithm then assigns x_j as a parent, blocking Algorithm 2")
    print("  from detecting the true hidden-cause (bidirected) relationship.")
