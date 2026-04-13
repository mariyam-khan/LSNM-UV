"""
Step-by-step debug of lsnm_uv_x.py.

We reuse the same 5-node graph from debug_datagen.py:
    Observed:  x0, x1, x2   (p=3)
    Hidden:    u0 (common cause), y0 (intermediate)

True ADMG:
    A: x0 → x2  (only remaining direct edge)
    B: x0 ↔ x1  (UCP via y0),  x1 ↔ x2  (UBP via u0)

We walk through _get_residual, the HSIC independence test,
Stage 1 (sink finding + parent assignment), and Stage 2 (checkVisible).

Run locally:
    python3 debug_algorithm.py
"""

import numpy as np
from lsnm_data_gen import _gen_lsnm_variable, compute_true_admg
from lsnm_uv_x import LSNMUV_X
from eval_metrics import parse_camuv_result, directed_metrics, bidirected_metrics
import networkx as nx

SEP = "=" * 60

# ── Rebuild the same graph and data as debug_datagen.py ──────────────────────
p    = 3
ntot = 5   # x0, x1, x2, u0, y0

G_full = np.zeros((ntot, ntot), dtype=int)
G_full[2, 0] = 1   # x0 → x2
G_full[1, 3] = 1   # u0 → x1
G_full[2, 3] = 1   # u0 → x2
G_full[4, 0] = 1   # x0 → y0
G_full[1, 4] = 1   # y0 → x1

cc_pairs  = [(1, 2)]   # UBP: x1 ↔ x2
int_pairs = [(0, 1)]   # UCP: x0 ↔ x1

A_true, B_true = compute_true_admg(G_full, p=3, cc_pairs=cc_pairs, int_pairs=int_pairs)

G_nx       = nx.from_numpy_array(G_full.T, create_using=nx.DiGraph)
topo_order = list(nx.topological_sort(G_nx))

n    = 2000
rng  = np.random.default_rng(42)
data = np.zeros((n, ntot))
for v in topo_order:
    parent_indices  = np.where(G_full[v, :] == 1)[0]
    obs_parent_vals = [data[:, par] for par in parent_indices if par < p]
    hid_parent_vals = [data[:, par] for par in parent_indices if par >= p]
    data[:, v]      = _gen_lsnm_variable(obs_parent_vals, hid_parent_vals, rng, n)

X = data[:, :p]   # observed data only

print(SEP)
print("Setup: same graph as debug_datagen.py")
print(SEP)
print(f"X shape: {X.shape}")
print(f"A_true:\n{A_true}")
print(f"B_true:\n{B_true}")
print("True edges: x0→x2 (directed), x0↔x1 (UCP), x1↔x2 (UBP)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — _get_residual: what does whitening do?
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 1 — _get_residual: GAMLSS whitening on x2")
print(SEP)
print("x2 has true parent x0 and hidden parent u0.")
print("After whitening by x0, residual = eta_2 = f^2(u0) + g^2(u0)*eps.")
print("eta_2 should be INDEPENDENT of x0 but DEPENDENT on x1 (via shared u0).")

model = LSNMUV_X(alpha=0.01, num_explanatory_vals=3)

eta2_true   = model._get_residual(X, 2, [0])    # x2 | x0  (true parent)
eta2_wrong  = model._get_residual(X, 2, [1])    # x2 | x1  (wrong parent)
eta2_raw    = X[:, 2]                            # x2 raw

def corr(a, b):
    a = a - a.mean();  b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

print(f"\n  corr(eta2|x0,  x0) = {corr(eta2_true,  X[:,0]):.4f}  (expect ~0 — x0 removed)")
print(f"  corr(eta2|x0,  x1) = {corr(eta2_true,  X[:,1]):.4f}  (expect nonzero — UBP with x1)")
print(f"  corr(eta2|x1,  x0) = {corr(eta2_wrong, X[:,0]):.4f}  (expect nonzero — x1 is wrong parent)")
print(f"  corr(raw x2,   x0) = {corr(eta2_raw,   X[:,0]):.4f}  (nonzero before whitening)")

print()
print("Heteroscedasticity check: Var(eta2|x0, given x0>0) vs Var(eta2|x0, given x0<=0)")
print("After correct whitening, scale dependence on x0 should be removed.")
var_pos = eta2_true[X[:,0] >  0].var()
var_neg = eta2_true[X[:,0] <= 0].var()
print(f"  Var(eta2|x0,  x0>0)  = {var_pos:.4f}")
print(f"  Var(eta2|x0,  x0<=0) = {var_neg:.4f}")
print(f"  Ratio = {var_pos/(var_neg+1e-10):.3f}  (expect ~1.0 — scale removed)")

var_pos_raw = eta2_raw[X[:,0] >  0].var()
var_neg_raw = eta2_raw[X[:,0] <= 0].var()
print(f"  Var(raw x2,  x0>0)   = {var_pos_raw:.4f}")
print(f"  Var(raw x2,  x0<=0)  = {var_neg_raw:.4f}")
print(f"  Ratio raw = {var_pos_raw/(var_neg_raw+1e-10):.3f}  (expect != 1 — scale NOT removed)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — HSIC independence test
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 2 — HSIC independence test (_is_independent)")
print(SEP)
print("_is_independent(a, b) returns True if HSIC p-value > alpha (0.01).")
print("i.e. True = cannot reject independence = looks independent.")
print()
print("Tests on x2:")
print("  eta2|x0  vs x0  → should be INDEPENDENT (True)  — x0 correctly whitened")
print("  eta2|x0  vs x1  → should be DEPENDENT   (False) — UBP with x1 remains")
print("  raw x2   vs x0  → should be DEPENDENT   (False) — x2 depends on x0")

r_eta2_true  = eta2_true.reshape(n, 1)
r_eta2_wrong = eta2_wrong.reshape(n, 1)
r_raw        = eta2_raw.reshape(n, 1)
r_x0         = X[:, 0].reshape(n, 1)
r_x1         = X[:, 1].reshape(n, 1)

ind_eta2_x0  = model._is_independent(r_eta2_true,  r_x0)
ind_eta2_x1  = model._is_independent(r_eta2_true,  r_x1)
ind_raw_x0   = model._is_independent(r_raw,         r_x0)

print(f"\n  HSIC(eta2|x0, x0) independent: {ind_eta2_x0}  (expect True)")
print(f"  HSIC(eta2|x0, x1) independent: {ind_eta2_x1}  (expect False)")
print(f"  HSIC(raw x2,  x0) independent: {ind_raw_x0}   (expect False)")

print()
print("Tests on x1:")
print("  eta1|x0  — x1 has NO observed parents (x0→x1 was replaced by UCP x0→y0→x1)")
print("  So x1's true obs parent set K_1 = {} — x0 does NOT appear in K_1!")
eta1_empty   = model._get_residual(X, 1, [])     # x1 | {} (true: no obs parents)
eta1_x0      = model._get_residual(X, 1, [0])    # x1 | x0 (x0 is NOT a true obs parent)
r_eta1_empty = eta1_empty.reshape(n, 1)
r_eta1_x0    = eta1_x0.reshape(n, 1)

ind_eta1_empty_x0 = model._is_independent(r_eta1_empty, r_x0)
ind_eta1_empty_x2 = model._is_independent(r_eta1_empty, X[:,2].reshape(n,1))
ind_eta1_x0_x0    = model._is_independent(r_eta1_x0,    r_x0)

print(f"\n  HSIC(eta1|{{}},  x0) independent: {ind_eta1_empty_x0}  (expect False — x0 is ancestor of x1)")
print(f"  HSIC(eta1|{{}},  x2) independent: {ind_eta1_empty_x2}  (expect False — x1↔x2 UBP)")
print(f"  HSIC(eta1|x0,  x0) independent: {ind_eta1_x0_x0}   (expect True or False — x0 is hidden ancestor not obs parent)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Stage 1: fit LSNMUV_X and inspect raw adjacency matrix
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 3 — Stage 1 output: raw adjacency matrix after CAMUV")
print(SEP)
print("CAMUV (Stage 1) identifies:")
print("  - visible parents (entries = 1)")
print("  - invisible pairs (entries = NaN)")
print("  - no edge (entries = 0)")
print()
print("Expected for our graph:")
print("  x2's parent = x0  → A[2,0]=1")
print("  x1 has no obs parents, and x1↔x2 UBP  → A[1,:]=0, NaN on x1-x2 pair")
print("  x0↔x1 UCP  → NaN on x0-x1 pair")

model.fit(X)

# Access the internal adjacency matrix BEFORE checkVisible modifies it
# We run Stage 1 only by using LSNMUV (parent class without checkVisible)
from lsnm_uv_x import LSNMUV
model_s1 = LSNMUV(alpha=0.01, num_explanatory_vals=3)
model_s1.fit(X)
mat_s1 = model_s1.adjacency_matrix_

print(f"\nStage 1 raw adjacency matrix:")
print(f"{mat_s1}")
print(f"\n  NaN positions (invisible pairs):")
for i in range(p):
    for j in range(p):
        if i != j and np.isnan(mat_s1[i,j]):
            print(f"    mat[{i},{j}] = NaN")

print(f"\n  Directed edge positions (1s):")
for i in range(p):
    for j in range(p):
        if mat_s1[i,j] == 1:
            print(f"    mat[{i},{j}] = 1  →  x{j} → x{i}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Stage 2: checkVisible resolves NaN pairs
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 4 — Stage 2 output: after checkVisible")
print(SEP)
print("checkVisible re-examines each NaN pair and tries to resolve direction.")
print("If it can confirm x_j is NOT a parent of x_i, it infers x_j → x_i.")
print("If both are not-parents, it's a visible non-edge (0).")
print("If neither resolved, it stays NaN (truly invisible).")

mat_s2 = model.adjacency_matrix_   # full LSNMUV_X (Stage 1 + Stage 2)
print(f"\nStage 2 adjacency matrix (after checkVisible):")
print(f"{mat_s2}")

print(f"\n  Changes from Stage 1 → Stage 2:")
for i in range(p):
    for j in range(p):
        if i == j:
            continue
        v1 = mat_s1[i,j]
        v2 = mat_s2[i,j]
        both_nan = np.isnan(v1) and np.isnan(v2)
        changed  = not both_nan and not (v1 == v2)
        if changed or (np.isnan(v1) and not np.isnan(v2)):
            s1_str = "NaN" if np.isnan(v1) else str(int(v1))
            s2_str = "NaN" if np.isnan(v2) else str(int(v2))
            print(f"    mat[{i},{j}]: {s1_str} → {s2_str}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Final metrics
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("STEP 5 — Final metrics vs ground truth")
print(SEP)

A_est, B_est = parse_camuv_result(model)

print(f"A_true:\n{A_true}")
print(f"A_est:\n{A_est.astype(int)}")
print(f"\nB_true:\n{B_true}")
print(f"B_est:\n{B_est.astype(int)}")

p_d, r_d, f1_d = directed_metrics(A_est, A_true)
p_b, r_b, f1_b = bidirected_metrics(B_est, B_true)

print(f"\nDirected  — P={p_d:.2f}  R={r_d:.2f}  F1={f1_d:.2f}")
print(f"Bidirected — P={p_b:.2f}  R={r_b:.2f}  F1={f1_b:.2f}")

# Entry-by-entry comparison
print(f"\nEntry-by-entry directed comparison (true vs est):")
for i in range(p):
    for j in range(p):
        if A_true[i,j] == 1 or A_est[i,j] == 1:
            t = A_true[i,j]; e = int(A_est[i,j])
            status = "TP" if t==1 and e==1 else ("FP" if e==1 else "FN")
            print(f"  x{j}→x{i}: true={t} est={e}  [{status}]")

print(f"\nEntry-by-entry bidirected comparison (upper triangle):")
for i in range(p):
    for j in range(i+1, p):
        if B_true[i,j] == 1 or B_est[i,j] == 1:
            t = B_true[i,j]; e = int(B_est[i,j])
            status = "TP" if t==1 and e==1 else ("FP" if e==1 else "FN")
            print(f"  x{i}↔x{j}: true={t} est={e}  [{status}]")
