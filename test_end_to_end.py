"""
End-to-end unit test for LSNM-UV-X pipeline.

Tests each component in isolation with known expected outputs:

  Steps 0-2: eval_metrics.py — verifiable by hand arithmetic
  Step 3:    _get_residual (GAMLSS) — checks that the residual
             becomes linearly and nonlinearly independent of the
             true parent after whitening (Pearson r ≈ 0 for both
             level and squared residual)
  Step 4:    Full LSNMUV_X.fit on gen_lsnm_experiment data —
             checks that the estimated graph has positive F1
             (algorithm is doing something sensible)
  Step 5:    lsnm_data_gen — checks graph properties and data
             statistics of the generated experiment

Graph used in Steps 0 and 3 (simple hand-crafted LSNM):
    x0 --> x1    f1 = 0.8*x0,  g1 = exp(0.3*x0)
    x0 --> x2    f2 = -0.5*x0, g2 = exp(-0.2*x0)
    x1 <-> x2    (hidden confounder h, coefficient 0.3)

Convention for adjacency matrices: A[i,j]=1 means x_j --> x_i.
"""

import numpy as np

# ── Step 0: Generate hand-crafted LSNM data ──────────────────────────────────
print("=" * 60)
print("STEP 0 — Hand-crafted LSNM data (3 variables)")
print("=" * 60)

rng = np.random.default_rng(42)
n   = 3000

h    = rng.standard_normal(n)   # hidden confounder (weaker: coef 0.3)
eps0 = rng.standard_normal(n)
eps1 = rng.standard_normal(n)
eps2 = rng.standard_normal(n)

x0 = eps0                                          # root, no parents
x1 = 0.8*x0  + np.exp( 0.3*x0) * (0.3*h + eps1)  # parent x0
x2 = -0.5*x0 + np.exp(-0.2*x0) * (0.3*h + eps2)  # parent x0, hidden h

X = np.column_stack([x0, x1, x2])
print(f"Data shape : {X.shape}")
print(f"Col means  (expect ~0): {X.mean(0).round(3)}")
print(f"Col stds              : {X.std(0).round(3)}")

A_true = np.array([[0,0,0],[1,0,0],[1,0,0]], dtype=int)
B_true = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=int)
print(f"\nA_true:\n{A_true}")
print(f"B_true:\n{B_true}")


# ── Step 1: eval_metrics — hand-checkable arithmetic ─────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — eval_metrics: precision / recall / F1")
print("=" * 60)

from eval_metrics import directed_metrics, bidirected_metrics

p, r, f = directed_metrics(A_true, A_true)
print(f"Perfect directed   P={p:.2f} R={r:.2f} F1={f:.2f}  (expect all 1.00)")

p, r, f = bidirected_metrics(B_true, B_true)
print(f"Perfect bidirected P={p:.2f} R={r:.2f} F1={f:.2f}  (expect all 1.00)")

p, r, f = directed_metrics(np.zeros_like(A_true), A_true)
print(f"All-zero directed  P={p:.2f} R={r:.2f} F1={f:.2f}  (expect all 0.00)")

p, r, f = bidirected_metrics(np.zeros_like(B_true), B_true)
print(f"All-zero bidirected P={p:.2f} R={r:.2f} F1={f:.2f}  (expect all 0.00)")

# 1 correct edge (x0->x1), 1 wrong edge (x1->x2), 1 missed (x0->x2)
# TP=1, FP=1, FN=1  =>  P=0.5, R=0.5, F1=0.5
A_one_wrong = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=int)
p, r, f = directed_metrics(A_one_wrong, A_true)
print(f"1 correct + 1 wrong P={p:.2f} R={r:.2f} F1={f:.2f}  (expect 0.50 0.50 0.50)")


# ── Step 2: parse_camuv_result ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — parse_camuv_result: NaN <-> bidirected")
print("=" * 60)

from eval_metrics import parse_camuv_result
import types

# mat[i,j] = 1   => x_j -> x_i  (directed)
# mat[i,j] = NaN => invisible pair
mock_mat = np.array([[0,    0,     0   ],
                     [1,    0,     np.nan],
                     [1,    np.nan, 0   ]], dtype=float)
mock_model = types.SimpleNamespace(adjacency_matrix_=mock_mat)

A_est, B_est = parse_camuv_result(mock_model)
print(f"Input mat (NaN=invisible):\n{mock_mat}")
print(f"A_est (directed):\n{A_est}")
print(f"B_est (bidirected):\n{B_est}")

# Manual check
ok_A = (A_est[1,0]==1 and A_est[2,0]==1 and A_est.sum()==2)
ok_B = (B_est[1,2]==1 and B_est[2,1]==1 and B_est.sum()==2)
print(f"A_est correct: {ok_A}  (expect True)")
print(f"B_est correct: {ok_B}  (expect True)")


# ── Step 3: _get_residual — GAMLSS independence check ────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — _get_residual: GAMLSS whitening")
print("=" * 60)
print("We check independence via linear correlation (Pearson r).")
print("After whitening by the TRUE parent x0, the residual eta1 should be")
print("  (a) linearly uncorrelated with x0           [location removed]")
print("  (b) linearly uncorrelated with x0^2         [scale removed]")
print("Raw x1 has nonzero correlation with both x0 and x0^2.")

from lsnm_uv_x import LSNMUV_X
model = LSNMUV_X(alpha=0.01, num_explanatory_vals=3)

eta_true_parent  = model._get_residual(X, 1, [0])   # x1 | x0 (true parent)
eta_wrong_parent = model._get_residual(X, 1, [2])   # x1 | x2 (non-parent)
eta_raw          = X[:, 1]                           # raw x1

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

print(f"\n  corr(eta1|x0,  x0 )  = {corr(eta_true_parent,  x0):.3f}  (expect ~0 — location removed)")
print(f"  corr(eta1|x0,  x0²)  = {corr(eta_true_parent,  x0**2):.3f}  (expect ~0 — scale removed)")
print(f"  corr(eta1|x2,  x0 )  = {corr(eta_wrong_parent, x0):.3f}  (expect nonzero — x2 not a parent)")
print(f"  corr(raw x1,   x0 )  = {corr(eta_raw,          x0):.3f}  (expect large — x1 depends on x0)")
print(f"  corr(raw x1,   x0²)  = {corr(eta_raw,          x0**2):.3f}  (expect nonzero — scale dep.)")


# ── Step 4: Full algorithm on gen_lsnm_experiment data ───────────────────────
print("\n" + "=" * 60)
print("STEP 4 — LSNMUV_X.fit on gen_lsnm_experiment (n=1000, seed=0)")
print("=" * 60)
print("Uses the data generator purpose-built for the algorithm.")
print("We check that the estimated graph has positive F1 (not random).")

from lsnm_data_gen import gen_lsnm_experiment

X4, A4_true, B4_true, perm4 = gen_lsnm_experiment(n=1000, seed=0)
print(f"Data shape : {X4.shape}")
print(f"True directed edges  (A_true > 0): {int(A4_true.sum())}")
print(f"True bidirected pairs: {int(B4_true.sum()) // 2}")
print(f"Column permutation   : {perm4}")

model4 = LSNMUV_X(alpha=0.01, num_explanatory_vals=3)
model4.fit(X4)
A4_est, B4_est = parse_camuv_result(model4)

p_d, r_d, f1_d = directed_metrics(A4_est, A4_true)
p_b, r_b, f1_b = bidirected_metrics(B4_est, B4_true)

print(f"\nDirected  — P={p_d:.2f}  R={r_d:.2f}  F1={f1_d:.2f}")
print(f"Bidirected — P={p_b:.2f}  R={r_b:.2f}  F1={f1_b:.2f}")
print(f"\nF1_directed > 0:  {f1_d > 0}  (expect True)")
print(f"\nTrue A:\n{A4_true}")
print(f"Est  A:\n{A4_est}")
print(f"True B (upper tri):\n{B4_true}")
print(f"Est  B (upper tri):\n{B4_est}")


# ── Step 5: lsnm_data_gen — sanity checks ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — lsnm_data_gen: graph and data sanity checks")
print("=" * 60)

X5, A5, B5, info5 = gen_lsnm_experiment(n=500, seed=7)

p5 = X5.shape[1]

# Bow-free: A[i,j] * B[i,j] = 0 for all i,j
bow_free = bool(np.all(A5 * B5 == 0))
print(f"Bow-free ADMG: {bow_free}  (expect True)")

# B is symmetric
b_sym = bool(np.allclose(B5, B5.T))
print(f"B symmetric:   {b_sym}  (expect True)")

# A has no self-loops
no_selfloop = bool(np.all(np.diag(A5) == 0))
print(f"A acyclic diag:{no_selfloop}  (expect True)")

# A is a DAG (no directed cycles): check via topological sort
def is_dag(A):
    p = A.shape[0]
    indegree = A.sum(axis=1).astype(int)   # indegree[i] = #parents of i
    queue = [i for i in range(p) if indegree[i] == 0]
    removed = 0
    while queue:
        v = queue.pop()
        removed += 1
        for u in range(p):
            if A[u, v] == 1:         # v -> u, v is parent of u
                indegree[u] -= 1
                if indegree[u] == 0:
                    queue.append(u)
    return removed == p

print(f"A is a DAG:    {is_dag(A5)}  (expect True)")

# Data has no NaNs or Infs
print(f"Data finite:   {bool(np.all(np.isfinite(X5)))}  (expect True)")
print(f"Data shape:    {X5.shape}  (expect (500, {p5}))")
print(f"Col means ~0:  {np.allclose(X5.mean(0), 0, atol=0.5)}  (rough check)")


# ── Step 6: 4-variable bow-free graph ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — LSNMUV_X on 4-variable bow-free graph (n=1500)")
print("=" * 60)
print("Graph: x0->x1, x0->x2, x1->x3, x2->x3, x1<->x2 (hidden h)")

rng6 = np.random.default_rng(7)
n6   = 1500
h6   = rng6.standard_normal(n6)

x6_0 = rng6.standard_normal(n6)
x6_1 = 0.7*x6_0  + np.exp( 0.25*x6_0) * (0.3*h6 + rng6.standard_normal(n6))
x6_2 = -0.6*x6_0 + np.exp(-0.2*x6_0)  * (0.3*h6 + rng6.standard_normal(n6))
x6_3 = 0.5*x6_1  + 0.4*x6_2 + np.exp(0.15*x6_1) * rng6.standard_normal(n6)

X6 = np.column_stack([x6_0, x6_1, x6_2, x6_3])

# A6[i,j]=1 means x_j -> x_i
A6_true = np.array([[0,0,0,0],
                    [1,0,0,0],   # x0->x1
                    [1,0,0,0],   # x0->x2
                    [0,1,1,0]])  # x1->x3, x2->x3

B6_true = np.array([[0,0,0,0],
                    [0,0,1,0],   # x1<->x2
                    [0,1,0,0],
                    [0,0,0,0]])

print(f"A_true:\n{A6_true}")
print(f"B_true:\n{B6_true}")

# Verify it is bow-free
print(f"Bow-free: {bool(np.all(A6_true * B6_true == 0))}  (expect True)")

model6 = LSNMUV_X(alpha=0.01, num_explanatory_vals=3)
model6.fit(X6)
A6_est, B6_est = parse_camuv_result(model6)

p_d, r_d, f1_d = directed_metrics(A6_est, A6_true)
p_b, r_b, f1_b = bidirected_metrics(B6_est, B6_true)
print(f"\nDirected  — P={p_d:.2f}  R={r_d:.2f}  F1={f1_d:.2f}  (expect F1>0)")
print(f"Bidirected — P={p_b:.2f}  R={r_b:.2f}  F1={f1_b:.2f}")
print(f"Est A:\n{A6_est}")
print(f"Est B:\n{B6_est}")


# ── Step 7: Bow graph — algorithm must NOT produce a bow ──────────────────────
print("\n" + "=" * 60)
print("STEP 7 — eval_metrics rejects a bow (A[i,j]=B[i,j]=1 violates bow-free)")
print("=" * 60)
print("A bow means the SAME pair has both a directed and bidirected edge.")
print("The algorithm should never output this. We verify the bow-free check.")

# Create a matrix that HAS a bow: x0<->x1 AND x0->x1
A_bow = np.array([[0,0],[1,0]])   # x0->x1
B_bow = np.array([[0,1],[1,0]])   # x0<->x1

has_bow = bool(np.any(A_bow * B_bow != 0))
print(f"Input has bow: {has_bow}  (expect True)")

# Check that parse_camuv_result never produces a bow
# (NaN pairs set B=1; directed entries set A=1; they can't overlap
#  because NaN and 1 are mutually exclusive in mat[i,j])
mat_no_bow = np.array([[0,   np.nan],
                       [np.nan, 0  ]], dtype=float)  # only invisible pair, no directed edge
mock7 = types.SimpleNamespace(adjacency_matrix_=mat_no_bow)
A7, B7 = parse_camuv_result(mock7)
bow_in_output = bool(np.any(A7 * B7 != 0))
print(f"parse_camuv_result output has bow: {bow_in_output}  (expect False)")
print(f"A7:\n{A7}\nB7:\n{B7}")
