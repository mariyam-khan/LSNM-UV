"""
End-to-end unit test for LSNM-UV-X pipeline.

Graph:  x0 --> x1
        x0 --> x2
        x1 <-> x2   (hidden confounder h ~ N(0,1) enters both x1 and x2)

True adjacency matrices (convention: A[i,j]=1 means x_j --> x_i):
    A_true = [[0, 0, 0],
              [1, 0, 0],   # x0 --> x1
              [1, 0, 0]]   # x0 --> x2

    B_true = [[0, 0, 0],
              [0, 0, 1],   # x1 <-> x2
              [0, 1, 0]]
"""

import numpy as np

# ── Step 0: fix random seed for reproducibility ───────────────────────────────
rng = np.random.default_rng(42)
n   = 2000   # large n so the test is reliable

print("=" * 60)
print("STEP 0 — Generate data from known LSNM graph")
print("=" * 60)

# Hidden confounder
h = rng.standard_normal(n)

# x0: root node, no parents
eps0 = rng.standard_normal(n)
x0   = eps0                              # f=0, g=1 (pure noise)

# x1: parent x0, plus hidden h
eps1 = rng.standard_normal(n)
f1   = 0.8 * x0                         # location depends on x0
g1   = np.exp(0.3 * x0)                 # scale depends on x0  (LSNM!)
x1   = f1 + g1 * (0.6 * h + eps1)       # hidden h enters the noise

# x2: parent x0, plus hidden h
eps2 = rng.standard_normal(n)
f2   = -0.5 * x0                        # location depends on x0
g2   = np.exp(-0.2 * x0)                # scale depends on x0
x2   = f2 + g2 * (0.6 * h + eps2)       # same hidden h -> x1 <-> x2

X = np.column_stack([x0, x1, x2])
print(f"Data shape: {X.shape}")
print(f"Column means  (should be ~0): {X.mean(axis=0).round(3)}")
print(f"Column stds   (should be ~1-3): {X.std(axis=0).round(3)}")

# ── Ground truth ──────────────────────────────────────────────────────────────
A_true = np.array([[0,0,0],
                   [1,0,0],
                   [1,0,0]], dtype=int)

B_true = np.array([[0,0,0],
                   [0,0,1],
                   [0,1,0]], dtype=int)

print(f"\nA_true (directed):\n{A_true}")
print(f"B_true (bidirected):\n{B_true}")


print("\n" + "=" * 60)
print("STEP 1 — Test eval_metrics helpers directly")
print("=" * 60)

from eval_metrics import directed_metrics, bidirected_metrics

# Perfect prediction
p_d, r_d, f1_d = directed_metrics(A_true, A_true)
p_b, r_b, f1_b = bidirected_metrics(B_true, B_true)
print(f"Perfect directed  — P={p_d:.2f} R={r_d:.2f} F1={f1_d:.2f}  (all should be 1.0)")
print(f"Perfect bidirected — P={p_b:.2f} R={r_b:.2f} F1={f1_b:.2f}  (all should be 1.0)")

# All-zeros prediction
A_zero = np.zeros_like(A_true)
B_zero = np.zeros_like(B_true)
p_d, r_d, f1_d = directed_metrics(A_zero, A_true)
p_b, r_b, f1_b = bidirected_metrics(B_zero, B_true)
print(f"Zero directed     — P={p_d:.2f} R={r_d:.2f} F1={f1_d:.2f}  (all should be 0.0)")
print(f"Zero bidirected   — P={p_b:.2f} R={r_b:.2f} F1={f1_b:.2f}  (all should be 0.0)")

# One wrong directed edge: predict A[2,1]=1 (x1->x2) instead of A[2,0]=1 (x0->x2)
A_wrong = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=int)
p_d, r_d, f1_d = directed_metrics(A_wrong, A_true)
print(f"1 wrong directed  — P={p_d:.2f} R={r_d:.2f} F1={f1_d:.2f}")
print(f"  Expected: TP=1(x0->x1 correct), FP=1(x1->x2 wrong), FN=1(x0->x2 missed)")
print(f"  P=1/2=0.50, R=1/2=0.50, F1=0.50")


print("\n" + "=" * 60)
print("STEP 2 — Test parse_camuv_result")
print("=" * 60)

from eval_metrics import parse_camuv_result
import types

# Mock a model object with the right adjacency matrix
# NaN marks invisible pairs; 1 marks directed edge x_j -> x_i
mock_mat = np.array([[0,   0,   0  ],
                     [1,   0,   np.nan],   # A[1,0]=1 (x0->x1), [1,2]=NaN (invisible pair)
                     [1,   np.nan, 0 ]], dtype=float)

mock_model       = types.SimpleNamespace()
mock_model.adjacency_matrix_ = mock_mat

A_est, B_est = parse_camuv_result(mock_model)
print(f"Mock adjacency matrix (NaN = invisible pair):\n{mock_mat}")
print(f"A_est (directed, NaN->0):\n{A_est}")
print(f"B_est (bidirected, 1 where NaN was):\n{B_est}")
print(f"Expected A_est[1,0]=1, A_est[2,0]=1, rest 0")
print(f"Expected B_est[1,2]=B_est[2,1]=1, rest 0")


print("\n" + "=" * 60)
print("STEP 3 — Test GAMLSS residual (_get_residual) directly")
print("=" * 60)

from lsnm_uv_x import LSNMUV_X
model = LSNMUV_X(alpha=0.01, num_explanatory_vals=3)

# Compute residual of x1 given x0 (its true parent)
eta_with_parent = model._get_residual(X, 1, [0])
# Compute residual of x1 given x2 (a non-parent)
eta_with_nonparent = model._get_residual(X, 1, [2])
# Compute residual of x1 given nothing
eta_no_parent = model._get_residual(X, 1, [])

print(f"Residual x1|x0  — std={eta_with_parent.std():.3f}  (should be small, ~1)")
print(f"Residual x1|x2  — std={eta_with_nonparent.std():.3f}")
print(f"Residual x1|()  — std={eta_no_parent.std():.3f}  (should be larger, raw x1)")
print("Note: residual std closer to 1 = better parent found")


print("\n" + "=" * 60)
print("STEP 4 — Fit LSNM-UV-X and check output")
print("=" * 60)

model.fit(X)
mat = model.adjacency_matrix_
print(f"Estimated adjacency matrix:\n{mat}")
print(f"(1=directed edge, NaN=invisible pair, 0=no edge)")

A_est, B_est = parse_camuv_result(model)
print(f"\nA_est:\n{A_est}")
print(f"B_est:\n{B_est}")

p_d, r_d, f1_d = directed_metrics(A_est, A_true)
p_b, r_b, f1_b = bidirected_metrics(B_est, B_true)
print(f"\nDirected  — P={p_d:.2f} R={r_d:.2f} F1={f1_d:.2f}")
print(f"Bidirected — P={p_b:.2f} R={r_b:.2f} F1={f1_b:.2f}")
print(f"\nTrue directed edges:  x0->x1 (A[1,0]=1), x0->x2 (A[2,0]=1)")
print(f"True bidirected edge: x1<->x2 (B[1,2]=B[2,1]=1)")

