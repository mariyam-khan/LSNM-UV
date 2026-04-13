"""
Evaluation metrics for causal discovery on bow-free ADMGs.

Metrics follow Maeda & Shimizu (2021) Section 5:
    TP        = correctly inferred directed edges (position + direction)
    Precision = TP / (total estimated directed edges)
    Recall    = TP / (total true directed edges)
    F-measure = 2 · precision · recall / (precision + recall)

Separate metrics are computed for:
    (1) Directed edges  (A matrix)
    (2) UBP/UCP pairs   (B matrix — invisible pairs)

FCI comparison strategy
-----------------------
FCI outputs a PAG (Partial Ancestral Graph).  Only *definite* edges are
extracted for evaluation — edges where both endpoint marks are fully resolved
(no circle 'o' marks).  This penalises FCI on recall but not on precision.

    Definite directed  x_j → x_i :  tail (–) at x_j,  arrowhead (>) at x_i
    Definite bidirected x_i ↔ x_j :  arrowhead (>) at both ends

Causal-learn PAG graph matrix encoding:
    G.graph[i, j]  =  mark at the end of the edge pointing towards node j
        -1  →  tail  (–)
         1  →  arrowhead  (>)
         2  →  circle  (o)
         0  →  no edge between i and j
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _prf(tp: int, fp: int, fn: int):
    """Return (precision, recall, f1) from TP, FP, FN counts."""
    prec = _safe_div(tp, tp + fp)
    rec  = _safe_div(tp, tp + fn)
    f1   = _safe_div(2 * prec * rec, prec + rec)
    return prec, rec, f1


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def directed_metrics(A_est: np.ndarray, A_true: np.ndarray):
    """
    Precision, recall, F1 for directed edges.

    NaN entries in A_est (invisible pairs) are treated as 0 (no directed edge
    estimated).  True directed edges that fall on NaN pairs are counted as FN.
    """
    A_e = np.where(np.isnan(A_est),  0, A_est).astype(int)
    A_t = np.where(np.isnan(A_true), 0, A_true).astype(int)

    tp = int(np.sum((A_e == 1) & (A_t == 1)))
    fp = int(np.sum((A_e == 1) & (A_t == 0)))
    fn = int(np.sum((A_e == 0) & (A_t == 1)))
    return _prf(tp, fp, fn)


def bidirected_metrics(B_est: np.ndarray, B_true: np.ndarray):
    """
    Precision, recall, F1 for UBP/UCP pair identification.

    B matrices are symmetric; evaluated on upper-triangle pairs only.
    """
    p   = B_true.shape[0]
    idx = np.triu_indices(p, k=1)
    est  = np.where(np.isnan(B_est[idx]),  0, B_est[idx]).astype(int)
    true = np.where(np.isnan(B_true[idx]), 0, B_true[idx]).astype(int)

    tp = int(np.sum((est == 1) & (true == 1)))
    fp = int(np.sum((est == 1) & (true == 0)))
    fn = int(np.sum((est == 0) & (true == 1)))
    return _prf(tp, fp, fn)


# ─────────────────────────────────────────────────────────────────────────────
# Result parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_camuv_result(model) -> tuple:
    """
    Extract (A_est, B_est) from a fitted CAMUV or LSNMUV_X instance.

    Adjacency matrix convention (lingam):
        mat[i, j] = 1    →  x_j → x_i  →  A_est[i,j] = 1
        mat[i, j] = NaN  →  invisible pair
        mat[i, j] = 0    →  no edge
    """
    mat = model.adjacency_matrix_
    p   = mat.shape[0]

    A_est = np.where(np.isnan(mat), 0, mat).astype(float)

    B_est = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(i + 1, p):
            if np.isnan(mat[i, j]) or np.isnan(mat[j, i]):
                B_est[i, j] = 1.0
                B_est[j, i] = 1.0

    return A_est, B_est


def parse_fci_result(pag, p: int) -> tuple:
    """
    Extract definite (A_est, B_est) from a causal-learn FCI PAG.

    Causal-learn graph matrix encoding:
        G.graph[i, j]  =  mark at the end pointing towards j
            -1  =  tail  (–)
             1  =  arrowhead  (>)
             2  =  circle  (o)
             0  =  no edge

    Definite directed  x_j → x_i  :  graph[j, i] = -1  AND  graph[i, j] = 1
    Definite bidirected x_i ↔ x_j :  graph[i, j] = 1   AND  graph[j, i] = 1

    Edges with circle marks on either endpoint are UNRESOLVED and not counted.
    """
    A_est = np.zeros((p, p), dtype=int)
    B_est = np.zeros((p, p), dtype=int)

    G = pag.graph   # (p, p) endpoint-mark matrix

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            mark_at_i = G[j, i]   # mark at end pointing towards i (= endpoint at i)
            mark_at_j = G[i, j]   # mark at end pointing towards j (= endpoint at j)

            # Skip if either mark is a circle (unresolved)
            if mark_at_i == 2 or mark_at_j == 2:
                continue

            # Definite directed  x_j → x_i  (arrow at i, tail at j)
            if mark_at_i == 1 and mark_at_j == -1:
                A_est[i, j] = 1

            # Definite bidirected  x_i ↔ x_j  (both arrowheads, record once)
            if mark_at_i == 1 and mark_at_j == 1 and i < j:
                B_est[i, j] = 1
                B_est[j, i] = 1

    return A_est, B_est


def parse_bang_result(r_output, p: int) -> tuple:
    """
    Parse ngBap::bang() R output into (A_est, B_est).

    ngBap::bang returns a named R list.  Relevant fields (confirmed from
    bang/bang.R lines 222-224):

        r_output.rx2('dEdge')
            Binary directed adjacency matrix.
            dEdge[i,j] = 1  means  x_j → x_i  (j is a parent of i).
            Matches our convention: A[i,j] = 1  →  x_j → x_i.

        r_output.rx2('bEdge')
            Bidirected adjacency matrix = siblings + identity.
            bEdge[i,j] = 1 (off-diagonal)  means  x_i ↔ x_j.
            The diagonal is always 1 (added by construction in R) and must
            be zeroed before use.
    """
    A_est = np.zeros((p, p), dtype=int)
    B_est = np.zeros((p, p), dtype=int)

    try:
        # Access named list elements — try rx2 first, fall back to [] for newer rpy2
        try:
            D = np.array(r_output.rx2('dEdge')).reshape(p, p)
            B = np.array(r_output.rx2('bEdge')).reshape(p, p)
        except AttributeError:
            D = np.array(r_output['dEdge']).reshape(p, p)
            B = np.array(r_output['bEdge']).reshape(p, p)

        # Remove the diagonal that R added (bEdge = siblings + diag(1,...,1))
        np.fill_diagonal(B, 0)

        # Directed edges: dEdge[i,j]=1 → x_j → x_i
        A_est = (D != 0).astype(int)

        # Bidirected edges: treat as symmetric, record on upper + lower triangle
        for i in range(p):
            for j in range(i + 1, p):
                if B[i, j] == 1 or B[j, i] == 1:
                    B_est[i, j] = 1
                    B_est[j, i] = 1

    except Exception as e:
        print(f"[parse_bang_result] Warning: {e}")

    return A_est, B_est
