"""
LSNM Data Generation — Section 6.1 simulation setup.
======================================================

Overview
--------
This file generates synthetic data for evaluating the LSNM-UV-X algorithm.
Each call to gen_lsnm_experiment() produces one simulation trial: a dataset of
observed variables, and the ground-truth ADMG that the algorithm is expected to
recover.  The graph structure follows Maeda & Shimizu (2021) Section 5.1, and
the data model extends it to the Location-Scale Noise Model of our paper (Eq. 1).


Step 1 — Graph Construction (_build_full_graph)
------------------------------------------------
We construct a full latent DAG over both observed and hidden variables.
Variable indices are laid out as:

    0 … p-1          →  observed variables  x_0, …, x_{p-1}
    p … p+n_cc-1     →  hidden common causes  (root nodes, no parents)
    p+n_cc … end     →  hidden intermediates

With default parameters: p=10 observed variables, n_cc=2 hidden common causes,
n_int=2 hidden intermediates (14 nodes total).

Observed skeleton.
    Direct edges among the p observed variables are drawn from an Erdős-Rényi
    DAG with edge probability 0.3, following Maeda21 Section 5.1 exactly.
    Node index order is the topological order (edge x_j → x_i only if j < i).

Hidden common causes (UBPs).
    For each hidden common cause u_k (a root node with no parents of its own):
      - We select a pair of observed variables (x_a, x_b) with no direct edge
        between them.
      - We add edges u_k → x_a and u_k → x_b.
      - In the projected ADMG this becomes a bidirected edge x_a ↔ x_b,
        representing an Unobserved Backdoor Path (UBP).

Hidden intermediates (UCPs).
    For each hidden intermediate y_k:
      - We select an existing observed-to-observed edge x_j → x_i.
      - We replace it with x_j → y_k → x_i (the direct edge is removed).
      - In the projected ADMG this becomes a bidirected edge x_j ↔ x_i,
        representing an Unobserved Causal Path (UCP).

Bow-free guarantee.
    The construction ensures no observed variable simultaneously has a direct
    edge and a bidirected edge to the same other variable, satisfying the
    bow-free ADMG condition required by the paper (Definition 2.5).


Step 2 — Data Generation (_gen_lsnm_variable)
----------------------------------------------
All variables (observed and hidden) are generated in topological order so that
when we generate variable v_i, all its parents already have values.

For each variable v_i we implement the two-level LSNM from Equation (1):

    v_i  =  f_i^1(K_i)  +  g_i^1(K_i) · η_i          (Layer 1)
    η_i  =  f_i^2(Q_i)  +  g_i^2(Q_i) · ε_i           (Layer 2)

where K_i are the observed parents of v_i, Q_i are the hidden parents of v_i,
and ε_i ~ N(0,1) are mutually independent across all variables.

The critical structural property is the separation of roles: f_i^1 and g_i^1
depend only on K_i; hidden parents Q_i enter v_i exclusively through η_i.
This is what makes Lemma 1 hold — the GAMLSS residual
    η̂_i = (x_i − f̂^1(K_i)) / ĝ^1(K_i)
recovers η_i, whose independence structure across variables reveals the
UBP/UCP structure.

Layer 2 — computing η_i:
  - If Q_i = ∅ (no hidden parents):  η_i = ε_i,  standard independent N(0,1).
  - If Q_i ≠ ∅:
        η_i = f_i^2(Q_i) + g_i^2(Q_i) · ε_i
    where f_i^2(Q_i) = Σ_{q∈Q_i} sign·((q+a)^c+b)  is the location shift from
    hidden parents, and g_i^2(Q_i) = exp(normalised Σ sign·((q+a')^c+b')) is
    the scale from hidden parents.

Layer 1 — computing v_i:
  - If K_i = ∅ (no observed parents):  v_i = η_i  directly.
  - If K_i ≠ ∅:
        v_i = f_i^1(K_i) + g_i^1(K_i) · η_i
    where f_i^1 and g_i^1 have the same additive polynomial form as above,
    computed from observed parents only.

Nonlinear function form.
    Both f and g are sums of polynomial terms over individual parents, following
    Maeda21 Equation (8):

        sign · ((v_j + a)^c + b)

    Parameters are sampled independently for every parent, every variable, and
    separately for location and scale:
        a    ~ Uniform(−5, 5)
        b    ~ Uniform(−1, 1)
        c    ~ {2, 3} with equal probability
        sign ~ {−1, +1} with equal probability

    The scale is always positive because it is obtained by exponentiating the
    sum, after normalising by its standard deviation to prevent numerical
    overflow.  Each variable is normalised to unit variance after generation.

Why heteroscedasticity matters.
    In Maeda21's model, the noise is additive and homoscedastic — the variance
    does not change with parent values.  In our model, the scale g_i^1(K_i)
    depends on parent values, so the noise variance of v_i changes as the
    observed parents change.  This is the key structural extension that
    LSNM-UV-X is designed to handle, and which CAM-UV is not designed for.


Step 3 — Ground-Truth ADMG (compute_true_admg)
-----------------------------------------------
After generating data for all variables, we project the full latent DAG down
to the p observed variables to obtain the ground-truth ADMG (A, B):

    A[i,j] = 1  if there is a direct edge x_j → x_i among observed variables.
    B[i,j] = 1  if there is a UBP or UCP between x_i and x_j (i.e., they
                share a hidden common cause, or are connected via a hidden
                intermediate).

The bow-free constraint is enforced: if A[i,j]=1 and B[i,j]=1 arise
simultaneously (which should not occur given correct construction), B takes
precedence.


Step 4 — Final Output (gen_lsnm_experiment)
--------------------------------------------
A random column permutation is applied to the observed data matrix before
returning it.  This ensures the algorithm cannot exploit the fact that variable
indices coincide with topological order.  The ground-truth A and B matrices are
permuted consistently so evaluation remains valid.

Returns:
    X_perm  —  (n, p) array of observed data, column-permuted
    A_true  —  (p, p) directed adjacency matrix  (A[i,j]=1 means x_j → x_i)
    B_true  —  (p, p) symmetric bidirected matrix (B[i,j]=1 means UBP or UCP)
    perm    —  the permutation applied, for reference


Summary of Design Choices
--------------------------
    p = 10 observed variables           Matches Maeda21 Section 5.1
    n_cc = 2 hidden common causes       Matches Maeda21; each creates one UBP
    n_int = 2 hidden intermediates      Extended from Maeda21; each creates one UCP
    Edge probability 0.3 (ER)          Matches Maeda21 Section 5.1
    Nonlinear form sign·((x+a)^c+b)    Matches Maeda21 Equation (8)
    Scale g_i^1(K_i) depends on K_i    Implements heteroscedastic LSNM; absent in Maeda21
    Two-level generation (K_i, Q_i)    Faithfully implements paper Eq.(1); ensures Lemma 1
    Bow-free construction               Required for identifiability (Theorem 2)

Requirements: numpy, networkx
"""

import numpy as np
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Low-level LSNM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nl_term(x: np.ndarray, a: float, b: float, c: int, sign: float) -> np.ndarray:
    """
    Single nonlinear term from Maeda21 Eq. (8):  sign · ((x + a)^c + b).

    This is the building block for both the location function f and the
    log-scale function log(g).  Parameters a, b, c, sign are sampled
    independently for each parent and each variable (see _lsnm_loc_scale).
    """
    return sign * ((x + a) ** c + b)


def _lsnm_loc_scale(parent_vals: list, rng: np.random.Generator, n: int):
    """
    Compute additive nonlinear location and scale from one set of parent values.

    For each parent v_j in parent_vals, two independent polynomial terms are
    drawn: one contributing to the location, one to the log-scale.

        loc(parents)   = Σ_j  sign_j · ((v_j + a_j)^{c_j} + b_j)
        scale(parents) = exp( normalised  Σ_j  sign_j · ((v_j + a'_j)^{c'_j} + b'_j) )

    Parameters are sampled i.i.d. for every parent and every variable:
        a, a'  ~ Uniform(−5, 5)
        b, b'  ~ Uniform(−1, 1)
        c, c'  ~ {2, 3} with equal probability        (polynomial degree)
        sign   ~ {−1, +1} with equal probability

    The log-scale sum is normalised by its standard deviation before
    exponentiation to prevent numerical overflow.  The scale is clipped
    below at 1e-6 to guarantee strict positivity.

    This function is called twice per variable: once for the hidden parents
    (Layer 2, producing f_i^2 and g_i^2) and once for the observed parents
    (Layer 1, producing f_i^1 and g_i^1).  Parameters are always drawn
    freshly and independently between the two calls.

    Returns (loc, scale) arrays of shape (n,).
    """
    loc       = np.zeros(n)
    scale_log = np.zeros(n)

    for pv in parent_vals:
        # ── Location contribution from this parent ────────────────────────────
        a    = rng.uniform(-5.0, 5.0)
        b    = rng.uniform(-1.0, 1.0)
        c    = int(rng.choice([2, 3]))
        sign = float(rng.choice([-1.0, 1.0]))
        loc += _nl_term(pv, a, b, c, sign)

        # ── Scale contribution from this parent (independent parameters) ──────
        a    = rng.uniform(-5.0, 5.0)
        b    = rng.uniform(-1.0, 1.0)
        c    = int(rng.choice([2, 3]))
        sign = float(rng.choice([-1.0, 1.0]))
        scale_log += _nl_term(pv, a, b, c, sign)

    # Normalise log-scale before exp to avoid numerical overflow
    scale_log_std = np.std(scale_log)
    if scale_log_std > 1e-8:
        scale_log /= scale_log_std
    scale = np.clip(np.exp(scale_log), 1e-6, None)

    return loc, scale


def _gen_lsnm_variable(
    obs_parent_vals: list,
    hid_parent_vals: list,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """
    Generate one variable from the two-level LSNM (Eq. 1 of the paper).

    The model separates the roles of observed parents K_i and hidden parents Q_i:

        Layer 2 (hidden parents):   η_i = f_i^2(Q_i) + g_i^2(Q_i) · ε_i
        Layer 1 (observed parents): v_i = f_i^1(K_i) + g_i^1(K_i) · η_i

    where ε_i ~ N(0,1) is the variable's own independent external noise.

    Key structural property.
        f_i^1 and g_i^1 depend ONLY on observed parents K_i.  Hidden parents
        Q_i enter v_i exclusively through η_i.  This separation ensures that
        when the algorithm fits a GAMLSS on the observed parents K_i, the
        residual it recovers,
            η̂_i = (x_i − f̂^1(K_i)) / ĝ^1(K_i),
        is exactly η_i — a function of hidden parents and ε_i only.  The
        independence structure of η_i across variables then follows Lemma 1:
        η_i ⊥⊥ η_j if and only if there is no UBP or UCP between x_i and x_j.

    Three cases handled:
        Q_i = ∅, K_i = ∅  →  root node:  v_i = ε_i  (pure independent noise)
        Q_i = ∅, K_i ≠ ∅  →  η_i = ε_i, v_i = f^1(K_i) + g^1(K_i)·ε_i
        Q_i ≠ ∅, K_i ≠ ∅  →  full two-level structure above

    Output is normalised to unit variance.
    """
    # ── Layer 2: compute η_i from hidden parents Q_i ──────────────────────────
    # If Q_i is empty, η_i reduces to a standard independent N(0,1) noise,
    # which is the correct behaviour for variables with no hidden parents.
    if len(hid_parent_vals) == 0:
        eta = rng.standard_normal(n)
    else:
        loc2, scale2 = _lsnm_loc_scale(hid_parent_vals, rng, n)
        eta = loc2 + scale2 * rng.standard_normal(n)   # f_i^2(Q_i) + g_i^2(Q_i)·ε_i

    # ── Layer 1: compute v_i from observed parents K_i and η_i ───────────────
    # If K_i is empty, v_i = η_i directly (no observed-parent contribution).
    if len(obs_parent_vals) == 0:
        h = eta
    else:
        loc1, scale1 = _lsnm_loc_scale(obs_parent_vals, rng, n)
        h = loc1 + scale1 * eta                        # f_i^1(K_i) + g_i^1(K_i)·η_i

    # Normalise to unit variance
    return h / (np.std(h) + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def gen_er_dag(p: int, er_prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Erdős-Rényi DAG over p observed variables, following Maeda21 Section 5.1.

    Each ordered pair (j, i) with j < i is connected by an edge x_j → x_i
    independently with probability er_prob.  Because j < i always, the index
    order is a valid topological order and the matrix is lower-triangular.

        G[i, j] = 1  ⟹  x_j → x_i  (j is a parent of x_i)
    """
    G = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i):
            if rng.random() < er_prob:
                G[i, j] = 1
    return G


def _build_full_graph(rng, p=10, n_cc=2, n_int=2, er_prob=0.3):
    """
    Build the full latent DAG over observed and hidden variables (Step 1).

    Variable index layout in G_full  (size = p + n_cc + n_int):
        0 … p-1          →  observed variables  x_0, …, x_{p-1}
        p … p+n_cc-1     →  hidden common causes  (root nodes, no parents)
        p+n_cc … end     →  hidden intermediates

    Hidden common causes (UBPs).
        Each hidden common cause u_k is a root node.  We select a pair of
        observed variables (x_a, x_b) that have no direct edge between them,
        and add edges u_k → x_a and u_k → x_b.  When the hidden variable is
        marginalised out, this induces a bidirected edge x_a ↔ x_b in the
        ADMG, representing an Unobserved Backdoor Path (UBP).

    Hidden intermediates (UCPs).
        Each hidden intermediate y_k replaces one existing observed edge.  We
        select an edge x_j → x_i, remove it, and insert x_j → y_k → x_i.
        When y_k is marginalised out, this induces a bidirected edge x_j ↔ x_i,
        representing an Unobserved Causal Path (UCP).

    Bow-free guarantee.
        Common causes are only connected to pairs with no direct edge, so no
        variable gains both a directed edge and a bidirected edge to the same
        partner (Definition 2.5 of the paper).  Intermediates replace an
        existing edge rather than adding a new one, preserving bow-freeness.

    Returns
    -------
    G_full    : (p+n_cc+n_int, p+n_cc+n_int) int adjacency matrix
    cc_pairs  : list[(child_a, child_b)]       observed children of each common cause
    int_pairs : list[(obs_parent, obs_child)]   observed endpoints of each intermediate
    """
    n_total = p + n_cc + n_int
    G_obs   = gen_er_dag(p, er_prob, rng)

    G_full  = np.zeros((n_total, n_total), dtype=int)
    G_full[:p, :p] = G_obs      # start with the observed skeleton

    cc_pairs  = []
    int_pairs = []

    # ── Hidden common causes: each is a root node with two observed children ──
    for k in range(n_cc):
        hc = p + k
        # Only connect to pairs with no existing direct edge (bow-free condition)
        candidates = [
            (i, j) for i in range(p) for j in range(i + 1, p)
            if G_full[i, j] == 0 and G_full[j, i] == 0
        ]
        if not candidates:
            candidates = [(i, j) for i in range(p) for j in range(i + 1, p)]
        idx  = int(rng.integers(0, len(candidates)))
        a, b = candidates[idx]
        G_full[a, hc] = 1   # u_k → x_a
        G_full[b, hc] = 1   # u_k → x_b  →  projects to UBP: x_a ↔ x_b
        cc_pairs.append((a, b))

    # ── Hidden intermediates: each replaces one existing observed edge ─────────
    for k in range(n_int):
        hi = p + n_cc + k
        edges = [(i, j) for i in range(p) for j in range(p) if G_full[i, j] == 1]
        if not edges:
            break
        idx  = int(rng.integers(0, len(edges)))
        i, j = edges[idx]          # existing edge: x_j → x_i
        G_full[i, j]  = 0          # remove direct edge
        G_full[hi, j] = 1          # x_j  → y_k
        G_full[i, hi] = 1          # y_k  → x_i  →  projects to UCP: x_j ↔ x_i
        int_pairs.append((j, i))   # record (obs_parent, obs_child)

    return G_full, cc_pairs, int_pairs


def compute_true_admg(G_full: np.ndarray, p: int, cc_pairs: list, int_pairs: list):
    """
    Project the full latent DAG onto the ground-truth ADMG (Step 3).

    After generating all data, we discard the hidden variables and record
    what their presence implies for the observed variables:

        A[i, j] = 1  iff  x_j → x_i is a direct edge among observed variables.
        B[i, j] = 1  iff  a UBP or UCP exists between x_i and x_j (symmetric).

    B is set from cc_pairs (UBPs) and int_pairs (UCPs) recorded during graph
    construction.  A and B are then checked for bow-free compliance: if both
    A[i,j]=1 and B[i,j]=1 arise (which correct construction prevents), B takes
    precedence and A is cleared.
    """
    A = np.zeros((p, p), dtype=int)
    B = np.zeros((p, p), dtype=int)

    # Directed edges: only between observed variables (both indices < p)
    for i in range(p):
        for j in range(p):
            if G_full[i, j] == 1:
                A[i, j] = 1

    # Bidirected edges from hidden common causes (UBPs)
    for (a, b) in cc_pairs:
        B[a, b] = 1;  B[b, a] = 1

    # Bidirected edges from hidden intermediates (UCPs)
    for (j, i) in int_pairs:
        B[i, j] = 1;  B[j, i] = 1

    # Enforce bow-free: clear any directed edge that is also a UBP/UCP target
    A[B == 1] = 0

    return A, B


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Main experiment generator
# ─────────────────────────────────────────────────────────────────────────────

def gen_lsnm_experiment(
    n:       int,
    seed:    int  = None,
    p:       int  = 10,
    n_cc:    int  = 2,
    n_int:   int  = 2,
    er_prob: float = 0.3,
):
    """
    Generate one simulation trial (Steps 1–4 of the data generation pipeline).

    Orchestrates graph construction, topological data generation, and output
    formatting for a single experiment.  All randomness is controlled through
    a single seed for full reproducibility.

    Parameters
    ----------
    n       : number of i.i.d. samples to generate
    seed    : integer random seed (None for unseeded / non-reproducible)
    p       : number of observed variables                     (default 10)
    n_cc    : number of hidden common causes (UBPs)            (default 2)
    n_int   : number of hidden intermediates (UCPs)            (default 2)
    er_prob : Erdős-Rényi edge probability for observed DAG    (default 0.3)

    Returns
    -------
    X_perm  : (n, p) ndarray  — observed data matrix, column-permuted
    A_true  : (p, p) ndarray  — true directed adjacency  (A[i,j]=1 ⟹ x_j→x_i)
    B_true  : (p, p) ndarray  — true bidirected matrix   (B[i,j]=1 ⟹ UBP/UCP)
    perm    : (p,)  ndarray   — column permutation applied to X and (A, B)
    """
    rng     = np.random.default_rng(seed)
    n_total = p + n_cc + n_int

    # ── Step 1: Build the full latent DAG ─────────────────────────────────────
    # Produces the observed ER-DAG skeleton plus hidden common causes and
    # hidden intermediates, with the bow-free property guaranteed by construction.
    G_full, cc_pairs, int_pairs = _build_full_graph(
        rng, p=p, n_cc=n_cc, n_int=n_int, er_prob=er_prob
    )

    # ── Step 2: Determine topological generation order ────────────────────────
    # networkx.topological_sort gives an order in which every variable is
    # generated after all its parents, for both observed and hidden variables.
    G_nx       = nx.from_numpy_array(G_full.T, create_using=nx.DiGraph)
    topo_order = list(nx.topological_sort(G_nx))

    # ── Step 2 (cont.): Generate data in topological order ───────────────────
    # For each variable v_i, parents are split into K_i (observed, index < p)
    # and Q_i (hidden, index >= p), and the two-level LSNM (Eq. 1) is applied:
    #     v_i = f^1(K_i) + g^1(K_i) · η_i,   η_i = f^2(Q_i) + g^2(Q_i) · ε_i
    # This ensures hidden parents only enter v_i through η_i, so that the
    # GAMLSS residual recovered by the algorithm exactly equals η_i.
    data = np.zeros((n, n_total))
    for v in topo_order:
        parent_indices  = np.where(G_full[v, :] == 1)[0]
        obs_parent_vals = [data[:, par] for par in parent_indices if par < p]
        hid_parent_vals = [data[:, par] for par in parent_indices if par >= p]
        data[:, v]      = _gen_lsnm_variable(obs_parent_vals, hid_parent_vals, rng, n)

    # ── Step 3: Extract observed variables and discard hidden ones ────────────
    # Only columns 0 … p-1 are returned; the hidden variable columns are
    # internal to the generation process and are not available to the algorithm.
    X_obs = data[:, :p]

    # ── Step 4: Random column permutation ─────────────────────────────────────
    # We randomly permute the observed variable columns so the algorithm cannot
    # exploit the fact that variable indices equal the topological order.
    # The ground-truth A and B matrices are permuted consistently.
    perm   = np.arange(p)
    rng.shuffle(perm)
    X_perm = X_obs[:, perm]

    A_raw, B_raw = compute_true_admg(G_full, p, cc_pairs, int_pairs)
    A_true = A_raw[np.ix_(perm, perm)]
    B_true = B_raw[np.ix_(perm, perm)]

    return X_perm, A_true, B_true, perm
