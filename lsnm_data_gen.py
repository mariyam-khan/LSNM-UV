"""
LSNM Data Generation -- Section 6.1 simulation setup.
======================================================

Overview
--------
This file generates synthetic data for evaluating the LSNM-UV-X algorithm.
Each call to gen_lsnm_experiment() produces one simulation trial: a dataset of
observed variables, and the ground-truth ADMG that the algorithm is expected to
recover.  The graph structure follows Maeda & Shimizu (2021) Section 5.1, and
the data model extends it to the Location-Scale Noise Model of our paper (Eq. 1).


Step 1 -- Graph Construction (_build_full_graph)
-------------------------------------------------
We construct a full latent DAG over both observed and hidden variables.
Variable indices are laid out as:

    0 ... p-1          ->  observed variables  x_0, ..., x_{p-1}
    p ... p+n_cc-1     ->  hidden common causes  (root nodes, no parents)
    p+n_cc ... end     ->  hidden intermediates

With default parameters: p=10 observed variables, n_cc=2 hidden common causes,
n_int=2 hidden intermediates (14 nodes total).

Observed skeleton.
    Direct edges among the p observed variables are drawn from an Erdos-Renyi
    DAG with edge probability 0.3, following Maeda21 Section 5.1 exactly.
    Node index order is the topological order (edge x_j -> x_i only if j < i).

Hidden common causes (UBPs).
    For each hidden common cause u_k (a root node with no parents of its own):
      - We select a pair of observed variables (x_a, x_b) with no direct edge
        between them.
      - We add edges u_k -> x_a and u_k -> x_b.
      - In the projected ADMG this becomes a bidirected edge x_a <-> x_b,
        representing an Unobserved Backdoor Path (UBP).

Hidden intermediates (UCPs).
    For each hidden intermediate y_k:
      - We select an existing observed-to-observed edge x_j -> x_i.
      - We replace it with x_j -> y_k -> x_i (the direct edge is removed).
      - In the projected ADMG this becomes a bidirected edge x_j <-> x_i,
        representing an Unobserved Causal Path (UCP).

Bow-free guarantee.
    The construction ensures no observed variable simultaneously has a direct
    edge and a bidirected edge to the same other variable, satisfying the
    bow-free ADMG condition required by the paper (Definition 2.5).


Step 2 -- Data Generation (_gen_lsnm_variable)
-----------------------------------------------
All variables (observed and hidden) are generated in topological order so that
when we generate variable v_i, all its parents already have values.

For each variable v_i we implement the two-level LSNM from Equation (1):

    v_i  =  f_i^1(K_i)  +  g_i^1(K_i) * eta_i          (Layer 1)
    eta_i  =  eps_i  +  sum_{q in Q_i} f(q)              (Layer 2)

where K_i are the observed parents of v_i, Q_i are the hidden parents of v_i,
and eps_i ~ N(0,1) are mutually independent across all variables.

Layer 2 uses Pham2025's additive loop: eta_i starts from eps_i and each hidden
parent adds a random_nlfunc term.  This is identical to Pham2025's gen_data_matrix.

Layer 1 is the LSNM-specific extension: observed parents enter both the location
f_i^1(K_i) and the log-scale log(g_i^1(K_i)), so that g_i^1(K_i) is
heteroscedastic -- the key structural property that CAMUV is not designed for.

The critical structural property is the separation of roles: f_i^1 and g_i^1
depend only on K_i; hidden parents Q_i enter v_i exclusively through eta_i.
This is what makes Lemma 1 hold -- the GAMLSS residual
    eta_hat_i = (x_i - f_hat^1(K_i)) / g_hat^1(K_i)
recovers eta_i, whose independence structure across variables reveals the
UBP/UCP structure.

Nonlinear function form.
    random_nlfunc follows Maeda & Shimizu (2021) Eq. (8):

        (v_j + a)^c + b,   a ~ U(-5,5),  b ~ U(-1,1),  c in {2, 3}

    This function is used identically for:
      - Layer 2 (hidden parents -> eta): additive sum over hidden parents
      - Layer 1 location sum (observed parents -> loc): additive sum
      - Layer 1 scale sum (observed parents -> scale_log): LSNM addition only

    The scale_log sum is normalised by its std before exponentiation, then
    clipped to [0.1, 10] to prevent overflow from rare large quadratic values.
    Each variable is normalised to zero mean and unit variance after generation.

Why heteroscedasticity matters.
    In Maeda21's model, the noise is additive and homoscedastic -- the variance
    does not change with parent values.  In our model, the scale g_i^1(K_i)
    depends on parent values, so the noise variance of v_i changes as the
    observed parents change.  This is the key structural extension that
    LSNM-UV-X is designed to handle, and which CAM-UV is not designed for.


Step 3 -- Ground-Truth ADMG (compute_true_admg)
------------------------------------------------
After generating data for all variables, we project the full latent DAG down
to the p observed variables to obtain the ground-truth ADMG (A, B):

    A[i,j] = 1  if there is a direct edge x_j -> x_i among observed variables.
    B[i,j] = 1  if there is a UBP or UCP between x_i and x_j (i.e., they
                share a hidden common cause, or are connected via a hidden
                intermediate).

The bow-free constraint is enforced: if A[i,j]=1 and B[i,j]=1 arise
simultaneously (which should not occur given correct construction), B takes
precedence.


Step 4 -- Final Output (gen_lsnm_experiment)
---------------------------------------------
A random column permutation is applied to the observed data matrix before
returning it.  This ensures the algorithm cannot exploit the fact that variable
indices coincide with topological order.  The ground-truth A and B matrices are
permuted consistently so evaluation remains valid.

Returns:
    X_perm  --  (n, p) array of observed data, column-permuted
    A_true  --  (p, p) directed adjacency matrix  (A[i,j]=1 means x_j -> x_i)
    B_true  --  (p, p) symmetric bidirected matrix (B[i,j]=1 means UBP or UCP)
    perm    --  the permutation applied, for reference


Summary of Design Choices
--------------------------
    p = 10 observed variables           Matches Maeda21 Section 5.1
    n_cc = 2 hidden common causes       Matches Maeda21; each creates one UBP
    n_int = 2 hidden intermediates      Extended from Maeda21; each creates one UCP
    Edge probability 0.3 (ER)          Matches Maeda21 Section 5.1
    random_nlfunc: (x+a)^c+b            Maeda21 Eq. (8): a~U(-5,5), c in {2,3}
    Layer 2 (eta): additive loop        Sum of random_nlfunc over hidden parents
    Layer 1: loc + scale*eta            LSNM-specific; scale depends on K_i
    Scale clip [0.1, 10]                LSNM-specific; prevents exp() overflow
    Mean centering after generation     Prevents one-sided distributions from x^2 terms
    Bow-free construction               Required for identifiability (Theorem 2)

Requirements: numpy, networkx
"""

import numpy as np
import networkx as nx


# -----------------------------------------------------------------------------
# Nonlinear function -- Maeda & Shimizu (2021) Eq. (8)
#
# Paper Section 6.1: "All nonlinear functions follow the polynomial form
# of Maeda & Shimizu (2021) Eq. (8)."
#
# gen_noise replaced by eps_i = rng.standard_normal(n) inside _gen_lsnm_variable
# for us each noise eps_i is N(0,1) independently drawn for each node
# -----------------------------------------------------------------------------

def random_nlfunc(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random nonlinear function following Maeda & Shimizu (2021) Eq. (8).

    Computes  (x + a)^c + b  with:
        a ~ Uniform(-5, 5)   (horizontal shift)
        b ~ Uniform(-1, 1)   (vertical shift)
        c ~ {2, 3} with equal probability  (polynomial degree)

    This matches the CAMUV-data-gene.ipynb reference (causal_func) and
    the paper's Section 6.1 specification.  Compared to Pham2025's form
    (a in (-1,1), c=2 always, ±1 sign flip), this uses a wider shift
    range and includes odd-degree (c=3) terms that produce asymmetric
    functions.
    """
    a = rng.uniform(-5.0, 5.0)
    b = rng.uniform(-1.0, 1.0)
    c = rng.choice([2, 3])
    return (x + a) ** c + b


# -----------------------------------------------------------------------------
# Variable generator -- Layer 2 (Pham-style) + Layer 1 (LSNM extension)

# Replces gen_data_matrix in Pham, 
# -----------------------------------------------------------------------------

def _gen_lsnm_variable(
    obs_parent_vals: list,
    hid_parent_vals: list,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """
    Generate one variable v_i from the two-level LSNM (paper Eq. 1 / Eq. 2).

    Paper Eq. (1):  v_i = f_i^1(K_i) + g_i^1(K_i) * f_i^2(Q_i) + g_i^1(K_i) * g_i^2(Q_i) * eps_i
    Paper Eq. (2):  v_i = f_i^1(K_i) + g_i^1(K_i) * eta_i
    Paper Eq. (7):  eta_i = f_i^2(Q_i) + g_i^2(Q_i) * eps_i

    where:
        K_i  = observed direct causes of v_i     (obs_parent_vals)
        Q_i  = unobservable direct causes of v_i (hid_parent_vals)
        eps_i ~ N(0,1)                           external noise, mutually independent across all v_i

    Key structural property (Lemma 1 / Eq. 14):
        f_i^1 and g_i^1 depend ONLY on K_i.
        Q_i enters v_i exclusively through eta_i.
        => GAMLSS residual  eta_hat_i = (v_i - f_hat_i^1(K_i)) / g_hat_i^1(K_i)  recovers eta_i,
           and eta_i _|_ eta_j  iff  no UBP/UCP between v_i and v_j.
    """
    # -------------------------------------------------------------------------
    # Layer 2 -- paper Eq. (7):  eta_i = f_i^2(Q_i) + g_i^2(Q_i) * eps_i
    # -------------------------------------------------------------------------

    # eps_i ~ N(0,1)  -- paper Eq. (1)
    eps_i = rng.standard_normal(n)

    if len(hid_parent_vals) == 0:
        # Q_i = empty: eta_i = eps_i  (no hidden parents)
        eta_i = eps_i
    else:
        # f_i^2(Q_i) = sum_{u_k in Q_i} phi_k(u_k)  -- location from hidden parents
        f_i2 = np.zeros(n)
        for u_k in hid_parent_vals:
            f_i2 = f_i2 + random_nlfunc(u_k, rng)

        # g_i^2(Q_i) = clip(exp(log_g_i2 / std(log_g_i2)), 0.1, 10)  -- scale from hidden parents
        # log_g_i2 = sum_{u_k in Q_i} tilde_phi_k(u_k), independent parameters from f_i2
        log_g_i2 = np.zeros(n)
        for u_k in hid_parent_vals:
            log_g_i2 = log_g_i2 + random_nlfunc(u_k, rng)
        s = np.std(log_g_i2)
        if s > 1e-8:
            log_g_i2 = log_g_i2 / s
        g_i2 = np.clip(np.exp(log_g_i2), 0.1, 10.0)

        # eta_i = f_i^2(Q_i) + g_i^2(Q_i) * eps_i  -- paper Eq. (7)
        eta_i = f_i2 + g_i2 * eps_i

    # Normalise eta_i to mean~0, std~1 for numerical stability.
    # Bijective transform: preserves eta_i _|_ eta_j iff no UBP/UCP (Lemma 1).
    eta_i = (eta_i - np.mean(eta_i)) / (np.std(eta_i) + 1e-8)

    # -------------------------------------------------------------------------
    # Layer 1 -- paper Eq. (2):  v_i = f_i^1(K_i) + g_i^1(K_i) * eta_i
    # -------------------------------------------------------------------------

    if len(obs_parent_vals) == 0:
        # K_i = empty: f_i^1 = 0, g_i^1 = 1  =>  v_i = eta_i
        v_i = eta_i
    else:
        # f_i^1(K_i) = sum_{x_k in K_i} phi_k(x_k)  -- location from observed parents
        f_i1 = np.zeros(n)
        for x_k in obs_parent_vals:
            f_i1 = f_i1 + random_nlfunc(x_k, rng)

        # g_i^1(K_i) = clip(exp(log_g_i1 / std(log_g_i1)), 0.1, 10)  -- scale from observed parents
        # log_g_i1 = sum_{x_k in K_i} tilde_phi_k(x_k), independent parameters from f_i1.
        # Dividing by std keeps the exponent moderate; clip to [0.1, 10] prevents exp() overflow
        # while retaining a 100x variance range (substantial heteroscedasticity).
        log_g_i1 = np.zeros(n)
        for x_k in obs_parent_vals:
            log_g_i1 = log_g_i1 + random_nlfunc(x_k, rng)
        s = np.std(log_g_i1)
        if s > 1e-8:
            log_g_i1 = log_g_i1 / s
        g_i1 = np.clip(np.exp(log_g_i1), 0.1, 10.0)

        # v_i = f_i^1(K_i) + g_i^1(K_i) * eta_i  -- paper Eq. (2)
        v_i = f_i1 + g_i1 * eta_i

    # Normalise v_i to zero mean and unit variance.
    # The constants (mean, std) are absorbed into f_i^1 and g_i^1 by the GAMLSS fit,
    # so this does not affect the identifiability argument (paper Eq. 14).
    # Mean centering is essential: (x+c)^2 >= 0 always, so without it the location
    # sum accumulates a positive bias that propagates through the DAG.
    return (v_i - np.mean(v_i)) / (np.std(v_i) + 1e-8)


# -----------------------------------------------------------------------------
# Step 1 -- Graph construction
# -----------------------------------------------------------------------------

def gen_er_dag(p: int, er_prob: float, rng: np.random.Generator) -> np.ndarray:
    """
    Erdos-Renyi DAG over p observed variables, following Maeda21 Section 5.1.

    Each ordered pair (j, i) with j < i is connected by an edge x_j -> x_i
    independently with probability er_prob.  Because j < i always, the index
    order is a valid topological order and the matrix is lower-triangular.

        G[i, j] = 1  =>  x_j -> x_i  (j is a parent of x_i)
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
        0 ... p-1          ->  observed variables  x_0, ..., x_{p-1}
        p ... p+n_cc-1     ->  hidden common causes  (root nodes, no parents)
        p+n_cc ... end     ->  hidden intermediates

    Hidden common causes (UBPs).
        Each hidden common cause u_k is a root node.  We select a pair of
        observed variables (x_a, x_b) that have no direct edge between them,
        and add edges u_k -> x_a and u_k -> x_b.  When the hidden variable is
        marginalised out, this induces a bidirected edge x_a <-> x_b in the
        ADMG, representing an Unobserved Backdoor Path (UBP).

    Hidden intermediates (UCPs).
        Each hidden intermediate y_k replaces one existing observed edge.  We
        select an edge x_j -> x_i, remove it, and insert x_j -> y_k -> x_i.
        When y_k is marginalised out, this induces a bidirected edge x_j <-> x_i,
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

    # -- Hidden common causes: each is a root node with two observed children --
    used_bidir_pairs: set = set()
    for k in range(n_cc):
        hc = p + k
        # Only connect to pairs with no existing direct edge (bow-free condition)
        # and not already targeted by a previous hidden common cause (no duplicates).
        candidates = [
            (i, j) for i in range(p) for j in range(i + 1, p)
            if G_full[i, j] == 0 and G_full[j, i] == 0
            and (i, j) not in used_bidir_pairs
        ]
        if not candidates:
            continue   # skip this hidden cause; adding one to a directed-edge pair
                       # would violate bow-freeness and corrupt A_true
        idx  = int(rng.integers(0, len(candidates)))
        a, b = candidates[idx]
        G_full[a, hc] = 1   # u_k -> x_a
        G_full[b, hc] = 1   # u_k -> x_b  ->  projects to UBP: x_a <-> x_b
        cc_pairs.append((a, b))
        used_bidir_pairs.add((a, b))

    # -- Hidden intermediates: each replaces one existing observed edge ---------
    for k in range(n_int):
        hi = p + n_cc + k
        edges = [(i, j) for i in range(p) for j in range(p) if G_full[i, j] == 1]
        if not edges:
            break
        idx  = int(rng.integers(0, len(edges)))
        i, j = edges[idx]          # existing edge: x_j -> x_i
        G_full[i, j]  = 0          # remove direct edge
        G_full[hi, j] = 1          # x_j  -> y_k
        G_full[i, hi] = 1          # y_k  -> x_i  ->  projects to UCP: x_j <-> x_i
        int_pairs.append((j, i))   # record (obs_parent, obs_child)

    return G_full, cc_pairs, int_pairs


def compute_true_admg(G_full: np.ndarray, p: int, cc_pairs: list, int_pairs: list):
    """
    Project the full latent DAG onto the ground-truth ADMG (Step 3).

    After generating all data, we discard the hidden variables and record
    what their presence implies for the observed variables:

        A[i, j] = 1  iff  x_j -> x_i is a direct edge among observed variables.
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

    # Verify bow-free by construction (should never fail given correct _build_full_graph)
    overlap = int(np.sum((A == 1) & (B == 1)))
    assert overlap == 0, (
        f"Bow-free violation: {overlap} pairs have both directed and bidirected edges. "
        f"This indicates a bug in _build_full_graph."
    )

    return A, B


# -----------------------------------------------------------------------------
# Step 4 -- Main experiment generator
# -----------------------------------------------------------------------------

def gen_lsnm_experiment(
    n:       int,
    seed:    int  = None,
    p:       int  = 10,
    n_cc:    int  = 2,
    n_int:   int  = 2,
    er_prob: float = 0.3,
):
    """
    Generate one simulation trial (Steps 1-4 of the data generation pipeline).

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
    er_prob : Erdos-Renyi edge probability for observed DAG    (default 0.3)

    Returns
    -------
    X_perm  : (n, p) ndarray  -- observed data matrix, column-permuted
    A_true  : (p, p) ndarray  -- true directed adjacency  (A[i,j]=1 => x_j->x_i)
    B_true  : (p, p) ndarray  -- true bidirected matrix   (B[i,j]=1 => UBP/UCP)
    perm    : (p,)  ndarray   -- column permutation applied to X and (A, B)
    """
    rng     = np.random.default_rng(seed)
    n_total = p + n_cc + n_int

    # -- Step 1: Build the full latent DAG -------------------------------------
    # Produces the observed ER-DAG skeleton plus hidden common causes and
    # hidden intermediates, with the bow-free property guaranteed by construction.
    
    # Parents split into obs_parent_vals ($K_i$) and hid_parent_vals ($Q_i$) — new, needed for two-level LSNM
    G_full, cc_pairs, int_pairs = _build_full_graph(
        rng, p=p, n_cc=n_cc, n_int=n_int, er_prob=er_prob
    )

    # -- Step 2: Determine topological generation order ------------------------
    # networkx.topological_sort gives an order in which every variable is
    # generated after all its parents, for both observed and hidden variables.
    G_nx       = nx.from_numpy_array(G_full.T, create_using=nx.DiGraph)
    topo_order = list(nx.topological_sort(G_nx))

    # -- Step 2 (cont.): Generate data in topological order -------------------
    # For each variable v_i, parents are split into K_i (observed, index < p)
    # and Q_i (hidden, index >= p), and the two-level LSNM is applied:
    #     eta_i = eps_i + sum_{q in Q_i} random_nlfunc(q)   [Pham2025 style]
    #     v_i   = loc(K_i) + scale(K_i) * eta_i             [LSNM extension]
    # This ensures hidden parents only enter v_i through eta_i, so that the
    # GAMLSS residual recovered by the algorithm exactly equals eta_i.
    data = np.zeros((n, n_total))
    for v in topo_order:
        parent_indices  = np.where(G_full[v, :] == 1)[0]
        obs_parent_vals = [data[:, par] for par in parent_indices if par < p]
        hid_parent_vals = [data[:, par] for par in parent_indices if par >= p]
        data[:, v]      = _gen_lsnm_variable(obs_parent_vals, hid_parent_vals, rng, n)

    # -- Step 3: Extract observed variables and discard hidden ones ------------
    # Only columns 0 ... p-1 are returned; the hidden variable columns are
    # internal to the generation process and are not available to the algorithm.
    X_obs = data[:, :p]

    # -- Step 4: Random column permutation -------------------------------------
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
