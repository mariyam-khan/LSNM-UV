"""
LSNM-UV-X Algorithm.

Two-stage algorithm for causal discovery in Location-Scale Noise Models with
hidden variables (bow-free ADMGs).

  Stage 1 — LSNMUV (LSNM-UV-Base):
      CAM-UV (Maeda & Shimizu, 2021, Algorithms 1 & 2) with the additive
      residual  x_i − Ĝ(K_i)  replaced by the LSNM residual

          η_i = (x_i − f̂(K_i)) / ĝ(K_i)                             [Eq. (8)]

      f̂ and ĝ are estimated by a two-step GAMLSS procedure:
        • f̂  — LinearGAM fit for the location (conditional mean)
        • ĝ  — exp( LinearGAM fit on log|residual| ) for the scale

  Stage 2 — checkVISIBLE:
      Re-examines invisible pairs (NaN entries) by searching over regression
      sets; follows Pham et al. (2025) / cam-uv-x_extended.py.

Requirements: lingam, pygam, numpy
"""

import itertools
import numpy as np
from lingam import CAMUV
from pygam import LinearGAM


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: LSNM-UV-Base
# ─────────────────────────────────────────────────────────────────────────────

class LSNMUV(CAMUV):
    """
    LSNM-UV-Base: CAM-UV with location-scale residuals.

    Two changes relative to CAMUV:

    1. ``_get_residual`` is overridden to use the two-step GAMLSS LSNM residual
       eta_hat_i = (x_i - f_hat_i^1(K_i)) / g_hat_i^1(K_i)  [paper Eq. 14].

    2. ``fit`` is overridden to replace CAMUV's neighbourhood-gated UBP/UCP
       detection with a full pairwise residual check.

       CAMUV's original Algorithm 2 only tests pairs (i,j) that pass two gates:
         (a) no directed edge found between them  (i not in P[j] and j not in P[i])
         (b) they appear correlated in raw X       (i in N[j] and j in N[i])

       Gate (b) uses raw X values (not residuals) and fails on LSNM data:
       the hidden-cause signal is diluted by two layers of normalisation, so
       true UBP/UCP pairs often appear uncorrelated in raw X and never enter N.
       Removing gate (b) and testing ALL non-parent pairs with LSNM residuals
       directly recovers pairs that the N-gate would otherwise discard.

       Cost: O(d^2) HSIC tests instead of O(|N|) — same asymptotic complexity.
       False-positive rate: ~alpha * C(d,2) per graph (e.g., 0.01 * 45 ≈ 0.45
       extra invisible pairs expected by chance for d=10, alpha=0.01).
    """

    def fit(self, X: np.ndarray):
        """
        Fit LSNM-UV-Base to X.

        Replaces CAMUV's neighbourhood-gated UBP/UCP detection (Algorithm 2)
        with a full pairwise LSNM-residual check over all pairs not already
        assigned a directed edge.
        """
        from sklearn.utils.validation import check_array
        X = check_array(X)
        n, d = X.shape

        # ── Algorithm 1: find directed edges (parent sets P) ──────────────────
        # Inherited from CAMUV; uses LSNM _get_residual via _find_parents.
        N = self._get_neighborhoods(X)
        P = self._find_parents(X, self._num_explanatory_vals, N)

        # ── Algorithm 2: UBP/UCP detection (full pairwise, no N gate) ─────────
        # For each pair with no directed edge, test independence of LSNM
        # residuals.  Dependence => invisible pair (UBP or UCP).
        U = []
        for i in range(d):
            for j in range(i + 1, d):
                # Skip pairs where a directed edge was already identified
                if (i in P[j]) or (j in P[i]):
                    continue
                r_i = self._get_residual(X, i, P[i]).reshape(n, 1)
                r_j = self._get_residual(X, j, P[j]).reshape(n, 1)
                if not self._is_independent(r_i, r_j):
                    U.append(set([i, j]))

        self._U = U
        self._P = P
        return self._estimate_adjacency_matrix(X, P, U)

    def _get_residual(self, X: np.ndarray, explained_i: int, explanatory_ids) -> np.ndarray:
        """
        Compute the LSNM residual  eta_hat_i = (x_i - f_hat_i^1(K_i)) / g_hat_i^1(K_i).

        This implements paper Eq. (14) via a two-step procedure.

        The LSNM model for x_i is (paper Eq. 2):
            x_i = f_i^1(K_i) + g_i^1(K_i) * eta_i

        where K_i = explanatory_ids are the observed parents of x_i.
        Both f_i^1 (location) and g_i^1 (scale) are unknown nonlinear functions
        of K_i that must be estimated from data before eta_i can be recovered.

        A LinearGAM fits an additive spline model:
            y = beta_0 + sum_k s_k(x_k) + eps
        where each s_k is a smooth spline over one predictor.  LinearGAM().fit(X_K, y)
        learns the spline coefficients; .predict(X_K) returns the fitted values
            y_hat = beta_0_hat + sum_k s_hat_k(x_k)
        which approximates E[y | K_i] nonparametrically.

        Step 1 uses LinearGAM to estimate f_i^1(K_i) = E[x_i | K_i].
        Step 2 uses LinearGAM again on log(r_i^2) to estimate 2*log(g_i^1(K_i)).

        Falls back to the plain additive residual (= CAM-UV behaviour) if either
        GAM fit fails (e.g. constant feature, too few samples).
        """
        explanatory_ids = list(explanatory_ids)

        # K_i = empty: no observed parents to regress out.
        # eta_hat_i = x_i  (paper Eq. 14 with f_i^1 = 0, g_i^1 = 1)
        if len(explanatory_ids) == 0:
            return X[:, explained_i]

        X_expl = X[:, explanatory_ids]   # shape (n, |K_i|) -- values of observed parents K_i
        xi     = X[:, explained_i]       # shape (n,)       -- values of x_i

        # ---------------------------------------------------------------------
        # Step 1 -- estimate location f_i^1(K_i) and compute location residual r_i
        #
        # Model:  x_i = f_i^1(K_i) + g_i^1(K_i) * eta_i
        #
        # Fit LinearGAM:
        #   x_i = beta_0 + sum_{k in K_i} s_k(x_k) + eps
        #
        # Prediction:
        #   f_hat_i^1(K_i) = beta_0_hat + sum_k s_hat_k(x_k)
        #                  = E_hat[x_i | K_i]
        #
        # Location residual:
        #   r_i = x_i - f_hat_i^1(K_i)
        #       ~= g_i^1(K_i) * eta_i          (scale g_i^1 still present)
        #
        # This is the same step as Pham's _get_residual.  For homoscedastic
        # models (g_i^1 = const) r_i already equals eta_i up to a constant.
        # For LSNM, the scale g_i^1(K_i) must still be removed (Step 2).
        # ---------------------------------------------------------------------
        try:
            gam_loc   = LinearGAM().fit(X_expl, xi)
            loc_pred  = gam_loc.predict(X_expl)    # f_hat_i^1(K_i)
            loc_resid = xi - loc_pred               # r_i = x_i - f_hat_i^1(K_i)
        except Exception:
            # Fallback: demean only (no spline fit)
            loc_resid = xi - xi.mean()

        # ---------------------------------------------------------------------
        # Step 2 -- estimate scale g_i^1(K_i) from log(r_i^2) and divide
        #
        # From Step 1:  r_i ~= g_i^1(K_i) * eta_i
        #
        # Squaring and taking log:
        #   log(r_i^2) = 2 * log(g_i^1(K_i)) + log(eta_i^2)
        #                \_____________________/  \__________/
        #                  depends on K_i          noise term, indep of K_i
        #
        # Therefore:
        #   E[ log(r_i^2) | K_i ] = 2 * log(g_i^1(K_i))
        #
        # Fit LinearGAM on log(r_i^2) to estimate this conditional mean:
        #   log(r_i^2) = beta_0 + sum_{k in K_i} s_k(x_k) + eps
        #
        # Prediction:
        #   log_scale_hat = E_hat[ log(r_i^2) | K_i ]
        #                 ~= 2 * log(g_hat_i^1(K_i))
        #
        # Recover g_hat_i^1(K_i):
        #   g_hat_i^1(K_i) = exp( 0.5 * log_scale_hat )
        #
        # Clip to [1e-6, inf) to avoid division by near-zero scale.
        #
        # Final LSNM residual (paper Eq. 14):
        #   eta_hat_i = r_i / g_hat_i^1(K_i)
        #             ~= g_i^1(K_i) * eta_i / g_hat_i^1(K_i)
        #             ~= eta_i
        # ---------------------------------------------------------------------
        try:
            log_sq    = np.log(loc_resid ** 2 + 1e-8)   # log(r_i^2), +1e-8 avoids log(0)
            gam_scale = LinearGAM().fit(X_expl, log_sq)
            log_scale = gam_scale.predict(X_expl)        # E_hat[log(r_i^2) | K_i]
            scale     = np.clip(np.exp(0.5 * log_scale), 1e-6, None)  # g_hat_i^1(K_i)
            return loc_resid / scale                     # eta_hat_i  -- paper Eq. (14)
        except Exception:
            return loc_resid    # scale fit failed: return r_i (= CAM-UV / Pham fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: checkVISIBLE
# ─────────────────────────────────────────────────────────────────────────────

class LSNMUV_X(LSNMUV):
    """
    LSNM-UV-X: LSNM-UV-Base + checkVISIBLE.

    Algorithm 1 of the LSNM-UV paper:
        A ← LSNM-UV-Base(X, d, α)           [Stage 1, via super().fit()]
        for each invisible pair (i,j):
            CHECKVISIBLE(i, j)               [Stage 2, this class]
        return A

    Parameters
    ----------
    alpha               : HSIC significance level (default 0.01)
    num_explanatory_vals: max |K| in parent search  (default 3, i.e. d=3)
    max_regress_size    : max regression-set size in checkVISIBLE (default 2)
    """

    def __init__(
        self,
        alpha:                float = 0.01,
        num_explanatory_vals: int   = 3,
        max_regress_size:     int   = 2,
    ):
        # Note: independence method is fixed to HSIC (CAMUV default).
        # ind_corr and the 'fcorr' option from CAMUV are not exposed here
        # because the paper (Assumption 3) is stated specifically for HSIC.
        super().__init__(alpha=alpha, num_explanatory_vals=num_explanatory_vals)
        self._max_regress_size = max_regress_size

    # ── Public fit ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray):
        """
        Fit LSNM-UV-X to data matrix X  (n_samples × p_features).

        Returns self; adjacency matrix accessible via .adjacency_matrix_.
        Convention: A[i,j] = 1 ⟹ x_j → x_i;  A[i,j] = NaN ⟹ invisible pair.
        """
        # Stage 1: LSNM-UV-Base (runs Maeda21 Alg 1 + Alg 2 with LSNM residuals)
        super().fit(X)
        mat = self._adjacency_matrix.copy()

        # Stage 2: checkVISIBLE
        mat = self._check_visible(X, mat)
        self._adjacency_matrix = mat
        return self

    # ── checkVISIBLE ──────────────────────────────────────────────────────────

    def _check_visible(self, X: np.ndarray, mat: np.ndarray) -> np.ndarray:
        """
        Re-examine invisible pairs (NaN entries) using broader regression sets.

        For each NaN pair (x_i, x_j) with i < j, search over pairs of regression
        sets (S_1, S_2) drawn from:
            Q = {parents of x_i} ∪ {parents of x_j}
              ∪ {other NaN-neighbours of x_i} ∪ {other NaN-neighbours of x_j}

        Tests  (using LSNM residuals h = _get_residual):

        (a) h(i, S_1) ⊥ h(j, S_2)
                → visible non-edge  (A[i,j] = A[j,i] = 0)

        (b) h(i, S_1 ∪ {x_j}) ⊥ h(j, S_2)   (x_j included in x_i's regression)
                → x_i is NOT a parent of x_j  (iNotParent)

        (c) h(i, S_1) ⊥ h(j, S_2 ∪ {x_i})   (x_i included in x_j's regression)
                → x_j is NOT a parent of x_i  (jNotParent)

        Resolution:
            iNotParent only   →  edge x_j → x_i  (A[i,j]=1, A[j,i]=0)
            jNotParent only   →  edge x_i → x_j  (A[j,i]=1, A[i,j]=0)
            both iNotParent and jNotParent simultaneously
                              →  visible non-edge
            neither           →  pair remains NaN (invisible)

        Follows Pham et al. (2025), Algorithm 3.
        """
        n     = X.shape[0]
        p     = mat.shape[0]
        mat_new = mat.copy()

        # Only process upper-triangle pairs to avoid processing each pair twice
        nan_pairs = [
            (i, j) for i in range(p) for j in range(i + 1, p)
            if np.isnan(mat[i, j]) and np.isnan(mat[j, i])
        ]

        for (x_i, x_j) in nan_pairs:
            # ── Build candidate regression set Q ──────────────────────────────
            nan_xi = set(np.where(np.isnan(mat_new[x_i, :]))[0]) - {x_i, x_j}
            nan_xj = set(np.where(np.isnan(mat_new[x_j, :]))[0]) - {x_i, x_j}
            P_i    = set(np.where(mat_new[x_i, :] == 1)[0])
            P_j    = set(np.where(mat_new[x_j, :] == 1)[0])
            Q      = nan_xi | nan_xj | P_i | P_j

            if len(Q) == 0:
                continue   # no candidates, leave as NaN

            iNotParent = False
            jNotParent = False
            isNonEdge  = False

            max_sz = min(self._max_regress_size, len(Q))

            outer_break = False
            for sz_i in range(1, max_sz + 1):
                if outer_break:
                    break
                for s1 in itertools.combinations(Q, sz_i):
                    if outer_break:
                        break
                    for sz_j in range(1, max_sz + 1):
                        if outer_break:
                            break
                        for s2 in itertools.combinations(Q, sz_j):
                            expl_i = set(s1)
                            expl_j = set(s2)

                            r_i = self._get_residual(X, x_i, expl_i).reshape(n, 1)
                            r_j = self._get_residual(X, x_j, expl_j).reshape(n, 1)

                            # (a) visible non-edge
                            if self._is_independent(r_i, r_j):
                                mat_new[x_i, x_j] = 0
                                mat_new[x_j, x_i] = 0
                                isNonEdge  = True
                                outer_break = True
                                break

                            # (b) x_i not parent of x_j
                            r_i2 = self._get_residual(
                                X, x_i, expl_i | {x_j}
                            ).reshape(n, 1)
                            if self._is_independent(r_i2, r_j):
                                iNotParent = True

                            # (c) x_j not parent of x_i
                            r_j2 = self._get_residual(
                                X, x_j, expl_j | {x_i}
                            ).reshape(n, 1)
                            if self._is_independent(r_i, r_j2):
                                jNotParent = True

                            # Both not-parent: visible non-edge
                            if iNotParent and jNotParent:
                                mat_new[x_i, x_j] = 0
                                mat_new[x_j, x_i] = 0
                                isNonEdge  = True
                                outer_break = True
                                break

            # ── Resolve direction if possible ─────────────────────────────────
            if not isNonEdge:
                if iNotParent and not jNotParent:
                    # x_j → x_i
                    mat_new[x_i, x_j] = 1
                    mat_new[x_j, x_i] = 0
                elif jNotParent and not iNotParent:
                    # x_i → x_j
                    mat_new[x_j, x_i] = 1
                    mat_new[x_i, x_j] = 0
                # if neither: leave as NaN

        return mat_new
