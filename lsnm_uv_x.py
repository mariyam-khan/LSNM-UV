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

    The only change relative to CAMUV is the override of ``_get_residual``.
    All independence tests, parent-finding logic, and UBP/UCP detection are
    inherited unchanged.
    """

    def _get_residual(self, X: np.ndarray, explained_i: int, explanatory_ids) -> np.ndarray:
        """
        LSNM residual  η_i = (x_i − f̂(K_i)) / ĝ(K_i).

        Two-step GAMLSS approximation:
          1. Fit location:  f̂  = LinearGAM(X_K).predict
          2. Fit scale:     ĝ  = exp( 0.5 · LinearGAM(X_K).predict( log(r²) ) )
             where  r = x_i − f̂(X_K)  is the location residual.

        Falls back to the plain additive residual if fitting fails (e.g. when
        the conditioning set is very small or near-degenerate).
        """
        explanatory_ids = list(explanatory_ids)

        if len(explanatory_ids) == 0:
            return X[:, explained_i]

        X_expl = X[:, explanatory_ids]
        xi     = X[:, explained_i]

        # ── Step 1: location ─────────────────────────────────────────────────
        # Fit a nonparametric regression f̂(K_i) ≈ E[x_i | K_i].
        # If fitting fails (e.g. constant feature, too few samples), fall back
        # to demeaning only — still better than returning the raw signal.
        try:
            gam_loc   = LinearGAM().fit(X_expl, xi)
            loc_pred  = gam_loc.predict(X_expl)
            loc_resid = xi - loc_pred
        except Exception:
            loc_resid = xi - xi.mean()

        # ── Step 2: scale via log(residual²) ─────────────────────────────────
        # Fit ĝ(K_i) ≈ E[|x_i − f̂| | K_i] by regressing log(r²) on K_i and
        # exponentiating (0.5 * prediction recovers log|r|, so exp gives |r̂|).
        # If scale fitting fails, fall back to the additive residual (= CAM-UV
        # behaviour), which is always well-defined because loc_resid already exists.
        try:
            log_sq    = np.log(loc_resid ** 2 + 1e-8)
            gam_scale = LinearGAM().fit(X_expl, log_sq)
            log_scale = gam_scale.predict(X_expl)
            scale     = np.clip(np.exp(0.5 * log_scale), 1e-6, None)
            return loc_resid / scale
        except Exception:
            return loc_resid    # scale failed: additive residual (CAM-UV fallback)


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
