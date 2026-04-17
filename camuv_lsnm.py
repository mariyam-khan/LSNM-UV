"""
CAM-UV with LSNM residuals (LSNM-UV-Base).

This is Maeda & Shimizu (2021) CAM-UV exactly as published, with a single
change: _get_residual computes the two-step GAMLSS LSNM residual

    eta_hat_i = (x_i - f_hat(K_i)) / g_hat(K_i)          [paper Eq. 14]

instead of the additive residual  x_i - GAM(K_i).

Everything else -- fit, Algorithm 1 (parent search), Algorithm 2
(neighbourhood construction + UBP/UCP detection), independence tests,
prior knowledge handling -- is identical to camuv-original.py.
"""

import copy
import itertools

import numpy as np
from pygam import LinearGAM
from sklearn.utils import check_array

from lingam.hsic import hsic_test_gamma
from lingam.utils import f_correlation


class CAMUV_LSNM:
    """CAM-UV with LSNM residuals.  See camuv-original.py for the original."""

    def __init__(
        self,
        alpha=0.01,
        num_explanatory_vals=2,
        independence="hsic",
        ind_corr=0.5,
        prior_knowledge=None,
    ):
        # Check parameters
        if num_explanatory_vals <= 0:
            raise ValueError("num_explanatory_vals must be > 0.")

        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        if independence not in ("hsic", "fcorr"):
            raise ValueError("independence must be 'hsic' or 'fcorr'.")

        if ind_corr < 0.0:
            raise ValueError("ind_corr must be an float greater than 0.")

        self._num_explanatory_vals = num_explanatory_vals
        self._alpha = alpha
        self._independence = independence
        self._ind_corr = ind_corr
        self._pk_dict = self._make_pk_dict(prior_knowledge)

    def fit(self, X):
        """Fit the model to X.  Identical to CAMUV.fit."""
        X = check_array(X)

        n = X.shape[0]
        d = X.shape[1]
        N = self._get_neighborhoods(X)
        P = self._find_parents(X, self._num_explanatory_vals, N)

        U = []

        for i in range(d):
            for j in range(d)[i + 1 :]:
                if (i in P[j]) or (j in P[i]):
                    continue
                if (i not in N[j]) or (j not in N[i]):
                    continue

                i_residual = self._get_residual(X, i, P[i])
                j_residual = self._get_residual(X, j, P[j])
                in_X = np.reshape(i_residual, [n, 1])
                in_Y = np.reshape(j_residual, [n, 1])
                if not self._is_independent(in_X, in_Y):
                    if not set([i, j]) in U:
                        U.append(set([i, j]))

        self._U = U
        self._P = P

        return self._estimate_adjacency_matrix(X, P, U)

    def _make_pk_dict(self, prior_knowledge):
        if prior_knowledge is None:
            return None

        pk_dict = dict()
        for pair in prior_knowledge:
            if not pair[1] in pk_dict:
                pk_dict[pair[1]] = [pair[0]]
            else:
                pk_dict[pair[1]].append(pair[0])
        return pk_dict

    # ─────────────────────────────────────────────────────────────────────────
    # ONLY CHANGE vs camuv-original.py: LSNM two-step GAMLSS residual
    # ─────────────────────────────────────────────────────────────────────────

    def _get_residual(self, X, explained_i, explanatory_ids):
        """
        LSNM residual: eta_hat_i = (x_i - f_hat(K_i)) / g_hat(K_i).

        Step 1: LinearGAM estimates location f_hat(K_i) = E[x_i | K_i].
                Location residual: r_i = x_i - f_hat(K_i).

        Step 2: LinearGAM on log(r_i^2) estimates 2*log(g_hat(K_i)).
                Scale: g_hat(K_i) = exp(0.5 * predicted log(r_i^2)).
                LSNM residual: eta_hat_i = r_i / g_hat(K_i).

        Falls back to additive residual (= original CAM-UV) if GAM fit fails.
        """
        explanatory_ids = list(explanatory_ids)

        if len(explanatory_ids) == 0:
            return X[:, explained_i]

        X_expl = X[:, explanatory_ids]
        xi     = X[:, explained_i]

        # Step 1 -- location
        try:
            gam_loc   = LinearGAM().fit(X_expl, xi)
            loc_pred  = gam_loc.predict(X_expl)
            loc_resid = xi - loc_pred
        except Exception:
            loc_resid = xi - xi.mean()

        # Step 2 -- scale
        try:
            log_sq    = np.log(loc_resid ** 2 + 1e-8)
            gam_scale = LinearGAM().fit(X_expl, log_sq)
            log_scale = gam_scale.predict(X_expl)
            scale     = np.clip(np.exp(0.5 * log_scale), 1e-6, None)
            return loc_resid / scale
        except Exception:
            return loc_resid

    # ─────────────────────────────────────────────────────────────────────────
    # Everything below is identical to camuv-original.py
    # ─────────────────────────────────────────────────────────────────────────

    def _is_independent(self, X, Y):
        if self._independence == "hsic":
            threshold = self._alpha
        elif self._independence == "fcorr":
            threshold = self._ind_corr
        is_independent, _ = self._is_independent_by(X, Y, threshold)
        return is_independent

    def _is_independent_by(self, X, Y, threshold):
        is_independent = False
        if self._independence == "hsic":
            _, value = hsic_test_gamma(X, Y)
            is_independent = value > threshold
        elif self._independence == "fcorr":
            value = f_correlation(X, Y)
            is_independent = value < threshold
        return is_independent, value

    def _get_neighborhoods(self, X):
        n = X.shape[0]
        d = X.shape[1]
        N = [set() for i in range(d)]
        for i in range(d):
            for j in range(d)[i + 1 :]:
                in_X = np.reshape(X[:, i], [n, 1])
                in_Y = np.reshape(X[:, j], [n, 1])
                if not self._is_independent(in_X, in_Y):
                    N[i].add(j)
                    N[j].add(i)
        return N

    def _find_parents(self, X, maxnum_vals, N):
        n = X.shape[0]
        d = X.shape[1]
        P = [set() for i in range(d)]  # Parents
        t = 2
        Y = copy.deepcopy(X)

        while True:
            changed = False
            variables_set_list = list(itertools.combinations(set(range(d)), t))
            for variables_set in variables_set_list:
                variables_set = set(variables_set)

                if not self._check_identified_causality(variables_set, P):
                    continue

                child, is_independence_with_K = self._get_child(
                    X, variables_set, P, N, Y
                )
                if child is None:
                    continue
                if not is_independence_with_K:
                    continue

                parents = variables_set - {child}
                if not self._check_independence_withou_K(parents, child, P, N, Y):
                    continue

                for parent in parents:
                    P[child].add(parent)
                    changed = True
                    Y = self._get_residuals_matrix(X, Y, P, child)

            if changed:
                t = 2
            else:
                t += 1
                if t > maxnum_vals:
                    break

        for i in range(d):
            non_parents = set()
            for j in P[i]:
                residual_i = self._get_residual(X, i, P[i] - {j})
                residual_j = self._get_residual(X, j, P[j])
                in_X = np.reshape(residual_i, [n, 1])
                in_Y = np.reshape(residual_j, [n, 1])
                if self._is_independent(in_X, in_Y):
                    non_parents.add(j)
            P[i] = P[i] - non_parents

        return P

    def _check_prior_knowledge(self, xj_list, xi):
        if self._pk_dict is not None:
            for xj in xj_list:
                if (xi in self._pk_dict) and (xj in self._pk_dict[xi]):
                    return True
        return False

    def _get_residuals_matrix(self, X, Y_old, P, child):
        Y = copy.deepcopy(Y_old)
        Y[:, child] = self._get_residual(X, child, P[child])
        return Y

    def _get_child(self, X, variables_set, P, N, Y):
        n = X.shape[0]

        prev_independence = 0.0 if self._independence == "hsic" else 1.0
        max_independence_child = None

        for child in variables_set:
            parents = variables_set - {child}

            if self._check_prior_knowledge(parents, child):
                continue

            if not self._check_correlation(child, parents, N):
                continue

            residual = self._get_residual(X, child, parents | P[child])
            in_X = np.reshape(residual, [n, 1])
            in_Y = np.reshape(Y[:, list(parents)], [n, len(parents)])
            is_ind, value = self._is_independent_by(in_X, in_Y, prev_independence)
            if is_ind:
                prev_independence = value
                max_independence_child = child

        if self._independence == "hsic":
            is_independent = prev_independence > self._alpha
        elif self._independence == "fcorr":
            is_independent = prev_independence < self._ind_corr

        return max_independence_child, is_independent

    def _check_independence_withou_K(self, parents, child, P, N, Y):
        n = Y.shape[0]
        for parent in parents:
            in_X = np.reshape(Y[:, child], [n, 1])
            in_Y = np.reshape(Y[:, parent], [n, 1])
            if self._is_independent(in_X, in_Y):
                return False
        return True

    def _check_identified_causality(self, variables_set, P):
        variables_list = list(variables_set)
        for i in variables_list:
            for j in variables_list[variables_list.index(i) + 1 :]:
                if (j in P[i]) or (i in P[j]):
                    return False
        return True

    def _check_correlation(self, child, parents, N):
        for parent in parents:
            if parent not in N[child]:
                return False
        return True

    def _estimate_adjacency_matrix(self, X, P, U):
        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i, parents in enumerate(P):
            for parent in parents:
                B[i, parent] = 1
        for confounded_pair in U:
            confounded_pair = list(confounded_pair)
            B[confounded_pair[0], confounded_pair[1]] = np.nan
            B[confounded_pair[1], confounded_pair[0]] = np.nan
        self._adjacency_matrix = B
        return self

    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix
