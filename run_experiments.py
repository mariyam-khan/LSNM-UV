"""
Experiment runner — LSNM-UV-X Section 6.1 simulations.

Reproduces the evaluation framework of Maeda & Shimizu (2021) Section 5.1
using LSNM data and comparing:
    LSNM-UV-X  (this paper)
    CAM-UV     (Maeda & Shimizu 2021,  lingam package)
    FCI        (Spirtes et al. 1999,   causal-learn package)
    BANG       (Wang et al.,           R package ngBap — optional)

Usage
-----
From terminal:
    python run_experiments.py

From JupyterHub notebook:
    from run_experiments import run_all_experiments, plot_results
    df = run_all_experiments(n_list=[100,300,500], n_trials=10)
    plot_results(df)

Requirements: numpy, pandas, joblib, lingam, pygam, causal-learn
Optional:     rpy2  +  R package ngBap  (for BANG)
"""

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lsnm_data_gen import gen_lsnm_experiment
from lsnm_uv_x    import LSNMUV_X
from eval_metrics import (
    directed_metrics, bidirected_metrics,
    parse_camuv_result, parse_fci_result, parse_bang_result,
)


# ─────────────────────────────────────────────────────────────────────────────
# Method wrappers
# ─────────────────────────────────────────────────────────────────────────────

def run_lsnm_uv_x(X: np.ndarray, alpha: float = 0.01, d: int = 3,
                  max_regress_size: int = 2):
    """Run LSNM-UV-X and return (A_est, B_est)."""
    model = LSNMUV_X(alpha=alpha, num_explanatory_vals=d,
                     max_regress_size=max_regress_size)
    model.fit(X)
    return parse_camuv_result(model)


def run_camuv(X: np.ndarray, alpha: float = 0.01, d: int = 3):
    """Run CAM-UV (lingam) and return (A_est, B_est)."""
    from lingam import CAMUV
    model = CAMUV(alpha=alpha, num_explanatory_vals=d)
    model.fit(X)
    return parse_camuv_result(model)


def run_fci(X: np.ndarray, alpha: float = 0.01):
    """
    Run FCI (causal-learn) and return (A_est, B_est).

    Uses Fisher's z independence test (assumes approximately Gaussian marginals
    after the LSNM normalisation step).  Replace with a non-parametric test
    (e.g. 'kci') for more robustness at the cost of runtime.
    """
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    p      = X.shape[1]
    pag, _ = fci(X, independence_test_method=fisherz, alpha=alpha, verbose=False)
    return parse_fci_result(pag, p)


def run_bang(X: np.ndarray):
    """
    Run BANG (ngBap::bang) via rpy2.

    Prerequisites on the server:
        install.packages("remotes")
        remotes::install_github("ysamwang/ngBap")

    Function confirmed from bang/bang.R:
        ngBap::bang(Y, K=3, level=0.01, verbose=FALSE, restrict=1)
    where K is the degree of the non-Gaussian moment (K=3 is standard from
    the paper's simulations) and restrict=1 tests all moments up to degree K.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri as numpy2ri

        numpy2ri.activate()
        ngbap = importr("ngBap")

        n, p  = X.shape
        r_X   = ro.r.matrix(X.flatten(), nrow=n, ncol=p, byrow=True)

        result = ngbap.bang(r_X, K=3, level=0.01, verbose=False, restrict=1)
        numpy2ri.deactivate()

        return parse_bang_result(result, p)

    except Exception as e:
        print(f"[BANG] skipped ({e})")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def run_single_trial(
    n:              int,
    seed:           int,
    alpha:          float = 0.01,
    d:              int   = 3,
    include_bang:   bool  = False,
) -> list:
    """
    One trial: generate LSNM data → run all methods → compute metrics.

    Returns a list of result dicts (one per method).
    """
    X, A_true, B_true, _ = gen_lsnm_experiment(n=n, seed=seed)

    methods = {
        "LSNM-UV-X": lambda: run_lsnm_uv_x(X, alpha=alpha, d=d),
        "CAM-UV":    lambda: run_camuv(X,     alpha=alpha, d=d),
        "FCI":       lambda: run_fci(X,       alpha=alpha),
    }
    if include_bang:
        methods["BANG"] = lambda: run_bang(X)

    rows = []
    for name, fn in methods.items():
        t0 = time.perf_counter()
        try:
            A_est, B_est = fn()
        except Exception as e:
            print(f"  [{name}] n={n} seed={seed}: {e}")
            A_est = np.zeros_like(A_true, dtype=float)
            B_est = np.zeros_like(B_true, dtype=float)
        runtime = time.perf_counter() - t0

        if A_est is None:
            A_est = np.zeros_like(A_true, dtype=float)
        if B_est is None:
            B_est = np.zeros_like(B_true, dtype=float)

        prec_d, rec_d, f1_d = directed_metrics(A_est, A_true)
        prec_b, rec_b, f1_b = bidirected_metrics(B_est, B_true)

        rows.append(dict(
            method        = name,
            n             = n,
            seed          = seed,
            prec_directed = prec_d,
            rec_directed  = rec_d,
            f1_directed   = f1_d,
            prec_bidir    = prec_b,
            rec_bidir     = rec_b,
            f1_bidir      = f1_b,
            runtime_sec   = runtime,
        ))

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment grid
# ─────────────────────────────────────────────────────────────────────────────

def run_all_experiments(
    n_list:       list  = None,
    n_trials:     int   = 100,
    alpha:        float = 0.01,
    d:            int   = 3,
    include_bang: bool  = False,
    n_jobs:       int   = -1,
    save_path:    str   = "results_section6.csv",
) -> pd.DataFrame:
    """
    Run the full experiment grid and save results to CSV.

    Parameters
    ----------
    n_list       : sample sizes  (default [100, 200, …, 1000])
    n_trials     : trials per sample size  (default 100)
    alpha        : HSIC significance level  (default 0.01)
    d            : max |K| for parent search  (default 3)
    include_bang : run BANG via rpy2  (default False)
    n_jobs       : joblib parallelism  (-1 = all cores)
    save_path    : path for CSV output

    Returns
    -------
    df : pandas DataFrame with all results
    """
    if n_list is None:
        n_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    tasks = [(n, seed) for n in n_list for seed in range(n_trials)]
    print(f"Launching {len(tasks)} trials "
          f"({len(n_list)} sample sizes × {n_trials} trials) …")

    all_rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_single_trial)(n, seed, alpha=alpha, d=d,
                                  include_bang=include_bang)
        for n, seed in tasks
    )

    df = pd.DataFrame([row for trial in all_rows for row in trial])
    df.to_csv(save_path, index=False)
    print(f"Saved → {save_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Alpha-sensitivity analysis (Figure 7 analogue)
# ─────────────────────────────────────────────────────────────────────────────

def run_alpha_sensitivity(
    alpha_list: list  = None,
    n:          int   = 500,
    n_trials:   int   = 100,
    d:          int   = 3,
    save_path:  str   = "results_alpha_sensitivity.csv",
) -> pd.DataFrame:
    """
    Vary significance level α at fixed n=500 for LSNM-UV-X only.
    """
    if alpha_list is None:
        alpha_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    rows = []
    for alpha in alpha_list:
        for seed in range(n_trials):
            X, A_true, B_true, _ = gen_lsnm_experiment(n=n, seed=seed)
            A_est, B_est = run_lsnm_uv_x(X, alpha=alpha, d=d)
            p_d, r_d, f1_d = directed_metrics(A_est, A_true)
            rows.append(dict(alpha=alpha, seed=seed,
                             prec=p_d, rec=r_d, f1=f1_d))

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved → {save_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    df:        pd.DataFrame,
    save_path: str = "figure_section6_directed.png",
):
    """
    Three-panel figure: average precision / recall / F-measure vs sample size,
    one line per method.  Reproduces the style of Maeda21 Figure 5.
    """
    import matplotlib.pyplot as plt

    methods  = df["method"].unique()
    n_list   = sorted(df["n"].unique())
    cols     = ["prec_directed", "rec_directed", "f1_directed"]
    titles   = ["Average Precision", "Average Recall", "Average F-measure"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, col, title in zip(axes, cols, titles):
        for method in methods:
            sub = df[df["method"] == method].groupby("n")[col].mean()
            ax.plot(sub.index, sub.values, marker="s", label=method)
        ax.set_xlabel("sample size")
        ax.set_ylabel(title.lower())
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.show()


def plot_bidir_results(
    df:        pd.DataFrame,
    method:    str = "LSNM-UV-X",
    save_path: str = "figure_section6_bidir.png",
):
    """
    UBP/UCP identification performance (precision/recall/F1 vs sample size)
    for a single method.  Reproduces Maeda21 Figure 6.
    """
    import matplotlib.pyplot as plt

    sub_df = df[df["method"] == method]
    n_list = sorted(sub_df["n"].unique())

    cols   = ["prec_bidir", "rec_bidir", "f1_bidir"]
    labels = ["average precision", "average recall", "average F-measure"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, label in zip(cols, labels):
        vals = sub_df.groupby("n")[col].mean()
        ax.plot(vals.index, vals.values, marker="s", label=label)
    ax.set_xlabel("sample size")
    ax.set_ylabel("value")
    ax.set_title(f"{method}: UBP/UCP identification")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.show()


def plot_alpha_sensitivity(
    df:        pd.DataFrame,
    save_path: str = "figure_alpha_sensitivity.png",
):
    """Reproduce Maeda21 Figure 7: precision/recall/F1 vs alpha."""
    import matplotlib.pyplot as plt

    cols   = ["prec", "rec", "f1"]
    labels = ["average precision", "average recall", "average F-measure"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, label in zip(cols, labels):
        vals = df.groupby("alpha")[col].mean()
        ax.plot(vals.index, vals.values, marker="s", label=label)
    ax.set_xlabel("α")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylabel("value")
    ax.set_title("Sensitivity to α  (LSNM-UV-X, n=500)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.show()


def plot_runtime(
    df:        pd.DataFrame,
    save_path: str = "figure_runtime.png",
):
    """Average runtime vs sample size (Maeda21 Figure 8 analogue)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in df["method"].unique():
        sub = df[df["method"] == method].groupby("n")["runtime_sec"].mean()
        ax.plot(sub.index, sub.values, marker="s", label=method)
    ax.set_xlabel("sample size")
    ax.set_ylabel("average run time (seconds)")
    ax.set_title("Average runtime")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Main experiment ───────────────────────────────────────────────────────
    df_main = run_all_experiments(
        n_list       = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        n_trials     = 100,
        alpha        = 0.01,
        d            = 3,
        include_bang = False,   # set True once R + ngBap is installed
        n_jobs       = -1,
        save_path    = "results_section6.csv",
    )
    plot_results(df_main,       save_path="figure_section6_directed.png")
    plot_bidir_results(df_main, save_path="figure_section6_bidir.png")
    plot_runtime(df_main,       save_path="figure_runtime.png")

    # ── Alpha sensitivity ─────────────────────────────────────────────────────
    df_alpha = run_alpha_sensitivity(
        alpha_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        n          = 500,
        n_trials   = 100,
        save_path  = "results_alpha_sensitivity.csv",
    )
    plot_alpha_sensitivity(df_alpha, save_path="figure_alpha_sensitivity.png")
