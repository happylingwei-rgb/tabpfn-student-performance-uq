"""Reproduction of Yin (2026), PLOS One 21(9): e0328634.

"Predicting student academic performance using TabPFN and SHAP"

Pipeline (follows the paper's Methodology section):
  1. Label / ordinal encoding + min-max normalization
  2. Stratified 80/20 train-test split
  3. GridSearchCV (10-fold, neg MSE) for 7 baselines, search spaces from S1 Table
  4. Test-set RMSE / MAE / R2 with 1,000-iteration bootstrap 95% CI     (Table 3 / 5)
  5. Paired t-test, Wilcoxon, Cohen's d on 10-fold CV RMSE             (Table 4 / S3)
  6. Sample-size experiment 500-3,500, 30 reps, seed 42+i               (Fig 5)
  7. SHAP via remove-and-recontextualize (exact, same as shapiq TabPFNExplainer)
                                                                         (Fig 6-8)

Usage:
  python reproduce.py --dataset math              # case-study dataset (exams.csv)
  python reproduce.py --dataset sap4000           # needs data/SAP-4000.csv
  python reproduce.py --dataset math --quick      # fast smoke test
"""

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
SEED = 42

# --------------------------------------------------------------------------- #
# Data loading / preprocessing
# --------------------------------------------------------------------------- #

PARENT_EDU_ORDER = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
MATH_PARENT_EDU_ORDER = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5,
}


def _find_col(df, *candidates):
    norm = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def _minmax(s):
    return (s - s.min()) / (s.max() - s.min())


def load_sap4000(path):
    """SAP-4000: target Exam_Score kept on its original 0-100 scale."""
    raw = pd.read_csv(path)
    col = lambda *c: raw[_find_col(raw, *c)]
    X = pd.DataFrame(
        {
            "HoursStudied/Week": _minmax(col("HoursStudied/Week", "HoursStudied", "Hours_Studied").astype(float)),
            "Attendance(%)": _minmax(col("Attendance(%)", "Attendance").astype(float)),
            "Tutoring_encoded": col("Tutoring").map({"No": 0, "Yes": 1}),
            "Gender_encoded": col("Gender").map({"Male": 0, "Female": 1}),
            "Region_encoded": col("Region").map({"Rural": 0, "Urban": 1}),
            "Parent Education_encoded": col("Parent Education", "ParentEducation", "Parent_Education").map(PARENT_EDU_ORDER),
        }
    )
    if X.isna().any().any():
        raise ValueError(f"Unmapped category values:\n{X.isna().sum()}")
    y = col("Exam_Score", "ExamScore").astype(float)
    return X, y.rename("Exam_Score")


def load_math(path):
    """Case study (Kaggle rkiattisak). Target math score, min-max scaled to [0, 1]
    (the paper's Table 5 RMSE ~0.06 implies a normalized target)."""
    raw = pd.read_csv(path)
    X = pd.DataFrame(
        {
            "gender_encoded": (raw["gender"] == "female").astype(int),
            "race/ethnicity_encoded": raw["race/ethnicity"].str[-1].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}),
            "parental education_encoded": raw["parental level of education"].map(MATH_PARENT_EDU_ORDER),
            "lunch_encoded": (raw["lunch"] == "standard").astype(int),
            "test preparation_encoded": (raw["test preparation course"] == "completed").astype(int),
            "reading score": _minmax(raw["reading score"].astype(float)),
            "writing score": _minmax(raw["writing score"].astype(float)),
        }
    )
    if X.isna().any().any():
        raise ValueError(f"Unmapped category values:\n{X.isna().sum()}")
    y = _minmax(raw["math score"].astype(float))
    return X, y.rename("math score")


DATASETS = {
    "sap4000": (load_sap4000, ROOT / "data" / "SAP-4000.csv"),
    "math": (load_math, ROOT / "data" / "student_performance_math" / "exams.csv"),
}


def strat_bins(y, q=10):
    """Quantile bins of a continuous target, used for stratified splitting."""
    return pd.qcut(y, q=q, labels=False, duplicates="drop")


def vif_table(X):
    rows = []
    for c in X.columns:
        others = X.drop(columns=c)
        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])
        rows.append({"Feature": c, "VIF": 1.0 / (1.0 - r2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# --------------------------------------------------------------------------- #
# Models (search spaces = S1 Table)
# --------------------------------------------------------------------------- #


def make_tabpfn(n_estimators, seed=SEED):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    # Paper uses TabPFN v2 (ref [48]); v2 weights are not license-gated.
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2,
        n_estimators=n_estimators,
        random_state=seed,
        ignore_pretraining_limits=True,
    )


BASELINES = {
    "SVR": (SVR(), {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"]}),
    "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9, 11, 15, 20, 40]}),
    "Ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
    "RandomForest": (
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15, 20], "min_samples_split": [4, 8, 12, 16]},
    ),
    "GBM": (
        GradientBoostingRegressor(random_state=SEED),
        {"n_estimators": [50, 100, 200, 300], "learning_rate": [0.01, 0.05, 0.1, 0.5], "max_depth": [3, 5, 10, 15]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 6, 10], "subsample": [0.8, 1.0]},
    ),
    "LightGBM": (
        LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
        {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 6, 10], "num_leaves": [5, 10, 15, 20, 30]},
    ),
}

# Optimal configurations reported in S1 Table (used when --no-grid is given).
PAPER_BEST = {
    "SVR": {"C": 10, "kernel": "linear"},
    "KNN": {"n_neighbors": 9},
    "Ridge": {"alpha": 1},
    "RandomForest": {"n_estimators": 200, "max_depth": 10, "min_samples_split": 12},
    "GBM": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    "XGBoost": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8},
    "LightGBM": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 10},
}

TABPFN_GRID = [8, 16, 24, 32]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def metrics(y, p):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "R2": float(r2_score(y, p)),
    }


def bootstrap_ci(y, p, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    out = {"RMSE": [], "MAE": [], "R2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for k, v in metrics(y[idx], p[idx]).items():
            out[k].append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def cv_rmse(model, X, y, folds):
    scores = []
    for tr, va in folds:
        m = clone(model) if not hasattr(model, "_tabpfn_factory") else model._tabpfn_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(np.sqrt(mean_squared_error(y.iloc[va], m.predict(X.iloc[va]))))
    return np.array(scores)


class _TabPFNFactory:
    """Marker so cv_rmse builds a fresh TabPFN per fold."""

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def _tabpfn_factory(self):
        return make_tabpfn(self.n_estimators)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #


def tune_models(X_tr, y_tr, args, log):
    best = {}
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for name, (est, grid) in BASELINES.items():
        t = time.time()
        if args.no_grid:
            params = PAPER_BEST[name]
        else:
            gs = GridSearchCV(est, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            gs.fit(X_tr, y_tr)
            params = gs.best_params_
        best[name] = clone(est).set_params(**params)
        log(f"  {name:<12} {params}  ({time.time() - t:.1f}s)")

    if args.tabpfn_grid:
        t = time.time()
        folds = list(cv.split(X_tr))
        res = {n: cv_rmse(_TabPFNFactory(n), X_tr, y_tr, folds).mean() for n in TABPFN_GRID}
        n_best = min(res, key=res.get)
        log(f"  {'TabPFN':<12} CV RMSE {res} -> n_estimators={n_best}  ({time.time() - t:.1f}s)")
    else:
        n_best = args.tabpfn_estimators
        log(f"  {'TabPFN':<12} n_estimators={n_best} (S1 Table optimum; use --tabpfn-grid to search)")
    return best, n_best


def plot_pred_vs_obs(preds, y_te, out):
    names = list(preds)
    ncol = 2
    nrow = math.ceil(len(names) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.5 * nrow))
    for ax, name in zip(axes.ravel(), names):
        p = preds[name]
        m = metrics(y_te, p)
        sc = ax.scatter(y_te, p, c=p, cmap="jet", s=10, alpha=0.8)
        lo, hi = min(y_te.min(), p.min()), max(y_te.max(), p.max())
        xs = np.linspace(lo, hi, 100)
        b, a = np.polyfit(y_te, p, 1)
        resid_sd = np.std(p - (a + b * y_te), ddof=2)
        ax.plot(xs, xs, "--", color="purple", label="Ideal Line (y=x)")
        ax.plot(xs, a + b * xs, "r-", label=f"Fitted Line: y={a:.3f}+{b:.3f}x")
        ax.fill_between(xs, a + b * xs - 1.96 * resid_sd, a + b * xs + 1.96 * resid_sd, color="pink", alpha=0.3, label="95% Confidence Interval")
        ax.text(0.03, 0.97, f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f}\nMAE = {m['MAE']:.3f}", transform=ax.transAxes, va="top", fontsize=9, bbox=dict(fc="white", ec="gray"))
        ax.set_title(f"{name} Model - Predicted vs Observed Values", fontsize=10)
        ax.set_xlabel("Observed Value")
        ax.set_ylabel("Predicted Value")
        ax.legend(loc="lower right", fontsize=7)
        fig.colorbar(sc, ax=ax, label="Predicted Value")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def exp_main(X, y, args, outdir, log):
    log("\n=== [1] Train/test split + hyperparameter tuning ===")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=strat_bins(y))
    log(f"  train={len(X_tr)}  test={len(X_te)}")
    best, n_tabpfn = tune_models(X_tr, y_tr, args, log)

    log("\n=== [2] Test-set performance (Table 3 / Table 5) ===")
    preds, rows, times = {}, [], []
    for name, model in [*best.items(), ("TabPFN", None)]:
        m = make_tabpfn(n_tabpfn) if name == "TabPFN" else clone(model)
        t0 = time.time()
        m.fit(X_tr, y_tr)
        t1 = time.time()
        p = m.predict(X_te)
        t2 = time.time()
        preds[name] = p
        mt, ci = metrics(y_te, p), bootstrap_ci(y_te, p, n_boot=args.n_boot)
        rows.append(
            {
                "Model": name,
                **{f"{k}": mt[k] for k in mt},
                **{f"{k} 95% CI": f"[{ci[k][0]:.4f}, {ci[k][1]:.4f}]" for k in ci},
            }
        )
        times.append({"Model": name, "Training Time (s)": t1 - t0, "Inference Time (s)": t2 - t1})
        log(f"  {name:<12} RMSE={mt['RMSE']:.4f} {ci['RMSE']}  MAE={mt['MAE']:.4f}  R2={mt['R2']:.4f}")
    perf = pd.DataFrame(rows)
    perf.to_csv(outdir / "table3_test_performance.csv", index=False)
    pd.DataFrame(times).to_csv(outdir / "S2_train_inference_time.csv", index=False)
    plot_pred_vs_obs(preds, y_te.to_numpy(), outdir / "fig4_pred_vs_obs.png")

    log("\n=== [3] Paired t-test / Wilcoxon / Cohen's d on 10-fold CV RMSE (Table 4 / S3) ===")
    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=SEED).split(X_tr))
    cv_scores = {name: cv_rmse(m, X_tr, y_tr, folds) for name, m in best.items()}
    cv_scores["TabPFN"] = cv_rmse(_TabPFNFactory(n_tabpfn), X_tr, y_tr, folds)
    pd.DataFrame(cv_scores).to_csv(outdir / "cv_fold_rmse.csv", index_label="fold")
    tab = cv_scores["TabPFN"]
    rows = []
    for name, s in cv_scores.items():
        if name == "TabPFN":
            continue
        diff = tab - s
        t, p_t = stats.ttest_rel(tab, s)
        w, p_w = stats.wilcoxon(tab, s)
        # Paper's d matches mean(diff) / std(diff, ddof=0), i.e. t / sqrt(n-1).
        d = diff.mean() / diff.std(ddof=0)
        size = "Large" if abs(d) >= 0.8 else "Medium" if abs(d) >= 0.5 else "Small" if abs(d) >= 0.2 else "Negligible"
        rows.append(
            {
                "Comparison": f"TabPFN vs {name}",
                "RMSE Diff": -diff.mean(),
                "t-statistic": t,
                "p-value (t)": p_t,
                "Cohen's d": d,
                "Effect Size": size,
                "Wilcoxon W": w,
                "p-value (Wilcoxon)": p_w,
            }
        )
        log(f"  TabPFN vs {name:<12} diff={-diff.mean():+.4f} t={t:.3f} p={p_t:.4g} d={d:.3f} | W={w:.0f} p={p_w:.4g}")
    pd.DataFrame(rows).to_csv(outdir / "table4_significance.csv", index=False)

    return X_tr, X_te, y_tr, y_te, best, n_tabpfn


def exp_sample_size(X, y, best, n_tabpfn, args, outdir, log):
    log("\n=== [4] Sample-size experiment (Fig 5) ===")
    sizes = [s for s in [500, 1000, 1500, 2000, 2500, 3000, 3500] if s < len(X)]
    if not sizes:
        log("  dataset too small, skipped")
        return
    models = {"TabPFN": None, "XGBoost": best["XGBoost"], "RandomForest": best["RandomForest"]}
    bins = strat_bins(y)
    rows = []
    for n in sizes:
        for rep in range(args.reps):
            seed = SEED + rep
            idx, _ = train_test_split(np.arange(len(X)), train_size=n, random_state=seed, stratify=bins)
            Xs, ys = X.iloc[idx], y.iloc[idx]
            X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.2, random_state=seed, stratify=strat_bins(ys))
            for name, m in models.items():
                m = make_tabpfn(n_tabpfn, seed) if name == "TabPFN" else clone(m)
                m.fit(X_tr, y_tr)
                rows.append({"n": n, "rep": rep, "Model": name, **metrics(y_te, m.predict(X_te))})
        sub = pd.DataFrame([r for r in rows if r["n"] == n]).groupby("Model")[["R2", "RMSE"]].agg(["mean", "std"])
        log(f"  n={n}\n{sub.round(4).to_string()}")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "fig5_sample_size_raw.csv", index=False)
    agg = df.groupby(["Model", "n"])[["R2", "RMSE"]].agg(["mean", "std"])
    agg.to_csv(outdir / "fig5_sample_size_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    style = {"TabPFN": ("o", "#d95f02"), "XGBoost": ("s", "#2c6fa8"), "RandomForest": ("^", "#7f6fbf")}
    for ax, met, title in [(axes[0], "R2", "(a) R²"), (axes[1], "RMSE", "(b) RMSE")]:
        for name, (mk, c) in style.items():
            a = agg.loc[name]
            ax.errorbar(a.index, a[(met, "mean")], yerr=a[(met, "std")], marker=mk, color=c, capsize=4, label=name)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(met.replace("R2", "R²"))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig5_sample_size.png", dpi=200)
    plt.close(fig)


def exact_shapley_recontextualize(X_tr, y_tr, X_te, n_estimators, log):
    """Exact Shapley values with the remove-and-recontextualize value function
    used by shapiq's TabPFNExplainer: v(S)(x) = TabPFN fit on features S only,
    v(empty) = mean(y_train). With d features this needs 2^d - 1 fits, each
    predicting all test rows at once (much faster than per-instance)."""
    cols = list(X_tr.columns)
    d = len(cols)
    V = np.zeros((2**d, len(X_te)))
    V[0] = y_tr.mean()
    t = time.time()
    for mask in range(1, 2**d):
        feats = [cols[j] for j in range(d) if mask >> j & 1]
        m = make_tabpfn(n_estimators)
        m.fit(X_tr[feats], y_tr)
        V[mask] = m.predict(X_te[feats])
        if mask % 16 == 0 or mask == 2**d - 1:
            log(f"    coalition {mask}/{2**d - 1}  ({time.time() - t:.0f}s)")
    phi = np.zeros((len(X_te), d))
    for i in range(d):
        for mask in range(2**d):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            w = math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
            phi[:, i] += w * (V[mask | (1 << i)] - V[mask])
    return phi, V[0][0], V[-1]


def exp_shap(X_tr, X_te, y_tr, y_te, args, outdir, log):
    import shap

    log(f"\n=== [5] SHAP (remove-and-recontextualize, exact, n_estimators={args.shap_estimators}) ===")
    phi, base, full_pred = exact_shapley_recontextualize(X_tr, y_tr, X_te, args.shap_estimators, log)
    cols = list(X_te.columns)
    pd.DataFrame(phi, columns=cols, index=X_te.index).to_csv(outdir / "shap_values_test.csv")

    # Fig 6: global importance with 95% bootstrap CI (over test samples)
    rng = np.random.default_rng(SEED)
    imp = np.abs(phi).mean(0)
    boots = np.array([np.abs(phi[rng.integers(0, len(phi), len(phi))]).mean(0) for _ in range(args.n_boot)])
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    order = np.argsort(imp)
    gi = pd.DataFrame({"Feature": cols, "mean|SHAP|": imp, "CI low": lo, "CI high": hi}).sort_values("mean|SHAP|", ascending=False)
    gi.to_csv(outdir / "fig6_global_importance.csv", index=False)
    log(gi.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(np.array(cols)[order], imp[order], xerr=[imp[order] - lo[order], hi[order] - imp[order]], capsize=3, color="#1f77b4")
    ax.set_xlabel("Mean Absolute SHAP Value (95% Bootstrap CI)")
    ax.set_title("Global Feature Importance with Confidence Intervals")
    fig.tight_layout()
    fig.savefig(outdir / "fig6_global_importance.png", dpi=200)
    plt.close(fig)

    # Fig 7: summary dot plot
    shap.summary_plot(phi, X_te, feature_names=cols, show=False)
    plt.title("SHAP Summary Dot Plot (TabPFN)")
    plt.savefig(outdir / "fig7_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close("all")

    # Fig 8: waterfall for one test sample (Sample 18 as in the paper, if present)
    k = min(18, len(X_te) - 1)
    exp = shap.Explanation(values=phi[k], base_values=base, data=X_te.iloc[k].to_numpy(), feature_names=cols)
    shap.plots.waterfall(exp, show=False)
    true, pred = y_te.iloc[k], base + phi[k].sum()
    plt.gcf().suptitle(f"Sample {k} SHAP Waterfall | Original index:{X_te.index[k]} | True:{true:.4f} Pred:{pred:.4f} Residual:{true - pred:.4f}", fontsize=9, y=1.02)
    plt.savefig(outdir / "fig8_shap_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close("all")
    log(f"  efficiency check: max|base+sum(phi) - f(x)| = {np.abs(base + phi.sum(1) - full_pred).max():.2e}")


# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, default="math")
    ap.add_argument("--data-path", type=Path, help="override CSV path")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--reps", type=int, default=30, help="repetitions per sample size (Fig 5)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tabpfn-estimators", type=int, default=32)
    ap.add_argument("--shap-estimators", type=int, default=8, help="TabPFN n_estimators inside SHAP (2^d fits)")
    ap.add_argument("--tabpfn-grid", action="store_true", help="grid-search TabPFN n_estimators [8,16,24,32] (slow on CPU)")
    ap.add_argument("--no-grid", action="store_true", help="skip GridSearchCV, use S1 Table optima")
    ap.add_argument("--skip-size", action="store_true")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 folds, 2 reps, 4 TabPFN estimators, 200 bootstrap")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.reps, args.n_boot = 3, 2, 200
        args.tabpfn_estimators = args.shap_estimators = 4
        args.no_grid = True

    loader, default_path = DATASETS[args.dataset]
    path = args.data_path or default_path
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}\nSee README.md for how to obtain it.")
    outdir = ROOT / "results" / (args.dataset + ("_quick" if args.quick else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "log.txt", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"dataset={args.dataset} path={path}\nargs={json.dumps({k: str(v) for k, v in vars(args).items()})}")
    X, y = loader(path)
    log(f"\nX shape={X.shape}\n{X.describe().T.round(3).to_string()}\n\n{y.describe().round(3).to_string()}")
    vif = vif_table(X)
    vif.to_csv(outdir / "table2_vif.csv", index=False)
    log(f"\nVIF (Table 2):\n{vif.round(4).to_string(index=False)}")

    t0 = time.time()
    X_tr, X_te, y_tr, y_te, best, n_tabpfn = exp_main(X, y, args, outdir, log)
    if not args.skip_size:
        exp_sample_size(X, y, best, n_tabpfn, args, outdir, log)
    if not args.skip_shap:
        exp_shap(X_tr, X_te, y_tr, y_te, args, outdir, log)
    log(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {outdir}")


if __name__ == "__main__":
    main()
