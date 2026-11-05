"""Extended benchmark for the follow-up study.

Improvements over Yin (2026, PLOS One e0328634):
  * 4 public datasets (395 - 6,607 students) instead of 1 (+1 case study)
  * leakage-free pipeline: imputation / min-max scaling fitted inside each training fold
  * repeated K-fold CV + Nadeau-Bengio corrected resampled t-test + Holm correction
    (the paper's plain paired t-test on CV folds is known to be anti-conservative)
  * new models: CatBoost and a TabPFN+CatBoost blend
  * uncertainty: TabPFN predictive quantiles + conformalized quantile regression (CQR),
    compared with split-conformal CatBoost
  * at-risk student detection (bottom 20 %) from the predictive distribution (ROC/PR-AUC)
  * subgroup (gender) error gaps
  * SHAP stability across folds (CatBoost TreeSHAP)

Usage:
  python benchmark.py                          # all datasets
  python benchmark.py --datasets math uci_por  # subset
  python benchmark.py --quick                  # smoke test
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from reproduce import make_tabpfn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 42
ALPHA = 0.10  # 90 % prediction intervals
RISK_Q = 0.20  # at-risk = bottom 20 % of the training-fold target
QGRID = [round(q, 3) for q in np.arange(0.025, 0.976, 0.025)]  # 39 quantile levels

# --------------------------------------------------------------------------- #
# Datasets  ->  (X with numeric/ordinal-encoded columns, may contain NaN), y, gender
# --------------------------------------------------------------------------- #


def _ordinal(s, order):
    m = {v: i for i, v in enumerate(order)}
    out = s.map(m)
    bad = s.notna() & out.isna()
    if bad.any():
        raise ValueError(f"{s.name}: unmapped values {s[bad].unique()}")
    return out


def load_math():
    df = pd.read_csv(DATA / "student_performance_math" / "exams.csv")
    edu = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    X = pd.DataFrame(
        {
            "gender": (df["gender"] == "female").astype(int),
            "race/ethnicity": _ordinal(df["race/ethnicity"], [f"group {c}" for c in "ABCDE"]),
            "parental_education": _ordinal(df["parental level of education"], edu),
            "lunch_standard": (df["lunch"] == "standard").astype(int),
            "test_prep_completed": (df["test preparation course"] == "completed").astype(int),
            "reading_score": df["reading score"],
            "writing_score": df["writing score"],
        }
    )
    y = df["math score"].astype(float)
    y = (y - y.min()) / (y.max() - y.min())  # same 0-1 scale as the paper's Table 5
    return X, y, X["gender"]


def _load_uci(fname):
    """UCI Student Performance (Cortez & Silva 2008). Early-warning setting:
    G1/G2 period grades are excluded, only information available before the term."""
    df = pd.read_csv(DATA / "uci_student_performance" / fname, sep=";")
    y = df.pop("G3").astype(float)
    df = df.drop(columns=["G1", "G2"])
    binary = {
        "school": "GP", "sex": "F", "address": "U", "famsize": "GT3", "Pstatus": "T",
        "schoolsup": "yes", "famsup": "yes", "paid": "yes", "activities": "yes", "nursery": "yes",
        "higher": "yes", "internet": "yes", "romantic": "yes",
    }
    for c, pos in binary.items():
        df[c] = (df[c] == pos).astype(int)
    for c in ["Mjob", "Fjob", "reason", "guardian"]:  # nominal -> one-hot
        df = pd.concat([df.drop(columns=c), pd.get_dummies(df[c], prefix=c, dtype=int)], axis=1)
    return df, y, df["sex"]


def load_uci_mat():
    return _load_uci("student-mat.csv")


def load_uci_por():
    return _load_uci("student-por.csv")


def load_spf():
    """Kaggle 'Student Performance Factors' (6,607 students). Feature set overlaps
    SAP-4000: study hours, attendance, tutoring, parental education, gender."""
    df = pd.read_csv(DATA / "student_performance_factors" / "StudentPerformanceFactors.csv")
    y = df.pop("Exam_Score").astype(float)
    lmh = ["Low", "Medium", "High"]
    for c in ["Parental_Involvement", "Access_to_Resources", "Motivation_Level", "Family_Income", "Teacher_Quality"]:
        df[c] = _ordinal(df[c], lmh)
    df["Peer_Influence"] = _ordinal(df["Peer_Influence"], ["Negative", "Neutral", "Positive"])
    df["Parental_Education_Level"] = _ordinal(df["Parental_Education_Level"], ["High School", "College", "Postgraduate"])
    df["Distance_from_Home"] = _ordinal(df["Distance_from_Home"], ["Near", "Moderate", "Far"])
    for c, pos in {"Extracurricular_Activities": "Yes", "Internet_Access": "Yes", "Learning_Disabilities": "Yes",
                   "School_Type": "Public", "Gender": "Female"}.items():
        df[c] = (df[c] == pos).astype(int)
    return df, y, df["Gender"]


DATASETS = {
    "math": ("Students Performance in Exams (US, Kaggle)", load_math),
    "uci_mat": ("UCI Student Performance - Mathematics (PT)", load_uci_mat),
    "uci_por": ("UCI Student Performance - Portuguese (PT)", load_uci_por),
    "spf": ("Student Performance Factors (Kaggle)", load_spf),
}

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

S1_SPACES = {  # search spaces from the original paper's S1 Table
    "SVR": (SVR(), {"svr__C": [0.1, 1, 10, 100], "svr__kernel": ["linear", "rbf"]}),
    "KNN": (KNeighborsRegressor(), {"kneighborsregressor__n_neighbors": [3, 5, 7, 9, 11, 15, 20, 40]}),
    "Ridge": (Ridge(), {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}),
    "RandomForest": (
        RandomForestRegressor(random_state=SEED, n_jobs=1),
        {"randomforestregressor__n_estimators": [100, 200, 300], "randomforestregressor__max_depth": [5, 10, 15, 20],
         "randomforestregressor__min_samples_split": [4, 8, 12, 16]},
    ),
    "GBM": (
        GradientBoostingRegressor(random_state=SEED),
        {"gradientboostingregressor__n_estimators": [50, 100, 200, 300],
         "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1, 0.5],
         "gradientboostingregressor__max_depth": [3, 5, 10, 15]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=SEED, n_jobs=1, verbosity=0),
        {"xgbregressor__n_estimators": [50, 100, 200], "xgbregressor__learning_rate": [0.01, 0.05, 0.1],
         "xgbregressor__max_depth": [3, 6, 10], "xgbregressor__subsample": [0.8, 1.0]},
    ),
    "LightGBM": (
        LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1),
        {"lgbmregressor__n_estimators": [50, 100, 200], "lgbmregressor__learning_rate": [0.01, 0.05, 0.1],
         "lgbmregressor__max_depth": [3, 6, 10], "lgbmregressor__num_leaves": [5, 10, 15, 20, 30]},
    ),
}


def _prep():
    return [SimpleImputer(strategy="most_frequent"), MinMaxScaler()]


def tuned(name, n_iter, inner_cv, seed):
    est, space = S1_SPACES[name]
    pipe = make_pipeline(*_prep(), clone(est))
    return RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=inner_cv, scoring="neg_mean_squared_error",
                              random_state=seed, n_jobs=-1)


def catboost(seed):
    return make_pipeline(*_prep(), CatBoostRegressor(random_seed=seed, verbose=0, thread_count=-1))


class TabPFNPipe:
    """Imputation + scaling in-fold, then TabPFN v2; exposes predictive quantiles."""

    def __init__(self, n_estimators, seed):
        self.prep = make_pipeline(*_prep())
        self.model = make_tabpfn(n_estimators, seed)

    def fit(self, X, y):
        self.model.fit(self.prep.fit_transform(X), np.asarray(y))
        return self

    def predict_main(self, X):
        out = self.model.predict(self.prep.transform(X), output_type="main", quantiles=QGRID)
        return np.asarray(out["mean"]), np.vstack(out["quantiles"]).T  # (n,), (n, len(QGRID))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def point_metrics(y, p):
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p)}


def risk_metrics(y, score, thr):
    lab = (np.asarray(y) < thr).astype(int)
    if lab.min() == lab.max():
        return {"ROC_AUC": np.nan, "PR_AUC": np.nan}
    return {"ROC_AUC": roc_auc_score(lab, score), "PR_AUC": average_precision_score(lab, score)}


def group_gap(y, p, g):
    y, p, g = map(np.asarray, (y, p, g))
    r = [np.sqrt(np.mean((y[g == v] - p[g == v]) ** 2)) for v in (0, 1)]
    return {"RMSE_g0": r[0], "RMSE_g1": r[1], "RMSE_gap": abs(r[0] - r[1])}


def prob_below(qpred, thr):
    """P(y < thr) from a quantile grid (piecewise-linear CDF)."""
    lv = np.array(QGRID)
    return np.array([np.interp(thr, np.sort(row), lv, left=0.0, right=1.0) for row in qpred])


def cqr(q_lo_cal, q_hi_cal, y_cal, alpha=ALPHA):
    """Conformalized quantile regression correction (Romano et al., 2019)."""
    s = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(s)[k - 1]


def split_conformal(res_cal, alpha=ALPHA):
    n = len(res_cal)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(np.abs(res_cal))[k - 1]


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return t, p


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def run_dataset(key, args, log):
    title, loader = DATASETS[key]
    X, y, g = loader()
    out = ROOT / "results" / ("benchmark_quick" if args.quick else "benchmark") / key
    out.mkdir(parents=True, exist_ok=True)
    log(f"\n##### {key}: {title}  n={len(X)}  d={X.shape[1]}  y[{y.min():.3g}, {y.max():.3g}] mean={y.mean():.3g}")

    rkf = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    rows, unc_rows, shap_rows = [], [], []
    for fold, (tr, te) in enumerate(rkf.split(X)):
        t0 = time.time()
        Xtr, Xte, ytr, yte, gte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], g.iloc[te]
        thr = np.quantile(ytr, RISK_Q)
        preds = {}

        for name in S1_SPACES:
            m = tuned(name, args.n_iter, args.inner_cv, SEED + fold).fit(Xtr, ytr)
            preds[name] = m.predict(Xte)

        cb = catboost(SEED + fold).fit(Xtr, ytr)
        preds["CatBoost"] = cb.predict(Xte)
        Xte_s = cb[:-1].transform(Xte)
        sv = cb[-1].get_feature_importance(type="ShapValues", data=Pool(Xte_s))[:, :-1]
        shap_rows.append(dict(zip(X.columns, np.abs(sv).mean(0)), fold=fold))

        tab = TabPFNPipe(args.tabpfn_estimators, SEED + fold).fit(Xtr, ytr)
        preds["TabPFN"], qte = tab.predict_main(Xte)
        preds["TabPFN+CatBoost"] = (preds["TabPFN"] + preds["CatBoost"]) / 2

        for name, p in preds.items():
            r = {"fold": fold, "Model": name, **point_metrics(yte, p), **risk_metrics(yte, -p, thr), **group_gap(yte, p, gte)}
            rows.append(r)
        # TabPFN risk score from its full predictive distribution instead of the point estimate
        rows.append({"fold": fold, "Model": "TabPFN (P(y<t))", **point_metrics(yte, preds["TabPFN"]),
                     **risk_metrics(yte, prob_below(qte, thr), thr), **group_gap(yte, preds["TabPFN"], gte)})

        # ---- uncertainty: fit on 80 % proper-train, calibrate on 20 % ----
        i_fit, i_cal = train_test_split(np.arange(len(Xtr)), test_size=0.2, random_state=SEED + fold)
        Xf, yf, Xc, yc = Xtr.iloc[i_fit], ytr.iloc[i_fit], Xtr.iloc[i_cal], ytr.iloc[i_cal].to_numpy()
        lo_i, hi_i = QGRID.index(round(ALPHA / 2, 3)), QGRID.index(round(1 - ALPHA / 2, 3))
        # raw TabPFN quantiles (full-train model)
        lo, hi = qte[:, lo_i], qte[:, hi_i]
        unc_rows.append({"fold": fold, "Method": "TabPFN raw quantiles", "Coverage": np.mean((yte >= lo) & (yte <= hi)), "Width": np.mean(hi - lo)})
        # CQR-TabPFN
        tab_c = TabPFNPipe(args.tabpfn_estimators, SEED + fold).fit(Xf, yf)
        _, qc = tab_c.predict_main(Xc)
        _, qt = tab_c.predict_main(Xte)
        e = cqr(qc[:, lo_i], qc[:, hi_i], yc)
        lo, hi = qt[:, lo_i] - e, qt[:, hi_i] + e
        unc_rows.append({"fold": fold, "Method": "CQR-TabPFN", "Coverage": np.mean((yte >= lo) & (yte <= hi)), "Width": np.mean(hi - lo)})
        # split-conformal CatBoost (constant-width baseline)
        cb_c = catboost(SEED + fold).fit(Xf, yf)
        e = split_conformal(yc - cb_c.predict(Xc))
        p = cb_c.predict(Xte)
        unc_rows.append({"fold": fold, "Method": "Split-conformal CatBoost", "Coverage": np.mean(np.abs(yte - p) <= e), "Width": 2 * e})

        best = min(preds, key=lambda k: point_metrics(yte, preds[k])["RMSE"])
        log(f"  fold {fold + 1}/{args.folds * args.repeats}  best={best}  TabPFN RMSE={point_metrics(yte, preds['TabPFN'])['RMSE']:.4f}  ({time.time() - t0:.0f}s)")

    folds = pd.DataFrame(rows)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    unc = pd.DataFrame(unc_rows)
    unc.to_csv(out / "uncertainty_folds.csv", index=False)
    shap_df = pd.DataFrame(shap_rows).set_index("fold")
    shap_df.to_csv(out / "catboost_shap_folds.csv")

    summ = folds.groupby("Model").agg(["mean", "std"]).drop(columns="fold", level=0)
    summ = summ.sort_values(("RMSE", "mean"))
    summ.to_csv(out / "summary.csv")
    unc_s = unc.groupby("Method")[["Coverage", "Width"]].agg(["mean", "std"])
    unc_s.to_csv(out / "uncertainty_summary.csv")

    # significance: best model and TabPFN vs every other model
    n_te = len(X) // args.folds
    n_tr = len(X) - n_te
    sig = []
    point_models = [m for m in folds["Model"].unique() if m != "TabPFN (P(y<t))"]
    pivot = folds[folds["Model"].isin(point_models)].pivot(index="fold", columns="Model", values="RMSE")
    for ref in dict.fromkeys([summ.index[0] if summ.index[0] in point_models else "TabPFN", "TabPFN"]):
        tmp = []
        for m in point_models:
            if m == ref:
                continue
            t, p = corrected_ttest(pivot[ref], pivot[m], n_tr, n_te)
            tmp.append({"Reference": ref, "vs": m, "mean RMSE diff": (pivot[ref] - pivot[m]).mean(), "t (corrected)": t, "p": p})
        adj = holm([r["p"] for r in tmp])
        for r, a in zip(tmp, adj):
            r["p (Holm)"] = a
        sig += tmp
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "significance.csv", index=False)

    imp = shap_df.mean().sort_values(ascending=False)
    rank_corr = shap_df.rank(axis=1).T.corr(method="spearman").values[np.triu_indices(len(shap_df), 1)].mean()

    log(f"\n  -- {key} summary (mean over {args.folds}x{args.repeats} folds) --")
    log(summ[[("RMSE", "mean"), ("RMSE", "std"), ("MAE", "mean"), ("R2", "mean"), ("ROC_AUC", "mean"), ("PR_AUC", "mean"), ("RMSE_gap", "mean")]].round(4).to_string())
    log(f"\n  uncertainty (target {1 - ALPHA:.0%}):\n{unc_s.round(4).to_string()}")
    log(f"\n  significance:\n{sig.round(4).to_string(index=False)}")
    log(f"\n  CatBoost SHAP top-5: {imp.head().round(4).to_dict()}  mean fold-to-fold Spearman={rank_corr:.3f}")
    return key, summ, unc_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations for baselines")
    ap.add_argument("--inner-cv", type=int, default=3)
    ap.add_argument("--tabpfn-estimators", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.repeats, args.n_iter, args.inner_cv, args.tabpfn_estimators = 2, 1, 3, 2, 2

    base = ROOT / "results" / ("benchmark_quick" if args.quick else "benchmark")
    base.mkdir(parents=True, exist_ok=True)
    logf = open(base / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"\n==== benchmark {time.strftime('%Y-%m-%d %H:%M')}  args={json.dumps(vars(args))}")
    t0 = time.time()
    overall = []
    for key in args.datasets:
        _, summ, _ = run_dataset(key, args, log)
        r = summ[("RMSE", "mean")].rename(key)
        overall.append(r)
    tab = pd.concat(overall, axis=1)
    tab.to_csv(base / "rmse_all_datasets.csv")
    ranks = tab.drop(index="TabPFN (P(y<t))", errors="ignore").rank(axis=0).mean(axis=1).sort_values()
    ranks.rename("mean rank").to_csv(base / "mean_rank.csv")
    log(f"\n==== mean RMSE rank across datasets ====\n{ranks.round(2).to_string()}")
    log(f"\nTotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
