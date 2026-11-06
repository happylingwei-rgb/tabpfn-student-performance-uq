"""Extensions of FairWarn-PFN (see guarantee.py).

  (1) A second protected attribute (socio-economic / location proxy) in addition to
      gender:  D1 lunch type, D2/D3 home address (urban/rural), D4 family income (3 groups).
  (2) Fairness baselines for at-risk screening:
        - Hardt et al. (2016) equal-opportunity post-processing (fairlearn ThresholdOptimizer,
          true-positive-rate parity, balanced-accuracy objective), fitted on the calibration set;
        - plug-in group thresholds: the empirical alpha_r quantile of the calibration at-risk
          scores of each group, i.e. FairWarn-PFN without the finite-sample correction.
  (3) FairWarn-PFN-CV+: cross-conformal screening that uses out-of-fold scores of the whole
      training fold instead of a 20 % calibration split (K = 5 inner folds), which removes the
      "flag everyone" behaviour of small groups. Run on the three small datasets.

Same outer folds / proper-train / calibration split as benchmark.py and guarantee.py.

Usage:
  python extensions.py
  python extensions.py --datasets uci_mat --quick
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split

from benchmark import ALPHA, DATASETS, RISK_Q, ROOT, SEED, catboost
from guarantee import ALPHA_R, HI, LO, FinePipe, cqr_e, prob_below, screen_threshold

CV_DATASETS = ("math", "uci_mat", "uci_por")  # FairWarn-PFN-CV+ (5 extra TabPFN fits per fold)
K_INNER = 5


def ses_attribute(key, X):
    if key == "math":
        return "lunch", X["lunch_standard"].map({0: "free/reduced", 1: "standard"})
    if key in ("uci_mat", "uci_por"):
        return "address", X["address"].map({0: "rural", 1: "urban"})
    return "income", X["Family_Income"].map({0: "low", 1: "medium", 2: "high"})


class _Score(BaseEstimator):
    """Pre-fitted 'estimator' whose prediction is the risk score itself (column 0)."""

    def __init__(self):
        self.fitted_ = True

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


def hardt_eo(s_cal, z_cal, g_cal, s_te, g_te, seed):
    to = ThresholdOptimizer(estimator=_Score(), constraints="true_positive_rate_parity",
                            objective="balanced_accuracy_score", prefit=True, predict_method="predict")
    to.fit(s_cal.reshape(-1, 1), z_cal.astype(int), sensitive_features=g_cal)
    return to.predict(s_te.reshape(-1, 1), sensitive_features=g_te, random_state=seed).astype(bool)


def group_rule(s_cal, z_cal, g_cal, s_te, g_te, thr_fn):
    flag = np.zeros(len(s_te), bool)
    for v in np.unique(g_te):
        lam = thr_fn(s_cal[z_cal & (g_cal == v)])
        flag[g_te == v] = s_te[g_te == v] >= lam
    return flag


def plugin_threshold(alpha):
    return lambda s: -np.inf if len(s) == 0 else np.quantile(s, alpha)


def cvplus_rule(S, fold_id, z_tr, g_tr, s_te_folds, g_te, alpha):
    """Flag x iff #{i in J_g : S_i <= s_{-k(i)}(x)} >= k, k = floor(alpha (n1 + 1))."""
    flag = np.zeros(s_te_folds.shape[1], bool)
    for v in np.unique(g_te):
        J = z_tr & (g_tr == v)
        k = int(np.floor(alpha * (J.sum() + 1)))
        m = g_te == v
        if k == 0:
            flag[m] = True
            continue
        cnt = np.zeros(m.sum())
        for f in range(s_te_folds.shape[0]):
            Sf = S[J & (fold_id == f)]
            cnt += (Sf[:, None] <= s_te_folds[f, m][None, :]).sum(0)
        flag[m] = cnt >= k
    return flag


def screen_rows(base, flag, z, g):
    rows = [{**base, "group": "ALL", "n_atrisk": int(z.sum()), "Recall": flag[z].mean() if z.any() else np.nan,
             "FlagRate": flag.mean(), "Precision": z[flag].mean() if flag.any() else np.nan}]
    for v in np.unique(g):
        m = z & (g == v)
        rows.append({**base, "group": v, "n_atrisk": int(m.sum()), "Recall": flag[m].mean() if m.any() else np.nan,
                     "FlagRate": flag[g == v].mean(), "Precision": np.nan})
    return rows


def run_dataset(key, args, log):
    title, loader = DATASETS[key]
    X, y, gender = loader()
    ses_name, ses = ses_attribute(key, X)
    attrs = {"gender": gender.map({0: "M", 1: "F"}).to_numpy(), ses_name: ses.to_numpy()}
    do_cv = key in CV_DATASETS and not args.no_cv
    out = ROOT / "results" / ("extensions_quick" if args.quick else "extensions") / key
    out.mkdir(parents=True, exist_ok=True)
    log(f"\n##### {key}: {title}  n={len(X)}  attrs={list(attrs)}  "
        f"{ses_name} counts={pd.Series(attrs[ses_name]).value_counts().to_dict()}  CV+={do_cv}")

    rkf = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    iv_rows, sc_rows = [], []
    for fold, (tr, te) in enumerate(rkf.split(X)):
        t0 = time.time()
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr]
        yte, ytr_np = y.iloc[te].to_numpy(), ytr.to_numpy()
        c = np.quantile(ytr, RISK_Q)
        i_fit, i_cal = train_test_split(np.arange(len(Xtr)), test_size=0.2, random_state=SEED + fold)
        Xf, yf, Xc, yc = Xtr.iloc[i_fit], ytr.iloc[i_fit], Xtr.iloc[i_cal], ytr_np[i_cal]
        scale = np.std(yf)

        tab = FinePipe(args.tabpfn_estimators, SEED + fold).fit(Xf, yf)
        mc, qc = tab.predict_main(Xc)
        mt, qt = tab.predict_main(Xte)
        cb = catboost(SEED + fold).fit(Xf, yf)
        cbc, cbt = cb.predict(Xc), cb.predict(Xte)
        s_c = prob_below(qc, c) - 1e-6 * (mc - c) / scale
        s_t = prob_below(qt, c) - 1e-6 * (mt - c) / scale
        zc, zt = yc < c, yte < c

        if do_cv:  # out-of-fold scores for the whole training fold
            S, fold_id = np.empty(len(tr)), np.empty(len(tr), int)
            s_te_folds = np.empty((K_INNER, len(te)))
            ik = KFold(K_INNER, shuffle=True, random_state=SEED + fold)
            for f, (a, b) in enumerate(ik.split(Xtr)):
                m_in = FinePipe(args.tabpfn_estimators, SEED + 100 * fold + f).fit(Xtr.iloc[a], ytr.iloc[a])
                sc_in = np.std(ytr_np[a])
                mb, qb = m_in.predict_main(Xtr.iloc[b])
                S[b], fold_id[b] = prob_below(qb, c) - 1e-6 * (mb - c) / sc_in, f
                mtt, qtt = m_in.predict_main(Xte)
                s_te_folds[f] = prob_below(qtt, c) - 1e-6 * (mtt - c) / sc_in
            z_tr = ytr_np < c

        for aname, g_all in attrs.items():
            gtr, gte = g_all[tr], g_all[te]
            gc = gtr[i_cal]
            # ---------- intervals ----------
            cr_e = np.sort(np.abs(yc - cbc))[min(int(np.ceil((len(yc) + 1) * (1 - ALPHA))), len(yc)) - 1]
            e = cqr_e(qc[:, LO], qc[:, HI], yc, ALPHA)
            ivs = {"CQR-TabPFN (marginal)": (qt[:, LO] - e, qt[:, HI] + e),
                   "Split-conformal CatBoost": (cbt - cr_e, cbt + cr_e)}
            lo, hi = np.empty(len(yte)), np.empty(len(yte))
            for v in np.unique(gte):
                ev = cqr_e(qc[gc == v, LO], qc[gc == v, HI], yc[gc == v], ALPHA)
                lo[gte == v], hi[gte == v] = qt[gte == v, LO] - ev, qt[gte == v, HI] + ev
            ivs["FairWarn-PFN"] = (lo, hi)
            for name, (l, h) in ivs.items():
                cov, w = (yte >= l) & (yte <= h), h - l
                iv_rows.append({"fold": fold, "attr": aname, "Method": name, "group": "ALL",
                                "Coverage": cov.mean(), "Width": np.mean(w)})
                for v in np.unique(gte):
                    iv_rows.append({"fold": fold, "attr": aname, "Method": name, "group": v,
                                    "Coverage": cov[gte == v].mean(), "Width": np.mean(w[gte == v])})
            # ---------- screening ----------
            rules = {("TabPFN: y_hat < c", None): mt < c,
                     ("Hardt EO (TabPFN pi)", None): hardt_eo(s_c, zc, gc, s_t, gte, SEED + fold)}
            for ar in ALPHA_R:
                rules[("Marginal conformal (TabPFN pi)", ar)] = s_t >= screen_threshold(s_c[zc], ar)
                rules[("Plug-in group quantile (TabPFN pi)", ar)] = group_rule(s_c, zc, gc, s_t, gte, plugin_threshold(ar))
                rules[("FairWarn-PFN", ar)] = group_rule(s_c, zc, gc, s_t, gte, lambda s, a=ar: screen_threshold(s, a))
                if do_cv:
                    rules[("FairWarn-PFN-CV+", ar)] = cvplus_rule(S, fold_id, z_tr, gtr, s_te_folds, gte, ar)
            for (name, ar), flag in rules.items():
                sc_rows += screen_rows({"fold": fold, "attr": aname, "Rule": name, "alpha_r": ar}, flag, zt, gte)
        log(f"  fold {fold + 1}/{args.folds * args.repeats}  ({time.time() - t0:.0f}s)")

    iv, sc = pd.DataFrame(iv_rows), pd.DataFrame(sc_rows)
    sc["alpha_r"] = sc["alpha_r"].fillna(-1)
    iv.to_csv(out / "interval_folds.csv", index=False)
    sc.to_csv(out / "screening_folds.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
        log(iv.groupby(["attr", "Method", "group"])[["Coverage", "Width"]].mean().round(4).to_string())
        log(sc.groupby(["attr", "Rule", "alpha_r", "group"])[["Recall", "FlagRate", "Precision"]].mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tabpfn-estimators", type=int, default=8)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.repeats, args.tabpfn_estimators = 2, 1, 2
    base = ROOT / "results" / ("extensions_quick" if args.quick else "extensions")
    base.mkdir(parents=True, exist_ok=True)
    logf = open(base / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"\n==== extensions {time.strftime('%Y-%m-%d %H:%M')}  args={json.dumps(vars(args))}")
    t0 = time.time()
    for key in args.datasets:
        run_dataset(key, args, log)
    log(f"\nTotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
