"""Guaranteed, group-fair early warning with TabPFN (proposed method).

Two finite-sample guarantees that hold for any model and any data distribution
under exchangeability, built on TabPFN's predictive distribution:

  (1) Group-conditional (Mondrian) CQR intervals: P(y in C(x) | G = g) >= 1 - alpha
      for every gender group g.
  (2) Recall-guaranteed at-risk screening ("equal-opportunity conformal screening"):
      flag x if s(x) >= lambda_g, with lambda_g calibrated on the at-risk students of
      group g, so that P(student is missed | at-risk, G = g) <= alpha_r for every g.

Same folds / proper-train / calibration split as benchmark.py, so the numbers are
directly comparable. Per-student predictions are saved for re-analysis.

Usage:
  python guarantee.py                         # all datasets
  python guarantee.py --datasets math uci_por
  python guarantee.py --quick
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, train_test_split

from benchmark import ALPHA, DATASETS, RISK_Q, ROOT, SEED, TabPFNPipe, catboost

# fine quantile grid so that P(y < c) is resolved in the tails as well
QFINE = sorted(set(np.round(np.concatenate([[0.001, 0.002, 0.003, 0.004], np.arange(0.005, 0.996, 0.005),
                                             [0.996, 0.997, 0.998, 0.999]]), 3).tolist()))
LO, HI = QFINE.index(round(ALPHA / 2, 3)), QFINE.index(round(1 - ALPHA / 2, 3))
ALPHA_R = (0.10, 0.20)  # screening: guaranteed recall of at least 90 % / 80 %


class FinePipe(TabPFNPipe):
    def predict_main(self, X):
        out = self.model.predict(self.prep.transform(X), output_type="main", quantiles=QFINE)
        return np.asarray(out["mean"]), np.vstack(out["quantiles"]).T


def prob_below(q, c):
    """pi(x) = F(c | x), piecewise-linear CDF through the predictive quantiles."""
    lv = np.array(QFINE)
    return np.array([np.interp(c, np.sort(row), lv, left=0.0, right=1.0) for row in q])


def cqr_e(lo, hi, y, alpha):
    """Finite-sample CQR correction; +inf if the calibration set is too small."""
    s = np.maximum(lo - y, y - hi)
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return np.inf if k > len(s) else np.sort(s)[k - 1]


def screen_threshold(s_atrisk, alpha):
    """lambda = k-th smallest score of calibration at-risk students, k = floor(alpha (n1+1)).
    Flag if s >= lambda  =>  P(at-risk student missed) <= k / (n1 + 1) <= alpha."""
    k = int(np.floor(alpha * (len(s_atrisk) + 1)))
    return -np.inf if k == 0 else np.sort(s_atrisk)[k - 1]


def screen_metrics(flag, z, g):
    out = {"FlagRate": flag.mean(), "Recall": flag[z].mean() if z.any() else np.nan,
           "Precision": z[flag].mean() if flag.any() else np.nan}
    for v, name in ((0, "M"), (1, "F")):
        m = z & (g == v)
        out[f"Recall_{name}"] = flag[m].mean() if m.any() else np.nan
    out["RecallGap"] = abs(out["Recall_M"] - out["Recall_F"])
    return out


def run_dataset(key, args, log):
    title, loader = DATASETS[key]
    X, y, g = loader()
    out = ROOT / "results" / ("guarantee_quick" if args.quick else "guarantee") / key
    out.mkdir(parents=True, exist_ok=True)
    log(f"\n##### {key}: {title}  n={len(X)}")

    rkf = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=SEED)
    iv_rows, sc_rows, pred_rows = [], [], []
    for fold, (tr, te) in enumerate(rkf.split(X)):
        t0 = time.time()
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr]
        yte, gte = y.iloc[te].to_numpy(), g.iloc[te].to_numpy()
        c = np.quantile(ytr, RISK_Q)
        i_fit, i_cal = train_test_split(np.arange(len(Xtr)), test_size=0.2, random_state=SEED + fold)
        Xf, yf, Xc = Xtr.iloc[i_fit], ytr.iloc[i_fit], Xtr.iloc[i_cal]
        yc, gc = ytr.iloc[i_cal].to_numpy(), g.iloc[tr].iloc[i_cal].to_numpy()

        tab = FinePipe(args.tabpfn_estimators, SEED + fold).fit(Xf, yf)
        mc, qc = tab.predict_main(Xc)
        mt, qt = tab.predict_main(Xte)
        cb = catboost(SEED + fold).fit(Xf, yf)
        cbc, cbt = cb.predict(Xc), cb.predict(Xte)

        # ---------- (1) prediction intervals ----------
        scale = np.std(yf)
        cr_e = np.sort(np.abs(yc - cbc))[min(int(np.ceil((len(yc) + 1) * (1 - ALPHA))), len(yc)) - 1]
        e = cqr_e(qc[:, LO], qc[:, HI], yc, ALPHA)
        ivs = {
            "CQR-TabPFN (marginal)": (qt[:, LO] - e, qt[:, HI] + e),
            "Split-conformal CatBoost": (cbt - cr_e, cbt + cr_e),
        }
        lo, hi = np.empty(len(yte)), np.empty(len(yte))
        for v in (0, 1):
            ev = cqr_e(qc[gc == v, LO], qc[gc == v, HI], yc[gc == v], ALPHA)
            lo[gte == v], hi[gte == v] = qt[gte == v, LO] - ev, qt[gte == v, HI] + ev
        ivs["Mondrian CQR-TabPFN (proposed)"] = (lo, hi)
        for name, (lo, hi) in ivs.items():
            cov = (yte >= lo) & (yte <= hi)
            iv_rows.append({"fold": fold, "Method": name, "Coverage": cov.mean(), "Width": np.mean(hi - lo),
                            "Coverage_M": cov[gte == 0].mean(), "Coverage_F": cov[gte == 1].mean(),
                            "Width_M": np.mean((hi - lo)[gte == 0]), "Width_F": np.mean((hi - lo)[gte == 1]),
                            "CoverageGap": abs(cov[gte == 0].mean() - cov[gte == 1].mean())})

        # ---------- (2) at-risk screening ----------
        zc, zt = yc < c, yte < c
        # tie-break pi with the point prediction (a fixed score function, so validity is unaffected)
        s_tab_c = prob_below(qc, c) - 1e-6 * (mc - c) / scale
        s_tab_t = prob_below(qt, c) - 1e-6 * (mt - c) / scale
        s_cb_c, s_cb_t = (c - cbc) / scale, (c - cbt) / scale
        rules = {
            "TabPFN: y_hat < c": mt < c,
            "TabPFN: pi >= 0.5": prob_below(qt, c) >= 0.5,
            "CatBoost: y_hat < c": cbt < c,
        }
        for ar in ALPHA_R:
            for sname, sc_, st_ in (("TabPFN pi", s_tab_c, s_tab_t), ("CatBoost", s_cb_c, s_cb_t)):
                lam = screen_threshold(sc_[zc], ar)
                rules[f"{sname}: conformal marginal (a={ar})"] = st_ >= lam
                flag = np.zeros(len(yte), bool)
                for v in (0, 1):
                    lam_v = screen_threshold(sc_[zc & (gc == v)], ar)
                    flag[gte == v] = st_[gte == v] >= lam_v
                rules[f"{sname}: conformal group (a={ar})"] = flag
        for name, flag in rules.items():
            sc_rows.append({"fold": fold, "Rule": name, **screen_metrics(flag, zt, gte)})

        pred_rows.append(pd.DataFrame({"fold": fold, "idx": te, "y": yte, "gender_F": gte, "c": c,
                                       "tabpfn_mean": mt, "tabpfn_q05": qt[:, LO], "tabpfn_q95": qt[:, HI],
                                       "tabpfn_pi": prob_below(qt, c), "catboost": cbt}))
        log(f"  fold {fold + 1}/{args.folds * args.repeats}  calib n={len(yc)} at-risk={zc.sum()}  ({time.time() - t0:.0f}s)")

    iv, sc = pd.DataFrame(iv_rows), pd.DataFrame(sc_rows)
    iv.to_csv(out / "interval_folds.csv", index=False)
    sc.to_csv(out / "screening_folds.csv", index=False)
    pd.concat(pred_rows).to_csv(out / "predictions.csv", index=False)
    iv_s = iv.drop(columns="fold").groupby("Method").agg(["mean", "std"])
    sc_s = sc.drop(columns="fold").groupby("Rule").agg(["mean", "std"])
    iv_s.to_csv(out / "interval_summary.csv")
    sc_s.to_csv(out / "screening_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log(iv.drop(columns="fold").groupby("Method").mean().round(4).to_string())
        log(sc.drop(columns="fold").groupby("Rule").mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tabpfn-estimators", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.repeats, args.tabpfn_estimators = 2, 1, 2
    base = ROOT / "results" / ("guarantee_quick" if args.quick else "guarantee")
    base.mkdir(parents=True, exist_ok=True)
    logf = open(base / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"\n==== guarantee {time.strftime('%Y-%m-%d %H:%M')}  args={json.dumps(vars(args))}")
    t0 = time.time()
    for key in args.datasets:
        run_dataset(key, args, log)
    log(f"\nTotal {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
