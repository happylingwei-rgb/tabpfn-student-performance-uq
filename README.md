# Beyond point accuracy: TabPFN for student performance prediction with uncertainty quantification

Code and results for the manuscript *Beyond point accuracy: A multi-dataset, uncertainty-aware evaluation of tabular foundation models for student performance prediction and at-risk identification* (Ling-Wei Lin, National Taichung University of Science and Technology).

The study re-examines [Yin (2026), PLOS One 21(9): e0328634](https://doi.org/10.1371/journal.pone.0328634), which reported that TabPFN outperforms classical machine-learning models for predicting secondary-school exam scores. Compared with that study, we evaluate on four public datasets with leakage-free preprocessing and corrected statistical tests. We also look beyond point accuracy:

- conformalized quantile regression (CQR) on TabPFN's predictive distribution;
- identification of at-risk students from the predictive distribution;
- gender error gaps;
- the stability of SHAP explanations across folds.

We also propose **FairWarn-PFN** (Group-Fair Guaranteed Early Warning with Prior-data Fitted Networks), an early-warning model built on TabPFN's predictive distribution. It uses group-conditional (Mondrian) CQR intervals and recall-guaranteed conformal at-risk screening. Under exchangeability, the method guarantees interval coverage of at least 1 − α and at-risk recall of at least 1 − α_r within every group, in finite samples. Groups are defined by gender and by a socio-economic or location attribute (D1 lunch type, D2/D3 home address, D4 family income). For small groups we add a cross-conformal variant, **FairWarn-PFN-CV+**.

## Datasets

The datasets are public but not redistributed here. Download them with `python download_data.py`.

| ID | Dataset | Students | Target | Source |
|---|---|---|---|---|
| D1 | Students Performance in Exams | 1,000 | math score (scaled to [0, 1]) | [Kaggle](https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics) |
| D2 | UCI Student Performance – Mathematics | 395 | final grade G3 (0–20), without G1/G2 | [UCI #320](https://archive.ics.uci.edu/dataset/320/student+performance) |
| D3 | UCI Student Performance – Portuguese | 649 | final grade G3 (0–20), without G1/G2 | [UCI #320](https://archive.ics.uci.edu/dataset/320/student+performance) |
| D4 | Student Performance Factors | 6,607 | exam score | [Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) |

## Setup

Python 3.13 was used. No GPU is required: everything here was run on a laptop CPU (Intel i7-1255U, 24 GB RAM).

```bash
pip install -r requirements.txt
python download_data.py
```

TabPFN is run with the openly available **v2** weights (`ModelVersion.V2`), so no Prior Labs account or licence login is needed.

## Experiments

| Script | What it does | Output |
|---|---|---|
| `python reproduce.py --dataset math` | Replicates the original protocol on D1. This covers an 80/20 split, 10-fold grid search over the original S1 search spaces, bootstrap CIs, paired *t*/Wilcoxon tests, the sample-size experiment and exact TabPFN Shapley values (remove-and-recontextualize, all 2^7 − 1 subsets). | `results/math/` |
| `python benchmark.py` | Runs the main benchmark: 4 datasets × repeated 5-fold CV (3 repeats) × 10 models, Nadeau–Bengio corrected *t*-tests with Holm adjustment, CQR intervals, at-risk ROC/PR-AUC, gender RMSE gaps and CatBoost SHAP stability. On a CPU this takes about 9 hours. | `results/benchmark/<dataset>/` |
| `python benchmark.py --quick` | Smoke test (2 folds, 1 repeat). | `results/benchmark_quick/` |
| `python guarantee.py` | Runs FairWarn-PFN. It uses the same folds and calibration split as `benchmark.py` and compares group-conditional CQR with marginal CQR and split-conformal CatBoost. It also compares conformal screening (marginal and per-group, α_r ∈ {0.10, 0.20}, TabPFN P(y < c) and CatBoost scores) against naive thresholds. Per-student predictions are saved. On a CPU this takes about 4 hours. | `results/guarantee/<dataset>/` |
| `python extensions.py` | Extends FairWarn-PFN on the same folds: a second group attribute (socio-economic or location), fairness baselines (Hardt et al. equal-opportunity post-processing via fairlearn's `ThresholdOptimizer`, and per-group empirical quantiles without the finite-sample correction), and FairWarn-PFN-CV+ (cross-conformal screening with 5 inner folds, D1–D3 only). On a CPU this takes about 8 hours. | `results/extensions/<dataset>/` |

`reproduce.py --dataset sap4000` expects `data/SAP-4000.csv`. That file is the main dataset of the original study, and we could not find it publicly.

## Main results (`results/benchmark/`)

- **Accuracy.** TabPFN has the best mean RMSE rank across the four datasets (2.25), followed by the TabPFN+CatBoost blend (2.50). However, after correction only 13 of the 36 TabPFN-versus-model comparisons are significant (Holm-adjusted *p* < 0.05). All 13 are against tree ensembles or KNN. TabPFN is never significantly better than Ridge or SVR.
- **Prediction intervals (target 90%).** TabPFN's raw quantile intervals cover 0.849–0.978. CQR-TabPFN restores coverage to 0.898–0.916. Its intervals are also 5.4%–36.4% narrower than split-conformal CatBoost at comparable coverage.
- **At-risk students (bottom 20%).** On the two Portuguese datasets, the probability P(y < c) taken from TabPFN's predictive distribution gives the highest ROC-AUC (0.756 and 0.842).
- **SHAP stability.** The mean fold-to-fold Spearman correlation of feature rankings ranges from 0.853 (D2) to 0.993 (D4).

## FairWarn-PFN (`results/guarantee/`)

- **Per-group coverage.** FairWarn-PFN's group-conditional intervals cover 0.898–0.923 in each gender group, and no group is significantly below 0.90. Split-conformal CatBoost under-covers women on D4 (0.890, one-sided *p* = 0.019).
- **Per-group recall.** None of the 64 group recalls of the conformal screening rules is significantly below its target. Naive thresholds (ŷ < c) catch only 37%–50% of at-risk students on D2 and D3.
- **Efficiency.** On D2–D4, TabPFN's P(y < c) needs a lower flag rate than CatBoost to meet the same guarantee in all 12 comparisons. The reduction reaches up to 20% of flagged students. On D1 the two scores are on par. On small calibration sets, per-group screening becomes conservative: with fewer than 1/α_r − 1 at-risk calibration students in a group, every student in that group is flagged.

## Extensions (`results/extensions/`)

- **Second group attribute.** Across all 17 groups (gender plus lunch type, address or income), FairWarn-PFN's coverage is 0.898–0.937, and no group is significantly below 0.90. Split-conformal CatBoost under-covers the high-income group on D4 (0.881, *p* = 0.001).
- **Hidden under-recall.** On D1, marginal conformal screening meets the overall recall target (0.923), but recall in the standard-lunch group is only 0.849 at α_r = 0.10 and 0.638 at α_r = 0.20 (both significant). FairWarn-PFN raises these to 0.905 and 0.852. None of FairWarn-PFN's 34 group recalls is significantly below target.
- **Fairness baselines.** Hardt equal-opportunity post-processing equalises recall across groups, but it does not control the recall level. Overall recall is 0.58–0.63 on D2 and 0.74–0.75 on D3; on D4 it flags more students than FairWarn-PFN. Removing the finite-sample correction leaves 15 of 34 group recalls significantly below target.
- **FairWarn-PFN-CV+.** At α_r = 0.10, it lowers the flag rate on D1–D3 by 21.5%–39.3% (Wilcoxon *p* ≤ 0.011); for example, D2 by gender goes from 0.986 to 0.712. None of its 24 group recalls is significantly below target. Its worst-case theoretical bound is weaker (about 1 − 2α_r; Barber et al., 2021), so it is evaluated empirically.

## Licence

The code is released under the MIT licence. The datasets remain under the licences of their original providers.
