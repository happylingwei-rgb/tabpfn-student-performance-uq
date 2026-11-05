# Beyond point accuracy: TabPFN for student performance prediction with uncertainty quantification

Code and results for the manuscript *Beyond point accuracy: A multi-dataset, uncertainty-aware evaluation of tabular foundation models for student performance prediction and at-risk identification* (Ling-Wei Lin, National Taichung University of Science and Technology).

The study re-examines [Yin (2026), PLOS One 21(9): e0328634](https://doi.org/10.1371/journal.pone.0328634), which reported that TabPFN outperforms classical machine-learning models for predicting secondary-school exam scores. Compared with that study, we evaluate on four public datasets with leakage-free preprocessing and corrected statistical tests. We also look beyond point accuracy:

- conformalized quantile regression (CQR) on TabPFN's predictive distribution;
- identification of at-risk students from the predictive distribution;
- gender error gaps;
- the stability of SHAP explanations across folds.

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

`reproduce.py --dataset sap4000` expects `data/SAP-4000.csv`. That file is the main dataset of the original study, and we could not find it publicly.

## Main results (`results/benchmark/`)

- **Accuracy.** TabPFN has the best mean RMSE rank across the four datasets (2.25), followed by the TabPFN+CatBoost blend (2.50). However, after correction only 13 of the 36 TabPFN-versus-model comparisons are significant (Holm-adjusted *p* < 0.05). All 13 are against tree ensembles or KNN. TabPFN is never significantly better than Ridge or SVR.
- **Prediction intervals (target 90%).** TabPFN's raw quantile intervals cover 0.849–0.978. CQR-TabPFN restores coverage to 0.898–0.916. Its intervals are also 5.4%–36.4% narrower than split-conformal CatBoost at comparable coverage.
- **At-risk students (bottom 20%).** On the two Portuguese datasets, the probability P(y < c) taken from TabPFN's predictive distribution gives the highest ROC-AUC (0.756 and 0.842).
- **SHAP stability.** The mean fold-to-fold Spearman correlation of feature rankings ranges from 0.853 (D2) to 0.993 (D4).

## Licence

The code is released under the MIT licence. The datasets remain under the licences of their original providers.
