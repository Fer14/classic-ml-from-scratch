# Phase 07: Trees and Ensembles

> One tree is easy to read and easy to fool. Hundreds of them, combined the right way, are the strongest models for tabular data.

This phase builds decision trees from first principles and then the ensemble methods that turn them into state-of-the-art tabular models. You start with CART: greedy, axis-aligned splits chosen by an impurity measure. You then control a tree's complexity with pruning, reduce its variance with bagging and random forests, and reduce its bias with boosting, from AdaBoost to gradient boosting to the histogram-based, second-order boosting behind XGBoost and LightGBM. Every algorithm is written from scratch in NumPy first and then compared with scikit-learn.

**Prerequisites:** Phase 02 (Calculus and Optimization), Phase 03 (Probability and Statistics), Phase 04 (ML Fundamentals), Phase 06 (Supervised Classification)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Decision Trees](01-decision-trees/) | A CART classifier and regressor from scratch: recursive greedy splits, leaf predictions, and tree traversal |
| 02 | [Entropy and Gini](02-entropy-and-gini/) | Impurity measures (entropy, Gini, misclassification error) and information gain for choosing splits |
| 03 | [Pruning](03-pruning/) | Pre-pruning (max depth, min samples) and cost-complexity post-pruning chosen by validation |
| 04 | [Bagging](04-bagging/) | Bootstrap-aggregated trees with out-of-bag error estimation |
| 05 | [Random Forests](05-random-forests/) | A random forest with per-split feature subsampling, OOB score, tree decorrelation measurement and impurity-based importance |
| 06 | [AdaBoost](06-adaboost/) | Discrete AdaBoost with weighted decision stumps, sample reweighting and a label-noise stress test |
| 07 | [Gradient Boosting](07-gradient-boosting/) | Gradient boosting for squared error and log-loss with shrinkage, subsampling and Newton leaf values |
| 08 | [XGBoost and LightGBM in Practice](08-xgboost-and-lightgbm-in-practice/) | Histogram binning and second-order gain from scratch, then HistGradientBoosting with early stopping, categorical features and monotonic constraints |

Run the first lesson:

```bash
uv run python phases/07-trees-and-ensembles/01-decision-trees/code/main.py
```
