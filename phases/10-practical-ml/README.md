# Phase 10: Practical ML

> Knowing the algorithms gets you a model. Knowing the practice gets you a model you can trust.

Every previous phase taught an algorithm. This phase teaches what happens around it in a real project: rare classes that make accuracy meaningless, columns that add cost without signal, hyperparameters that decide whether a model works at all, preprocessing that silently leaks validation data, and stakeholders who want to know why the model said what it said. Each lesson builds the technique from scratch in NumPy, then shows the scikit-learn tool you will use day to day. The phase ends with a complete project that puts every piece together in the right order.

**Prerequisites:** Phase 04 (ML Fundamentals), Phase 05 (Supervised Regression), Phase 06 (Supervised Classification), Phase 07 (Trees and Ensembles)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Imbalanced Data](01-imbalanced-data/) | Class weights, random over/undersampling, SMOTE and threshold tuning for a 5%-positive problem |
| 02 | [Feature Selection](02-feature-selection/) | Variance and correlation filters, forward selection and backward elimination with CV, compared with Lasso |
| 03 | [Hyperparameter Tuning](03-hyperparameter-tuning/) | Grid search and log-uniform random search with k-fold CV for RBF kernel ridge regression |
| 04 | [Scikit-learn Pipelines](04-sklearn-pipelines/) | A minimal Pipeline and column transformer, plus a demonstration of 81% "accuracy" on pure noise from leakage |
| 05 | [Interpretability](05-interpretability/) | Permutation importance, partial dependence, ICE curves and exact Shapley values by enumerating coalitions |
| 06 | [End-to-End Project](06-end-to-end-project/) | A full breast cancer classification project: framing, baselines, model comparison, threshold choice, test evaluation |

Run the first lesson:

```bash
uv run python phases/10-practical-ml/01-imbalanced-data/code/main.py
```
