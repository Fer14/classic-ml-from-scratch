# Phase 04: ML Fundamentals

> Before you learn any algorithm, learn how to tell whether it works.

This phase covers the ideas every machine learning algorithm shares, independent of the model: what learning means, how to split data so your scores are honest, which loss to train on and which metric to report, why models overfit and how regularization and the bias-variance tradeoff explain it, how cross-validation gives stable estimates, and how to scale and encode features so models see fair numbers. Every later phase (regression, classification, trees, clustering, practical ML) relies on these tools, so you build each one from scratch in NumPy and then check it against scikit-learn.

**Prerequisites:** Phase 00 (Python Foundations), Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization), Phase 03 (Probability and Statistics)

| # | Lesson | What you build |
|---|---|---|
| 01 | [What Is Machine Learning](01-what-is-machine-learning/) | A learned threshold classifier, a nearest-centroid classifier and a least-squares regressor, compared with a hand-written rule |
| 02 | [Train, Validation, and Test Sets](02-train-validation-test/) | Three-way, stratified and time-ordered splits, plus a demo of how selecting on the test set inflates scores |
| 03 | [Loss Functions](03-loss-functions/) | MSE, MAE, Huber, cross-entropy and hinge losses, and line fits that show how each loss treats outliers |
| 04 | [Overfitting and Regularization](04-overfitting-and-regularization/) | Ridge regression in closed form on polynomial features, plus more data and early stopping as regularizers |
| 05 | [The Bias-Variance Tradeoff](05-bias-variance/) | A simulation that decomposes test error into bias^2, variance and noise for polynomials and trees |
| 06 | [Cross-Validation](06-cross-validation/) | K-fold, stratified k-fold and leave-one-out splitters, and a lambda search with cross-validated ridge |
| 07 | [Evaluation Metrics](07-metrics/) | Confusion matrix, precision, recall, F1, ROC AUC, average precision and regression metrics |
| 08 | [Feature Scaling and Encoding](08-feature-scaling-and-encoding/) | Standard, min-max and robust scalers, one-hot and ordinal encoders, and their effect on k-NN and gradient descent |

Run the first lesson:

```bash
uv run python phases/04-ml-fundamentals/01-what-is-machine-learning/code/main.py
```
