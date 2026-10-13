# Phase 06: Supervised Classification

> Draw the boundary between classes, then learn how to bend it.

This phase covers the classic classifiers that every practitioner should be able to build by hand. You start with probabilistic linear models (logistic regression), move to instance-based learning (k-nearest neighbors), then to generative models that model each class with a probability distribution (naive Bayes, LDA, QDA). From there you go back to geometry: the perceptron, maximum-margin SVMs, and the kernel trick that turns linear models into nonlinear ones. The phase closes with the strategies that turn any binary classifier into a multiclass one. Every lesson builds the algorithm in NumPy first and then checks it against scikit-learn.

**Prerequisites:** Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization), Phase 03 (Probability and Statistics), Phase 04 (ML Fundamentals), Phase 05 (Supervised Regression)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Logistic Regression](01-logistic-regression/) | A sigmoid + cross-entropy binary classifier trained with gradient descent |
| 02 | [K-Nearest Neighbors](02-k-nearest-neighbors/) | A vectorized kNN classifier with distance weighting and a sweep over k |
| 03 | [Naive Bayes](03-naive-bayes/) | Gaussian and multinomial naive Bayes with log-probabilities and smoothing |
| 04 | [LDA and QDA](04-lda-and-qda/) | Linear and quadratic discriminant analysis from class means and covariances |
| 05 | [Perceptron](05-perceptron/) | The mistake-driven perceptron, its averaged variant, and the XOR failure case |
| 06 | [Support Vector Machines](06-support-vector-machines/) | A soft-margin linear SVM trained with hinge-loss subgradient descent |
| 07 | [Kernel Trick](07-kernel-trick/) | A kernel SVM solved with simplified SMO, with linear, polynomial, and RBF kernels |
| 08 | [Multiclass Strategies](08-multiclass-strategies/) | One-vs-rest, one-vs-one, and softmax regression on a 4-class problem |

Run the first lesson:

```bash
uv run python phases/06-supervised-classification/01-logistic-regression/code/main.py
```
