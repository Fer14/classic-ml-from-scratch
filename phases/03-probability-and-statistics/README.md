# Phase 03: Probability and Statistics

> Every model makes assumptions about randomness. This phase teaches you to read them.

Machine learning models are statistical models. Classifiers output probabilities, loss functions are negative log-likelihoods, regularization is a prior, and every reported score is an estimate with uncertainty around it. This phase builds the probability and statistics you need to understand those ideas instead of memorizing them: events and conditional probability, distributions and how to sample from them, expectation and variance, Bayes' theorem, maximum likelihood, the central limit theorem with confidence intervals and the bootstrap, and hypothesis testing for A/B tests and model comparisons. Everything is built from scratch in NumPy and then checked against `scipy.stats`.

**Prerequisites:** Phase 00 (Python Foundations), Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [Probability Basics](01-probability-basics/) | Exact and simulated dice probabilities, conditional probability, and independence checks |
| 02 | [Random Variables and Distributions](02-random-variables-and-distributions/) | PMFs, PDFs, and CDFs for six distributions, plus inverse transform and Box-Muller samplers |
| 03 | [Expectation and Variance](03-expectation-and-variance/) | Expectation, variance, Bessel's correction, covariance, correlation, and a covariance matrix |
| 04 | [Bayes' Theorem](04-bayes-theorem/) | A medical-test calculator, sequential updating, and a grid posterior for a coin's bias |
| 05 | [Maximum Likelihood](05-maximum-likelihood/) | MLE by grid search, closed form, and gradient ascent, and the proof that MSE is Gaussian MLE |
| 06 | [Sampling and the Central Limit Theorem](06-sampling-and-central-limit-theorem/) | Sampling distributions, standard errors, confidence interval coverage, and a bootstrap |
| 07 | [Hypothesis Testing](07-hypothesis-testing/) | z-test, Welch t-test, and permutation test for an A/B test, plus power and multiple-testing simulations |

Run the first lesson:

```bash
uv run python phases/03-probability-and-statistics/01-probability-basics/code/main.py
```
