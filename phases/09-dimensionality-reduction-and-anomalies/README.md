# Phase 09: Dimensionality Reduction and Anomalies

> Fewer dimensions, clearer pictures, and a sharp eye for what does not belong.

Real datasets have many correlated features, hidden structure that no single column shows, and rare events that matter more than the rest. This phase gives you the unsupervised tools for all three. You build PCA and SVD from the linear algebra of Phase 01, then move to nonlinear neighbor embeddings (t-SNE and the ideas behind UMAP) for visualization. You finish with two practical unsupervised tasks: flagging anomalies with robust statistics and isolation forests, and mining association rules from transaction data with Apriori. Every algorithm is built from scratch in NumPy and then checked against scikit-learn.

**Prerequisites:** Phase 01 (Linear Algebra), Phase 02 (Calculus and Optimization), Phase 03 (Probability and Statistics), Phase 07 (Trees and Ensembles), Phase 08 (Unsupervised Clustering)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [PCA](01-pca/) | PCA from the covariance eigen-decomposition, explained variance, reconstruction and loadings on iris |
| 02 | [SVD](02-svd/) | SVD by power iteration and deflation, low-rank image compression, latent factors in a ratings matrix |
| 03 | [t-SNE](03-t-sne/) | Exact t-SNE with perplexity calibration, early exaggeration and gains, compared with PCA on digits |
| 04 | [UMAP Intuition](04-umap-intuition/) | A tiny UMAP: fuzzy k-NN graph, fitted similarity curve, negative-sampling layout, local vs global scores |
| 05 | [Anomaly Detection](05-anomaly-detection/) | z-score and robust MAD z-score, an isolation forest from scratch, evaluation with ROC AUC |
| 06 | [Association Rules](06-association-rules/) | Apriori with candidate pruning, rule generation with support, confidence and lift |

Run the first lesson:

```bash
uv run python phases/09-dimensionality-reduction-and-anomalies/01-pca/code/main.py
```
