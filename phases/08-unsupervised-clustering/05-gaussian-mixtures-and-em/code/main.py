import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def make_elongated_clusters(seed):
    rng = np.random.default_rng(seed)
    specs = [
        (np.array([0.0, 0.0]), np.array([[4.0, 3.6], [3.6, 4.0]]), 250),
        (np.array([4.0, -1.0]), np.array([[4.0, -3.6], [-3.6, 4.0]]), 250),
        (np.array([7.0, 4.0]), np.array([[0.3, 0.0], [0.0, 0.3]]), 100),
    ]
    points, labels = [], []
    for label, (mean, cov, size) in enumerate(specs):
        points.append(rng.multivariate_normal(mean, cov, size))
        labels.append(np.full(size, label))
    return np.vstack(points), np.concatenate(labels)


def log_gaussian(X, mean, cov):
    d = X.shape[1]
    cholesky = np.linalg.cholesky(cov)
    solved = np.linalg.solve(cholesky, (X - mean).T)
    mahalanobis_sq = np.sum(solved**2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(cholesky)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis_sq)


def logsumexp(values, axis):
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


class GaussianMixtureEM:
    def __init__(self, n_components, max_iter=200, tol=1e-6, reg_covar=1e-6, seed=0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed
        self.log_likelihoods = []

    def initialize(self, X):
        rng = np.random.default_rng(self.seed)
        self.means = X[rng.choice(len(X), self.n_components, replace=False)].copy()
        shared_cov = np.cov(X.T) + self.reg_covar * np.eye(X.shape[1])
        self.covariances = np.array([shared_cov.copy() for _ in range(self.n_components)])
        self.weights = np.full(self.n_components, 1.0 / self.n_components)

    def weighted_log_probs(self, X):
        return np.column_stack([
            np.log(self.weights[k]) + log_gaussian(X, self.means[k], self.covariances[k])
            for k in range(self.n_components)
        ])

    def e_step(self, X):
        weighted = self.weighted_log_probs(X)
        log_norm = logsumexp(weighted, axis=1)
        responsibilities = np.exp(weighted - log_norm[:, None])
        return responsibilities, float(np.mean(log_norm))

    def m_step(self, X, responsibilities):
        n, d = X.shape
        effective_counts = responsibilities.sum(axis=0)
        self.weights = effective_counts / n
        self.means = (responsibilities.T @ X) / effective_counts[:, None]
        for k in range(self.n_components):
            centered = X - self.means[k]
            weighted_cov = (responsibilities[:, k, None] * centered).T @ centered / effective_counts[k]
            self.covariances[k] = weighted_cov + self.reg_covar * np.eye(d)

    def fit(self, X):
        self.initialize(X)
        previous = -np.inf
        for _ in range(self.max_iter):
            responsibilities, mean_log_likelihood = self.e_step(X)
            self.log_likelihoods.append(mean_log_likelihood)
            if mean_log_likelihood - previous < self.tol:
                break
            previous = mean_log_likelihood
            self.m_step(X, responsibilities)
        return self

    def predict_proba(self, X):
        return self.e_step(X)[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def bic(self, X):
        n, d = X.shape
        n_parameters = self.n_components * (d + d * (d + 1) / 2) + self.n_components - 1
        total_log_likelihood = np.sum(logsumexp(self.weighted_log_probs(X), axis=1))
        return -2 * total_log_likelihood + n_parameters * np.log(n)


def best_of(X, n_components, n_init):
    runs = [GaussianMixtureEM(n_components, seed=seed).fit(X) for seed in range(n_init)]
    return max(runs, key=lambda model: model.log_likelihoods[-1])


def main():
    X, y_true = make_elongated_clusters(seed=1)
    print(f"Data: {len(X)} points, two crossing elongated clusters and one small round one")

    print("\n=== EM for a Gaussian mixture, from scratch ===")
    model = best_of(X, 3, n_init=5)
    for step in [0, 1, 2, 5, 10, 20]:
        if step < len(model.log_likelihoods):
            print(f"  iteration {step:3d} | mean log-likelihood {model.log_likelihoods[step]:.5f}")
    print(f"  converged after {len(model.log_likelihoods)} iterations at {model.log_likelihoods[-1]:.5f}")
    order = np.argsort(model.means[:, 0])
    for k in order:
        cov = model.covariances[k]
        print(f"  weight {model.weights[k]:.3f} | mean [{model.means[k][0]:5.2f}, {model.means[k][1]:5.2f}] | "
              f"cov [[{cov[0, 0]:5.2f}, {cov[0, 1]:5.2f}], [{cov[1, 0]:5.2f}, {cov[1, 1]:5.2f}]]")

    gmm_labels = model.predict(X)
    kmeans_labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)
    print(f"\n  ARI GMM (from scratch): {adjusted_rand_score(y_true, gmm_labels):.3f}")
    print(f"  ARI k-means:            {adjusted_rand_score(y_true, kmeans_labels):.3f}")

    probabilities = model.predict_proba(X)
    uncertain = np.sum(probabilities.max(axis=1) < 0.9)
    print(f"  Points with max membership probability < 0.9: {uncertain} (where the clusters cross)")

    print("\n=== Choosing the number of components with BIC ===")
    for n_components in range(1, 7):
        bic = best_of(X, n_components, n_init=3).bic(X)
        print(f"  components {n_components} | BIC {bic:9.1f}")

    print("\n=== scikit-learn GaussianMixture ===")
    sk_model = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=0).fit(X)
    sk_order = np.argsort(sk_model.means_[:, 0])
    for k in sk_order:
        print(f"  weight {sk_model.weights_[k]:.3f} | mean [{sk_model.means_[k][0]:5.2f}, {sk_model.means_[k][1]:5.2f}]")
    print(f"  mean log-likelihood: sklearn {sk_model.score(X):.5f} | from scratch {model.log_likelihoods[-1]:.5f}")
    print(f"  BIC: sklearn {sk_model.bic(X):.1f} | from scratch {model.bic(X):.1f}")
    print(f"  ARI between label sets: {adjusted_rand_score(gmm_labels, sk_model.predict(X)):.3f}")


if __name__ == "__main__":
    main()
