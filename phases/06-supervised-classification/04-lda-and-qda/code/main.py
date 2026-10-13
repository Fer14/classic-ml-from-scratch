import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def train_test_split_simple(X, y, test_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_curved_classes(n_per_class, seed):
    rng = np.random.default_rng(seed)
    tight = rng.multivariate_normal([0, 0], [[0.3, 0.0], [0.0, 0.3]], n_per_class)
    wide = rng.multivariate_normal([0.5, 0.5], [[4.0, 1.5], [1.5, 2.0]], n_per_class)
    X = np.vstack([tight, wide])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


class LDAScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.priors = np.array([np.mean(y == c) for c in self.classes])
        pooled = np.zeros((n_features, n_features))
        for c, mean in zip(self.classes, self.means):
            centered = X[y == c] - mean
            pooled += centered.T @ centered
        self.covariance = pooled / (n_samples - len(self.classes))
        precision = np.linalg.inv(self.covariance)
        self.coef = self.means @ precision
        self.intercept = -0.5 * np.sum(self.coef * self.means, axis=1) + np.log(self.priors)
        return self

    def decision_function(self, X):
        return X @ self.coef.T + self.intercept

    def predict(self, X):
        return self.classes[np.argmax(self.decision_function(X), axis=1)]

    def projection_directions(self, n_components):
        overall_mean = self.means.T @ self.priors
        between = sum(
            p * np.outer(m - overall_mean, m - overall_mean) for p, m in zip(self.priors, self.means)
        )
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(self.covariance) @ between)
        order = np.argsort(eigenvalues.real)[::-1]
        return eigenvectors.real[:, order[:n_components]], eigenvalues.real[order[:n_components]]


class QDAScratch:
    def __init__(self, reg=0.0):
        self.reg = reg

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.means, self.precisions, self.log_dets, self.log_priors = [], [], [], []
        for c in self.classes:
            Xc = X[y == c]
            mean = Xc.mean(axis=0)
            cov = np.cov(Xc, rowvar=False) + self.reg * np.eye(n_features)
            sign, log_det = np.linalg.slogdet(cov)
            self.means.append(mean)
            self.precisions.append(np.linalg.inv(cov))
            self.log_dets.append(log_det)
            self.log_priors.append(np.log(len(Xc) / len(X)))
        return self

    def decision_function(self, X):
        scores = []
        for mean, precision, log_det, log_prior in zip(
            self.means, self.precisions, self.log_dets, self.log_priors
        ):
            centered = X - mean
            mahalanobis = np.einsum("ij,jk,ik->i", centered, precision, centered)
            scores.append(-0.5 * log_det - 0.5 * mahalanobis + log_prior)
        return np.column_stack(scores)

    def predict(self, X):
        return self.classes[np.argmax(self.decision_function(X), axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def standardize(X_train, X_test):
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def main():
    print("=== Step 1: LDA on the wine dataset ===")
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split_simple(wine.data, wine.target, 0.3, seed=0)
    X_train, X_test = standardize(X_train, X_test)
    lda = LDAScratch().fit(X_train, y_train)
    lda_pred = lda.predict(X_test)
    print(f"  shared covariance shape = {lda.covariance.shape}")
    print(f"  LDA test accuracy = {accuracy(y_test, lda_pred):.4f}")

    print("\n=== Step 2: QDA on the wine dataset ===")
    qda = QDAScratch(reg=0.0).fit(X_train, y_train)
    qda_pred = qda.predict(X_test)
    n_params_lda = 13 * 14 // 2
    n_params_qda = 3 * n_params_lda
    print(f"  QDA test accuracy = {accuracy(y_test, qda_pred):.4f}")
    print(f"  covariance parameters: LDA {n_params_lda}, QDA {n_params_qda}")

    print("\n=== Step 3: when the covariances differ, QDA wins ===")
    X_curve, y_curve = make_curved_classes(300, seed=1)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split_simple(X_curve, y_curve, 0.3, seed=2)
    lda_c = LDAScratch().fit(Xc_train, yc_train)
    qda_c = QDAScratch().fit(Xc_train, yc_train)
    print(f"  LDA accuracy (linear boundary)    = {accuracy(yc_test, lda_c.predict(Xc_test)):.4f}")
    print(f"  QDA accuracy (quadratic boundary) = {accuracy(yc_test, qda_c.predict(Xc_test)):.4f}")

    print("\n=== Step 4: LDA as dimensionality reduction ===")
    directions, eigenvalues = lda.projection_directions(2)
    projected = X_train @ directions
    explained = eigenvalues / eigenvalues.sum()
    print(f"  13 features -> 2 discriminant axes, separation share {np.round(explained, 3)}")
    for c in lda.classes:
        print(f"  class {c} centroid on axes: {np.round(projected[y_train == c].mean(axis=0), 3)}")

    print("\n=== Use It: scikit-learn ===")
    sk_lda = LinearDiscriminantAnalysis(solver="lsqr").fit(X_train, y_train)
    sk_qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    sk_lda_pred = sk_lda.predict(X_test)
    sk_qda_pred = sk_qda.predict(X_test)
    print(f"  sklearn LDA accuracy = {accuracy(y_test, sk_lda_pred):.4f} | agreement {np.mean(sk_lda_pred == lda_pred):.4f}")
    print(f"  sklearn QDA accuracy = {accuracy(y_test, sk_qda_pred):.4f} | agreement {np.mean(sk_qda_pred == qda_pred):.4f}")
    sk_qda_c = QuadraticDiscriminantAnalysis().fit(Xc_train, yc_train)
    print(f"  sklearn QDA on curved data = {accuracy(yc_test, sk_qda_c.predict(Xc_test)):.4f}")
    reduced = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
    print(f"  sklearn explained_variance_ratio_ = {np.round(reduced.explained_variance_ratio_, 3)}")


if __name__ == "__main__":
    main()
