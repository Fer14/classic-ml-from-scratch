import numpy as np
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def train_test_split_simple(X, y, test_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test_idx, train_idx = order[:n_test], order[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class GaussianNBScratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        epsilon = self.var_smoothing * X.var(axis=0).max()
        self.means = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.variances = np.array([X[y == c].var(axis=0) + epsilon for c in self.classes])
        self.log_priors = np.log(np.array([np.mean(y == c) for c in self.classes]))
        return self

    def joint_log_likelihood(self, X):
        scores = []
        for mean, var, log_prior in zip(self.means, self.variances, self.log_priors):
            log_gauss = -0.5 * np.log(2 * np.pi * var) - (X - mean) ** 2 / (2 * var)
            scores.append(log_prior + log_gauss.sum(axis=1))
        return np.column_stack(scores)

    def predict_proba(self, X):
        jll = self.joint_log_likelihood(X)
        jll = jll - jll.max(axis=1, keepdims=True)
        probs = np.exp(jll)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes[np.argmax(self.joint_log_likelihood(X), axis=1)]


class MultinomialNBScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        counts = np.array([X[y == c].sum(axis=0) for c in self.classes]) + self.alpha
        self.log_word_probs = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.log_priors = np.log(np.array([np.mean(y == c) for c in self.classes]))
        return self

    def joint_log_likelihood(self, X):
        return X @ self.log_word_probs.T + self.log_priors

    def predict(self, X):
        return self.classes[np.argmax(self.joint_log_likelihood(X), axis=1)]


def build_vocabulary(documents):
    words = sorted({word for doc in documents for word in doc.split()})
    return {word: i for i, word in enumerate(words)}


def bag_of_words(documents, vocabulary):
    matrix = np.zeros((len(documents), len(vocabulary)))
    for row, doc in enumerate(documents):
        for word in doc.split():
            if word in vocabulary:
                matrix[row, vocabulary[word]] += 1
    return matrix


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    print("=== Step 1: Bayes rule on one feature ===")
    prior_spam = 0.3
    p_free_given_spam = 0.6
    p_free_given_ham = 0.05
    evidence = p_free_given_spam * prior_spam + p_free_given_ham * (1 - prior_spam)
    posterior = p_free_given_spam * prior_spam / evidence
    print(f"  P(spam | 'free') = {posterior:.4f}")

    print("\n=== Step 2: Gaussian Naive Bayes on the wine dataset ===")
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split_simple(wine.data, wine.target, 0.3, seed=0)
    gnb = GaussianNBScratch().fit(X_train, y_train)
    print(f"  {len(gnb.classes)} classes, {X_train.shape[1]} features")
    print(f"  priors = {np.round(np.exp(gnb.log_priors), 3)}")
    print(f"  class means of alcohol = {np.round(gnb.means[:, 0], 3)}")
    gnb_pred = gnb.predict(X_test)
    print(f"  test accuracy = {accuracy(y_test, gnb_pred):.4f}")
    print(f"  first 3 test posteriors:\n{np.round(gnb.predict_proba(X_test[:3]), 4)}")

    print("\n=== Step 3: why we work in log space ===")
    raw_product = np.prod(np.full(500, 0.01))
    log_sum = np.sum(np.log(np.full(500, 0.01)))
    print(f"  product of 500 probabilities of 0.01 = {raw_product}")
    print(f"  sum of their logs = {log_sum:.2f}")

    print("\n=== Step 4: Multinomial Naive Bayes for text ===")
    train_docs = [
        "win free money now",
        "free prize claim now",
        "cheap pills free offer",
        "win a free trip",
        "claim your prize money",
        "meeting moved to monday",
        "lunch with the team today",
        "project report due friday",
        "can we review the report",
        "team meeting about the project",
    ]
    train_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    test_docs = ["free money offer", "report for the meeting", "claim free lunch", "project prize"]
    vocabulary = build_vocabulary(train_docs)
    X_text_train = bag_of_words(train_docs, vocabulary)
    X_text_test = bag_of_words(test_docs, vocabulary)
    print(f"  vocabulary size = {len(vocabulary)}")
    mnb = MultinomialNBScratch(alpha=1.0).fit(X_text_train, train_labels)
    mnb_pred = mnb.predict(X_text_test)
    for doc, label in zip(test_docs, mnb_pred):
        print(f"  '{doc}' -> {'spam' if label == 1 else 'ham'}")

    print("\n=== Step 5: Laplace smoothing ===")
    no_smoothing = MultinomialNBScratch(alpha=1e-12).fit(X_text_train, train_labels)
    lunch = vocabulary["lunch"]
    print(f"  P('lunch' | spam), alpha~0: {np.exp(no_smoothing.log_word_probs[1, lunch]):.2e}")
    print(f"  P('lunch' | spam), alpha=1: {np.exp(mnb.log_word_probs[1, lunch]):.4f}")
    for name, model in [("alpha~0", no_smoothing), ("alpha=1", mnb)]:
        jll = model.joint_log_likelihood(X_text_test[2:3])[0]
        p_spam = 1.0 / (1.0 + np.exp(jll[0] - jll[1]))
        print(f"  'claim free lunch' {name}: log scores {np.round(jll, 2)} -> P(spam) = {p_spam:.10f}")

    print("\n=== Use It: scikit-learn ===")
    sk_gnb = GaussianNB().fit(X_train, y_train)
    sk_gnb_pred = sk_gnb.predict(X_test)
    print(f"  GaussianNB test accuracy = {accuracy(y_test, sk_gnb_pred):.4f}")
    print(f"  agreement with scratch = {np.mean(sk_gnb_pred == gnb_pred):.4f}")
    print(f"  max prob difference = {np.abs(sk_gnb.predict_proba(X_test) - gnb.predict_proba(X_test)).max():.2e}")
    sk_mnb = MultinomialNB(alpha=1.0).fit(X_text_train, train_labels)
    sk_mnb_pred = sk_mnb.predict(X_text_test)
    print(f"  MultinomialNB predictions = {sk_mnb_pred}, scratch = {mnb_pred}")


if __name__ == "__main__":
    main()
