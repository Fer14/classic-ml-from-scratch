import numpy as np
from sklearn.model_selection import train_test_split


def make_data(n_samples, rng):
    x = rng.uniform(-3, 3, n_samples)
    y = np.sin(x) + 0.3 * x + rng.normal(0, 0.3, n_samples)
    return x, y


def split_indices(n_samples, fractions, rng):
    order = rng.permutation(n_samples)
    train_end = int(fractions[0] * n_samples)
    val_end = train_end + int(fractions[1] * n_samples)
    return order[:train_end], order[train_end:val_end], order[val_end:]


def stratified_split(labels, test_fraction, rng):
    train_parts, test_parts = [], []
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        n_test = int(round(test_fraction * len(members)))
        test_parts.append(members[:n_test])
        train_parts.append(members[n_test:])
    return np.concatenate(train_parts), np.concatenate(test_parts)


def polynomial_design(x, degree):
    return np.vander(x, degree + 1, increasing=True)


def fit_polynomial(x, y, degree):
    return np.linalg.lstsq(polynomial_design(x, degree), y, rcond=None)[0]


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def main():
    rng = np.random.default_rng(42)
    x, y = make_data(300, rng)

    print("=== Step 1: three-way split ===")
    train_idx, val_idx, test_idx = split_indices(len(x), (0.6, 0.2), rng)
    print(f"train={len(train_idx)}  validation={len(val_idx)}  test={len(test_idx)}")
    print(f"overlap train/val: {len(np.intersect1d(train_idx, val_idx))}, train/test: {len(np.intersect1d(train_idx, test_idx))}")

    print("\n=== Step 2: choose the degree on validation, never on test ===")
    results = []
    for degree in range(1, 13):
        coefficients = fit_polynomial(x[train_idx], y[train_idx], degree)
        train_error = mse(y[train_idx], polynomial_design(x[train_idx], degree) @ coefficients)
        val_error = mse(y[val_idx], polynomial_design(x[val_idx], degree) @ coefficients)
        results.append((degree, train_error, val_error))
        print(f"degree {degree:2d} | train MSE {train_error:.4f} | val MSE {val_error:.4f}")
    best_degree = min(results, key=lambda row: row[2])[0]
    print(f"Best degree on validation: {best_degree}")

    print("\n=== Step 3: refit on train+val, evaluate once on test ===")
    final_idx = np.concatenate([train_idx, val_idx])
    final_coefficients = fit_polynomial(x[final_idx], y[final_idx], best_degree)
    test_error = mse(y[test_idx], polynomial_design(x[test_idx], best_degree) @ final_coefficients)
    print(f"Test MSE for degree {best_degree}: {test_error:.4f}")

    print("\n=== Step 4: why peeking at the test set lies to you ===")
    small_test_labels = rng.integers(0, 2, 50)
    fresh_labels = rng.integers(0, 2, 5000)
    candidate_scores = []
    for candidate in range(500):
        candidate_rng = np.random.default_rng(candidate)
        test_guess = candidate_rng.integers(0, 2, 50)
        fresh_guess = candidate_rng.integers(0, 2, 5000)
        candidate_scores.append((np.mean(test_guess == small_test_labels), np.mean(fresh_guess == fresh_labels)))
    best_test, best_fresh = max(candidate_scores, key=lambda row: row[0])
    print("500 coin-flip 'models', labels are pure noise, best one picked by test accuracy")
    print(f"accuracy on the test set used to pick it: {best_test:.3f}")
    print(f"accuracy on 5000 fresh points:            {best_fresh:.3f}")

    print("\n=== Step 5: stratified split for imbalanced labels ===")
    labels = (rng.uniform(size=1000) < 0.1).astype(int)
    plain_train, plain_val, plain_test = split_indices(1000, (0.8, 0.0), np.random.default_rng(7))
    strat_train, strat_test = stratified_split(labels, 0.2, np.random.default_rng(7))
    print(f"overall positive rate:        {labels.mean():.3f}")
    print(f"random split test rate:       {labels[plain_test].mean():.3f}")
    print(f"stratified split test rate:   {labels[strat_test].mean():.3f}")

    print("\n=== Step 6: time-ordered split ===")
    n_days = 100
    cutoff = int(0.8 * n_days)
    print(f"train days 0..{cutoff - 1}, test days {cutoff}..{n_days - 1} (no shuffling for time series)")

    print("\n=== Use It: scikit-learn ===")
    indices = np.arange(len(x))
    temp_idx, sk_test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    sk_train_idx, sk_val_idx = train_test_split(temp_idx, test_size=0.25, random_state=42)
    print(f"sklearn train={len(sk_train_idx)}  validation={len(sk_val_idx)}  test={len(sk_test_idx)}")
    _, sk_strat_test = train_test_split(np.arange(1000), test_size=0.2, stratify=labels, random_state=42)
    print(f"sklearn stratified test positive rate: {labels[sk_strat_test].mean():.3f}")


if __name__ == "__main__":
    main()
