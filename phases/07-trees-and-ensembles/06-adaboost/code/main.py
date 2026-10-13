import numpy as np
from sklearn.datasets import make_hastie_10_2, make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class WeightedStump:
    def fit(self, X, y, sample_weight):
        n_samples, n_features = X.shape
        best_error = np.inf
        for feature in range(n_features):
            order = np.argsort(X[:, feature])
            sorted_values = X[order, feature]
            sorted_labels = y[order]
            sorted_weights = sample_weight[order]
            positive_weight_left = np.concatenate(([0.0], np.cumsum(sorted_weights * (sorted_labels == 1))))
            negative_weight_left = np.concatenate(([0.0], np.cumsum(sorted_weights * (sorted_labels == -1))))
            total_positive = positive_weight_left[-1]
            total_negative = negative_weight_left[-1]
            error_left_negative = positive_weight_left + (total_negative - negative_weight_left)
            error_left_positive = negative_weight_left + (total_positive - positive_weight_left)
            boundaries = np.concatenate(([True], sorted_values[:-1] < sorted_values[1:], [True]))
            for polarity, errors in ((1, error_left_negative), (-1, error_left_positive)):
                masked = np.where(boundaries, errors, np.inf)
                i = int(np.argmin(masked))
                if masked[i] < best_error:
                    best_error = masked[i]
                    if i == 0:
                        threshold = sorted_values[0] - 1.0
                    elif i == n_samples:
                        threshold = sorted_values[-1] + 1.0
                    else:
                        threshold = (sorted_values[i - 1] + sorted_values[i]) / 2
                    self.feature, self.threshold, self.polarity = feature, threshold, polarity
        self.weighted_error = best_error / sample_weight.sum()
        return self

    def predict(self, X):
        right_side = X[:, self.feature] > self.threshold
        return np.where(right_side, self.polarity, -self.polarity)


class AdaBoost:
    def __init__(self, n_rounds=200):
        self.n_rounds = n_rounds

    def fit(self, X, y, X_eval=None, y_eval=None):
        n_samples = len(y)
        weights = np.full(n_samples, 1.0 / n_samples)
        self.stumps, self.alphas = [], []
        self.train_curve, self.eval_curve = [], []
        train_scores = np.zeros(n_samples)
        eval_scores = np.zeros(len(y_eval)) if y_eval is not None else None
        for _ in range(self.n_rounds):
            stump = WeightedStump().fit(X, y, weights)
            error = np.clip(stump.weighted_error, 1e-10, 1 - 1e-10)
            if error >= 0.5:
                break
            alpha = 0.5 * np.log((1 - error) / error)
            predictions = stump.predict(X)
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / weights.sum()
            self.stumps.append(stump)
            self.alphas.append(alpha)
            train_scores += alpha * predictions
            self.train_curve.append(np.mean(np.sign(train_scores) != y))
            if eval_scores is not None:
                eval_scores += alpha * stump.predict(X_eval)
                self.eval_curve.append(np.mean(np.sign(eval_scores) != y_eval))
        self.final_weights = weights
        return self

    def decision_function(self, X):
        return sum(alpha * stump.predict(X) for alpha, stump in zip(self.alphas, self.stumps))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def main():
    X, y01 = make_hastie_10_2(n_samples=3000, random_state=1)
    y = y01.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    print("=== Step 1: a single weighted stump ===")
    uniform = np.full(len(y_train), 1.0 / len(y_train))
    stump = WeightedStump().fit(X_train, y_train, uniform)
    stump_error = np.mean(stump.predict(X_test) != y_test)
    print(f"feature={stump.feature}  threshold={stump.threshold:.3f}  polarity={stump.polarity:+d}")
    print(f"stump test error: {stump_error:.3f}")

    print("\n=== Step 2: AdaBoost with 300 stumps ===")
    model = AdaBoost(n_rounds=300).fit(X_train, y_train, X_test, y_test)
    for rounds in [1, 10, 50, 100, 200, 300]:
        print(f"rounds={rounds:3d}  train err={model.train_curve[rounds - 1]:.3f}  test err={model.eval_curve[rounds - 1]:.3f}")
    print(f"first alphas: {np.round(model.alphas[:5], 3).tolist()}")

    print("\n=== Step 3: where does the weight go? ===")
    margins = y_train * model.decision_function(X_train) / np.sum(model.alphas)
    heaviest = np.argsort(model.final_weights)[::-1][:200]
    print(f"mean |margin| of all points:          {np.mean(np.abs(margins)):.3f}")
    print(f"mean |margin| of 200 heaviest points: {np.mean(np.abs(margins[heaviest])):.3f}")
    print(f"weight held by heaviest 10% of rows:  {np.sort(model.final_weights)[::-1][:len(y_train) // 10].sum():.3f}")

    print("\n=== Step 4: sensitivity to label noise ===")
    Xn, yn01 = make_classification(n_samples=1500, n_features=10, n_informative=5, flip_y=0.0, random_state=4)
    yn = np.where(yn01 == 1, 1, -1)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, yn, test_size=0.4, random_state=4)
    rng = np.random.default_rng(4)
    for noise in [0.0, 0.2]:
        noisy = yn_train.copy()
        flip = rng.random(len(noisy)) < noise
        noisy[flip] *= -1
        noisy_model = AdaBoost(n_rounds=300).fit(Xn_train, noisy, Xn_test, yn_test)
        best_round = int(np.argmin(noisy_model.eval_curve)) + 1
        print(f"label noise={noise:.0%}  best test err={min(noisy_model.eval_curve):.3f} at round {best_round}  "
              f"final test err={noisy_model.eval_curve[-1]:.3f}")

    print("\n=== Use It: sklearn AdaBoostClassifier ===")
    sk = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), n_estimators=300, learning_rate=1.0, random_state=0,
    ).fit(X_train, y_train)
    ours = np.mean(model.predict(X_test) != y_test)
    print(f"ours test error:    {ours:.3f}")
    print(f"sklearn test error: {1 - sk.score(X_test, y_test):.3f}")
    deeper = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3), n_estimators=300, learning_rate=0.5, random_state=0,
    ).fit(X_train, y_train)
    print(f"sklearn depth-3 trees, lr=0.5 test error: {1 - deeper.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
