import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier


def make_fruit_data(n_per_class, rng):
    apple_weight = rng.normal(150, 20, n_per_class)
    apple_sugar = rng.normal(10, 1.5, n_per_class)
    orange_weight = rng.normal(200, 25, n_per_class)
    orange_sugar = rng.normal(8, 1.5, n_per_class)
    features = np.vstack([
        np.column_stack([apple_weight, apple_sugar]),
        np.column_stack([orange_weight, orange_sugar]),
    ])
    labels = np.array([0] * n_per_class + [1] * n_per_class)
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def hand_written_rule(features):
    return (features[:, 0] > 190).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


class ThresholdClassifier:
    def fit(self, features, labels):
        best_accuracy = -1.0
        for column in range(features.shape[1]):
            values = np.sort(np.unique(features[:, column]))
            candidates = (values[:-1] + values[1:]) / 2
            for threshold in candidates:
                for direction in (1, -1):
                    predictions = (direction * (features[:, column] - threshold) > 0).astype(int)
                    score = accuracy(labels, predictions)
                    if score > best_accuracy:
                        best_accuracy = score
                        self.column = column
                        self.threshold = threshold
                        self.direction = direction
        return self

    def predict(self, features):
        return (self.direction * (features[:, self.column] - self.threshold) > 0).astype(int)


class NearestCentroidClassifier:
    def fit(self, features, labels):
        self.classes = np.unique(labels)
        self.centroids = np.array([features[labels == c].mean(axis=0) for c in self.classes])
        return self

    def predict(self, features):
        distances = np.linalg.norm(features[:, None, :] - self.centroids[None, :, :], axis=2)
        return self.classes[np.argmin(distances, axis=1)]


class LeastSquaresRegressor:
    def fit(self, features, targets):
        design = np.column_stack([np.ones(len(features)), features])
        self.coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        return self

    def predict(self, features):
        return self.coefficients[0] + features @ self.coefficients[1:]


def main():
    rng = np.random.default_rng(0)
    train_x, train_y = make_fruit_data(100, rng)
    test_x, test_y = make_fruit_data(100, rng)

    print("=== Traditional programming: a hand-written rule ===")
    print(f"Rule 'weight > 190 means orange' test accuracy: {accuracy(test_y, hand_written_rule(test_x)):.3f}")

    print("\n=== Machine learning: learn the rule from data ===")
    stump = ThresholdClassifier().fit(train_x, train_y)
    feature_names = ["weight", "sugar"]
    print(f"Learned rule: feature={feature_names[stump.column]}, threshold={stump.threshold:.2f}, direction={stump.direction:+d}")
    print(f"Train accuracy: {accuracy(train_y, stump.predict(train_x)):.3f}")
    print(f"Test accuracy:  {accuracy(test_y, stump.predict(test_x)):.3f}")

    centroid_model = NearestCentroidClassifier().fit(train_x, train_y)
    print(f"\nNearest centroid centroids:\n{np.round(centroid_model.centroids, 2)}")
    print(f"Nearest centroid test accuracy: {accuracy(test_y, centroid_model.predict(test_x)):.3f}")

    print("\n=== Regression: predict a number ===")
    sizes = rng.uniform(40, 200, 200)
    prices = 2.5 * sizes + 30 + rng.normal(0, 20, 200)
    regressor = LeastSquaresRegressor().fit(sizes[:150, None], prices[:150])
    test_predictions = regressor.predict(sizes[150:, None])
    test_mae = np.mean(np.abs(test_predictions - prices[150:]))
    print(f"Learned price = {regressor.coefficients[1]:.3f} * size + {regressor.coefficients[0]:.3f}")
    print(f"Test mean absolute error: {test_mae:.2f}")

    print("\n=== Use It: scikit-learn ===")
    sk_stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(train_x, train_y)
    print(f"sklearn stump: feature={feature_names[sk_stump.tree_.feature[0]]}, threshold={sk_stump.tree_.threshold[0]:.2f}")
    print(f"sklearn stump test accuracy: {accuracy(test_y, sk_stump.predict(test_x)):.3f}")
    sk_centroid = NearestCentroid().fit(train_x, train_y)
    print(f"sklearn NearestCentroid test accuracy: {accuracy(test_y, sk_centroid.predict(test_x)):.3f}")
    sk_linear = LinearRegression().fit(sizes[:150, None], prices[:150])
    print(f"sklearn LinearRegression: price = {sk_linear.coef_[0]:.3f} * size + {sk_linear.intercept_:.3f}")


if __name__ == "__main__":
    main()
