import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import hinge_loss, log_loss, mean_absolute_error, mean_squared_error


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def huber_loss(y_true, y_pred, delta=1.0):
    residual = np.abs(y_true - y_pred)
    quadratic = 0.5 * residual ** 2
    linear = delta * (residual - 0.5 * delta)
    return np.mean(np.where(residual <= delta, quadratic, linear))


def binary_cross_entropy(y_true, probabilities, eps=1e-15):
    p = np.clip(probabilities, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def hinge(y_signed, scores):
    return np.mean(np.maximum(0, 1 - y_signed * scores))


def categorical_cross_entropy(y_true_index, probabilities, eps=1e-15):
    p = np.clip(probabilities, eps, 1)
    return -np.mean(np.log(p[np.arange(len(y_true_index)), y_true_index]))


def gradient(loss_name, residual, delta=1.0):
    if loss_name == "mse":
        return 2 * residual
    if loss_name == "mae":
        return np.sign(residual)
    return np.clip(residual, -delta, delta)


def fit_line(x, y, loss_name, learning_rate=0.05, epochs=4000):
    slope, intercept = 0.0, 0.0
    for _ in range(epochs):
        residual = slope * x + intercept - y
        g = gradient(loss_name, residual)
        slope -= learning_rate * np.mean(g * x)
        intercept -= learning_rate * np.mean(g)
    return slope, intercept


def main():
    rng = np.random.default_rng(3)

    print("=== Step 1: regression losses on the same predictions ===")
    y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.0])
    y_pred = np.array([2.5, 5.0, 3.0, 8.0, 10.0])
    print(f"residuals: {y_pred - y_true}")
    print(f"MSE:   {mse_loss(y_true, y_pred):.4f}")
    print(f"MAE:   {mae_loss(y_true, y_pred):.4f}")
    print(f"Huber: {huber_loss(y_true, y_pred):.4f}")

    print("\n=== Step 2: best constant prediction under each loss ===")
    salaries = np.array([40, 42, 45, 47, 50, 52, 55, 400], dtype=float)
    grid = np.linspace(0, 450, 45001)
    best_mse = grid[np.argmin([mse_loss(salaries, c) for c in grid])]
    best_mae = grid[np.argmin([mae_loss(salaries, c) for c in grid])]
    print(f"MSE-optimal constant: {best_mse:.2f}  (mean   = {salaries.mean():.2f})")
    print(f"MAE-optimal constant: {best_mae:.2f}  (median = any value in [{np.sort(salaries)[3]:.0f}, {np.sort(salaries)[4]:.0f}])")

    print("\n=== Step 3: fitting a line with outliers ===")
    x = rng.uniform(0, 5, 60)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.3, 60)
    y[:5] += 15
    for loss_name in ("mse", "mae", "huber"):
        slope, intercept = fit_line(x, y, loss_name)
        print(f"{loss_name:5s} fit: y = {slope:.3f} x + {intercept:.3f}   (true: y = 2.000 x + 1.000)")

    print("\n=== Step 4: classification losses ===")
    labels = np.array([1, 0, 1, 1, 0])
    confident_right = np.array([0.95, 0.05, 0.9, 0.85, 0.1])
    unsure = np.array([0.6, 0.4, 0.55, 0.6, 0.45])
    confident_wrong = np.array([0.95, 0.05, 0.9, 0.02, 0.1])
    print(f"BCE confident and right: {binary_cross_entropy(labels, confident_right):.4f}")
    print(f"BCE unsure but right:    {binary_cross_entropy(labels, unsure):.4f}")
    print(f"BCE one confident miss:  {binary_cross_entropy(labels, confident_wrong):.4f}")
    signed = 2 * labels - 1
    scores = np.array([2.0, -0.5, 0.3, 1.5, 0.2])
    print(f"hinge loss with scores {scores}: {hinge(signed, scores):.4f}")
    three_class_probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])
    three_class_labels = np.array([0, 1, 2])
    print(f"categorical cross-entropy: {categorical_cross_entropy(three_class_labels, three_class_probs):.4f}")

    print("\n=== Step 5: per-example loss as the prediction moves ===")
    print(" p(true class) |   -log p")
    for p in (0.99, 0.9, 0.5, 0.1, 0.01):
        print(f"     {p:5.2f}    | {-np.log(p):8.4f}")

    print("\n=== Use It: scikit-learn ===")
    print(f"sklearn MSE:      {mean_squared_error(y_true, y_pred):.4f}")
    print(f"sklearn MAE:      {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"sklearn log_loss: {log_loss(labels, confident_wrong):.4f}")
    print(f"sklearn hinge:    {hinge_loss(signed, scores):.4f}")
    print(f"sklearn multiclass log_loss: {log_loss(three_class_labels, three_class_probs):.4f}")
    ols = LinearRegression().fit(x[:, None], y)
    huber = HuberRegressor(epsilon=1.35).fit(x[:, None], y)
    print(f"LinearRegression (MSE): y = {ols.coef_[0]:.3f} x + {ols.intercept_:.3f}")
    print(f"HuberRegressor:         y = {huber.coef_[0]:.3f} x + {huber.intercept_:.3f}")


if __name__ == "__main__":
    main()
