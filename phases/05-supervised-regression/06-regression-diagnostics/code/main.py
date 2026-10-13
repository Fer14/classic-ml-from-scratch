from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def make_data(n_samples, rng):
    size_m2 = rng.uniform(40, 200, n_samples)
    rooms = np.round(size_m2 / 30 + rng.normal(0, 0.4, n_samples))
    age_years = rng.uniform(0, 50, n_samples)
    price_k = 20 + 2.0 * size_m2 + 5 * rooms - 0.6 * age_years + rng.normal(0, 12, n_samples)
    features = np.column_stack([size_m2, rooms, age_years])
    features = np.vstack([features, [[190.0, 3.0, 5.0]]])
    price_k = np.append(price_k, 120.0)
    return features, price_k


def design_matrix(features):
    return np.hstack([np.ones((features.shape[0], 1)), features])


def fit_ols(features, targets):
    return np.linalg.lstsq(design_matrix(features), targets, rcond=None)[0]


def r2_and_adjusted(targets, predictions, n_predictors):
    n_samples = len(targets)
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
    adjusted = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)
    return r2, adjusted


def leverage(features):
    design = design_matrix(features)
    hat_matrix = design @ np.linalg.inv(design.T @ design) @ design.T
    return np.diag(hat_matrix)


def cooks_distance(residuals, leverages, n_params):
    sigma2 = np.sum(residuals ** 2) / (len(residuals) - n_params)
    return residuals ** 2 / (n_params * sigma2) * leverages / (1 - leverages) ** 2


def coefficient_table(features, targets, beta):
    design = design_matrix(features)
    n_samples, n_params = design.shape
    residuals = targets - design @ beta
    sigma2 = residuals @ residuals / (n_samples - n_params)
    standard_errors = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
    t_values = beta / standard_errors
    p_values = 2 * stats.t.sf(np.abs(t_values), n_samples - n_params)
    return standard_errors, t_values, p_values


def variance_inflation_factors(features):
    factors = []
    for j in range(features.shape[1]):
        others = np.delete(features, j, axis=1)
        beta = fit_ols(others, features[:, j])
        predictions = design_matrix(others) @ beta
        r2, _ = r2_and_adjusted(features[:, j], predictions, others.shape[1])
        factors.append(1 / (1 - r2))
    return np.array(factors)


def main():
    rng = np.random.default_rng(5)
    features, prices = make_data(120, rng)
    names = ["intercept", "size_m2", "rooms", "age_years"]

    print("=== Step 1: fit and goodness of fit ===")
    beta = fit_ols(features, prices)
    predictions = design_matrix(features) @ beta
    residuals = prices - predictions
    r2, adjusted = r2_and_adjusted(prices, predictions, features.shape[1])
    print(f"R^2 {r2:.4f}   adjusted R^2 {adjusted:.4f}")
    noise_rng = np.random.default_rng(9)
    padded = np.hstack([features, noise_rng.normal(0, 1, (len(prices), 10))])
    padded_pred = design_matrix(padded) @ fit_ols(padded, prices)
    r2_pad, adjusted_pad = r2_and_adjusted(prices, padded_pred, padded.shape[1])
    print(f"with 10 pure-noise features: R^2 {r2_pad:.4f}   adjusted R^2 {adjusted_pad:.4f}")

    print("\n=== Step 2: coefficients with standard errors ===")
    standard_errors, t_values, p_values = coefficient_table(features, prices, beta)
    print(f"{'term':>10} {'coef':>8} {'std err':>8} {'t':>7} {'p':>8}")
    for name, coef, se, t, p in zip(names, beta, standard_errors, t_values, p_values):
        print(f"{name:>10} {coef:>8.3f} {se:>8.3f} {t:>7.2f} {p:>8.4f}")

    print("\n=== Step 3: residual checks ===")
    print(f"mean residual {residuals.mean():.2e}")
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk normality test p = {shapiro_p:.4f}")
    first_half, second_half = np.array_split(np.argsort(predictions), 2)
    print(f"residual std, low vs high predictions: {residuals[first_half].std():.2f} vs {residuals[second_half].std():.2f}")

    print("\n=== Step 4: leverage and Cook's distance ===")
    leverages = leverage(features)
    n_params = features.shape[1] + 1
    cooks = cooks_distance(residuals, leverages, n_params)
    print(f"sum of leverages = {leverages.sum():.3f} (equals number of parameters {n_params})")
    worst = np.argsort(cooks)[::-1][:3]
    for index in worst:
        print(f"row {index:>3}: leverage {leverages[index]:.3f} residual {residuals[index]:8.2f} Cook's D {cooks[index]:.3f}")
    print(f"rule of thumb threshold 4/n = {4 / len(prices):.3f}")
    keep = np.ones(len(prices), dtype=bool)
    keep[worst[0]] = False
    beta_clean = fit_ols(features[keep], prices[keep])
    print(f"coefficients without row {worst[0]}: {np.round(beta_clean, 3)}")

    print("\n=== Step 5: multicollinearity with VIF ===")
    for name, vif in zip(names[1:], variance_inflation_factors(features)):
        print(f"VIF {name:>10}: {vif:6.2f}")

    print("\n=== Step 6: diagnostic plots ===")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(predictions, residuals, s=12)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Residuals vs fitted")
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q")
    axes[2].scatter(leverages, residuals / residuals.std(), s=12, c=cooks, cmap="viridis")
    axes[2].set_title("Standardized residuals vs leverage")
    figure.tight_layout()
    output_path = Path(__file__).parent / "diagnostics.png"
    figure.savefig(output_path, dpi=100)
    print(f"saved {output_path.name}")

    print("\n=== Use It: scikit-learn plus scipy ===")
    model = LinearRegression().fit(features, prices)
    print(f"sklearn R^2 {model.score(features, prices):.4f}, same coefficients: {np.allclose(np.r_[model.intercept_, model.coef_], beta)}")
    slope_test = stats.linregress(features[:, 0], prices)
    print(f"scipy linregress on size only: slope {slope_test.slope:.3f}, p = {slope_test.pvalue:.2e}")


if __name__ == "__main__":
    main()
