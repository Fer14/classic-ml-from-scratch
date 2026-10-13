import numpy as np
from scipy import stats


def discrete_expectation(values, probabilities, function=lambda x: x):
    return sum(function(v) * p for v, p in zip(values, probabilities))


def discrete_variance(values, probabilities):
    mean = discrete_expectation(values, probabilities)
    return discrete_expectation(values, probabilities, lambda x: (x - mean) ** 2)


def continuous_expectation(pdf, grid, function=lambda x: x):
    return np.trapezoid(function(grid) * pdf(grid), grid)


def sample_mean(data):
    return sum(data) / len(data)


def sample_variance(data, ddof=0):
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / (len(data) - ddof)


def covariance(x, y):
    mean_x, mean_y = sample_mean(x), sample_mean(y)
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / (len(x) - 1)


def correlation(x, y):
    return covariance(x, y) / np.sqrt(sample_variance(x, ddof=1) * sample_variance(y, ddof=1))


def covariance_matrix(data):
    centered = data - data.mean(axis=0)
    return centered.T @ centered / (len(data) - 1)


def main():
    rng = np.random.default_rng(42)

    print("=== Step 1: Expectation and variance of a die ===")
    faces = [1, 2, 3, 4, 5, 6]
    fair = [1 / 6] * 6
    loaded = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    for name, probabilities in [("fair", fair), ("loaded", loaded)]:
        mean = discrete_expectation(faces, probabilities)
        variance = discrete_variance(faces, probabilities)
        second_moment = discrete_expectation(faces, probabilities, lambda x: x**2)
        print(f"{name:6s} E[X]={mean:.4f}  Var={variance:.4f}  E[X^2]-E[X]^2={second_moment - mean**2:.4f}")

    print("\n=== Step 2: Continuous expectation by integration ===")
    rate = 2.0
    grid = np.linspace(0, 30, 300_001)
    exponential_pdf = lambda x: rate * np.exp(-rate * x)
    mean = continuous_expectation(exponential_pdf, grid)
    second = continuous_expectation(exponential_pdf, grid, lambda x: x**2)
    print(f"Exponential(rate=2): E[X]={mean:.4f} (theory 0.5)  Var={second - mean**2:.4f} (theory 0.25)")

    print("\n=== Step 3: Linearity and scaling ===")
    x = rng.normal(3, 2, 200_000)
    y = rng.exponential(1.0, 200_000)
    print(f"E[2X + 5]       = {np.mean(2 * x + 5):.4f}  vs 2E[X]+5 = {2 * np.mean(x) + 5:.4f}")
    print(f"E[X + Y]        = {np.mean(x + y):.4f}  vs E[X]+E[Y] = {np.mean(x) + np.mean(y):.4f}")
    print(f"Var(2X + 5)     = {np.var(2 * x + 5):.4f}  vs 4Var(X) = {4 * np.var(x):.4f}")
    print(f"Var(X + Y) indep= {np.var(x + y):.4f}  vs Var(X)+Var(Y) = {np.var(x) + np.var(y):.4f}")
    print(f"Var(X + X)      = {np.var(x + x):.4f}  vs 2Var(X) = {2 * np.var(x):.4f}  (not independent)")

    print("\n=== Step 4: Biased vs unbiased sample variance ===")
    true_variance = 4.0
    n_experiments, sample_size = 20_000, 5
    samples = rng.normal(0, np.sqrt(true_variance), size=(n_experiments, sample_size))
    biased = np.array([sample_variance(row, ddof=0) for row in samples])
    unbiased = np.array([sample_variance(row, ddof=1) for row in samples])
    print(f"True variance: {true_variance}")
    print(f"Average of divide-by-n     estimates: {biased.mean():.4f}  (theory {true_variance * (sample_size - 1) / sample_size:.2f})")
    print(f"Average of divide-by-(n-1) estimates: {unbiased.mean():.4f}")

    print("\n=== Step 5: Covariance and correlation ===")
    hours_studied = rng.uniform(0, 10, 500)
    exam_score = 50 + 4 * hours_studied + rng.normal(0, 5, 500)
    shoe_size = rng.normal(42, 2, 500)
    print(f"cov(hours, score) = {covariance(hours_studied, exam_score):.4f}")
    print(f"corr(hours, score)= {correlation(hours_studied, exam_score):.4f}")
    print(f"corr(hours, shoe) = {correlation(hours_studied, shoe_size):.4f}")
    u = rng.uniform(-1, 1, 500)
    print(f"corr(u, u^2)      = {correlation(u, u**2):.4f}  (dependent but uncorrelated)")
    data = np.column_stack([hours_studied, exam_score, shoe_size])
    print("Covariance matrix (scratch):")
    print(np.round(covariance_matrix(data), 3))

    print("\n=== Use It: NumPy and SciPy ===")
    print(f"np.var ddof=0 {np.var(exam_score):.4f}  scratch {sample_variance(exam_score, ddof=0):.4f}")
    print(f"np.var ddof=1 {np.var(exam_score, ddof=1):.4f}  scratch {sample_variance(exam_score, ddof=1):.4f}")
    print(f"np.cov matches scratch: {np.allclose(np.cov(data, rowvar=False), covariance_matrix(data))}")
    print(f"np.corrcoef(hours, score) = {np.corrcoef(hours_studied, exam_score)[0, 1]:.4f}")
    print(f"scipy expon(scale=0.5).mean()={stats.expon(scale=0.5).mean():.4f} var()={stats.expon(scale=0.5).var():.4f}")
    summary = stats.describe(exam_score)
    print(f"scipy describe: n={summary.nobs} mean={summary.mean:.4f} variance={summary.variance:.4f} skewness={summary.skewness:.4f}")


if __name__ == "__main__":
    main()
