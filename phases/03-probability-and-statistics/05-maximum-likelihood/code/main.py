import numpy as np
from scipy import optimize, stats


def bernoulli_log_likelihood(p, data):
    return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))


def normal_log_likelihood(mean, std, data):
    return np.sum(-0.5 * np.log(2 * np.pi * std**2) - (data - mean) ** 2 / (2 * std**2))


def poisson_log_likelihood(rate, data):
    log_factorials = np.array([np.sum(np.log(np.arange(1, k + 1))) for k in data])
    return np.sum(data * np.log(rate) - rate - log_factorials)


def normal_mle(data):
    mean = data.mean()
    std = np.sqrt(np.mean((data - mean) ** 2))
    return mean, std


def fit_normal_gradient_ascent(data, learning_rate=0.1, n_steps=500):
    mean, log_std = 0.0, 0.0
    n = len(data)
    for _ in range(n_steps):
        std = np.exp(log_std)
        residuals = data - mean
        gradient_mean = np.sum(residuals) / std**2
        gradient_log_std = -n + np.sum(residuals**2) / std**2
        mean += learning_rate * gradient_mean / n
        log_std += learning_rate * gradient_log_std / n
    return mean, np.exp(log_std)


def gaussian_negative_log_likelihood_line(params, x, y, noise_std):
    slope, intercept = params
    predictions = slope * x + intercept
    return -normal_log_likelihood(predictions, noise_std, y)


def main():
    rng = np.random.default_rng(42)

    print("=== Step 1: Bernoulli MLE by scanning the log-likelihood ===")
    clicks = (rng.random(200) < 0.12).astype(float)
    candidate_p = np.linspace(0.01, 0.99, 981)
    log_likelihoods = np.array([bernoulli_log_likelihood(p, clicks) for p in candidate_p])
    best_p = candidate_p[np.argmax(log_likelihoods)]
    print(f"{int(clicks.sum())} clicks out of {len(clicks)} views")
    print(f"Grid argmax p = {best_p:.3f}   closed form mean = {clicks.mean():.3f}")
    print(f"Raw likelihood at best p = {np.exp(log_likelihoods.max()):.3e}  (why we use logs)")

    print("\n=== Step 2: Normal MLE closed form ===")
    heights = rng.normal(170, 8, 1000)
    mle_mean, mle_std = normal_mle(heights)
    print(f"MLE mean={mle_mean:.4f}  MLE std={mle_std:.4f}  (true 170, 8)")
    print(f"log-likelihood at MLE        = {normal_log_likelihood(mle_mean, mle_std, heights):.2f}")
    print(f"log-likelihood at (168, 10)  = {normal_log_likelihood(168, 10, heights):.2f}")

    print("\n=== Step 3: Normal MLE by gradient ascent ===")
    standardized_heights = (heights - 170) / 10
    gradient_mean, gradient_std = fit_normal_gradient_ascent(standardized_heights)
    print(f"Gradient ascent: mean={gradient_mean * 10 + 170:.4f} std={gradient_std * 10:.4f}")

    print("\n=== Step 4: Poisson MLE ===")
    support_tickets = rng.poisson(4.2, 300)
    candidate_rates = np.linspace(1, 8, 701)
    poisson_scores = [poisson_log_likelihood(rate, support_tickets) for rate in candidate_rates]
    print(f"Grid argmax rate = {candidate_rates[np.argmax(poisson_scores)]:.3f}  sample mean = {support_tickets.mean():.3f}")

    print("\n=== Step 5: MSE is Gaussian maximum likelihood ===")
    x = rng.uniform(0, 10, 100)
    y = 2.5 * x + 1.0 + rng.normal(0, 1.5, 100)
    result = optimize.minimize(gaussian_negative_log_likelihood_line, x0=[0.0, 0.0], args=(x, y, 1.5))
    least_squares_slope, least_squares_intercept = np.polyfit(x, y, 1)
    print(f"Max likelihood line: slope={result.x[0]:.4f} intercept={result.x[1]:.4f}")
    print(f"Least squares line : slope={least_squares_slope:.4f} intercept={least_squares_intercept:.4f}")

    print("\n=== Use It: scipy.stats fit ===")
    scipy_mean, scipy_std = stats.norm.fit(heights)
    print(f"stats.norm.fit     mean={scipy_mean:.4f} std={scipy_std:.4f}")
    waiting_times = rng.exponential(1 / 0.8, 500)
    _, scipy_scale = stats.expon.fit(waiting_times, floc=0)
    print(f"stats.expon.fit    rate={1 / scipy_scale:.4f}  scratch 1/mean={1 / waiting_times.mean():.4f}  (true 0.8)")
    bernoulli_result = optimize.minimize_scalar(lambda p: -bernoulli_log_likelihood(p, clicks), bounds=(0.001, 0.999), method="bounded")
    print(f"minimize_scalar Bernoulli p={bernoulli_result.x:.4f}")


if __name__ == "__main__":
    main()
