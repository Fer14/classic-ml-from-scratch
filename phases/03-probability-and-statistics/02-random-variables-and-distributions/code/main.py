from math import comb, factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def bernoulli_pmf(k, p):
    return p if k == 1 else 1 - p


def binomial_pmf(k, n, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def poisson_pmf(k, rate):
    return rate**k * np.exp(-rate) / factorial(k)


def uniform_pdf(x, low, high):
    return np.where((x >= low) & (x <= high), 1.0 / (high - low), 0.0)


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def exponential_pdf(x, rate):
    return np.where(x >= 0, rate * np.exp(-rate * x), 0.0)


def numeric_cdf(pdf_values, grid):
    step = grid[1] - grid[0]
    return np.cumsum(pdf_values) * step


def sample_exponential_inverse_transform(rng, rate, size):
    uniforms = rng.uniform(0, 1, size)
    return -np.log(1 - uniforms) / rate


def sample_normal_box_muller(rng, size):
    u1 = rng.uniform(0, 1, size)
    u2 = rng.uniform(0, 1, size)
    return np.sqrt(-2 * np.log(1 - u1)) * np.cos(2 * np.pi * u2)


def main():
    rng = np.random.default_rng(42)

    print("=== Step 1: Discrete distributions (PMF) ===")
    print(f"Bernoulli(p=0.3): P(1)={bernoulli_pmf(1, 0.3):.2f}  P(0)={bernoulli_pmf(0, 0.3):.2f}")
    n_flips, p_heads = 10, 0.5
    binomial_probs = np.array([binomial_pmf(k, n_flips, p_heads) for k in range(n_flips + 1)])
    print(f"Binomial(n=10, p=0.5) PMF: {np.round(binomial_probs, 4)}")
    print(f"  sums to {binomial_probs.sum():.6f}")
    poisson_probs = np.array([poisson_pmf(k, 3.0) for k in range(10)])
    print(f"Poisson(rate=3) P(0..9): {np.round(poisson_probs, 4)}")

    print("\n=== Step 2: Continuous distributions (PDF) ===")
    grid = np.linspace(-6, 10, 16001)
    for name, density in [
        ("Uniform(0, 4)", uniform_pdf(grid, 0, 4)),
        ("Normal(0, 1)", normal_pdf(grid, 0, 1)),
        ("Exponential(1.5)", exponential_pdf(grid, 1.5)),
    ]:
        area = np.trapezoid(density, grid)
        print(f"{name:18s} area under PDF = {area:.4f}  max density = {density.max():.4f}")
    print(f"Normal(0, 0.1) peak density = {normal_pdf(np.array([0.0]), 0, 0.1)[0]:.3f}  (> 1 is fine)")

    print("\n=== Step 3: CDF by accumulating the PDF ===")
    normal_cdf_values = numeric_cdf(normal_pdf(grid, 0, 1), grid)
    for z in (-1.96, -1.0, 0.0, 1.0, 1.96):
        index = np.searchsorted(grid, z)
        print(f"P(Z <= {z:5.2f}) = {normal_cdf_values[index]:.4f}")

    print("\n=== Step 4: Sampling from scratch ===")
    exponential_samples = sample_exponential_inverse_transform(rng, 1.5, 100_000)
    normal_samples = sample_normal_box_muller(rng, 100_000)
    print(f"Exponential(1.5) inverse transform: mean={exponential_samples.mean():.4f} (theory {1 / 1.5:.4f})")
    print(f"Normal Box-Muller: mean={normal_samples.mean():.4f} std={normal_samples.std():.4f}")
    print(f"  fraction within 1 std: {(np.abs(normal_samples) < 1).mean():.4f} (theory 0.6827)")

    print("\n=== Use It: scipy.stats ===")
    print(f"binom.pmf(5, 10, 0.5)  scratch={binomial_pmf(5, 10, 0.5):.6f}  scipy={stats.binom.pmf(5, 10, 0.5):.6f}")
    print(f"poisson.pmf(2, 3)      scratch={poisson_pmf(2, 3.0):.6f}  scipy={stats.poisson.pmf(2, 3.0):.6f}")
    print(f"norm.pdf(0.5)          scratch={normal_pdf(np.array([0.5]), 0, 1)[0]:.6f}  scipy={stats.norm.pdf(0.5):.6f}")
    print(f"norm.cdf(1.96)         scratch={normal_cdf_values[np.searchsorted(grid, 1.96)]:.6f}  scipy={stats.norm.cdf(1.96):.6f}")
    print(f"norm.ppf(0.975)        scipy={stats.norm.ppf(0.975):.6f}")
    ks_statistic, ks_p_value = stats.kstest(exponential_samples, stats.expon(scale=1 / 1.5).cdf)
    print(f"KS test scratch exponential samples vs scipy expon: stat={ks_statistic:.4f} p={ks_p_value:.3f}")

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].bar(range(n_flips + 1), binomial_probs, color="tab:blue")
    axes[0].set_title("Binomial(10, 0.5) PMF")
    plot_grid = np.linspace(-4, 4, 400)
    axes[1].hist(normal_samples, bins=80, density=True, alpha=0.5, label="Box-Muller samples")
    axes[1].plot(plot_grid, normal_pdf(plot_grid, 0, 1), color="black", label="PDF")
    axes[1].set_title("Normal(0, 1)")
    axes[1].legend()
    positive_grid = np.linspace(0, 4, 400)
    axes[2].hist(exponential_samples, bins=80, range=(0, 4), density=True, alpha=0.5, label="inverse transform")
    axes[2].plot(positive_grid, exponential_pdf(positive_grid, 1.5), color="black", label="PDF")
    axes[2].set_title("Exponential(1.5)")
    axes[2].legend()
    figure.tight_layout()
    output_path = Path(__file__).parent / "distributions.png"
    figure.savefig(output_path, dpi=100)
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
