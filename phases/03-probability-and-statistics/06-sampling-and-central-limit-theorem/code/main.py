from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def sampling_distribution_of_mean(rng, population, sample_size, n_repeats):
    indices = rng.integers(0, len(population), size=(n_repeats, sample_size))
    return population[indices].mean(axis=1)


def skewness(values):
    centered = values - values.mean()
    return np.mean(centered**3) / np.mean(centered**2) ** 1.5


def normal_confidence_interval(sample, z=1.96):
    standard_error = sample.std(ddof=1) / np.sqrt(len(sample))
    return sample.mean() - z * standard_error, sample.mean() + z * standard_error


def bootstrap_confidence_interval(rng, sample, statistic, n_resamples=5000, level=0.95):
    indices = rng.integers(0, len(sample), size=(n_resamples, len(sample)))
    replicates = statistic(sample[indices], axis=1)
    tail = (1 - level) / 2
    return np.quantile(replicates, [tail, 1 - tail])


def main():
    rng = np.random.default_rng(42)
    population = rng.exponential(scale=10.0, size=1_000_000)
    population_mean, population_std = population.mean(), population.std()

    print("=== Step 1: A skewed population ===")
    print(f"Population mean={population_mean:.4f} std={population_std:.4f} skewness={skewness(population):.3f}")
    print(f"Median={np.median(population):.4f} (mean > median: right-skewed)")

    print("\n=== Step 2: Sampling distribution of the mean ===")
    print(" n    mean of means   std of means   sigma/sqrt(n)   skewness")
    sampling_distributions = {}
    for sample_size in (1, 2, 5, 30, 100):
        means = sampling_distribution_of_mean(rng, population, sample_size, 20_000)
        sampling_distributions[sample_size] = means
        print(f"{sample_size:3d}   {means.mean():12.4f}   {means.std():12.4f}   {population_std / np.sqrt(sample_size):13.4f}   {skewness(means):8.3f}")

    print("\n=== Step 3: Coverage of the 95% normal interval ===")
    for sample_size in (5, 30, 200):
        n_trials, covered = 5000, 0
        for _ in range(n_trials):
            sample = population[rng.integers(0, len(population), sample_size)]
            low, high = normal_confidence_interval(sample)
            covered += low <= population_mean <= high
        print(f"n={sample_size:4d}: interval contains true mean in {covered / n_trials:.3f} of samples")

    print("\n=== Step 4: Bootstrap ===")
    sample = population[rng.integers(0, len(population), 60)]
    mean_low, mean_high = bootstrap_confidence_interval(rng, sample, np.mean)
    median_low, median_high = bootstrap_confidence_interval(rng, sample, np.median)
    print(f"One sample of 60: mean={sample.mean():.3f}  median={np.median(sample):.3f}")
    print(f"Bootstrap 95% CI for mean:   [{mean_low:.3f}, {mean_high:.3f}]  (true {population_mean:.3f})")
    print(f"Bootstrap 95% CI for median: [{median_low:.3f}, {median_high:.3f}]  (true {np.median(population):.3f})")
    normal_low, normal_high = normal_confidence_interval(sample)
    print(f"Normal 95% CI for mean:      [{normal_low:.3f}, {normal_high:.3f}]")

    print("\n=== Use It: scipy.stats ===")
    t_low, t_high = stats.t.interval(0.95, df=len(sample) - 1, loc=sample.mean(), scale=stats.sem(sample))
    print(f"stats.sem={stats.sem(sample):.4f}  scratch={sample.std(ddof=1) / np.sqrt(len(sample)):.4f}")
    print(f"t interval:             [{t_low:.3f}, {t_high:.3f}]")
    scipy_bootstrap = stats.bootstrap((sample,), np.mean, n_resamples=5000, method="percentile", random_state=42)
    interval = scipy_bootstrap.confidence_interval
    print(f"stats.bootstrap (mean): [{interval.low:.3f}, {interval.high:.3f}]")
    print(f"stats.skew of means n=30: scipy={stats.skew(sampling_distributions[30]):.4f}  scratch={skewness(sampling_distributions[30]):.4f}")

    figure, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for axis, sample_size in zip(axes, (1, 2, 5, 30)):
        means = sampling_distributions[sample_size]
        axis.hist(means, bins=60, density=True, alpha=0.6)
        grid = np.linspace(means.min(), means.max(), 300)
        axis.plot(grid, stats.norm.pdf(grid, population_mean, population_std / np.sqrt(sample_size)), color="black")
        axis.set_title(f"means of n={sample_size}")
    figure.tight_layout()
    output_path = Path(__file__).parent / "clt.png"
    figure.savefig(output_path, dpi=100)
    print(f"\nSaved plot to {output_path.name}")


if __name__ == "__main__":
    main()
