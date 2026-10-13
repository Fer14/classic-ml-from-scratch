from math import erf, sqrt

import numpy as np
from scipy import stats


def standard_normal_cdf(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def two_proportion_z_test(successes_a, n_a, successes_b, n_b):
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    pooled = (successes_a + successes_b) / (n_a + n_b)
    standard_error = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (rate_b - rate_a) / standard_error
    p_value = 2 * (1 - standard_normal_cdf(abs(z)))
    return z, p_value


def welch_t_statistic(sample_a, sample_b):
    variance_a, variance_b = sample_a.var(ddof=1), sample_b.var(ddof=1)
    n_a, n_b = len(sample_a), len(sample_b)
    standard_error = np.sqrt(variance_a / n_a + variance_b / n_b)
    t = (sample_b.mean() - sample_a.mean()) / standard_error
    degrees_of_freedom = (variance_a / n_a + variance_b / n_b) ** 2 / (
        (variance_a / n_a) ** 2 / (n_a - 1) + (variance_b / n_b) ** 2 / (n_b - 1)
    )
    return t, degrees_of_freedom


def permutation_test(rng, sample_a, sample_b, n_permutations=10_000):
    observed = sample_b.mean() - sample_a.mean()
    pooled = np.concatenate([sample_a, sample_b])
    n_a = len(sample_a)
    count_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        difference = shuffled[n_a:].mean() - shuffled[:n_a].mean()
        count_extreme += abs(difference) >= abs(observed)
    return observed, (count_extreme + 1) / (n_permutations + 1)


def simulate_rejection_rate(rng, true_difference, n_per_group, n_experiments, alpha=0.05):
    rejections = 0
    for _ in range(n_experiments):
        control = rng.normal(0, 1, n_per_group)
        treatment = rng.normal(true_difference, 1, n_per_group)
        t, _ = welch_t_statistic(control, treatment)
        p_value = 2 * (1 - standard_normal_cdf(abs(t)))
        rejections += p_value < alpha
    return rejections / n_experiments


def main():
    rng = np.random.default_rng(42)

    print("=== Step 1: A/B test on conversion rates ===")
    visitors_a, conversions_a = 5000, 500
    visitors_b, conversions_b = 5000, 565
    z, p_value = two_proportion_z_test(conversions_a, visitors_a, conversions_b, visitors_b)
    print(f"A: {conversions_a / visitors_a:.3%}  B: {conversions_b / visitors_b:.3%}")
    print(f"z = {z:.4f}   two-sided p-value = {p_value:.4f}")
    print("Reject H0 at alpha=0.05" if p_value < 0.05 else "Fail to reject H0 at alpha=0.05")

    print("\n=== Step 2: Welch t statistic on session times ===")
    old_design = rng.normal(5.0, 2.0, 80)
    new_design = rng.normal(5.8, 2.5, 70)
    t, degrees_of_freedom = welch_t_statistic(old_design, new_design)
    normal_p_value = 2 * (1 - standard_normal_cdf(abs(t)))
    print(f"means: old={old_design.mean():.3f} new={new_design.mean():.3f}")
    print(f"t = {t:.4f}   df = {degrees_of_freedom:.2f}   p (normal approx) = {normal_p_value:.4f}")

    print("\n=== Step 3: Permutation test (no distribution assumptions) ===")
    observed, permutation_p_value = permutation_test(rng, old_design, new_design)
    print(f"observed difference = {observed:.4f}   permutation p-value = {permutation_p_value:.4f}")

    print("\n=== Step 4: Type I error and power by simulation ===")
    false_positive_rate = simulate_rejection_rate(rng, 0.0, 50, 4000)
    print(f"No real effect, n=50: rejected H0 in {false_positive_rate:.3f} of experiments (alpha=0.05)")
    for n_per_group in (20, 50, 100, 200):
        power = simulate_rejection_rate(rng, 0.3, n_per_group, 2000)
        print(f"Effect 0.3 sd, n={n_per_group:3d} per group: power = {power:.3f}")

    print("\n=== Step 5: Multiple testing ===")
    n_tests, n_experiments = 20, 2000
    p_values = np.array([
        [2 * (1 - standard_normal_cdf(abs(welch_t_statistic(rng.normal(0, 1, 40), rng.normal(0, 1, 40))[0]))) for _ in range(n_tests)]
        for _ in range(n_experiments)
    ])
    any_false_positive = (p_values < 0.05).any(axis=1).mean()
    any_after_bonferroni = (p_values < 0.05 / n_tests).any(axis=1).mean()
    print(f"20 null tests: P(at least one p<0.05) = {any_false_positive:.3f}  (theory {1 - 0.95**20:.3f})")
    print(f"With Bonferroni (p < 0.05/20):        = {any_after_bonferroni:.3f}")

    print("\n=== Use It: scipy.stats ===")
    scipy_t = stats.ttest_ind(new_design, old_design, equal_var=False)
    print(f"ttest_ind Welch: t={scipy_t.statistic:.4f} p={scipy_t.pvalue:.4f} df={scipy_t.df:.2f}")
    print(f"scratch t with exact t-distribution p: {2 * stats.t.sf(abs(t), degrees_of_freedom):.4f}")
    scipy_permutation = stats.permutation_test(
        (new_design, old_design), lambda x, y: np.mean(x) - np.mean(y), n_resamples=10_000, random_state=42
    )
    print(f"permutation_test p={scipy_permutation.pvalue:.4f}")
    counts = np.array([[conversions_a, visitors_a - conversions_a], [conversions_b, visitors_b - conversions_b]])
    chi2, chi2_p_value, _, _ = stats.chi2_contingency(counts, correction=False)
    print(f"chi2_contingency: chi2={chi2:.4f} (z^2={z**2:.4f}) p={chi2_p_value:.4f}")
    mann_whitney = stats.mannwhitneyu(new_design, old_design)
    print(f"mannwhitneyu (rank-based): p={mann_whitney.pvalue:.4f}")


if __name__ == "__main__":
    main()
