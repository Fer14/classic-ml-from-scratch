import numpy as np
from scipy import stats


def bayes_posterior(prior, likelihood_if_true, likelihood_if_false):
    evidence = likelihood_if_true * prior + likelihood_if_false * (1 - prior)
    return likelihood_if_true * prior / evidence


def simulate_screening(rng, n_people, prevalence, sensitivity, false_positive_rate):
    is_sick = rng.random(n_people) < prevalence
    test_random = rng.random(n_people)
    tests_positive = np.where(is_sick, test_random < sensitivity, test_random < false_positive_rate)
    return is_sick, tests_positive


def grid_posterior(grid, prior_weights, heads, tails):
    likelihood = grid**heads * (1 - grid) ** tails
    unnormalized = likelihood * prior_weights
    return unnormalized / np.trapezoid(unnormalized, grid)


def main():
    rng = np.random.default_rng(42)
    prevalence, sensitivity, false_positive_rate = 0.01, 0.95, 0.05

    print("=== Step 1: Medical test with Bayes' theorem ===")
    posterior = bayes_posterior(prevalence, sensitivity, false_positive_rate)
    print(f"P(sick)              = {prevalence:.3f}")
    print(f"P(+ | sick)          = {sensitivity:.3f}")
    print(f"P(+ | healthy)       = {false_positive_rate:.3f}")
    print(f"P(sick | +)          = {posterior:.4f}")

    print("\n=== Step 2: Check by simulating 1,000,000 people ===")
    is_sick, tests_positive = simulate_screening(rng, 1_000_000, prevalence, sensitivity, false_positive_rate)
    true_positives = np.sum(is_sick & tests_positive)
    false_positives = np.sum(~is_sick & tests_positive)
    print(f"Positive tests: {tests_positive.sum()}  (true {true_positives}, false {false_positives})")
    print(f"Fraction of positives that are sick: {true_positives / tests_positive.sum():.4f}")

    print("\n=== Step 3: Sequential updating (yesterday's posterior is today's prior) ===")
    belief = prevalence
    for test_number in range(1, 4):
        belief = bayes_posterior(belief, sensitivity, false_positive_rate)
        print(f"After positive test {test_number}: P(sick) = {belief:.4f}")
    for base_rate in (0.001, 0.01, 0.1, 0.5):
        print(f"Prior {base_rate:5.3f} -> posterior after one positive {bayes_posterior(base_rate, sensitivity, false_positive_rate):.4f}")

    print("\n=== Step 4: Posterior over a coin's bias on a grid ===")
    grid = np.linspace(0, 1, 1001)
    uniform_prior = np.ones_like(grid)
    skeptical_prior = stats.norm.pdf(grid, 0.5, 0.05)
    flips = rng.random(50) < 0.7
    heads, tails = int(flips.sum()), int((~flips).sum())
    print(f"Observed {heads} heads, {tails} tails")
    for name, prior_weights in [("uniform", uniform_prior), ("skeptical", skeptical_prior)]:
        density = grid_posterior(grid, prior_weights, heads, tails)
        mean = np.trapezoid(grid * density, grid)
        map_estimate = grid[np.argmax(density)]
        print(f"{name:9s} prior: posterior mean={mean:.4f}  MAP={map_estimate:.3f}")

    print("\n=== Use It: conjugate Beta posterior in scipy ===")
    alpha_prior, beta_prior = 1, 1
    exact_posterior = stats.beta(alpha_prior + heads, beta_prior + tails)
    grid_density = grid_posterior(grid, uniform_prior, heads, tails)
    print(f"Beta({alpha_prior + heads}, {beta_prior + tails}) mean={exact_posterior.mean():.4f}")
    print(f"Max |grid - scipy beta pdf| = {np.max(np.abs(grid_density - exact_posterior.pdf(grid))):.6f}")
    low, high = exact_posterior.ppf([0.025, 0.975])
    print(f"95% credible interval for bias: [{low:.3f}, {high:.3f}]")
    print(f"P(bias > 0.75 | data) = {exact_posterior.sf(0.75):.4f}")


if __name__ == "__main__":
    main()
