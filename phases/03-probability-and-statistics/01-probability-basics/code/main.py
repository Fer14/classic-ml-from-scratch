from itertools import product

import numpy as np


def exact_probability(sample_space, event):
    favorable = [outcome for outcome in sample_space if event(outcome)]
    return len(favorable) / len(sample_space)


def simulate_probability(rng, n_trials, event):
    rolls = rng.integers(1, 7, size=(n_trials, 2))
    hits = np.array([event(tuple(row)) for row in rolls])
    return hits.mean()


def conditional_probability(sample_space, event_a, event_b):
    in_b = [outcome for outcome in sample_space if event_b(outcome)]
    in_a_and_b = [outcome for outcome in in_b if event_a(outcome)]
    return len(in_a_and_b) / len(in_b)


def sum_is_seven(outcome):
    return outcome[0] + outcome[1] == 7


def first_is_even(outcome):
    return outcome[0] % 2 == 0


def sum_at_least_ten(outcome):
    return outcome[0] + outcome[1] >= 10


def has_a_six(outcome):
    return 6 in outcome


def main():
    rng = np.random.default_rng(42)
    sample_space = list(product(range(1, 7), repeat=2))
    print(f"Sample space size: {len(sample_space)}")
    print(f"First outcomes: {sample_space[:6]}")

    print("\n=== Step 2: Exact vs simulated probability ===")
    for name, event in [("sum == 7", sum_is_seven), ("first die even", first_is_even), ("sum >= 10", sum_at_least_ten)]:
        exact = exact_probability(sample_space, event)
        for n_trials in (100, 10_000, 200_000):
            estimate = simulate_probability(rng, n_trials, event)
            print(f"P({name:15s}) exact={exact:.4f}  sim(n={n_trials:>7d})={estimate:.4f}")

    print("\n=== Step 3: Complement, union, intersection ===")
    p_seven = exact_probability(sample_space, sum_is_seven)
    p_even = exact_probability(sample_space, first_is_even)
    p_both = exact_probability(sample_space, lambda o: sum_is_seven(o) and first_is_even(o))
    p_either = exact_probability(sample_space, lambda o: sum_is_seven(o) or first_is_even(o))
    print(f"P(not sum==7)         = {1 - p_seven:.4f}")
    print(f"P(A or B) direct      = {p_either:.4f}")
    print(f"P(A)+P(B)-P(A and B)  = {p_seven + p_even - p_both:.4f}")

    print("\n=== Step 4: Conditional probability and independence ===")
    p_ten_given_six = conditional_probability(sample_space, sum_at_least_ten, has_a_six)
    p_ten = exact_probability(sample_space, sum_at_least_ten)
    print(f"P(sum>=10)            = {p_ten:.4f}")
    print(f"P(sum>=10 | has a 6)  = {p_ten_given_six:.4f}  -> dependent")
    print(f"P(A and B)            = {p_both:.4f}")
    print(f"P(A) * P(B)           = {p_seven * p_even:.4f}  -> independent")

    print("\n=== Use It: vectorized NumPy ===")
    rolls = rng.integers(1, 7, size=(1_000_000, 2))
    totals = rolls.sum(axis=1)
    counts = np.bincount(totals, minlength=13)[2:]
    exact_counts = np.array([sum(1 for o in sample_space if sum(o) == s) for s in range(2, 13)])
    print("sum  simulated  exact")
    for total, count, exact in zip(range(2, 13), counts / len(totals), exact_counts / 36):
        print(f"{total:3d}  {count:.4f}     {exact:.4f}")
    has_six_mask = (rolls == 6).any(axis=1)
    print(f"P(sum>=10 | has a 6) vectorized: {(totals[has_six_mask] >= 10).mean():.4f}")


if __name__ == "__main__":
    main()
