from itertools import combinations

import numpy as np
import pandas as pd


def make_baskets(n_baskets, seed):
    rng = np.random.default_rng(seed)
    filler_items = ["apples", "bananas", "cereal", "coffee", "eggs", "rice", "soap", "tea"]
    baskets = []
    for _ in range(n_baskets):
        basket = set()
        if rng.random() < 0.35:
            basket.add("bread")
            if rng.random() < 0.7:
                basket.add("butter")
            if rng.random() < 0.5:
                basket.add("jam")
        if rng.random() < 0.25:
            basket.add("diapers")
            if rng.random() < 0.5:
                basket.add("beer")
        if rng.random() < 0.3:
            basket.add("milk")
            if "cereal" not in basket and rng.random() < 0.5:
                basket.add("cereal")
        if rng.random() < 0.15:
            basket.add("beer")
        if rng.random() < 0.6:
            basket.add("water")
        n_filler = rng.integers(1, 4)
        basket |= {str(item) for item in rng.choice(filler_items, size=n_filler, replace=False)}
        baskets.append(frozenset(basket))
    return baskets


def support_counts(baskets, candidates):
    counts = {candidate: 0 for candidate in candidates}
    for basket in baskets:
        for candidate in candidates:
            if candidate <= basket:
                counts[candidate] += 1
    return counts


def generate_candidates(frequent_previous, size):
    candidates = set()
    previous_list = list(frequent_previous)
    for i in range(len(previous_list)):
        for j in range(i + 1, len(previous_list)):
            union = previous_list[i] | previous_list[j]
            if len(union) != size:
                continue
            all_subsets_frequent = all(frozenset(subset) in frequent_previous for subset in combinations(union, size - 1))
            if all_subsets_frequent:
                candidates.add(union)
    return candidates


def apriori(baskets, min_support):
    n = len(baskets)
    items = {item for basket in baskets for item in basket}
    candidates = {frozenset([item]) for item in items}
    frequent = {}
    size = 1
    while candidates:
        counts = support_counts(baskets, candidates)
        level = {itemset: count / n for itemset, count in counts.items() if count / n >= min_support}
        print(f"  size {size}: {len(candidates):3d} candidates -> {len(level):3d} frequent")
        frequent.update(level)
        size += 1
        candidates = generate_candidates(set(level), size)
    return frequent


def association_rules(frequent, min_confidence):
    rules = []
    for itemset, itemset_support in frequent.items():
        if len(itemset) < 2:
            continue
        for antecedent_size in range(1, len(itemset)):
            for antecedent in combinations(sorted(itemset), antecedent_size):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = itemset_support / frequent[antecedent]
                if confidence < min_confidence:
                    continue
                lift = confidence / frequent[consequent]
                rules.append(
                    {
                        "antecedent": ", ".join(sorted(antecedent)),
                        "consequent": ", ".join(sorted(consequent)),
                        "support": itemset_support,
                        "confidence": confidence,
                        "lift": lift,
                    }
                )
    return pd.DataFrame(rules).sort_values("lift", ascending=False).reset_index(drop=True)


baskets = make_baskets(2000, seed=7)

print("=== Step 1: Transactions ===")
print(f"{len(baskets)} baskets, {len({item for basket in baskets for item in basket})} distinct items")
for basket in baskets[:3]:
    print(f"  {sorted(basket)}")

print("\n=== Step 2: Apriori frequent itemsets (min_support=0.08) ===")
frequent = apriori(baskets, min_support=0.08)
top_multi = sorted(((s, sorted(i)) for i, s in frequent.items() if len(i) >= 2), reverse=True)[:6]
for support, itemset in top_multi:
    print(f"  {support:.3f}  {itemset}")

print("\n=== Step 3: Rules (min_confidence=0.5), sorted by lift ===")
rules = association_rules(frequent, min_confidence=0.5)
pd.set_option("display.width", 120)
print(rules.head(10).round(3).to_string())

print("\n=== Step 4: Why lift matters ===")
filler_rules = rules[rules["lift"].between(0.9, 1.1)]
print(f"Rules with confidence >= 0.5 but lift near 1 (no real association): {len(filler_rules)}")
if len(filler_rules):
    print(filler_rules.head(3).round(3).to_string())

print("\n=== Use It: pandas one-hot check ===")
all_items = sorted({item for basket in baskets for item in basket})
one_hot = pd.DataFrame([[item in basket for item in all_items] for basket in baskets], columns=all_items)
triple_support = (one_hot["bread"] & one_hot["butter"] & one_hot["jam"]).mean()
print(f"support(bread, butter, jam) via pandas: {triple_support:.3f}  via apriori: {frequent[frozenset({"bread", "butter", "jam"})]:.3f}")
print(f"Item supports:\n{one_hot.mean().sort_values(ascending=False).round(3).to_string()}")
