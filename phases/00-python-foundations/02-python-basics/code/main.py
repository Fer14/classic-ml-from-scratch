import math
import random
import statistics

random.seed(7)

city = "Madrid"
days_recorded = 14
temperatures_celsius = [round(random.gauss(22.0, 4.0), 1) for _ in range(days_recorded)]
is_summer = True

print("Step 1: values and types")
print(f"  city                 {city!r:<12} type={type(city).__name__}")
print(f"  days_recorded        {days_recorded!r:<12} type={type(days_recorded).__name__}")
print(f"  first temperature    {temperatures_celsius[0]!r:<12} type={type(temperatures_celsius[0]).__name__}")
print(f"  is_summer            {is_summer!r:<12} type={type(is_summer).__name__}")
print(f"  7 / 2 = {7 / 2}   7 // 2 = {7 // 2}   7 % 2 = {7 % 2}   2 ** 10 = {2 ** 10}")
print(f"  0.1 + 0.2 == 0.3 -> {0.1 + 0.2 == 0.3}   math.isclose -> {math.isclose(0.1 + 0.2, 0.3)}")


def mean_of(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def min_and_max(values):
    smallest = values[0]
    largest = values[0]
    for value in values[1:]:
        if value < smallest:
            smallest = value
        if value > largest:
            largest = value
    return smallest, largest


def variance_of(values):
    center = mean_of(values)
    squared_deviation_sum = 0.0
    for value in values:
        squared_deviation_sum += (value - center) ** 2
    return squared_deviation_sum / (len(values) - 1)


def median_of(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def square_root(number, tolerance=1e-12):
    guess = number if number > 1 else 1.0
    iterations = 0
    while abs(guess * guess - number) > tolerance:
        guess = 0.5 * (guess + number / guess)
        iterations += 1
    return guess, iterations


def describe_temperature(celsius):
    if celsius < 18:
        return "cold"
    elif celsius < 25:
        return "mild"
    else:
        return "hot"


print()
print("Step 2: summary statistics with loops")
print(f"  temperatures {temperatures_celsius}")
average = mean_of(temperatures_celsius)
lowest, highest = min_and_max(temperatures_celsius)
sample_variance = variance_of(temperatures_celsius)
standard_deviation, newton_steps = square_root(sample_variance)
middle_value = median_of(temperatures_celsius)
print(f"  mean     {average:.3f}")
print(f"  median   {middle_value:.3f}")
print(f"  min/max  {lowest} / {highest}")
print(f"  variance {sample_variance:.3f}")
print(f"  std      {standard_deviation:.3f}  (Newton's method, {newton_steps} iterations)")

print()
print("Step 3: control flow to label each day")
label_counts = {"cold": 0, "mild": 0, "hot": 0}
for day_number, celsius in enumerate(temperatures_celsius, start=1):
    label = describe_temperature(celsius)
    label_counts[label] += 1
    if day_number <= 5:
        print(f"  day {day_number:2d}: {celsius:5.1f} C -> {label}")
print(f"  counts: {label_counts}")

first_hot_day = None
for day_number, celsius in enumerate(temperatures_celsius, start=1):
    if celsius >= 25:
        first_hot_day = day_number
        break
print(f"  first hot day: {first_hot_day}")

print()
print("Use It: the standard library equivalents")
library_results = {
    "mean": (average, statistics.mean(temperatures_celsius)),
    "median": (middle_value, statistics.median(temperatures_celsius)),
    "variance": (sample_variance, statistics.variance(temperatures_celsius)),
    "std": (standard_deviation, statistics.stdev(temperatures_celsius)),
    "min": (lowest, min(temperatures_celsius)),
    "max": (highest, max(temperatures_celsius)),
}
for name, (ours, theirs) in library_results.items():
    match = "match" if math.isclose(ours, theirs, rel_tol=1e-9) else "MISMATCH"
    print(f"  {name:<9} ours={ours:9.4f}  library={theirs:9.4f}  {match}")
