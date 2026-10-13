import csv
import io
import math
import random

import pandas as pd

random.seed(11)
CITIES = ["Madrid", "Valencia", "Sevilla"]
BASE_PRICE_PER_M2 = {"Madrid": 4200, "Valencia": 2300, "Sevilla": 2000}

csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(["listing_id", "city", "size_m2", "rooms", "price_eur"])
for listing_id in range(1, 31):
    city = random.choice(CITIES)
    size = round(random.uniform(40, 140))
    rooms = max(1, round(size / 35))
    price = round(size * BASE_PRICE_PER_M2[city] * random.uniform(0.85, 1.15), -3)
    size_text = "" if listing_id in (4, 17) else str(size)
    writer.writerow([listing_id, city, size_text, rooms, price])
CSV_TEXT = csv_buffer.getvalue()

CITY_INFO_TEXT = "city,region,population_millions\nMadrid,Comunidad de Madrid,3.4\nValencia,Comunitat Valenciana,0.8\nSevilla,Andalucia,0.7\n"


def read_records(text):
    records = []
    for row in csv.DictReader(io.StringIO(text)):
        records.append({
            "listing_id": int(row["listing_id"]),
            "city": row["city"],
            "size_m2": float(row["size_m2"]) if row["size_m2"] else None,
            "rooms": int(row["rooms"]),
            "price_eur": float(row["price_eur"]),
        })
    return records


def fill_missing_with_mean(records, column):
    present = [record[column] for record in records if record[column] is not None]
    column_mean = sum(present) / len(present)
    return [{**record, column: column_mean if record[column] is None else record[column]} for record in records]


def group_mean(records, key, value):
    sums = {}
    counts = {}
    for record in records:
        group = record[key]
        sums[group] = sums.get(group, 0.0) + record[value]
        counts[group] = counts.get(group, 0) + 1
    return {group: sums[group] / counts[group] for group in sorted(sums)}


def left_join(left_records, right_records, key):
    right_by_key = {record[key]: record for record in right_records}
    return [{**left, **right_by_key.get(left[key], {})} for left in left_records]


print("Build It: plain Python records")
records = read_records(CSV_TEXT)
print(f"  {len(records)} records, first: {records[0]}")
missing_sizes = [record["listing_id"] for record in records if record["size_m2"] is None]
print(f"  listings with missing size: {missing_sizes}")
records = fill_missing_with_mean(records, "size_m2")
records = [{**record, "price_per_m2": record["price_eur"] / record["size_m2"]} for record in records]
ours_group_means = group_mean(records, "city", "price_per_m2")
print(f"  mean price per m2 by city: { {city: round(value) for city, value in ours_group_means.items()} }")
city_info = [{"city": row["city"], "region": row["region"]} for row in csv.DictReader(io.StringIO(CITY_INFO_TEXT))]
joined = left_join(records, city_info, "city")
print(f"  joined first record region: {joined[0]['region']}")

print()
print("Use It: pandas")
listings = pd.read_csv(io.StringIO(CSV_TEXT))
print(f"  shape {listings.shape}")
print(f"  dtypes:\n{listings.dtypes.to_string()}")
print(f"\n  head(5):\n{listings.head(5).to_string()}")
print(f"\n  describe() of numeric columns:\n{listings[['size_m2', 'rooms', 'price_eur']].describe().round(1).to_string()}")

print(f"\n  missing values per column:\n{listings.isna().sum().to_string()}")
listings["size_m2"] = listings["size_m2"].fillna(listings["size_m2"].mean())
listings["price_per_m2"] = listings["price_eur"] / listings["size_m2"]

big_in_madrid = listings[(listings["city"] == "Madrid") & (listings["size_m2"] > 100)]
print(f"\n  Madrid listings over 100 m2:\n{big_in_madrid[['listing_id', 'size_m2', 'price_eur']].to_string(index=False)}")

summary = listings.groupby("city").agg(
    listings=("listing_id", "count"),
    mean_price=("price_eur", "mean"),
    mean_price_per_m2=("price_per_m2", "mean"),
).round(0)
print(f"\n  groupby('city').agg(...):\n{summary.to_string()}")

city_table = pd.read_csv(io.StringIO(CITY_INFO_TEXT))
merged = listings.merge(city_table, on="city", how="left")
print(f"\n  after merge: columns {list(merged.columns)}")

pivot = listings.pivot_table(index="city", columns="rooms", values="price_eur", aggfunc="mean").round(-3)
print(f"\n  pivot_table mean price by city x rooms:\n{pivot.to_string()}")

pandas_group_means = listings.groupby("city")["price_per_m2"].mean().to_dict()
all_match = all(math.isclose(ours_group_means[city], pandas_group_means[city]) for city in CITIES)
print(f"\n  plain-Python group means match pandas: {all_match}")
print(f"  plain-Python join matches pandas merge: {[r['region'] for r in joined] == merged['region'].tolist()}")
feature_matrix = merged[["size_m2", "rooms"]].to_numpy()
print(f"  to_numpy() for a model: shape {feature_matrix.shape}, dtype {feature_matrix.dtype}")
