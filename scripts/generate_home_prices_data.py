"""Synthetic Bay Area home-price data generator.

Generates ~20,000 home listing rows across three Silicon Valley cities:
San Jose, Saratoga, and Los Gatos. The model predicts ``price`` (regression)
primarily from square footage, with city/zip as the dominant price driver.

Pricing hierarchy (highest -> lowest):
  Saratoga > Los Gatos > San Jose

San Jose median is calibrated to ~$1.8MM for a 1600 sq ft home
(~$1,125/sq ft), per the user's anchoring constraint.

Price is driven by:
  - city / zip_code (dominant — Saratoga ~$1,750/sq ft, Los Gatos ~$1,400,
    San Jose ~$1,100 with per-zip variation)
  - sqft (primary continuous driver, with diminishing returns at high sqft)
  - bedrooms / bathrooms (small premia)
  - year_built (newer homes command a premium)
  - lot_size_sqft (larger lots add value)
  - garage stalls (small premium)
  - zip-level noise so no single feature perfectly determines price
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


# City -> list of (zip_code, price_per_sqft_base)
# San Jose zips span a range; 95124/95125/95129 are pricier (Willow Glen, Cambrian)
# while 95112/95122 are more affordable (downtown/east SJ).
CITY_ZIPS: dict[str, list[tuple[str, float]]] = {
    "San Jose": [
        ("95112", 1020.0),
        ("95116", 990.0),
        ("95122", 980.0),
        ("95118", 1125.0),
        ("95124", 1198.0),
        ("95125", 1218.0),
        ("95126", 1145.0),
        ("95129", 1250.0),
        ("95132", 1093.0),
        ("95135", 1166.0),
        ("95148", 1072.0),
    ],
    "Los Gatos": [
        ("95030", 1450.0),
        ("95032", 1380.0),
    ],
    "Saratoga": [
        ("95070", 1750.0),
        ("95071", 1700.0),
    ],
}

# Sampling weights per city so San Jose dominates the dataset (more inventory),
# Los Gatos is mid-sized, Saratoga is smallest.
CITY_WEIGHTS: dict[str, float] = {
    "San Jose": 0.70,
    "Los Gatos": 0.18,
    "Saratoga": 0.12,
}


def _pick_city_zip(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample (city, zip) pairs according to CITY_WEIGHTS, uniform within city."""
    cities = list(CITY_ZIPS.keys())
    city_probs = np.array([CITY_WEIGHTS[c] for c in cities])
    city_probs = city_probs / city_probs.sum()
    city_choices = rng.choice(cities, size=n, p=city_probs)

    zips = np.empty(n, dtype=object)
    for i, city in enumerate(city_choices):
        zips[i] = rng.choice([z for z, _ in CITY_ZIPS[city]])
    return city_choices, zips


def generate_homes(
    n_homes: int = 20_000,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic Bay Area home-price dataset."""
    rng = np.random.default_rng(seed)

    # --- Identifiers ---
    property_ids = [f"PROP_{i:06d}" for i in range(n_homes)]

    # Listing timestamps spread over ~18 months, chronological.
    start = datetime(2024, 6, 1)
    timestamps = [start + timedelta(hours=int(i * (13140.0 / n_homes)))
                  for i in range(n_homes)]

    # --- City / zip ---
    cities, zips = _pick_city_zip(rng, n_homes)

    # --- Square footage: 1500 - 3800, skewed toward mid-range ---
    sqft = rng.triangular(1500, 2400, 3800, n_homes).astype(int)
    sqft = np.clip(sqft, 1500, 3800)

    # --- Bedrooms / bathrooms derived from sqft (with noise) ---
    # Rough heuristic: 1 bedroom per ~600 sqft, 1 bathroom per ~700 sqft.
    bedrooms = np.clip(np.round(sqft / 600.0 + rng.normal(0, 0.4, n_homes)), 2, 6).astype(int)
    bathrooms = np.clip(np.round(sqft / 700.0 + rng.normal(0, 0.3, n_homes)) * 0.5, 1.0, 5.0)
    bathrooms = np.round(bathrooms * 2) / 2  # snap to half-baths

    # --- Lot size: correlated with sqft but with wide variation ---
    lot_size_sqft = sqft * rng.uniform(2.5, 6.0, n_homes)
    lot_size_sqft = lot_size_sqft.astype(int)

    # --- Year built: 1950 - 2024, weighted older in San Jose ---
    year_built = np.empty(n_homes, dtype=int)
    for i, city in enumerate(cities):
        if city == "San Jose":
            year_built[i] = int(rng.normal(1972, 18))
        elif city == "Los Gatos":
            year_built[i] = int(rng.normal(1985, 16))
        else:  # Saratoga
            year_built[i] = int(rng.normal(1978, 17))
    year_built = np.clip(year_built, 1950, 2024)

    # --- Garage stalls: 0 - 3 ---
    garage = rng.choice([1, 2, 2, 2, 3], size=n_homes)

    # --- Price model ---
    # Base price-per-sqft lookup by zip.
    zip_ppsqft = {z: p for zips_list in CITY_ZIPS.values() for z, p in zips_list}

    price = np.empty(n_homes, dtype=float)
    for i in range(n_homes):
        zip_code = zips[i]
        base_ppsqft = zip_ppsqft[zip_code]

        # Diminishing returns: larger homes have slightly lower $/sqft.
        sqft_factor = 1.0 - 0.00012 * (sqft[i] - 1600)
        ppsqft = base_ppsqft * sqft_factor

        # Bedroom premium (each extra bedroom above 3 adds ~1.5%).
        ppsqft *= 1.0 + 0.015 * max(0, bedrooms[i] - 3)

        # Bathroom premium (each full bath above 2 adds ~1%).
        ppsqft *= 1.0 + 0.01 * max(0, bathrooms[i] - 2.0)

        # Year-built premium: newer than 2000 adds up to ~6%, older than 1960 loses ~3%.
        if year_built[i] >= 2000:
            ppsqft *= 1.0 + 0.003 * (year_built[i] - 2000)
        elif year_built[i] < 1960:
            ppsqft *= 1.0 - 0.001 * (1960 - year_built[i])

        # Lot-size premium: each extra 1000 sqft of lot above 6000 adds ~0.4%.
        if lot_size_sqft[i] > 6000:
            ppsqft *= 1.0 + 0.004 * (lot_size_sqft[i] - 6000) / 1000.0

        # Garage premium: 2-car vs 1-car adds ~1%, 3-car adds ~2.5%.
        ppsqft *= 1.0 + 0.01 * (garage[i] - 1)

        # Zip-level noise so the relationship is learnable but imperfect.
        ppsqft *= rng.normal(1.0, 0.04)

        price[i] = ppsqft * sqft[i]

    # Round price to nearest $1,000.
    price = np.round(price / 1000.0) * 1000.0

    # --- Build DataFrame ---
    df = pl.DataFrame({
        "timestamp": timestamps,
        "property_id": property_ids,
        "city": cities,
        "zip_code": zips,
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "lot_size_sqft": lot_size_sqft,
        "year_built": year_built,
        "garage": garage,
        "price": price.astype(int),
    })

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Bay Area home-price data.")
    parser.add_argument("--homes", type=int, default=20_000, help="Number of home rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="data/home_prices/data.parquet",
                        help="Output Parquet path.")
    args = parser.parse_args()

    print(f"Generating {args.homes} home rows (seed={args.seed})...")
    df = generate_homes(n_homes=args.homes, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"Wrote {df.height} rows x {df.width} columns to {out}")

    # Quick summary.
    price_col = df["price"]
    print(f"  price: min=${price_col.min():,}, mean=${price_col.mean():,.0f}, max=${price_col.max():,}")

    # Verify city hierarchy and San Jose anchor.
    for city in ["San Jose", "Los Gatos", "Saratoga"]:
        sub = df.filter(pl.col("city") == city)
        print(f"  {city}: n={sub.height}, median=${sub['price'].median():,.0f}, "
              f"median $/sqft=${(sub['price'] / sub['sqft']).median():,.0f}")

    sj_1600 = df.filter((pl.col("city") == "San Jose") & (pl.col("sqft").is_between(1550, 1650)))
    print(f"  San Jose ~1600 sqft: n={sj_1600.height}, median=${sj_1600['price'].median():,.0f} "
          f"(target ~$1.8MM)")


if __name__ == "__main__":
    main()
