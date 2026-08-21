#!/usr/bin/env python3
"""Generate synthetic weekly supply history for the semiconductor memory catalog.

Reads the fixed 2,000-product reference catalog and generates weekly supply
observations for each product_id. The catalog itself is never modified —
this script produces a separate temporal supply history dataset used for
training the supply_risk_next_4w (classification) and
lead_time_next_4w_days (regression) prediction models.

Design:
  - 2,000 products × 25 weeks = 50,000 rows
  - Fixed seed (42) for reproducibility
  - Per-product baseline supply state drawn from the catalog snapshot
  - Weekly dynamics: demand noise, inventory drift, backlog fluctuations,
    supplier OTD variation, allocation status transitions
  - Targets derived from forward-looking supply state:
      supply_risk_next_4w: 1 if inventory coverage drops below 1.0 week
        OR allocation becomes Constrained within the next 4 weeks
      lead_time_next_4w_days: expected replenishment lead time, driven by
        supplier_lead_time_days, OTD, and allocation pressure

Usage:
    python scripts/generate_semiconductor_memory_supply.py
    # -> data/semiconductor_memory_supply/data.parquet (50,000 rows)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "semiconductor_memory" / "data.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "semiconductor_memory_supply" / "data.parquet"

SEED = 42
N_WEEKS = 25
START_WEEK_OFFSET = 0  # weeks from 2025-01-06 (first Monday)


def generate_supply_history(catalog: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Generate weekly supply history for every product in the catalog."""
    product_ids = catalog["product_id"].to_list()
    n_products = len(product_ids)

    # --- Per-product baselines from the catalog snapshot ---
    # Use the snapshot values as the initial state for week 0, then evolve.
    base_inventory = catalog["inventory_units"].to_numpy().astype(float)
    base_demand = catalog["weekly_demand_units"].to_numpy().astype(float)
    base_backlog = catalog["backlog_units"].to_numpy().astype(float)
    base_open_po = catalog["open_po_units"].to_numpy().astype(float)
    base_lead_time = catalog["supplier_lead_time_days"].to_numpy().astype(float)
    base_otd = catalog["supplier_otd_pct"].to_numpy().astype(float)

    # Product family influences supply dynamics (modules have longer lead times).
    family = catalog["product_family"].to_list()
    family_factor = np.array([
        1.0 if f == "DDR5_COMPONENT" else
        1.1 if f == "LPDDR5X_COMPONENT" else
        1.3 if f == "DDR5_RDIMM" else
        1.2  # DDR5_UDIMM
        for f in family
    ], dtype=float)

    # --- Generate week-by-week evolution ---
    rows: list[dict] = []

    # State arrays (per-product, updated each week).
    inventory = base_inventory.copy()
    demand = base_demand.copy()
    backlog = base_backlog.copy()
    open_po = base_open_po.copy()
    lead_time = base_lead_time.copy()
    otd = base_otd.copy()

    # Allocation status: 0=None, 1=Watch, 2=Constrained
    alloc_status = catalog["allocation_status"].to_list()
    alloc_code = np.array([0 if a == "None" else 1 if a == "Watch" else 2 for a in alloc_status], dtype=int)

    # Demand trend (4-week rolling pct change).
    demand_history = np.tile(demand, (4, 1))  # last 4 weeks for trend calc

    for week_idx in range(N_WEEKS):
        week_date = np.datetime64("2025-01-06") + np.timedelta64(week_idx * 7, "D")

        # --- Evolve supply state ---
        # Demand: random walk with mean reversion to baseline.
        demand_noise = rng.normal(0, 0.08, n_products)
        demand = demand * (1 + demand_noise) + 0.1 * (base_demand - demand)
        demand = np.clip(demand, 10, 1000)

        # Inventory: decreases by demand, increases by arrivals (from open POs).
        arrival_rate = rng.uniform(0.15, 0.35, n_products)
        arrivals = open_po * arrival_rate
        inventory = inventory - demand + arrivals
        inventory = np.clip(inventory, 0, 10000)

        # Backlog: random walk, increases when demand > arrivals.
        backlog_drift = (demand - arrivals) * 0.3
        backlog_noise = rng.normal(0, 5, n_products)
        backlog = np.clip(backlog + backlog_drift + backlog_noise, 0, 3000)

        # Open POs: replenished randomly, depleted by arrivals.
        po_replenish = rng.uniform(50, 300, n_products) * family_factor
        open_po = np.clip(open_po - arrivals + po_replenish, 0, 5000)

        # Supplier lead time: random walk with family factor.
        lt_noise = rng.normal(0, 1.5, n_products)
        lead_time = np.clip(lead_time + lt_noise * family_factor, 10, 120)

        # OTD: random walk, mean-reverting to baseline.
        otd_noise = rng.normal(0, 0.8, n_products)
        otd = otd + otd_noise + 0.05 * (base_otd - otd)
        otd = np.clip(otd, 60, 100)

        # Allocation status transitions based on inventory coverage.
        coverage = inventory / np.maximum(demand, 1)
        # Low coverage → Watch or Constrained; high coverage → None.
        for i in range(n_products):
            if coverage[i] < 0.8:
                alloc_code[i] = 2  # Constrained
            elif coverage[i] < 1.5:
                # Transition toward Watch with some probability.
                if rng.random() < 0.3:
                    alloc_code[i] = max(alloc_code[i], 1)
            elif coverage[i] > 3.0:
                if rng.random() < 0.2:
                    alloc_code[i] = max(0, alloc_code[i] - 1)

        # Demand trend: pct change over last 4 weeks.
        demand_history = np.roll(demand_history, -1, axis=0)
        demand_history[-1] = demand
        avg_past = demand_history[:-1].mean(axis=0)
        demand_trend = ((demand - avg_past) / np.maximum(avg_past, 1)) * 100
        demand_trend = np.clip(demand_trend, -50, 100)

        # Derived fields.
        inventory_coverage = inventory / np.maximum(demand, 1)
        backlog_ratio = backlog / np.maximum(demand, 1)
        open_po_coverage = open_po / np.maximum(demand, 1)

        # --- Compute forward-looking targets (next 4 weeks) ---
        # Simulate 4 weeks ahead to determine targets.
        future_inventory = inventory.copy()
        future_demand = demand.copy()
        future_alloc = alloc_code.copy()
        future_lead_time = lead_time.copy()
        future_otd = otd.copy()

        min_coverage_next_4w = inventory_coverage.copy()
        max_lead_time_next_4w = lead_time.copy()
        became_constrained = np.zeros(n_products, dtype=bool)

        for fw in range(4):
            f_demand_noise = rng.normal(0, 0.08, n_products)
            future_demand = future_demand * (1 + f_demand_noise) + 0.1 * (base_demand - future_demand)
            future_demand = np.clip(future_demand, 10, 1000)

            f_arrival_rate = rng.uniform(0.15, 0.35, n_products)
            f_arrivals = open_po * f_arrival_rate * (future_otd / 100)
            future_inventory = future_inventory - future_demand + f_arrivals
            future_inventory = np.clip(future_inventory, 0, 10000)

            f_coverage = future_inventory / np.maximum(future_demand, 1)
            min_coverage_next_4w = np.minimum(min_coverage_next_4w, f_coverage)

            # Allocation can worsen.
            for i in range(n_products):
                if f_coverage[i] < 0.8:
                    if future_alloc[i] < 2:
                        became_constrained[i] = True
                    future_alloc[i] = 2

            # Lead time evolution.
            f_lt_noise = rng.normal(0, 1.5, n_products)
            future_lead_time = np.clip(future_lead_time + f_lt_noise * family_factor, 10, 120)
            max_lead_time_next_4w = np.maximum(max_lead_time_next_4w, future_lead_time)

            f_otd_noise = rng.normal(0, 0.8, n_products)
            future_otd = future_otd + f_otd_noise + 0.05 * (base_otd - future_otd)
            future_otd = np.clip(future_otd, 60, 100)

        # supply_risk_next_4w: 1 if coverage drops below 1.0 week OR
        # allocation becomes Constrained.
        supply_risk = (
            (min_coverage_next_4w < 1.0) | became_constrained
        ).astype(int)

        # lead_time_next_4w_days: weighted average of current and max future
        # lead time, adjusted by OTD and allocation pressure.
        alloc_pressure = alloc_code / 2.0  # 0, 0.5, 1.0
        otd_factor = 1.0 + (100 - otd) / 200  # worse OTD → longer effective lead time
        lead_time_target = (
            lead_time * 0.4 + max_lead_time_next_4w * 0.6
        ) * otd_factor * (1 + alloc_pressure * 0.2)
        lead_time_target = np.clip(lead_time_target, 7, 160)

        # --- Emit rows for this week ---
        alloc_labels = np.where(alloc_code == 0, "None",
                       np.where(alloc_code == 1, "Watch", "Constrained"))

        for i in range(n_products):
            rows.append({
                "product_id": product_ids[i],
                "week": str(week_date),
                "inventory_units": int(round(inventory[i])),
                "weekly_demand_units": int(round(demand[i])),
                "inventory_coverage_weeks": round(float(inventory_coverage[i]), 2),
                "backlog_units": int(round(backlog[i])),
                "backlog_ratio": round(float(backlog_ratio[i]), 3),
                "open_po_units": int(round(open_po[i])),
                "open_po_coverage": round(float(open_po_coverage[i]), 2),
                "supplier_lead_time_days": int(round(lead_time[i])),
                "supplier_otd_pct": round(float(otd[i]), 1),
                "demand_trend_4w_pct": round(float(demand_trend[i]), 1),
                "allocation_status": alloc_labels[i],
                "supply_risk_next_4w": int(supply_risk[i]),
                "lead_time_next_4w_days": round(float(lead_time_target[i]), 1),
            })

    df = pl.DataFrame(rows)
    # Cast week string to Date for temporal rolling features.
    df = df.with_columns(pl.col("week").str.to_date("%Y-%m-%d"))
    return df


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"ERROR: catalog not found at {CATALOG_PATH}", file=sys.stderr)
        print("Run setup_datasets.py first to copy the catalog CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading catalog: {CATALOG_PATH}")
    catalog = pl.read_csv(str(CATALOG_PATH), try_parse_dates=True)
    print(f"  {catalog.height} products")

    print(f"Generating supply history (seed={SEED}, {N_WEEKS} weeks)...")
    rng = np.random.default_rng(SEED)
    df = generate_supply_history(catalog, rng)
    print(f"  Generated {df.height} rows, {df.width} columns")

    # Persist as Parquet.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(OUTPUT_PATH))
    print(f"  Written to {OUTPUT_PATH}")

    # Summary stats.
    print("\nSummary:")
    print(f"  Products: {df['product_id'].n_unique()}")
    print(f"  Weeks: {df['week'].n_unique()}")
    print(f"  supply_risk_next_4w distribution: {df['supply_risk_next_4w'].value_counts().sort('supply_risk_next_4w').to_dicts()}")
    print(f"  lead_time_next_4w_days: min={df['lead_time_next_4w_days'].min()}, max={df['lead_time_next_4w_days'].max()}, mean={df['lead_time_next_4w_days'].mean():.1f}")
    print(f"  allocation_status: {df['allocation_status'].value_counts().sort('count', descending=True).to_dicts()}")


if __name__ == "__main__":
    main()
