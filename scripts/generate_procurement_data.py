"""Synthetic semiconductor materials procurement & supply-planning data generator.

Generates ~100,000 weekly rows at the ``week x fab_id x material_id`` grain,
covering ~3 years of history across 8 fabs and 80 materials (8 categories)
with a sparse fab-material mapping.

The data contains realistic temporal relationships so XGBoost can learn:
  - planned_wafer_starts ↑ → material_consumption_qty ↑
  - fab_utilization_pct ↑ → consumption ↑
  - recent consumption ↑ → material_demand_next_4w ↑  (via lag features)
  - inventory ↓ + lead_time ↑ → shortage_risk_next_4w ↑
  - scheduled_receipt_qty ↑ → shortage_risk ↓
  - supplier_otd_pct ↓ → shortage_risk ↑

Also includes trends, seasonal cycles, demand spikes, fab shutdowns, and
supplier lead-time disruptions.

Pre-computed columns (because Anistroph has no lag transform):
  - consumption_lag_{1,2,4,8,13}w  — per (fab, material) series, past-only
  - material_demand_next_4w        — sum of next 4 weeks' consumption (regression target)
  - shortage_risk_next_4w          — 1 if inventory < safety_stock in next 4 weeks (classification target)

Output: data/semiconductor_procurement/data.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_FABS = 8
N_MATERIALS = 100
N_WEEKS = 160  # ~3 years
SPARSE_DENSITY = 0.78  # fraction of (fab, material) pairs that are active

FAB_IDS = [f"FAB_{chr(ord('A') + i)}" for i in range(N_FABS)]

# Fab capacity tiers (planned wafer starts per week)
FAB_BASE_STARTS = {
    "FAB_A": 5000,
    "FAB_B": 4500,
    "FAB_C": 4200,
    "FAB_D": 3800,
    "FAB_E": 3500,
    "FAB_F": 3000,
    "FAB_G": 2500,
    "FAB_H": 2000,
}

# Material categories with consumption characteristics
# (category_name, n_materials, base_consumption_per_1k_wafers, unit_cost_range, lead_time_range)
MATERIAL_CATEGORIES = [
    ("Silicon Wafers",        13, 12.0, (50, 300),   (14, 42)),
    ("Photoresists",          13,  8.0, (200, 800),  (7, 28)),
    ("Process Gases",         13, 15.0, (100, 500),  (10, 35)),
    ("Wet Chemicals",         12, 20.0, (30, 150),   (7, 21)),
    ("Deposition Precursors", 12,  6.0, (500, 2000), (21, 60)),
    ("Sputtering Targets",    12,  3.0, (800, 3000), (28, 75)),
    ("CMP Materials",         12, 10.0, (100, 400),  (14, 35)),
    ("Packaging Materials",   13, 25.0, (20, 100),   (7, 21)),
]

N_SUPPLIERS = 15
SUPPLIER_IDS = [f"SUP_{i:02d}" for i in range(1, N_SUPPLIERS + 1)]

# Lag weeks to pre-compute
LAG_WEEKS = [1, 2, 4, 8, 13]

# Forecast horizon for target
FORECAST_HORIZON_WEEKS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_material_catalog(rng: np.random.Generator) -> list[dict]:
    """Build the 80-material catalog with category, spec, cost, lead time, MOQ."""
    materials = []
    idx = 1
    for cat_name, n_mats, base_consumption, cost_range, lead_range in MATERIAL_CATEGORIES:
        for _ in range(n_mats):
            mat_id = f"MAT_{idx:04d}"
            idx += 1
            # Per-material variation around category base
            consumption_factor = rng.uniform(0.7, 1.3)
            unit_cost = rng.uniform(*cost_range)
            base_lead_time = rng.uniform(*lead_range)
            moq = int(rng.choice([50, 100, 100, 200, 200, 500]))
            # Generate a spec string
            spec = f"SPEC_{cat_name.split()[0][:4].upper()}_{rng.choice(['A', 'B', 'C'])}{rng.integers(1, 9)}"
            materials.append({
                "material_id": mat_id,
                "material_category": cat_name,
                "material_spec": spec,
                "base_consumption_per_1k": base_consumption * consumption_factor,
                "unit_cost": round(unit_cost, 2),
                "base_lead_time_days": base_lead_time,
                "minimum_order_qty": moq,
            })
    return materials


def _build_fab_material_mapping(
    rng: np.random.Generator, materials: list[dict]
) -> list[tuple[str, str, str]]:
    """Build sparse (fab_id, material_id, supplier_id) mapping."""
    pairs = []
    for fab_id in FAB_IDS:
        for mat in materials:
            if rng.random() < SPARSE_DENSITY:
                supplier = rng.choice(SUPPLIER_IDS)
                pairs.append((fab_id, mat["material_id"], supplier))
    return pairs


def _generate_series(
    rng: np.random.Generator,
    fab_id: str,
    mat: dict,
    supplier_id: str,
    weeks: list[datetime],
    fab_base_starts: float,
) -> pl.DataFrame:
    """Generate one (fab, material) weekly time series."""
    n = len(weeks)
    mat_id = mat["material_id"]
    series_id = f"{fab_id}__{mat_id}"

    # --- Planned wafer starts (with trend + seasonality + ramps) ---
    t = np.arange(n, dtype=float)
    trend = 1.0 + 0.15 * t / n  # 15% growth over 3 years
    seasonal = 1.0 + 0.08 * np.sin(2 * np.pi * t / 52.0)  # annual cycle
    quarterly = 1.0 + 0.04 * np.sin(2 * np.pi * t / 13.0)  # quarterly cycle
    noise = rng.normal(1.0, 0.05, n)

    # Random production ramp (sustained increase) for some series
    ramp_factor = np.ones(n)
    if rng.random() < 0.3:
        ramp_start = rng.integers(20, n - 40)
        ramp_end = min(ramp_start + rng.integers(10, 30), n)
        ramp_magnitude = rng.uniform(1.1, 1.3)
        ramp_factor[ramp_start:] = np.linspace(1.0, ramp_magnitude, n - ramp_start)

    # Fab shutdown / maintenance (2-4 week gaps)
    shutdown_mask = np.zeros(n, dtype=bool)
    if rng.random() < 0.25:
        shut_start = rng.integers(10, n - 20)
        shut_len = rng.integers(2, 5)
        shut_end = min(shut_start + shut_len, n)
        shutdown_mask[shut_start:shut_end] = True

    planned_starts = fab_base_starts * trend * seasonal * quarterly * noise * ramp_factor
    planned_starts = np.clip(planned_starts, 500, None)
    planned_starts[shutdown_mask] = 0.0

    # --- Actual wafer starts (planned + small variance, 0 during shutdown) ---
    actual_starts = planned_starts * rng.uniform(0.90, 1.05, n)
    actual_starts[shutdown_mask] = 0.0

    # --- Fab utilization ---
    utilization = np.clip(40 + 50 * (actual_starts / fab_base_starts) +
                          rng.normal(0, 3, n), 30, 98)
    utilization[shutdown_mask] = 0.0

    # --- Material consumption qty ---
    # Driven by actual wafer starts and utilization
    base_consumption = mat["base_consumption_per_1k"]
    consumption = base_consumption * (actual_starts / 1000.0) * (utilization / 100.0)

    # Add demand spikes (random events)
    if rng.random() < 0.15:
        spike_idx = rng.integers(10, n - 10)
        spike_mag = rng.uniform(1.5, 2.5)
        consumption[spike_idx] *= spike_mag

    # Consumption noise
    consumption *= rng.normal(1.0, 0.06, n)
    consumption = np.clip(consumption, 0, None)
    consumption[shutdown_mask] = 0.0

    # --- Supplier lead time (with disruptions) ---
    lead_time = np.full(n, mat["base_lead_time_days"])
    if rng.random() < 0.2:
        # Lead time disruption (step change for a period)
        disrupt_start = rng.integers(20, n - 40)
        disrupt_len = rng.integers(8, 20)
        disrupt_end = min(disrupt_start + disrupt_len, n)
        disrupt_factor = rng.uniform(1.5, 2.5)
        lead_time[disrupt_start:disrupt_end] *= disrupt_factor
    lead_time += rng.normal(0, 2, n)
    lead_time = np.clip(lead_time, 3, 120)

    # --- Supplier OTD (on-time delivery %) ---
    otd_base = rng.uniform(82, 96)
    otd = np.full(n, otd_base)
    if rng.random() < 0.2:
        # OTD degradation period
        otd_start = rng.integers(20, n - 30)
        otd_len = rng.integers(10, 25)
        otd_end = min(otd_start + otd_len, n)
        otd_drop = rng.uniform(10, 20)
        otd[otd_start:otd_end] -= otd_drop
    otd += rng.normal(0, 1.5, n)
    otd = np.clip(otd, 60, 99)

    # --- Safety stock (relatively stable, category-dependent) ---
    # Set safety stock at ~1.5-2.5 weeks of average consumption (tight enough for shortages)
    avg_consumption = base_consumption * (fab_base_starts / 1000.0) * 0.75
    safety_stock = avg_consumption * rng.uniform(1.5, 2.5)
    safety_stock_qty = np.full(n, safety_stock)

    # --- Inventory simulation with lead-time-aware reorder ---
    # Inventory depletes with consumption each week.
    # When inventory drops below reorder_point, an order is placed.
    # The order arrives after ceil(lead_time / 7) weeks, adjusted by OTD.
    # Poor OTD → order may be short or delayed → shortage risk.
    reorder_point = safety_stock * rng.uniform(1.0, 1.3)
    inventory = np.empty(n)
    open_po = np.zeros(n)
    scheduled_receipt = np.zeros(n)
    # Pending orders: list of (arrival_week, qty)
    pending_orders: list[tuple[int, float]] = []

    inventory[0] = safety_stock * rng.uniform(2.0, 3.5)

    for i in range(n):
        # Receive any orders arriving this week
        arrived = [(wk, qty) for (wk, qty) in pending_orders if wk <= i]
        pending_orders = [(wk, qty) for (wk, qty) in pending_orders if wk > i]
        total_arrived = sum(qty for _, qty in arrived)
        scheduled_receipt[i] = total_arrived

        if i > 0:
            inventory[i] = inventory[i-1] - consumption[i-1] + total_arrived
        else:
            inventory[i] = inventory[0] + total_arrived
        inventory[i] = max(inventory[i], 0.0)

        # Place reorder if inventory below reorder point
        if inventory[i] < reorder_point and consumption[i] > 0:
            # Order enough for ~3-4 weeks of consumption
            order_qty = max(mat["minimum_order_qty"] * rng.uniform(2, 5),
                           avg_consumption * rng.uniform(3, 5))
            # Lead time in weeks (ceil), with OTD penalty
            lead_weeks = int(np.ceil(lead_time[i] / 7.0))
            # Poor OTD adds extra delay
            if rng.random() > otd[i] / 100.0:
                lead_weeks += rng.integers(1, 3)
            arrival_week = i + lead_weeks
            if arrival_week < n:
                pending_orders.append((arrival_week, order_qty))
            open_po[i] = order_qty
        else:
            open_po[i] = max(0.0, safety_stock * rng.uniform(0.1, 0.5) - inventory[i])

    open_po = np.clip(open_po, 0, None)

    # --- Unit cost (slight drift) ---
    unit_cost = np.full(n, mat["unit_cost"])
    unit_cost *= (1.0 + 0.05 * t / n + rng.normal(0, 0.02, n))
    unit_cost = np.clip(unit_cost, mat["unit_cost"] * 0.8, mat["unit_cost"] * 1.3)

    # --- Minimum order qty (stable) ---
    moq = np.full(n, mat["minimum_order_qty"], dtype=float)

    # --- Pre-compute lag features ---
    lag_cols = {}
    for lag in LAG_WEEKS:
        lag_vals = np.full(n, np.nan)
        if lag < n:
            lag_vals[lag:] = consumption[:n - lag]
        # Fill initial nulls with 0 (no history)
        lag_vals[:lag] = 0.0
        lag_cols[f"consumption_lag_{lag}w"] = lag_vals

    # --- Pre-compute targets ---
    # material_demand_next_4w: sum of next 4 weeks' consumption
    demand_next_4w = np.full(n, np.nan)
    for i in range(n):
        end = min(i + FORECAST_HORIZON_WEEKS, n)
        if end > i + 1:
            demand_next_4w[i] = consumption[i+1:end].sum()
    # Fill last 4 weeks with 0 (no future data — will be dropped before training)
    demand_next_4w[n - FORECAST_HORIZON_WEEKS:] = 0.0

    # shortage_risk_next_4w: 1 if inventory < safety_stock in any of next 4 weeks
    shortage_risk = np.zeros(n, dtype=np.int8)
    for i in range(n - FORECAST_HORIZON_WEEKS):
        future_inv = inventory[i+1:i+1+FORECAST_HORIZON_WEEKS]
        if np.any(future_inv < safety_stock_qty[i+1:i+1+FORECAST_HORIZON_WEEKS]):
            shortage_risk[i] = 1

    # --- Build DataFrame ---
    data = {
        "week": weeks,
        "series_id": [series_id] * n,
        "fab_id": [fab_id] * n,
        "material_id": [mat_id] * n,
        "material_category": [mat["material_category"]] * n,
        "material_spec": [mat["material_spec"]] * n,
        "supplier_id": [supplier_id] * n,
        "planned_wafer_starts": planned_starts,
        "actual_wafer_starts": actual_starts,
        "fab_utilization_pct": utilization,
        "material_consumption_qty": consumption,
        "inventory_on_hand": inventory,
        "safety_stock_qty": safety_stock_qty,
        "open_po_qty": open_po,
        "scheduled_receipt_qty": scheduled_receipt,
        "supplier_lead_time_days": lead_time,
        "supplier_otd_pct": otd,
        "unit_cost": unit_cost,
        "minimum_order_qty": moq,
    }
    data.update(lag_cols)
    data["material_demand_next_4w"] = demand_next_4w
    data["shortage_risk_next_4w"] = shortage_risk

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_procurement_data(
    n_fabs: int = N_FABS,
    n_materials: int = N_MATERIALS,
    n_weeks: int = N_WEEKS,
    density: float = SPARSE_DENSITY,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate the full semiconductor procurement dataset."""
    rng = np.random.default_rng(seed)

    # Build material catalog
    materials = _build_material_catalog(rng)
    assert len(materials) == n_materials

    # Build sparse fab-material-supplier mapping
    pairs = _build_fab_material_mapping(rng, materials)

    # Generate weekly timestamps (Mondays)
    start = datetime(2023, 1, 1)
    # Align to first Monday
    while start.weekday() != 0:
        start += timedelta(days=1)
    weeks = [start + timedelta(weeks=i) for i in range(n_weeks)]

    print(f"  Generating {len(pairs)} series x {n_weeks} weeks = ~{len(pairs) * n_weeks} rows...")

    # Generate each series
    frames = []
    for fab_id, mat_id, supplier_id in pairs:
        mat = next(m for m in materials if m["material_id"] == mat_id)
        fab_base = FAB_BASE_STARTS[fab_id]
        df = _generate_series(rng, fab_id, mat, supplier_id, weeks, fab_base)
        frames.append(df)

    # Concatenate all series
    df = pl.concat(frames, how="vertical_relaxed")

    # Sort by series_id then week for clean ordering
    df = df.sort(["series_id", "week"])

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic semiconductor procurement & supply-planning data."
    )
    parser.add_argument("--fabs", type=int, default=N_FABS, help="Number of fabs.")
    parser.add_argument("--materials", type=int, default=N_MATERIALS, help="Number of materials.")
    parser.add_argument("--weeks", type=int, default=N_WEEKS, help="Number of weeks of history.")
    parser.add_argument("--density", type=float, default=SPARSE_DENSITY,
                        help="Fraction of (fab, material) pairs that are active.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default="data/semiconductor_procurement/data.parquet",
                        help="Output Parquet path.")
    args = parser.parse_args()

    print(f"Generating semiconductor procurement data (seed={args.seed})...")
    print(f"  {args.fabs} fabs, {args.materials} materials, {args.weeks} weeks, "
          f"density={args.density}")

    df = generate_procurement_data(
        n_fabs=args.fabs,
        n_materials=args.materials,
        n_weeks=args.weeks,
        density=args.density,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"\nWrote {df.height} rows x {df.width} columns to {out}")

    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"  Series (fab x material): {df['series_id'].n_unique()}")
    print(f"  Fabs: {df['fab_id'].n_unique()}")
    print(f"  Materials: {df['material_id'].n_unique()}")
    print(f"  Suppliers: {df['supplier_id'].n_unique()}")
    print(f"  Categories: {df['material_category'].n_unique()}")
    print(f"  Date range: {df['week'].min()} to {df['week'].max()}")

    print(f"\n  material_consumption_qty: mean={df['material_consumption_qty'].mean():.1f}, "
          f"min={df['material_consumption_qty'].min():.1f}, "
          f"max={df['material_consumption_qty'].max():.1f}")
    print(f"  material_demand_next_4w: mean={df['material_demand_next_4w'].mean():.1f}, "
          f"min={df['material_demand_next_4w'].min():.1f}, "
          f"max={df['material_demand_next_4w'].max():.1f}")
    print(f"  shortage_risk_next_4w: positive_rate={df['shortage_risk_next_4w'].mean():.3f}")

    print(f"\n  By category (mean consumption):")
    for cat in sorted(df['material_category'].unique()):
        sub = df.filter(pl.col('material_category') == cat)
        print(f"    {cat}: n={sub.height}, mean_consumption={sub['material_consumption_qty'].mean():.1f}")


if __name__ == "__main__":
    main()
