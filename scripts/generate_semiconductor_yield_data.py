"""Synthetic semiconductor wafer-yield data generator.

Generates ~50,000 wafer rows (one row = one completed wafer) with
intentionally learnable but imperfect yield relationships.

Three regression targets:
  - wafer_yield: overall yield (0.0-1.0), driven by etch/deposition interactions
  - critical_dimension_nm: measured CD after lithography/etch (~37-38 nm)
  - film_thickness_nm: measured deposited film thickness (~500-520 nm)

Yield is driven by:
  - etch_tool / etch_chamber interactions (ETCH_02 + CH_B = bad)
  - etch temperature variability
  - deposition tool / pressure variability
  - product × recipe interactions
  - maintenance age × tool interactions
  - process_route × temperature variability

CD is driven by:
  - exposure_dose (higher dose → smaller CD)
  - focus_offset (larger |offset| → wider CD, more variation)
  - etch_tool / etch_recipe (ETCH_02 tends to over-etch → smaller CD)
  - etch_temperature (higher temp → faster etch → smaller CD)

Film thickness is driven by:
  - deposition_tool / deposition_recipe (DEP_02 tends thinner)
  - deposition_process_time (longer → thicker, with diminishing returns)
  - deposition_pressure (higher pressure → denser, slightly thicker)
  - deposition_temperature (higher temp → more uniform, slightly thicker)

Baseline yield ~96-98%. Degraded conditions can drop yield to ~87-93%.
No single feature perfectly determines any target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


ETCH_TOOLS = ["ETCH_01", "ETCH_02", "ETCH_03"]
ETCH_CHAMBERS = ["CH_A", "CH_B"]
ETCH_RECIPES = ["RECIPE_A", "RECIPE_B", "RECIPE_C"]

DEPOSITION_TOOLS = ["DEP_01", "DEP_02", "DEP_03"]
DEPOSITION_CHAMBERS = ["DCH_A", "DCH_B"]
DEPOSITION_RECIPES = ["DEP_RECIPE_A", "DEP_RECIPE_B"]

PRODUCTS = ["PROD_A", "PROD_B", "PROD_C"]
FABS = ["FAB_01", "FAB_02"]
PROCESS_ROUTES = ["ROUTE_1", "ROUTE_2", "ROUTE_3"]


def generate_wafers(
    n_wafers: int = 50_000,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic semiconductor wafer-yield dataset."""
    rng = np.random.default_rng(seed)

    # --- Lot / wafer structure ---
    wafers_per_lot = 25
    n_lots = max(1, n_wafers // wafers_per_lot)
    lot_ids = [f"LOT_{i:05d}" for i in range(n_lots)]
    wafer_ids = [f"WAFER_{i:06d}" for i in range(n_wafers)]

    # Assign lots to wafers (25 wafers per lot).
    lot_assignment = [lot_ids[i // wafers_per_lot] for i in range(n_wafers)]

    # Timestamps: spread over ~2 years, chronological.
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(hours=int(i * (17520.0 / n_wafers)))
                  for i in range(n_wafers)]

    # --- Categorical assignments (random) ---
    product = rng.choice(PRODUCTS, size=n_wafers)
    fab = rng.choice(FABS, size=n_wafers)
    process_route = rng.choice(PROCESS_ROUTES, size=n_wafers)

    etch_tool = rng.choice(ETCH_TOOLS, size=n_wafers)
    etch_chamber = rng.choice(ETCH_CHAMBERS, size=n_wafers)
    etch_recipe = rng.choice(ETCH_RECIPES, size=n_wafers)

    dep_tool = rng.choice(DEPOSITION_TOOLS, size=n_wafers)
    dep_chamber = rng.choice(DEPOSITION_CHAMBERS, size=n_wafers)
    dep_recipe = rng.choice(DEPOSITION_RECIPES, size=n_wafers)

    # --- Numeric process parameters ---
    # Etch process: temperature, pressure, gas flow, RF power, process time.
    etch_temp_mean = rng.normal(85.0, 3.0, n_wafers)
    # Temperature std is a key driver — higher variability = worse yield.
    etch_temp_std = np.abs(rng.normal(1.5, 0.8, n_wafers))
    etch_pressure_mean = rng.normal(50.0, 2.0, n_wafers)
    etch_pressure_std = np.abs(rng.normal(0.8, 0.4, n_wafers))
    etch_gas_flow_mean = rng.normal(200.0, 10.0, n_wafers)
    etch_rf_power_mean = rng.normal(1500.0, 30.0, n_wafers)
    etch_process_time = rng.normal(120.0, 10.0, n_wafers)

    # Deposition process.
    dep_temp_mean = rng.normal(400.0, 8.0, n_wafers)
    dep_temp_std = np.abs(rng.normal(3.0, 1.5, n_wafers))
    dep_pressure_mean = rng.normal(5.0, 0.3, n_wafers)
    # Pressure std is a secondary driver for deposition_tool × pressure_std.
    dep_pressure_std = np.abs(rng.normal(0.15, 0.08, n_wafers))
    dep_process_time = rng.normal(180.0, 15.0, n_wafers)

    # Lithography.
    exposure_dose = rng.normal(25.0, 0.5, n_wafers)
    focus_offset = rng.normal(0.0, 0.05, n_wafers)

    # Maintenance age (hours since last maintenance on the tool).
    maintenance_age_etch = rng.uniform(0, 500, n_wafers)
    maintenance_age_deposition = rng.uniform(0, 600, n_wafers)

    # --- Yield model ---
    # Start with a high baseline and subtract yield penalties.
    yield_penalty = np.zeros(n_wafers)
    noise = rng.normal(0, 0.008, n_wafers)  # ~0.8% noise

    # Effect 1: ETCH_02 → small negative effect
    is_etch_02 = (etch_tool == "ETCH_02")
    yield_penalty += np.where(is_etch_02, 0.008, 0.0)

    # Effect 2: CH_B → small negative effect
    is_ch_b = (etch_chamber == "CH_B")
    yield_penalty += np.where(is_ch_b, 0.006, 0.0)

    # Effect 3: high temperature_std → small negative effect
    high_temp_std = etch_temp_std > 2.5
    yield_penalty += np.where(high_temp_std, 0.005, 0.0)

    # Interaction 1: ETCH_02 + CH_B → larger negative effect
    is_etch02_chb = is_etch_02 & is_ch_b
    yield_penalty += np.where(is_etch02_chb, 0.025, 0.0)

    # Interaction 2: ETCH_02 + CH_B + high temp_std → large negative effect
    is_triple = is_etch02_chb & high_temp_std
    yield_penalty += np.where(is_triple, 0.035, 0.0)

    # Weaker relationship: product × recipe
    is_prod_a_recipe_b = (product == "PROD_A") & (etch_recipe == "RECIPE_B")
    yield_penalty += np.where(is_prod_a_recipe_b, 0.004, 0.0)

    is_prod_c_recipe_c = (product == "PROD_C") & (etch_recipe == "RECIPE_C")
    yield_penalty += np.where(is_prod_c_recipe_c, 0.003, 0.0)

    # Weaker: deposition_tool × pressure_std
    is_dep_03_high_pstd = (dep_tool == "DEP_03") & (dep_pressure_std > 0.2)
    yield_penalty += np.where(is_dep_03_high_pstd, 0.005, 0.0)

    # Weaker: maintenance_age × tool (older maintenance on ETCH_02 is bad)
    is_etch02_old_maint = is_etch_02 & (maintenance_age_etch > 350)
    yield_penalty += np.where(is_etch02_old_maint, 0.006, 0.0)

    # Weaker: process_route × temperature variability
    is_route3_high_temp = (process_route == "ROUTE_3") & high_temp_std
    yield_penalty += np.where(is_route3_high_temp, 0.004, 0.0)

    # Continuous effects: temperature_std has a gradual linear effect
    yield_penalty += (etch_temp_std - 1.5) * 0.003

    # Continuous effect: maintenance age (general wear)
    yield_penalty += (maintenance_age_etch / 500.0) * 0.005

    # Continuous effect: focus offset (lithography)
    yield_penalty += np.abs(focus_offset) * 0.02

    # Compute final yield.
    baseline = 0.975
    wafer_yield = baseline - yield_penalty + noise
    wafer_yield = np.clip(wafer_yield, 0.0, 1.0)

    # --- Critical Dimension (CD) model ---
    # CD is measured in nm after lithography + etch.
    # Nominal target: 38.0 nm. Driven by exposure dose, focus, etch conditions.
    cd_noise = rng.normal(0, 0.3, n_wafers)  # ~0.3 nm measurement noise

    # Exposure dose: higher dose → more resist removed → smaller CD
    cd_dose_effect = (exposure_dose - 25.0) * -0.8  # -0.8 nm per unit dose

    # Focus offset: larger |offset| → wider CD (defocus broadens the pattern)
    cd_focus_effect = np.abs(focus_offset) * 15.0  # 15 nm per unit offset

    # Etch tool: ETCH_02 tends to over-etch → smaller CD
    cd_etch_effect = np.where(etch_tool == "ETCH_02", -0.5, 0.0)
    cd_etch_effect += np.where(etch_tool == "ETCH_03", 0.3, 0.0)

    # Etch recipe: RECIPE_C is more aggressive → smaller CD
    cd_recipe_effect = np.where(etch_recipe == "RECIPE_C", -0.4, 0.0)
    cd_recipe_effect += np.where(etch_recipe == "RECIPE_A", 0.3, 0.0)

    # Etch temperature: higher temp → faster etch → slightly smaller CD
    cd_temp_effect = (etch_temp_mean - 85.0) * -0.05

    # Product: PROD_A has a slightly tighter CD spec
    cd_product_effect = np.where(product == "PROD_A", -0.2, 0.0)

    # Interaction: ETCH_02 + RECIPE_C → over-etch → significantly smaller CD
    cd_interaction = np.where(
        (etch_tool == "ETCH_02") & (etch_recipe == "RECIPE_C"), -0.6, 0.0
    )

    critical_dimension_nm = (
        38.0  # nominal
        + cd_dose_effect
        + cd_focus_effect
        + cd_etch_effect
        + cd_recipe_effect
        + cd_temp_effect
        + cd_product_effect
        + cd_interaction
        + cd_noise
    )

    # --- Film Thickness model ---
    # Film thickness measured in nm after deposition.
    # Nominal target: 510.0 nm. Driven by deposition conditions.
    ft_noise = rng.normal(0, 2.0, n_wafers)  # ~2 nm measurement noise

    # Process time: longer → thicker (with diminishing returns, log-like)
    ft_time_effect = (dep_process_time - 180.0) * 0.8  # 0.8 nm per minute

    # Deposition tool: DEP_02 tends thinner, DEP_03 tends thicker
    ft_tool_effect = np.where(dep_tool == "DEP_02", -8.0, 0.0)
    ft_tool_effect += np.where(dep_tool == "DEP_03", 6.0, 0.0)

    # Deposition recipe: DEP_RECIPE_B is a thicker deposition process
    ft_recipe_effect = np.where(dep_recipe == "DEP_RECIPE_B", 5.0, 0.0)

    # Pressure: higher pressure → slightly thicker (denser deposition)
    ft_pressure_effect = (dep_pressure_mean - 5.0) * 8.0  # 8 nm per unit pressure

    # Temperature: higher temp → more uniform, slightly thicker
    ft_temp_effect = (dep_temp_mean - 400.0) * 0.3  # 0.3 nm per degree

    # Maintenance age: older maintenance on deposition tool → more variation
    ft_maint_effect = (maintenance_age_deposition / 600.0) * -3.0  # slight thinning

    # Interaction: DEP_02 + DEP_RECIPE_A → significantly thinner
    ft_interaction = np.where(
        (dep_tool == "DEP_02") & (dep_recipe == "DEP_RECIPE_A"), -5.0, 0.0
    )

    film_thickness_nm = (
        510.0  # nominal
        + ft_time_effect
        + ft_tool_effect
        + ft_recipe_effect
        + ft_pressure_effect
        + ft_temp_effect
        + ft_maint_effect
        + ft_interaction
        + ft_noise
    )

    # --- Build DataFrame ---
    df = pl.DataFrame({
        "timestamp": timestamps,
        "lot_id": lot_assignment,
        "wafer_id": wafer_ids,
        "product_id": product,
        "fab_id": fab,
        "process_route": process_route,
        "etch_tool": etch_tool,
        "etch_chamber": etch_chamber,
        "etch_recipe": etch_recipe,
        "deposition_tool": dep_tool,
        "deposition_chamber": dep_chamber,
        "deposition_recipe": dep_recipe,
        "etch_temperature_mean": etch_temp_mean,
        "etch_temperature_std": etch_temp_std,
        "etch_pressure_mean": etch_pressure_mean,
        "etch_pressure_std": etch_pressure_std,
        "etch_gas_flow_mean": etch_gas_flow_mean,
        "etch_rf_power_mean": etch_rf_power_mean,
        "etch_process_time": etch_process_time,
        "deposition_temperature_mean": dep_temp_mean,
        "deposition_temperature_std": dep_temp_std,
        "deposition_pressure_mean": dep_pressure_mean,
        "deposition_pressure_std": dep_pressure_std,
        "deposition_process_time": dep_process_time,
        "exposure_dose": exposure_dose,
        "focus_offset": focus_offset,
        "maintenance_age_etch": maintenance_age_etch,
        "maintenance_age_deposition": maintenance_age_deposition,
        "wafer_yield": wafer_yield,
        "critical_dimension_nm": critical_dimension_nm,
        "film_thickness_nm": film_thickness_nm,
    })

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic semiconductor wafer-yield data.")
    parser.add_argument("--wafers", type=int, default=50_000, help="Number of wafer rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output", type=str, default="data/semiconductor_yield/data.parquet",
                        help="Output Parquet path.")
    args = parser.parse_args()

    print(f"Generating {args.wafers} wafer rows (seed={args.seed})...")
    df = generate_wafers(n_wafers=args.wafers, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    print(f"Wrote {df.height} rows × {df.width} columns to {out}")

    # Quick summary.
    yield_col = df["wafer_yield"]
    print(f"  wafer_yield: min={yield_col.min():.4f}, mean={yield_col.mean():.4f}, max={yield_col.max():.4f}")
    cd_col = df["critical_dimension_nm"]
    print(f"  critical_dimension_nm: min={cd_col.min():.2f}, mean={cd_col.mean():.2f}, max={cd_col.max():.2f}")
    ft_col = df["film_thickness_nm"]
    print(f"  film_thickness_nm: min={ft_col.min():.2f}, mean={ft_col.mean():.2f}, max={ft_col.max():.2f}")

    # Verify hidden interactions.
    etch02_chb = df.filter((pl.col("etch_tool") == "ETCH_02") & (pl.col("etch_chamber") == "CH_B"))
    overall = df["wafer_yield"].mean()
    combo = etch02_chb["wafer_yield"].mean()
    print(f"  Overall mean yield: {overall:.4f}")
    print(f"  ETCH_02+CH_B mean yield: {combo:.4f} (diff: {combo - overall:.4f})")


if __name__ == "__main__":
    main()
