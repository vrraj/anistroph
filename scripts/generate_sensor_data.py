"""Synthetic semiconductor/manufacturing sensor data generator.

Generates 50 machines × 60 days × 5-minute observations with intentionally
learnable but imperfect deterioration patterns leading to failures.

Multiple targets:
  - failure: binary (0/1) — will the tool fail at this timestamp?
  - failure_mode: categorical (NONE/THERMAL/PRESSURE/VIBRATION/POWER) — what kind of failure?
  - remaining_useful_life_hours: regression — hours until next failure
  - maintenance_required: binary (0/1) — does the tool need maintenance now?

Failure probability is driven by:
  - increasing vibration
  - temperature drift
  - pressure instability
  - maintenance age

Failures are NOT randomly assigned independently of sensor behavior.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


MACHINE_TYPES = ["TYPE_A", "TYPE_B", "TYPE_C"]
TYPE_BASE_PARAMS = {
    "TYPE_A": {"temp_base": 70.0, "vib_base": 2.0, "press_base": 100.0, "deterioration": 1.0},
    "TYPE_B": {"temp_base": 75.0, "vib_base": 2.5, "press_base": 95.0, "deterioration": 1.3},
    "TYPE_C": {"temp_base": 68.0, "vib_base": 1.8, "press_base": 105.0, "deterioration": 0.8},
}


def _machine_deterioration_profile(rng: np.random.Generator, machine_idx: int, mtype: str) -> dict:
    """Per-machine baseline + deterioration rate so machines differ."""
    base = TYPE_BASE_PARAMS[mtype]
    jitter = rng.normal(0, 1, size=4)
    det_factor = base["deterioration"] * rng.uniform(0.7, 1.4)
    return {
        "temp_base": base["temp_base"] + jitter[0],
        "vib_base": base["vib_base"] + jitter[1] * 0.3,
        "press_base": base["press_base"] + jitter[2] * 2.0,
        "deterioration": det_factor,
        "temp_drift_rate": rng.uniform(0.5, 1.5) * det_factor,  # degC per day
        "vib_drift_rate": rng.uniform(0.05, 0.20) * det_factor,  # units per day
        "press_instability": rng.uniform(0.5, 2.0) * det_factor,
        "noise_scale": rng.uniform(0.8, 1.2),
    }


def _failure_risk_score(params: dict, elapsed_days: float, maintenance_age_hours: float,
                        temp: float, vib: float, press: float) -> float:
    """Compute a learnable risk score from sensor state + maintenance age."""
    temp_excess = max(0.0, temp - params["temp_base"] - 10.0)
    vib_excess = max(0.0, vib - params["vib_base"] - 1.0)
    press_dev = abs(press - params["press_base"])
    age_factor = maintenance_age_hours / 720.0  # normalise by ~30 days
    score = (
        0.30 * vib_excess
        + 0.25 * temp_excess * 0.1
        + 0.20 * press_dev * 0.05
        + 0.15 * age_factor
        + 0.10 * elapsed_days / 60.0
    )
    return score


def generate_dataset(
    n_machines: int = 50,
    n_days: int = 60,
    interval_minutes: int = 5,
    seed: int = 42,
    out_csv: str | Path | None = None,
    out_parquet: str | Path | None = None,
) -> pl.DataFrame:
    """Generate the synthetic predictive-maintenance dataset."""
    rng = np.random.default_rng(seed)
    interval = timedelta(minutes=interval_minutes)
    start = datetime(2026, 6, 1, 0, 0, 0)
    n_steps = n_days * 24 * 60 // interval_minutes

    rows: list[dict] = []
    for m in range(n_machines):
        machine_id = f"TOOL_{m:03d}"
        mtype = MACHINE_TYPES[m % len(MACHINE_TYPES)]
        params = _machine_deterioration_profile(rng, m, mtype)

        # Maintenance cycle: reset maintenance_age periodically with some randomness.
        maintenance_interval_hours = rng.uniform(300, 500)
        last_maintenance_step = 0
        operating_hours_offset = rng.uniform(0, 5000)

        # Per-machine failure cooldown so failures don't cluster.
        failure_cooldown_until = 0

        for t in range(n_steps):
            ts = start + t * interval
            elapsed_days = t * interval_minutes / (60 * 24)
            hours_since_maint = (t - last_maintenance_step) * interval_minutes / 60.0

            # Deterioration grows with time since maintenance, resets after maintenance.
            det_progress = min(1.0, hours_since_maint / maintenance_interval_hours)

            # Sensor values with drift + noise + diurnal cycle.
            diurnal = math.sin(2 * math.pi * (ts.hour + ts.minute / 60) / 24.0)
            temp = (
                params["temp_base"]
                + params["temp_drift_rate"] * det_progress * 20
                + diurnal * 1.5
                + rng.normal(0, 2.0 * params["noise_scale"])
            )
            vib = (
                params["vib_base"]
                + params["vib_drift_rate"] * det_progress * 5
                + rng.normal(0, 0.15 * params["noise_scale"])
            )
            press = (
                params["press_base"]
                + rng.normal(0, params["press_instability"] * (1 + det_progress))
            )
            current = 10.0 + 0.05 * (temp - params["temp_base"]) + rng.normal(0, 0.3)
            voltage = 230.0 + rng.normal(0, 1.5)
            rpm = 1800.0 - 50.0 * det_progress + rng.normal(0, 20.0)
            flow_rate = 50.0 + 0.1 * press + rng.normal(0, 0.5)
            operating_hours = operating_hours_offset + t * interval_minutes / 60.0

            # Occasional anomalous readings (~0.5% of rows).
            anomaly = rng.random() < 0.005
            if anomaly:
                sensor = rng.choice(["temperature", "vibration", "pressure"])
                if sensor == "temperature":
                    temp += rng.uniform(10, 25)
                elif sensor == "vibration":
                    vib += rng.uniform(2, 5)
                else:
                    press += rng.uniform(-15, 15)

            # Failure probability driven by sensor state + maintenance age.
            risk = _failure_risk_score(
                params, elapsed_days, hours_since_maint, temp, vib, press
            )
            # Base rate is low; risk amplifies it.
            base_rate = 0.0008
            fail_prob = base_rate + 0.05 * max(0.0, risk)
            # Enforce cooldown to avoid back-to-back failures.
            if t < failure_cooldown_until:
                fail_prob = 0.0

            failure = 0
            failure_mode = "NONE"
            if rng.random() < fail_prob:
                failure = 1
                # Failure mode correlated with dominant driver.
                press_deviation = abs(press - params["press_base"])
                if vib - params["vib_base"] > 1.5:
                    failure_mode = "VIBRATION"
                elif temp - params["temp_base"] > 8:
                    failure_mode = "THERMAL"
                elif press_deviation > 8:
                    failure_mode = "PRESSURE"
                else:
                    failure_mode = "POWER"
                # Trigger maintenance reset after failure.
                last_maintenance_step = t + int(rng.uniform(6, 24) * 60 / interval_minutes)
                failure_cooldown_until = t + int(rng.uniform(48, 120) * 60 / interval_minutes)
            else:
                # Scheduled maintenance when age exceeds interval.
                if hours_since_maint >= maintenance_interval_hours and rng.random() < 0.01:
                    last_maintenance_step = t

            # Maintenance required: 1 if maintenance age is high OR risk is elevated.
            maint_required = 1 if (hours_since_maint >= maintenance_interval_hours * 0.8 or risk > 0.5) else 0

            rows.append(
                {
                    "timestamp": ts,
                    "machine_id": machine_id,
                    "machine_type": mtype,
                    "temperature": round(temp, 3),
                    "vibration": round(vib, 4),
                    "pressure": round(press, 3),
                    "current": round(current, 4),
                    "voltage": round(voltage, 3),
                    "rpm": round(rpm, 2),
                    "flow_rate": round(flow_rate, 3),
                    "maintenance_age_hours": round(hours_since_maint, 2),
                    "operating_hours": round(operating_hours, 2),
                    "failure": failure,
                    "failure_mode": failure_mode,
                    "maintenance_required": maint_required,
                    # remaining_useful_life_hours is filled in post-hoc
                    "remaining_useful_life_hours": 0.0,
                }
            )

    df = pl.DataFrame(rows).sort(["machine_id", "timestamp"])

    # --- Compute remaining_useful_life_hours (RUL) ---
    # For each machine, look forward from each row to the next failure.
    # RUL = hours until the next failure event (0 at the failure timestamp).
    # If no future failure, RUL = a large value (capped at max lookahead).
    interval_hours = interval_minutes / 60.0
    rul_values = []
    for machine_id in df["machine_id"].unique().to_list():
        mdf = df.filter(pl.col("machine_id") == machine_id)
        fail_indices = mdf["failure"].to_numpy()
        n = len(fail_indices)
        rul = np.full(n, 9999.0)  # default: no future failure
        # Find failure positions
        fail_positions = np.where(fail_indices == 1)[0]
        for i in range(n):
            # Find the next failure at or after position i
            future_fails = fail_positions[fail_positions >= i]
            if len(future_fails) > 0:
                rul[i] = (future_fails[0] - i) * interval_hours
        rul_values.extend(rul.tolist())
    df = df.with_columns(pl.Series("remaining_useful_life_hours", rul_values))

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(str(out_csv))
    if out_parquet:
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(out_parquet))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic predictive-maintenance sensor data")
    parser.add_argument("--machines", type=int, default=50)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--interval", type=int, default=5, help="minutes between observations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", default="data/synthetic/predictive_maintenance.csv")
    parser.add_argument("--out-parquet", default="data/raw/predictive_maintenance.parquet")
    args = parser.parse_args()

    df = generate_dataset(
        n_machines=args.machines,
        n_days=args.days,
        interval_minutes=args.interval,
        seed=args.seed,
        out_csv=args.out_csv,
        out_parquet=args.out_parquet,
    )
    n_fail = df.filter(pl.col("failure") == 1).height
    n_maint = df.filter(pl.col("maintenance_required") == 1).height
    print(f"Generated {df.height} rows, {df['machine_id'].n_unique()} machines, {n_fail} failures")
    print(f"Failure rate: {n_fail / df.height:.4%}")
    print(f"Maintenance required rate: {n_maint / df.height:.4%}")
    rul_col = df["remaining_useful_life_hours"]
    print(f"RUL: min={rul_col.min():.1f}h, mean={rul_col.mean():.1f}h, max={rul_col.max():.1f}h")
    print(f"Failure modes: {df['failure_mode'].unique().to_list()}")
    print(f"CSV: {args.out_csv}")
    print(f"Parquet: {args.out_parquet}")


if __name__ == "__main__":
    main()
