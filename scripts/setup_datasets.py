"""One-shot dataset setup for Anistroph.

Generates all synthetic source datasets and registers all dataset configs
(multi-target + staged-prediction configs share source parquet files).
Idempotent: skips generation/registration for datasets that are already
present in the registry.

Usage:
    python scripts/setup_datasets.py            # generate + register all
    python scripts/setup_datasets.py --skip-gen # register only (data already on disk)
    python scripts/setup_datasets.py --force    # re-register even if already present

After this script runs, the platform is ready for `make start` or
`uvicorn backend.main:app --reload --port 9500`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.services import get_services

# --- Source data generators ----------------------------------------------
# (display name, script, default output parquet relative to repo root)
GENERATORS = [
    ("Predictive maintenance sensor history", "scripts/generate_sensor_data.py", "data/raw/predictive_maintenance.parquet"),
    ("Semiconductor manufacturing wafer data", "scripts/generate_semiconductor_yield_data.py", "data/semiconductor_yield/data.parquet"),
    ("Bay Area home prices", "scripts/generate_home_prices_data.py", "data/home_prices/data.parquet"),
    ("Semiconductor materials procurement", "scripts/generate_procurement_data.py", "data/semiconductor_procurement/data.parquet"),
    ("Semiconductor memory supply history", "scripts/generate_semiconductor_memory_supply.py", "data/semiconductor_memory_supply/data.parquet"),
]

# --- Dataset configs -----------------------------------------------------
# (config_path, source_path) — source_path is what register_dataset_from_config ingests.
# Multi-target and staged configs reuse the same source parquet as their parent dataset.
DATASETS = [
    # Predictive maintenance — 3 targets, one source
    ("datasets/predictive_maintenance/dataset.yaml", "data/synthetic/predictive_maintenance.csv"),
    ("datasets/predictive_maintenance_rul/dataset.yaml", "data/synthetic/predictive_maintenance.csv"),
    ("datasets/predictive_maintenance_maint/dataset.yaml", "data/synthetic/predictive_maintenance.csv"),
    # Semiconductor — 3 targets (yield, CD, film thickness), one source
    ("datasets/semiconductor_yield/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    ("datasets/semiconductor_cd/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    ("datasets/semiconductor_film_thickness/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    # Semiconductor staged prediction — 4 stages, same source
    ("datasets/semiconductor_yield_stage_a/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    ("datasets/semiconductor_yield_stage_b/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    ("datasets/semiconductor_yield_stage_c/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    ("datasets/semiconductor_yield_stage_d/dataset.yaml", "data/semiconductor_yield/data.parquet"),
    # Home prices — 1 target
    ("datasets/home_prices/dataset.yaml", "data/home_prices/data.parquet"),
    # Semiconductor procurement — 2 targets (demand + shortage risk), one source
    ("datasets/semiconductor_procurement_demand/dataset.yaml", "data/semiconductor_procurement/data.parquet"),
    ("datasets/semiconductor_procurement_shortage/dataset.yaml", "data/semiconductor_procurement/data.parquet"),
    # Semiconductor memory — catalog (search + analysis, no model trained)
    ("datasets/semiconductor_memory/dataset.yaml", "data/semiconductor_memory/data.csv"),
    # Semiconductor memory supply — 2 targets (risk + lead time), one source
    ("datasets/semiconductor_memory_supply_risk/dataset.yaml", "data/semiconductor_memory_supply/data.parquet"),
    ("datasets/semiconductor_memory_supply_lead_time/dataset.yaml", "data/semiconductor_memory_supply/data.parquet"),
]


def _run(cmd: list[str]) -> None:
    import subprocess
    print(f"      Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def generate_all(skip_gen: bool) -> None:
    """Run the five synthetic data generators unless outputs already exist."""
    if skip_gen:
        print("[1/2] Skipping data generation (--skip-gen)")
        return

    missing_outputs = [
        out_rel for _, _, out_rel in GENERATORS
        if not (_REPO_ROOT / out_rel).exists()
    ]
    print("[1/2] Preparing synthetic reference data")
    if missing_outputs:
        print("      First-time data generation may take a few minutes.")
        print("      Progress will be shown for each dataset family.")
    else:
        print("      All generated source files already exist; checking each one.")

    total = len(GENERATORS)
    for index, (display_name, script, out_rel) in enumerate(GENERATORS, start=1):
        out_path = _REPO_ROOT / out_rel
        if out_path.exists():
            print(f"      [{index}/{total}] {display_name}: already present (skipping)")
            continue
        print(f"      [{index}/{total}] Generating data: {display_name}", flush=True)
        print(f"            Output: {out_rel}", flush=True)
        _run(["python", script])

    print("      Reference source data is ready.")


def register_all(force: bool) -> None:
    """Register all dataset configs."""
    print(f"[2/2] Registering {len(DATASETS)} dataset configurations")
    svc = get_services()
    existing = {d.dataset_id for d in svc.list_datasets()}

    n_registered = 0
    n_skipped = 0
    total = len(DATASETS)
    for index, (config_rel, source_rel) in enumerate(DATASETS, start=1):
        config_path = _REPO_ROOT / config_rel
        source_path = _REPO_ROOT / source_rel
        if not config_path.exists():
            print(f"  ! missing config: {config_rel}", file=sys.stderr)
            sys.exit(1)
        if not source_path.exists():
            print(f"  ! missing source data: {source_rel}", file=sys.stderr)
            print(f"    Run `python scripts/setup_datasets.py` without --skip-gen first.", file=sys.stderr)
            sys.exit(1)

        # Read dataset_id from YAML to check existence (cheap, avoids re-registering).
        from backend.datasets.config import load_dataset_config
        cfg = load_dataset_config(config_path)
        dataset_id = cfg.dataset_spec.dataset_id

        if dataset_id in existing and not force:
            print(f"      [{index:02}/{total}] {dataset_id}: already registered (skipping)")
            n_skipped += 1
            continue

        print(f"      [{index:02}/{total}] Registering dataset: {dataset_id}", flush=True)
        meta = svc.register_dataset_from_config(config_path, source_path)
        train_name = Path(meta.train_parquet_path).name if meta.train_parquet_path else "none"
        print(f"            Ready: {meta.row_count} rows, train={train_name}")
        n_registered += 1

    print(f"\nDone. Registered {n_registered} dataset(s), skipped {n_skipped} already-registered.")
    print(f"Total registered: {len(svc.list_datasets())} dataset(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate + register all Anistroph reference datasets.")
    parser.add_argument("--skip-gen", action="store_true", help="Skip synthetic data generation (use existing parquet files).")
    parser.add_argument("--force", action="store_true", help="Re-register datasets even if already in the registry.")
    args = parser.parse_args()

    generate_all(skip_gen=args.skip_gen)
    register_all(force=args.force)

    print("\nNext steps:")
    print("  make start                                    # Docker Compose (port 9500)")
    print("  uvicorn backend.main:app --reload --port 9500 # native Python (no Docker)")
    print("  Web UI:   http://localhost:9500")
    print("  OpenAPI:  http://localhost:9500/docs")
    print("  MCP HTTP: http://localhost:9500/mcp")


if __name__ == "__main__":
    main()
