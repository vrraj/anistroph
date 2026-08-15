"""Admin model training CLI.

Usage:
    python scripts/train_model.py --dataset semiconductor_yield --model-type xgboost_regressor
    python scripts/train_model.py --dataset predictive_maintenance --model-type xgboost --model-id my-model

Training is an admin operation. It is NOT exposed through MCP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.services import get_services


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Anistroph model (admin operation).")
    parser.add_argument("--dataset", required=True, help="Dataset ID to train on.")
    parser.add_argument("--model-type", required=True,
                        help="Model type (xgboost, logistic_regression, xgboost_regressor, linear_regression).")
    parser.add_argument("--target", default=None,
                        help="Target name (defaults to the dataset's configured target).")
    parser.add_argument("--model-id", default=None, help="Explicit model ID (auto-generated if None).")
    parser.add_argument("--params", default=None,
                        help="JSON string of model hyperparameters.")
    args = parser.parse_args()

    svc = get_services()

    # Resolve target name from dataset config if not specified.
    target_name = args.target
    if target_name is None:
        config = svc.get_config(args.dataset)
        if config.target_spec is None:
            print(f"Error: no target spec in dataset {args.dataset!r} and --target not provided", file=sys.stderr)
            sys.exit(1)
        target_name = config.target_spec.name

    model_parameters = None
    if args.params:
        model_parameters = json.loads(args.params)

    print(f"Training {args.model_type} on dataset {args.dataset!r}, target={target_name!r}...")
    result = svc.train(
        dataset_id=args.dataset,
        target_name=target_name,
        model_type=args.model_type,
        model_parameters=model_parameters,
        model_id=args.model_id,
    )

    model_id = result["model_id"]
    metrics = result["metrics"]
    print(f"\nModel trained: {model_id}")
    print(f"  model_type: {result['model_type']}")
    print(f"  dataset_id: {result['dataset_id']}")
    print(f"  features: {len(result['feature_names'])}")
    print(f"\nMetrics:")
    # Print key metrics (avoid dumping huge arrays like pr_curve).
    for k, v in metrics.items():
        if isinstance(v, dict) and k not in ("confusion_matrix", "baseline"):
            print(f"  {k}: (object with {len(v)} keys)")
        elif isinstance(v, (int, float, str, type(None))) or (isinstance(v, dict) and k in ("confusion_matrix", "baseline")):
            print(f"  {k}: {v}")
    print(f"\nModel registered. Use model_id={model_id!r} for predictions.")


if __name__ == "__main__":
    main()
