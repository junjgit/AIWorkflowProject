#!/usr/bin/env python
"""Train and persist the aggregate plus top-country forecasting models."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import train_all_models


def main() -> None:
    result = train_all_models(
        ROOT / "cs-train",
        ROOT / "models",
        ROOT / "logs",
        ROOT / "reports" / "metrics" / "model_comparison.csv",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
