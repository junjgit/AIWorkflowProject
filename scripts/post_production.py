#!/usr/bin/env python
"""Run post-production monitoring against the held-out production period."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring import evaluate_post_production


def main() -> None:
    result = evaluate_post_production(
        ROOT / "cs-train",
        ROOT / "cs-production",
        ROOT / "models",
        ROOT / "reports" / "metrics",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
