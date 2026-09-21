#!/usr/bin/env python
"""Run the complete unit-test suite with one command."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
raise SystemExit(
    subprocess.call([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT, env=env)
)
