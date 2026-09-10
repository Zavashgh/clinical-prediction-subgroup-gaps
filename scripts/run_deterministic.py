"""Launch a Python command with the approved single-thread environment.

Examples:
    python scripts/run_deterministic.py run_age_decomposition.py
    python scripts/run_deterministic.py -m jupyter nbconvert --execute notebook.ipynb
"""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.runtime import THREAD_ENVIRONMENT


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_deterministic.py <python arguments>")
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    completed = subprocess.run(
        [sys.executable, *sys.argv[1:]],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
