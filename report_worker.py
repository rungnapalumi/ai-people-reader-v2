#!/usr/bin/env python3
"""
Canonical report worker implementation: **src/report_worker.py**

This file is a thin entrypoint only. Use either:

  python src/report_worker.py
  python report_worker.py

Both run the same code. Render (`render.yaml`) should keep using `src/report_worker.py`
for an explicit path in logs; local users may use either form.
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(root, "src", "report_worker.py")
    if not os.path.isfile(target):
        print(f"ERROR: missing {target}", file=sys.stderr)
        sys.exit(1)
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
