"""
Compatibility shim: single implementation lives in src/report_core.py.

Anything that does ``import report_core`` from the repo root (e.g. app.py admin preview)
must use the same MediaPipe fallbacks and REPORT_CORE_VERSION as ``src/report_worker.py``.
Do not duplicate report logic here — load src/report_core only.
"""
from __future__ import annotations

import importlib.util
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_DIR, "src", "report_core.py")
if not os.path.isfile(_SRC):
    raise RuntimeError(f"Missing canonical report_core at {_SRC}")

_spec = importlib.util.spec_from_file_location("_report_core_canonical", _SRC)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not create import spec for {_SRC}")
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)

__doc__ = getattr(_impl, "__doc__", "") or __doc__
for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_impl, _name)
