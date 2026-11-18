"""Pytest configuration for Device 3 tests.

Ensures the Device 3 package root is present in sys.path so that tests can be
executed either from within `modules/device3/` or from the repository root.
"""

from pathlib import Path
import sys

_DEVICE3_ROOT = Path(__file__).resolve().parent.parent
if str(_DEVICE3_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEVICE3_ROOT))
