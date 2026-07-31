from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

VERSIONED_VENDOR = ROOT / f".vendor-py{sys.version_info.major}{sys.version_info.minor}"
DEFAULT_VENDOR = ROOT / ".vendor"
for vendor in (VERSIONED_VENDOR, DEFAULT_VENDOR):
    if vendor.exists():
        sys.path.insert(0, str(vendor))
        break
sys.path.insert(0, str(SRC))

from kiosk_accessibility.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
