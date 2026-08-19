"""Resolve project root for source runs and PyInstaller bundles."""

from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / "calibration").is_dir():
            return exe_dir
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return exe_dir
    return Path(__file__).resolve().parent
