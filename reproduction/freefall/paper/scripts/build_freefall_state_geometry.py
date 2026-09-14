#!/usr/bin/env python3
"""Rebuild F2 coordinate figure.

The release is built atomically by build_panel_kit.py so all tables, videos,
figures, and hashes remain synchronized. See REBUILD.md for the exact command.
"""
from pathlib import Path
print((Path(__file__).resolve().parents[1] / "REBUILD.md").read_text(encoding="utf-8"))
