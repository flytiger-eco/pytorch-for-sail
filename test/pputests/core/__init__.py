"""Core building blocks of the PPU operator accuracy-comparison framework.

The package is deliberately dependency-light: only ``torch`` (and, for the
YAML config files, ``pyyaml``) are required at runtime.  Every module can be
imported on a machine without PPU hardware; device availability is probed
lazily inside :mod:`core.runner`.
"""

from __future__ import annotations

__all__ = [
    "tolerance",
    "inputgen",
    "opspec",
    "metrics",
    "reference",
    "runner",
    "report",
    "dump",
]
