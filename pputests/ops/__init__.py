"""Auto-discover and import every operator module in this package.

Importing :mod:`ops` walks the package directory and imports each ``*.py``
module, firing their ``@register`` side-effects and populating
:data:`core.opspec.OP_REGISTRY`.  This means **adding a new operator requires
zero edits to any existing file** -- just drop a new ``ops/<name>.py`` that
declares ``@register`` factories (DoD #5).
"""

from __future__ import annotations

import importlib
import pkgutil

__all__ = ["load_all"]


def load_all() -> None:
    """Import all sibling modules of this package (idempotent)."""
    for mod in pkgutil.iter_modules(__path__):
        name = mod.name
        if name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{name}")


load_all()
