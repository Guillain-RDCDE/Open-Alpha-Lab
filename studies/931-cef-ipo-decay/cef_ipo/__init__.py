"""Study 931 — the closed-end-fund IPO hole.

``data``    — the hardcoded CEF IPO list, the shared-cache loaders, and the synthetic panel.
``strategy`` — the abnormal-return event study, its inference, and the tradable mirror.
"""

from __future__ import annotations

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
