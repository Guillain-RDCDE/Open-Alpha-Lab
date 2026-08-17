"""Study 956 — the ADR custody (depositary) fee drag.

``data``     the ten ADR / home-line / FX triples, the shared-cache loaders, and the
             deterministic synthetic generators with a planted fee.
``strategy`` the trend estimator with break segmentation, the withholding-versus-custody
             decomposition, the sweeps, and the one traded "own the home line" race.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
