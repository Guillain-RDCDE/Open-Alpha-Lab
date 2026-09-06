"""Study 987 — Gold's Loud Cousin.

"Silver is gold on leverage" is the oldest line in the precious-metals trade, and
unlike most market folklore it is close to arithmetically checkable. If it is exactly true then
silver is redundant: hold gold, size it up, and you have replicated silver with a cheaper, more
liquid instrument. If it is not exactly true, the residual is a second asset with its own
drivers — and the interesting question becomes what that residual is made of and whether anyone
is paid for holding it.

- :mod:`loudcousin.data` — the real tape (shared desk cache, offline loader) and the
  deterministic synthetic generator used by the whole test-suite.
- :mod:`loudcousin.strategy` — the measurement, the inference and the sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
