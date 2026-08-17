"""Study 955 — ADR Catch-Up.

Does a US-listed ADR still owe its home market a move by the time it closes?

- ``data``     — the real tape (yfinance daily total-return closes, cache-only offline)
                 and the deterministic synthetic panel with a planted catch-up lag.
- ``strategy`` — the stale-price regression, the Fama-MacBeth predictive tests, the
                 costed long/short catch-up rule, and the robustness cuts.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
