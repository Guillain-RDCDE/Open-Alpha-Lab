"""Study 318 — Election-Volatility.

Does US presidential-election uncertainty spike equity volatility, and does selling
that elevated volatility into the event pay a risk premium?

This is the **volatility / event** angle on US elections — deliberately distinct from
the desk's two *return-cycle* election studies:

- Study 81 (Four-Year-Itch) — the year-3 presidential-cycle *return* premium.
- Study 248 (Presidential-Honeymoon) — the first-100-day post-inauguration *return*.

Here we never ask "do stocks go up around an election". We ask whether realized and
implied volatility are *elevated* in the run-up to election day, whether implied vol
(VIX) over-prices the volatility that actually shows up (a variance-risk-premium story),
and whether a short-vol carry trade timed to the event is investable after costs.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
