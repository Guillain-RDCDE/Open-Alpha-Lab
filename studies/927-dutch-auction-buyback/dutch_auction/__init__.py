"""Study 927 — Dutch Auction: do issuer self-tender offers mark the bottom?

``data``     — the hardcoded, EDGAR-derived event table plus the cached price tapes and
               the deterministic synthetic generators.
``strategy`` — the abnormal-return event study, the calendar-time portfolio, the placebo,
               and the cost / borrow / expiry / date sweeps.
"""

from __future__ import annotations

__all__ = ["data", "strategy"]
