"""Study 520 — External-Financing-Anomaly (Bradshaw–Richardson–Sloan 2006).

Firms that raise a lot of external finance (debt *and* equity together) subsequently
underperform. We compute net external financing from the cash-flow statement's financing
activities, scale it by average total assets, sort a large-cap survivor basket each year, and
form a long-short (long the *retirers* of capital, short the big *raisers*).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
