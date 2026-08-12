"""Study 877 — GDPNow Revisions.

Does the *daily revision* of the Atlanta Fed's GDPNow nowcast — a real-time growth
surprise — predict next-day / next-week SPY returns? Do large downward revisions
precede weakness? Engine: :mod:`gdpnow.data` (real GDPNow + SPY tape and a seeded
synthetic control) and :mod:`gdpnow.strategy` (the predictive regression, decile
conditional test, era cut, costed timer, and inference primitives).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
