"""Study 312 — Debt-Ceiling: is the brinkmanship a *volatility* trade?

Distinct from Study 311 (Government-Shutdown), which tests the *directional* "buy the
dip" forward return around funding-gap *shutdowns*. This study asks the orthogonal
question on a *different* event table — the binding US debt-ceiling **X-dates** — and on
a *different* signal axis: does *volatility* (VIX level + realized vol) ramp into the
deadline and collapse on resolution? i.e. is it a long-vol-into / short-vol-out trade?
"""

from . import data, strategy

__all__ = ["data", "strategy"]
