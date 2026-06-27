"""Study 518 -- Time-Series-Momentum (Moskowitz-Ooi-Pedersen own-sign trend).

The canonical "trend everywhere" factor, replicated honestly on a fixed ~12-name cross-asset
ETF basket (equity / bond / commodity / FX / gold): trade each asset long if its OWN trailing
12-month return is positive else short, vol-scale each position to a common risk target, and
average across the diversified basket. Distinct from cross-sectional momentum (Study 507) and
from SMA trend timing (Study 110 Faber).

Public surface:
    data.synthetic_panel / data.fetch_panel / data.fingerprint
    strategy.own_sign_signal / strategy.tsmom_book / strategy.summary /
    strategy.placebo_pvalue / strategy.synthetic_control / strategy.synthetic_null_robustness
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
