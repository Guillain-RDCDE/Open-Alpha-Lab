"""Study 61 — Slow-Burn: do leveraged ETFs decay, or amplify?

Folk wisdom says 3x leveraged ETFs (TQQQ) bleed to nothing via "volatility drag". The truth is
regime-dependent: in the 2010s bull TQQQ turned QQQ's +20%/yr into +44%/yr — but at *no* risk-adjusted
benefit (Sharpe 0.90 vs 0.98), an −82% drawdown, and −79% in 2022 alone. The drag is real and matches
theory (~13%/yr), but it isn't "decay to zero" — it's ruinous tail risk with no Sharpe gain.
"""
from . import data, strategy  # noqa: F401
