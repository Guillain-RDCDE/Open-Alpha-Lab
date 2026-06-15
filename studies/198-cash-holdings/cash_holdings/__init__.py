"""Study 198 — Cash-Holdings.

The corporate cash-holdings anomaly (Palazzo 2012): do firms with high cash-to-assets
ratios earn higher returns? The hypothesis is that cash-rich firms command a financial-
constraint premium — they hoard cash because they face high external financing costs, and
investors demand a premium for holding these constrained firms.

Signal: Cash-to-Assets = CashAndCashEquivalentsAtCarryingValue / Total Assets
Universe: Current S&P 500 (survivorship-biased — named on the Signal axis).
Lag: Fundamentals from fiscal year y → forward returns in calendar year y+1.
Controls: equal-weight market; random same-size portfolio.
Verdict: NONE / MIRAGE (the anomaly is weak or absent in large-cap survivors).
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cash_holdings")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["data", "strategy"]
