"""Study 233 — Shareholder-Yield: total return of capital (dividends + buybacks) as an equity signal.

Shareholder yield = dividend yield + net buyback yield (shrinkage in diluted shares, adjusted for
price). A broader capital-return metric championed by Meb Faber (2007) and Boudoukh et al. (2007).
On SEC EDGAR data for large-cap S&P 500 survivors, the signal produces a Weak/Mirage verdict:
buybacks now dominate dividends, the combined signal shows no stable premium on these liquid names.
"""
from . import data, strategy  # noqa: F401
