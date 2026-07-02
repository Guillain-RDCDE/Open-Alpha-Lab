"""Study 575 — CDS-Equity-Basis.

The cross-asset **CDS-equity basis**: for a single name, its credit-default-swap spread
prices the credit market's view of default risk, while the firm's equity (via a Merton-style
structural model) implies its *own* default risk from the stock price and its volatility. When
the two markets disagree — the *basis* — folklore says the gap predicts convergence: the market
that is "wrong" reprices toward the other, so the basis forecasts subsequent equity returns.

This study is **synthetic-only**: clean single-name CDS spreads and firm-level structural default
probabilities are not available on a no-key retail data stack (CDS is an OTC, licensed market).
So the effect can never earn a REAL stamp here (that needs a robust real tape); it is capped at
WEAK / NONE, and the data-availability limitation is stated openly on the SIGNAL axis.
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
