"""Study 501 -- Idiosyncratic-Volatility (the IVOL puzzle).

Ang, Hodrick, Xing & Zhang (2006): stocks with HIGH idiosyncratic volatility -- the
*residual* volatility left after stripping out market beta -- earn LOWER subsequent
returns. We rank a survivor large-cap panel into IVOL quintiles each month and hold a
**low-minus-high** long-short, testing the puzzling *negative* IVOL-return relation.

IVOL is distinct from the total-vol low-volatility anomaly (Study 330): it is the standard
deviation of the *CAPM residual* (return after removing the market factor), not raw return
vol. We test on a survivorship-biased large-cap panel (current S&P 500) via yfinance.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
