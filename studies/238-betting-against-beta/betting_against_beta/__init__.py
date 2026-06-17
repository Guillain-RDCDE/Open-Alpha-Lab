"""Study 238 -- Betting-Against-Beta (BAB): does leveraging low-beta / shorting high-beta earn a premium?

Frazzini & Pedersen (2014): rank stocks by beta, long the low-beta half (leveraged to
beta=1), short the high-beta half (deleveraged to beta=1), holding a beta-neutral portfolio.
We test this on a survivorship-biased large-cap panel (current S&P 500) using yfinance.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
