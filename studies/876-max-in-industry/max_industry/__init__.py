"""Study 876 — Industry-Relative MAX.

Refine the lottery / MAX effect (study 365, Bali-Cakici-Whitelaw 2011) by sorting on a
name's **industry-relative** MAX — its own maximum daily return minus its sector peers'
median MAX — to strip out sector-wide volatility and isolate idiosyncratic lottery demand.
"""

from . import data, strategy

__all__ = ["data", "strategy"]
