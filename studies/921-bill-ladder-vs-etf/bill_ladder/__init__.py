"""Study 921 — Bill Ladder vs ETF.

A home-made rolling 3-month Treasury-bill ladder, simulated from the ^IRX discount
yield, raced against the cash ETFs that charge a fee to run one for you (BIL, SGOV, SHV).
"""

from . import data, strategy  # noqa: F401

__all__ = ["data", "strategy"]
