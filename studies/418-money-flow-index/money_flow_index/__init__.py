"""Study 418 — Money Flow Index (the volume-weighted RSI).

Does folding volume into the RSI ("Money Flow Index") buy you anything the plain
RSI doesn't? This package builds the MFI and the RSI, turns each into the same
long/flat timing rule on SPY, and races NET excess-of-cash Sharpe vs buy-and-hold
under one execution lag and one-way costs.
"""

from . import data, strategy  # noqa: F401
