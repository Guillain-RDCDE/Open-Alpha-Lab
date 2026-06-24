"""Study 439 — Linear Regression Channel.

A rolling-OLS slope timing rule on daily closes, raced NET against buy-and-hold and
against the obvious simpler benchmark (an SMA crossover). See ``data`` for the tapes
and ``strategy`` for the detector, the timing rules and the honest arbiters.
"""

from . import data, strategy  # noqa: F401
