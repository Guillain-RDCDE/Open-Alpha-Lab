"""Study 729 — "the ramen index" (instant-noodle sales as a downturn tell, tested).

Does instant-noodle demand *lead* a recession / an equity drawdown, and are the noodle
makers a defensive place to hide? We test the strongest tradable version of the claim:
the WINA world-demand series (hardcoded, cited, **approximate** — a labelled proxy, not a
live feed) for the leading-indicator test, and the two listed noodle makers you can buy —
Nissin Foods Holdings (``2897.T``) and Toyo Suisan (``2875.T``) — benchmarked against the
Nikkei 225 (``^N225``) on beta, recession-window returns and the opportunity cost the
folklore never charges.

See :mod:`ramen_recession.data` (hardcoded/cited WINA demand + yfinance noodle equities +
the NBER recession windows + deterministic synthetic controls) and
:mod:`ramen_recession.strategy` (the lead-lag cross-correlation, bull/bear beta, Newey-West
CAPM alpha, the recession-window paired *t*, and the double look-ahead that kills the trade).
"""

from . import data, strategy

__all__ = ["data", "strategy"]
