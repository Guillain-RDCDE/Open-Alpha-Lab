"""Study 769 — "Theme-park attendance/pricing as a DIS consumer tell".

Does theme-park momentum lead Disney the stock? We test the strongest strictly-lagged,
no-look-ahead version: a **cited, approximate** annual Walt Disney Attractions attendance
series and the WDW ticket-price series (labelled proxies, reconstructed from the public
TEA/AECOM Theme Index), released with the Theme Index's real ~mid-following-year lag, vs
``DIS`` and ``SPY`` (month-end Adj Close, yfinance). Beats the benchmark? Leads it? Pays
after costs?

See :mod:`disney_parks.data` (hardcoded parks proxies + release lag + yfinance equities +
a deterministic synthetic control) and :mod:`disney_parks.strategy` (CAGR/vol/MDD,
annual-excess *t*, Newey-West lead-lag *t*, Welch regime split, and the costed timing
backtest)."""

from . import data, strategy

__all__ = ["data", "strategy"]
