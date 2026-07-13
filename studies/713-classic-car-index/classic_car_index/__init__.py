"""Study 713 — "Classic cars are an asset class that beats equities" (HAGI / Hagerty, tested).

Do collector cars beat the S&P? We test the strongest tradable version of the claim: the
collector-car price index (hardcoded, cited, **approximate** — a labelled proxy, not a
live feed, reconstructed from the HAGI Top Index / Knight Frank Luxury Investment Index /
Hagerty reporting), benchmarked against the S&P on both a **total-return** (``SPY``) and a
**price-only** (``^GSPC``) clock — because the pitch quotes a *price* index against
*total-return* equities — and the only listed ways to buy the trade (Ferrari ``RACE``,
Aston Martin ``AML.L``). Then the haircut the pitch never charges: the auction round-trip
spread + storage / insurance / maintenance carry.

See :mod:`classic_car_index.data` (hardcoded index + yfinance equity/benchmark proxies +
a deterministic synthetic control) and :mod:`classic_car_index.strategy` (CAGR/vol/MDD,
annual-excess *t*, Newey-West proxy alpha, the carry haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
