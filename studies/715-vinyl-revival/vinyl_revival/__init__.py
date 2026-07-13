"""Study 715 — "Vinyl is back — a trend to trade?" (the vinyl-revival, tested).

Vinyl records went from a nostalgia footnote to the fastest-growing format in
recorded music. The pitch: *"vinyl is back — that's a trend you can trade."* We test
the strongest tradable version of the claim: the **RIAA U.S. vinyl-revenue series**
(hardcoded, cited, **approximate** — a labelled proxy, not a live feed) as the trend
everyone quotes, and the only listed ways to buy the trade — **Warner Music Group
``WMG``, Spotify ``SPOT``, Universal Music Group ``UMG.AS``** — all benchmarked against
``SPY`` on return, volatility, drawdown and Newey-West alpha, plus the part the pitch
never charges: the dealer-spread + storage carry of physically collecting the records.

See :mod:`vinyl_revival.data` (hardcoded RIAA vinyl-revenue index + yfinance equity
proxies + a deterministic synthetic revival control) and :mod:`vinyl_revival.strategy`
(CAGR/vol/MDD, annual-excess *t*, Newey-West proxy alpha, and the collector carry haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
