"""Study 714 — "Contemporary art is an asset class" (the auction-index bubble, tested).

Do contemporary-art auction returns beat the S&P? We test the strongest tradable version
of the claim: a secondary-market **auction price index** (hardcoded, cited, **approximate**
— a labelled proxy, not a live feed; the shape of the Artprice Global / Sotheby's Mei Moses
reporting) and the only listed ways to touch the trade (MCH Group ``MCHN.SW`` — organiser of
Art Basel; Kering ``KER.PA`` — whose controlling shareholder Pinault owns Christie's), all
benchmarked against ``SPY`` on return, volatility, drawdown and — the part the pitch never
charges — the ~25% buyer's premium + seller's commission round-trip.

See :mod:`art_auction_index.data` (hardcoded art index + yfinance equity proxies + a
deterministic synthetic bubble control) and :mod:`art_auction_index.strategy` (CAGR/vol/MDD,
annual-excess *t*, Newey-West proxy alpha, and the premium haircut)."""

from . import data, strategy

__all__ = ["data", "strategy"]
