"""Data layer for Study 738 — Pollen-Season.

The claim under test: **allergy-brand owners get a springtime tailwind.** Every March→May
tens of millions of hay-fever sufferers restock antihistamines and nasal sprays, so the
listed companies that own the big allergy brands — Claritin, Allegra, Zyrtec/Benadryl,
Flonase, plus the private-label giant that makes the store-brand versions — ought to
carry a repeatable spring seasonal in their share price *relative to the market*.

Three ingredients, all offline-friendly once cached:

* **The pollen-season window, a LABELLED, CITED calendar rule.** There is no free,
  machine-readable "US pollen index" time series to trade against, so — exactly as
  ``358-watch-index`` and ``708-eurovision-effect`` do with their hand-built calendars —
  the season itself is a small, clearly-cited, hardcoded *calendar window*, not a data
  feed dressed up as a tape. US spring pollen (tree then early grass) runs roughly
  **March through May** (AAFA "Allergy Capitals" reports; AAAAI regional pollen
  calendars; NAB/pollen.com station data), so the study's window is **the last trading
  session of February → the last trading session on/before May 31** each year. It is a
  pure *calendar-known* rule (the dates are fixed years in advance), so — like a
  turn-of-month study — it needs **no execution lag at all**: you always know when
  spring is.

* **Real tape.** Daily total-return closes for the allergy basket and two benchmarks,
  all from yfinance (no key), cached as CSV under the study's own ``_cache/``. The
  basket is five listed owners of the household allergy brands (``ALLERGY_TICKERS``);
  the market benchmark is **SPY**, and **XLP** (consumer-staples sector) is a second,
  fairer-comparison benchmark for a robustness cut. Two of the five basket names are
  recent **spin-offs** (Kenvue from J&J in 2023, Haleon from GSK in 2022) and therefore
  only enter the basket for their post-listing spring windows — coverage is **named
  honestly** via ``n_names`` per year and cross-checked against a full-history
  3-name core basket, never silently back-filled.

* **Synthetic world.** A deterministic, seeded pair of daily basket/market "Close" tapes
  spanning ~30 spring windows, with a TUNABLE planted spring bump (``bump`` in window
  total-return units) sprinkled across the basket's pollen-season sessions only.
  ``bump = 0`` is the null world — spring windows statistically identical to the rest —
  and the one-sample-t machinery must NOT manufacture a seasonal from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

START = "1996-01-01"
AS_OF = "2026-06-30"        # last complete month at publication; the 2026 spring window
                            # (ends 2026-05-30) is fully closed, so 2026 is a complete event.

# --------------------------------------------------------------------------- #
# The pollen-season window — a labelled, cited CALENDAR rule (not a data feed).
# US spring pollen (tree -> early grass) runs ~March through May. The tradable window
# is the last session of February (enter here, the season has not started) through the
# last session on/before May 31 (exit here, the season is ending). Fixed calendar dates
# known years in advance => a calendar-known rule, no execution lag needed.
# --------------------------------------------------------------------------- #
SEASON_START_MMDD = (3, 1)    # nominal pollen-season start (enter at the prior close)
SEASON_END_MMDD = (5, 31)     # nominal pollen-season end   (exit at the last close <= here)
SEASON_LABEL = "Mar 1 -> May 31 (US spring tree/grass pollen; AAFA/AAAAI)"

# --------------------------------------------------------------------------- #
# The allergy basket: listed owners of the household allergy brands. Each row:
# ticker -> (brand-line, one-line provenance note). Equal-weighted each year over
# whichever names have full-window coverage that spring (see strategy.build_spread_table).
# --------------------------------------------------------------------------- #
ALLERGY_BRANDS: dict[str, tuple[str, str]] = {
    "BAYRY": ("Claritin (loratadine)",
              "Bayer AG ADR; acquired the Claritin OTC line with Merck consumer health 2014"),
    "SNY":   ("Allegra (fexofenadine) + Dupixent",
              "Sanofi ADR; Allegra OTC + the Regeneron-partnered allergy/asthma biologic Dupixent"),
    "PRGO":  ("store-brand / private-label OTC allergy meds",
              "Perrigo — the largest US private-label maker of the generic antihistamines/nasal sprays"),
    "KVUE":  ("Zyrtec (cetirizine) + Benadryl",
              "Kenvue — spun off from Johnson & Johnson 2023-05 (first full spring window 2024)"),
    "HLN":   ("Flonase (fluticasone)",
              "Haleon — spun off from GSK 2022-07 (first full spring window 2023)"),
}
ALLERGY_TICKERS = tuple(ALLERGY_BRANDS.keys())
CORE_TICKERS = ("BAYRY", "SNY", "PRGO")   # the full-history 3-name robustness basket

MARKET = "SPY"      # the market benchmark (total-return)
STAPLES = "XLP"     # consumer-staples sector — a fairer second benchmark (robustness)


def all_tickers() -> list[str]:
    """Every distinct ticker this study ever needs: the basket plus both benchmarks."""
    return list(ALLERGY_TICKERS) + [MARKET, STAPLES]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"pollen_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download total-return daily closes for every ticker; cache them as CSV.

    Network; run once. ``auto_adjust=True`` folds splits and dividends into the close
    (total-return, not price-only) — the honest comparison for equities, so the window
    returns below are a plain close/close ratio.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d[["Close"]].dropna().to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: total-return close Series}, each sliced to [START, asof]."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(asof))]
    return out


def sample_years(asof: str = AS_OF) -> list[int]:
    """Every spring window that is fully in-sample: 1996 .. the last complete May."""
    hi = pd.Timestamp(asof)
    last = hi.year if hi >= pd.Timestamp(hi.year, SEASON_END_MMDD[0], SEASON_END_MMDD[1]) \
        else hi.year - 1
    return list(range(1996, last + 1))


# --------------------------------------------------------------------------- #
# Synthetic world — a planted spring bump on a daily basket/market tape (the control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 738,
                    start: str = "1996-01-02", n_years: int = 30,
                    daily_vol: float = 0.011, rho: float = 0.6,
                    ) -> tuple[pd.Series, pd.Series]:
    """Deterministic paired (basket_close, market_close) daily tapes with a planted bump.

    Both tapes are correlated random walks (log returns, correlation ``rho`` — a sector
    basket vs the market). On every session that falls inside the pollen-season window
    (the SAME ``SEASON_*_MMDD`` calendar rule the real study uses), the basket earns an
    extra ``bump / w`` log-return, where ``w`` is that window's session count — so the
    planted spring seasonal cumulates to ``bump`` total-return units of basket-minus-market
    per year. ``bump = 0`` is the null world: spring is statistically identical to the
    rest of the year, and the detector must not fire.

    ~30 years of business days — far below the pandas ns-timestamp overflow. Returns
    (basket_close, market_close), both plain price Series indexed by real dates.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_years * 260)
    common = rng.normal(0.0, daily_vol, len(idx))
    idio_b = rng.normal(0.0, daily_vol, len(idx))
    idio_m = rng.normal(0.0, daily_vol, len(idx))
    lb = rho * common + np.sqrt(1 - rho**2) * idio_b
    lm = rho * common + np.sqrt(1 - rho**2) * idio_m

    if bump != 0.0:
        for yr in sorted({d.year for d in idx}):
            in_win = _season_mask(idx, yr)
            w = int(in_win.sum())
            if w > 0:
                lb[in_win] += bump / w
    basket = pd.Series(100.0 * np.exp(np.cumsum(lb)), index=idx)
    market = pd.Series(100.0 * np.exp(np.cumsum(lm)), index=idx)
    return basket, market


def _season_mask(idx: pd.DatetimeIndex, year: int) -> np.ndarray:
    """Boolean mask of sessions in ``idx`` strictly inside [Mar 1, May 31] of ``year``."""
    lo = pd.Timestamp(year, SEASON_START_MMDD[0], SEASON_START_MMDD[1])
    hi = pd.Timestamp(year, SEASON_END_MMDD[0], SEASON_END_MMDD[1])
    return (idx >= lo) & (idx <= hi)
