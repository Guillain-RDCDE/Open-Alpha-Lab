"""Data layer for Study 742 — Friday-17th (the Italian *Venerdì 17* superstition).

The claim under test: in Italy the unlucky day is **Friday the 17th**, not the 13th
(the Roman numeral for 17, ``XVII``, is an anagram of the Latin ``VIXI`` — "I have
lived", i.e. "I am dead"). If superstition-driven mood really pushed the tape, Italy's
own stock market should trade *weak* on Venerdì 17. This is the Latin-market cousin of
the desk's Friday-13th teardown (study 163) — same mechanism, a different calendar
slot, a different tape.

Two real series, one shape (a daily return series per instrument):

* **``FTSEMIB.MI``** — the FTSE MIB, Italy's blue-chip index, **in local currency
  (EUR), price-only** (no dividend adjustment on Yahoo's index series). This is the
  *purest* sentiment test: a superstition about Italian mood should show up on the
  Italian tape, priced by Italians, in euro. (Yahoo's ``FTSEMIB.MI`` history splices
  the earlier S&P/MIB into the current FTSE MIB brand; it starts 1997-12-31.)
* **``EWI``** — the iShares MSCI Italy ETF, **US-listed, USD, total-return** (dividends
  reinvested). This is the vehicle a non-Italian could actually *trade*, and it is the
  instrument the "could you trade it?" timer runs on. It is USD-denominated, so it
  blends the Italian tape with the EUR/USD cross — a caveat named out loud, not the
  clean local-sentiment read. Inception 1996-03-18.

* **Synthetic tape** — a deterministic, seeded daily generator with a TUNABLE
  ``f17_effect`` knob (a mean-return bump applied only on Friday-the-17th business
  days). ``f17_effect = 0`` is the null world — Venerdì 17 is just another Friday — so
  the one-sample-t / placebo machinery must NOT manufacture significance from it.

**No look-ahead.** Every signal is formed from the *calendar date label alone* — is
today a Friday AND the 17th? That fact is known before the open, so there is no
estimation lag at all (calendar-known rules need none — see METHODOLOGY.md). The
outcome is the close-to-close return of that session.

**Friday-17th dates are pure date arithmetic** (no fetch, no hardcoded table): a date
is a Venerdì 17 iff ``weekday() == 4`` (Friday) and ``day == 17``. Deterministic and
verifiable; ~1.7 events/year, so the effective n over the modern EWI tape is ~50.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete calendar month at publication

# The two real instruments. MIB is the local-currency, price-only sentiment tape;
# EWI is the US-listed, USD, total-return vehicle the tradability timer runs on.
MIB = "FTSEMIB.MI"          # FTSE MIB, EUR, price-only index
EWI = "EWI"                 # iShares MSCI Italy ETF, USD, total-return
TICKERS = (MIB, EWI)

# How each instrument is quoted -- carried into every table so gross/net and
# price-only/total-return are never mislabelled.
QUOTE = {
    MIB: dict(ccy="EUR", adj="price-only", label="FTSE MIB (local, EUR, price-only)"),
    EWI: dict(ccy="USD", adj="total-return", label="EWI iShares MSCI Italy (USD, total-return)"),
}


# --------------------------------------------------------------------------- #
# Calendar arithmetic -- the only "event table" this study needs
# --------------------------------------------------------------------------- #
def is_friday_17th(dates: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: True on every Friday-the-17th (Italy's *Venerdì 17*).

    Pure date arithmetic -- no network, no hardcoded table, because "Friday the
    17th" is a deterministic calendar property: ``weekday == 4`` and ``day == 17``.
    """
    return (dates.weekday == 4) & (dates.day == 17)


def is_friday_dom(dates: pd.DatetimeIndex, dom: int) -> np.ndarray:
    """Mask: True on every Friday whose day-of-month equals ``dom`` (a matched,
    superstition-free control slot -- e.g. the 10th or the 24th)."""
    return (dates.weekday == 4) & (dates.day == dom)


def is_friday(dates: pd.DatetimeIndex) -> np.ndarray:
    """All Fridays (including the 17th) -- the matched baseline weekday."""
    return dates.weekday == 4


# The day-of-month slots a "middle Friday" could plausibly land on, spaced 7 apart
# around the 17th (17 +/- 7k). A snooper testing all of these and picking the most
# extreme would need a multiple-comparisons correction -- the Bonferroni sweep.
SWEEP_DOMS = (3, 10, 17, 24, 31)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"friday17_{ticker.lower().replace('.', '_')}.csv")


def fetch(start: str = "1996-01-01", end: str = "2026-07-01") -> None:
    """Download daily closes for the FTSE MIB index and the EWI ETF; cache them.

    ``auto_adjust=True`` -- for EWI (an equity ETF) that yields total-return closes;
    for the ``FTSEMIB.MI`` *index* there is nothing to adjust (no dividends on the
    price index), so its closes are price-only, and that is labelled everywhere.
    Network; run once. Never imported by the offline notebook cells.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].dropna()
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        d.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: close Series}, each sliced to <= asof."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic tape -- deterministic offline core, planted Friday-17 effect (control)
# --------------------------------------------------------------------------- #
def synthetic_daily(n_years: int = 28, annual_vol: float = 0.22,
                    annual_drift: float = 0.05, f17_effect: float = 0.0,
                    seed: int = 742, start: str = "1998-01-02",
                    ) -> tuple[pd.Series, dict]:
    """A reproducible daily FTSE-MIB-like tape with a KNOWN Friday-17th anomaly.

    Daily log-returns ~ N(drift/252, (vol/sqrt(252))**2), with an additive bump of
    ``f17_effect * daily_sigma`` (in daily-vol units) applied ONLY on Friday-the-17th
    business days.

    * ``f17_effect = 0``  -> pure random walk; Venerdì 17 is a fair coin (the null).
    * ``f17_effect < 0``  -> the folk fear: negative bias on Friday-17.
    * ``f17_effect > 0``  -> contrarian positive bias.

    Returns ``(close, truth)`` where ``close`` is a price Series (business-day index)
    and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    f17 = is_friday_17th(idx)
    ret[f17] += f17_effect * daily_sig

    close = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="Close")
    truth = dict(n_years=n_years, annual_vol=annual_vol, annual_drift=annual_drift,
                 f17_effect=f17_effect, n_days=n, n_f17=int(f17.sum()), seed=seed)
    return close, truth
