"""Data layer for Study 88 (Dogs-of-the-Dow).

Three tapes, one job: let the real data decide whether buying the ten highest-yielding
Dow stocks each January beats the Dow itself, *without* manufacturing the answer with a
survivorship-biased universe.

The pieces:

- ``DOW_CHANGES`` / ``members_asof`` — a **point-in-time** Dow-30 membership timeline,
  encoded from public record (Wikipedia, *"Historical components of the Dow Jones
  Industrial Average"*). Using today's 30 names back to the 1990s would be textbook
  survivorship bias — the names that got *removed* (the losers) would be invisible, and
  the names that got *added* (the winners) would be back-dated into a portfolio that
  never held them. So each January's Dogs are picked only from the members **as of that
  January**. The timeline starts at **1999-11-01** (a clean, well-documented reshuffle:
  +MSFT/INTC/HD/SBC, −CHV/GT/S/UK), the earliest date this desk can reconstruct the
  exact 30 with confidence. Earlier years are deliberately **not** sampled — a shorter
  honest window beats a longer biased one (see ``docs/results.md``; no silent caps).

- ``load_panel`` — the real loader. Daily **total-return** prices and the **dividend**
  stream for every ticker that was ever a Dow member in the window, via yfinance
  (``yf.Ticker(t).dividends`` and ``yf.download(..., auto_adjust=...)``), cached to
  ``_cache/`` as parquet so notebooks run offline after the first fetch.

- ``synthetic_panel`` — a deterministic offline control. ``planted=True`` builds a panel
  where high trailing yield genuinely forecasts a higher forward total return (the
  harness *must* detect it); ``planted=False`` builds one where yield is pure noise (the
  harness must *not*). The study's spine tests both ends.

No look-ahead is baked in here: a January rebalance ranks on the trailing-12-month
dividend yield measured **at the prior December close**, and holds for the next calendar
year — the selection uses only information available at rebalance time.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Point-in-time Dow-30 membership timeline (public record)
# ---------------------------------------------------------------------------
# Anchor: the 1999-11-01 reshuffle. From that date the set of 30 below is exact;
# each later event is a (date, removed, added) tuple from the public record. Tickers
# are the *modern* Yahoo symbols for the same listed entity where the ticker changed
# but the company is continuous (e.g. SBC -> T after the AT&T merger/rename), so the
# price/dividend tape is the one a holder actually experienced.
DOW_1999_11_01 = [
    "AA", "AXP", "T", "BA", "CAT", "C", "KO", "DD", "EK", "GE",
    "GM", "HWP", "HD", "HON", "INTC", "IBM", "IP", "JPM", "JNJ", "MCD",
    "MRK", "MSFT", "MMM", "MO", "PG", "SBC", "UTX", "WMT", "DIS", "XOM",
]
# Note: in 1999 "T" was AT&T Corp and "SBC" was SBC Communications; SBC later bought
# AT&T and took the "T" ticker. To avoid a ticker collision in the panel we keep "SBC" as
# the single telecom line that continues into the modern "T", and bring VZ in at the
# anchor (the legacy AT&T Corp slot). The residual naming ambiguity is named on the Signal
# axis in ``docs/results.md`` — these entity renames cannot manufacture an edge, only blur
# one name's tape at the margin.


def _normalize_changes() -> list[tuple[str, list[str], list[str]]]:
    """The hand-curated, de-duplicated Dow-30 change stream applied by ``members_asof``.

    Each event is ``(effective-date, removed, added)`` from the public record (Wikipedia,
    *"Historical components of the Dow Jones Industrial Average"*), with entity renames
    (SBC->T, HWP->HPQ, the DuPont/Dow saga, UTX/RTX) already collapsed into the
    price-continuation ticker so the panel tracks the tape a holder actually lived.
    """
    return [
        ("2004-04-08", ["EK", "IP"], ["AIG", "PFE"]),       # T(AT&T Corp) handled at anchor
        ("2008-02-19", ["HON", "MO"], ["BAC", "CVX"]),
        ("2008-09-22", ["AIG"], ["KFT"]),
        ("2009-06-08", ["GM", "C"], ["CSCO", "TRV"]),
        ("2012-09-24", ["KFT"], ["UNH"]),
        ("2013-09-23", ["AA", "BAC", "HPQ"], ["GS", "NKE", "V"]),
        ("2015-03-19", ["SBC"], ["AAPL"]),                  # SBC->T (AT&T) replaced by Apple
        ("2018-06-26", ["GE"], ["WBA"]),
        ("2019-06-03", ["DD"], ["DOW"]),                    # DuPont line -> Dow Inc
        ("2020-04-06", ["XOM", "PFE", "UTX"], ["CRM", "AMGN", "HON"]),
        ("2024-02-26", ["WBA"], ["AMZN"]),
        ("2024-11-08", ["DOW"], ["NVDA"]),
    ]


def _anchor_set() -> list[str]:
    """The 30 members as of 1999-11-01, with telecom tickers mapped to price-continuation.

    AT&T Corp ("T" in 1999) was acquired by SBC in 2005; we drop the legacy "T" slot and
    keep "SBC" as the single telecom line (it became the modern "T"), then VZ enters at
    the 2004 reshuffle in place of the legacy AT&T. HWP -> HPQ (Hewlett-Packard),
    UTX/HON/MMM kept as listed. EK = Eastman Kodak, IP = Int'l Paper.
    """
    s = list(DOW_1999_11_01)
    s = [x for x in s if x != "T"]          # drop legacy AT&T Corp slot
    s = ["HPQ" if x == "HWP" else x for x in s]  # Hewlett-Packard ticker rename
    s.append("VZ")                          # telecom slot freed by legacy-T removal
    return s


def members_asof(date: str | pd.Timestamp) -> list[str]:
    """The point-in-time Dow-30 membership as of ``date`` (inclusive of changes up to it).

    Replays the anchor set (1999-11-01) forward through the normalized change stream,
    applying every (removed, added) event dated on or before ``date``. Returns the sorted
    list of tickers in the index on that date. Dates before the anchor return an empty
    list (the study does not sample pre-1999-11 — see the module docstring).
    """
    ts = pd.Timestamp(date)
    anchor_ts = pd.Timestamp("1999-11-01")
    if ts < anchor_ts:
        return []
    members = set(_anchor_set())
    for d, removed, added in _normalize_changes():
        if pd.Timestamp(d) <= ts:
            for r in removed:
                members.discard(r)
            for a in added:
                members.add(a)
    return sorted(members)


def all_tickers_in_window(start: str = "1999-11-01", end: str | None = None) -> list[str]:
    """Every ticker that was a Dow member at any rebalance January in the window.

    The panel must hold prices/dividends for the union of all members ever selected from,
    so that a year's Dogs come from the *then-current* 30 — never from a back-dated set.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    names: set[str] = set()
    for yr in range(pd.Timestamp(start).year, end_ts.year + 1):
        names.update(members_asof(f"{yr}-12-31"))
    return sorted(names)


# ---------------------------------------------------------------------------
# Real loader — prices + dividends via yfinance, cache-first
# ---------------------------------------------------------------------------
# Yahoo download alias: the membership ticker -> the symbol whose tape is the
# *price continuation* of the same listed entity. UTX was renamed RTX in 2020 (one
# continuous listing); Kraft Foods Inc. (the 1999-2012 Dow member, ticker KFT) became
# Mondelez (MDLZ) — the price line a KFT holder actually lived through. These are renames
# of one listing, not substitutions of a different company, so they cannot manufacture an
# edge. Names with NO recoverable Yahoo tape (EK = the old Eastman Kodak entity, bankrupt
# 2012; WBA = Walgreens, taken private and delisted late 2024) are left out and disclosed
# in ``docs/results.md`` — a named gap, never a silent one.
TICKER_ALIAS = {"UTX": "RTX", "KFT": "MDLZ"}
UNRECOVERABLE = ("EK", "WBA")


def _cache_path(name: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, name)


def load_panel(
    start: str = "1999-12-01",
    end: str | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Load the real Dow panel: total-return prices + dividends for every member ticker.

    Returns a dict with:

    - ``prices`` — a wide DataFrame of **total-return** (auto-adjusted) daily closes,
      one column per ticker (used to compound the held portfolio's total return).
    - ``divs`` — a wide DataFrame of per-share **dividends** on their ex-dates (used to
      build the trailing-12-month dividend yield at each December rebalance).
    - ``bench`` — the Dow total-return benchmark (DIA, the SPDR Dow ETF, total return,
      since 1998) as a one-column ``close`` frame; the fair same-basis yardstick.

    Cache-first: each ticker's price/dividend series and the benchmark are cached to
    parquet under ``_cache/``. The network is touched only on a cache miss.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    tickers = all_tickers_in_window(end=end)

    prices = {}
    divs = {}
    for t in tickers:
        px, dv = _load_one(t, cache_dir, use_cache)
        if px is not None and len(px):
            prices[t] = px
        if dv is not None and len(dv):
            divs[t] = dv

    price_df = pd.DataFrame(prices).sort_index()
    div_df = pd.DataFrame(divs).sort_index()
    price_df = price_df.loc[start:end]
    bench = _load_bench(cache_dir, use_cache).loc[start:end]
    return {"prices": price_df, "divs": div_df, "bench": bench, "tickers": tickers}


def _load_one(ticker: str, cache_dir: str, use_cache: bool):
    """Total-return close + dividend stream for one ticker, cached to parquet.

    ``ticker`` is the *membership* symbol (the panel column); the Yahoo fetch uses its
    price-continuation alias (``TICKER_ALIAS``) where the listing was renamed.
    """
    px_path = _cache_path(f"{ticker}_px.parquet", cache_dir)
    dv_path = _cache_path(f"{ticker}_div.parquet", cache_dir)
    if use_cache and os.path.exists(px_path) and os.path.exists(dv_path):
        px = pd.read_parquet(px_path)["close"]
        dv = pd.read_parquet(dv_path)["div"]
        return px, dv

    import yfinance as yf

    fetch_symbol = TICKER_ALIAS.get(ticker, ticker)
    tk = yf.Ticker(fetch_symbol)
    raw = tk.history(period="max", auto_adjust=True)  # total-return closes
    if raw is None or raw.empty:
        return None, None
    px = raw["Close"].rename("close")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    dv = tk.dividends.rename("div")
    if dv is None or len(dv) == 0:
        dv = pd.Series(dtype=float, name="div")
    dv.index = pd.to_datetime(dv.index).tz_localize(None)
    if use_cache:
        px.to_frame().to_parquet(px_path)
        dv.to_frame().to_parquet(dv_path)
    return px, dv


def _load_bench(cache_dir: str, use_cache: bool) -> pd.DataFrame:
    """DIA (SPDR Dow Jones Industrial Average ETF), total return — the same-basis yardstick."""
    path = _cache_path("DIA_bench.parquet", cache_dir)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.Ticker("DIA").history(period="max", auto_adjust=True)
    close = raw["Close"].rename("close")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    out = close.to_frame()
    if use_cache:
        out.to_parquet(path)
    return out


# ---------------------------------------------------------------------------
# Synthetic panel control — deterministic, offline
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 30,
    n_years: int = 25,
    planted: bool = True,
    excess: float = 0.04,
    seed: int = 88,
) -> dict:
    """A deterministic annual panel for the positive/negative control.

    Builds ``n_names`` stocks over ``n_years`` calendar years. Each name has an annual
    total return and a trailing dividend yield. When ``planted=True``, a name's yield in
    year *y* carries a genuine forecasting edge: the top-decile-yield names earn
    ``excess`` extra total return next year (the harness MUST detect this). When
    ``planted=False``, yield is drawn independently of forward return (the harness must
    NOT detect an edge).

    Returns a dict with ``yields`` (years x names), ``fwd_returns`` (years x names) and a
    ``truth`` block. The strategy's annual Dogs engine consumes exactly this shape, so the
    spine can run the *same* selection logic on a tape where the answer is known.
    """
    rng = np.random.default_rng(seed)
    years = list(range(2000, 2000 + n_years))
    names = [f"N{i:02d}" for i in range(n_names)]

    base_ret = 0.08
    vol = 0.16
    yields = rng.uniform(0.005, 0.06, size=(n_years, n_names))
    noise = rng.normal(0.0, vol, size=(n_years, n_names))
    fwd = base_ret + noise

    if planted:
        # Rank within each year; top-10 (highest yield) get the planted excess next year.
        for y in range(n_years):
            order = np.argsort(-yields[y])
            top = order[:10]
            fwd[y, top] += excess

    yld_df = pd.DataFrame(yields, index=years, columns=names)
    fwd_df = pd.DataFrame(fwd, index=years, columns=names)
    truth = {
        "planted": planted, "excess": excess if planted else 0.0,
        "n_names": n_names, "n_years": n_years, "seed": seed,
    }
    return {"yields": yld_df, "fwd_returns": fwd_df, "truth": truth}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """A short content fingerprint of a tape, for the as-of stamp.

    Accepts a DataFrame, Series, or the panel dict (hashes the benchmark close + the
    flattened price matrix), so a reader who reruns and matches the digest holds the same
    inputs; a mismatch flags drift loudly.
    """
    h = hashlib.sha1()
    if isinstance(obj, dict):
        for key in ("bench", "prices"):
            if key in obj and obj[key] is not None:
                arr = np.ascontiguousarray(
                    np.nan_to_num(obj[key].to_numpy(dtype=float))
                )
                h.update(arr.tobytes())
    elif isinstance(obj, pd.DataFrame):
        h.update(np.ascontiguousarray(np.nan_to_num(obj.to_numpy(dtype=float))).tobytes())
    else:
        h.update(np.ascontiguousarray(np.nan_to_num(np.asarray(obj, dtype=float))).tobytes())
    return h.hexdigest()[:12]
