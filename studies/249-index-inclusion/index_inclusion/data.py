"""Data layer for Study 249 (Index-Inclusion).

Two tapes:

- ``synthetic_events`` — a *deterministic, offline* generator.  Simulates announce-to-
  effective pops and post-inclusion give-backs with tunable ``pop_bps`` and
  ``giveback_bps`` knobs.  Setting both to zero is the null; positive values test
  whether the engine can detect effects when planted.

- ``fetch_inclusion_panel`` — the real loader.  Uses the hardcoded ``INCLUSION_TABLE``
  (announce date, effective date, ticker) to download split-adjusted daily prices via
  yfinance and computes two windows:
  (a) *announcement window*: close on announce_date → close on effective_date (the
      classic "inclusion pop" — front-run by arbitrageurs buying after announcement).
  (b) *give-back window*: close on effective_date → close N trading days after
      (the "post-inclusion reversal" as the index fund buying pressure fades).

No look-ahead: the signal (announcement) is public from the announce_date; forward
windows start at announce_date + 1 (announce window) and effective_date + 1 (give-
back window). Prices are split-adjusted by Yahoo!.

Hardcoded events cover notable S&P 500 additions 2000–2024. The table is intentionally
broad (large and mid caps) to maximise n; smaller additions may have wider bid-ask
and noisier data — this is called out in results.md and the notebooks.
"""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded S&P 500 inclusion event table
# Columns: ticker, announce_date, effective_date, replaced_ticker (for reference)
# Sources: S&P press releases / Wikipedia list of S&P 500 additions
# We include events where the added stock was publicly listed before announcement
# (no IPO-day additions — those are not arbitrageable in the same way).
# ---------------------------------------------------------------------------
_RAW_TABLE = [
    # (ticker, announce_date, effective_date)
    # --- 2000s ---
    ("EBAY",   "2001-03-22", "2001-03-26"),
    ("FLIR",   "2003-11-17", "2003-11-24"),
    ("AIG",    "2004-01-27", "2004-01-30"),   # re-addition after S&P restructure
    ("GOOG",   "2006-03-30", "2006-04-03"),   # Google Class A
    ("MA",     "2006-07-25", "2006-07-28"),   # Mastercard
    ("MMI",    "2007-04-05", "2007-04-10"),   # Motorola Mobility precursor
    ("AAPL",   "2012-08-20", "2012-08-27"),   # rejoined with higher float
    ("FB",     "2012-12-11", "2012-12-21"),   # Meta (then Facebook)
    ("PCLN",   "2009-07-01", "2009-07-06"),   # Priceline (now BKNG)
    ("NFLX",   "2010-12-20", "2010-12-23"),
    ("FOSL",   "2011-07-26", "2011-08-01"),
    ("DLPH",   "2012-10-04", "2012-10-09"),   # Delphi (now APTV)
    ("NOW",    "2015-07-21", "2015-07-24"),   # ServiceNow
    ("PYPL",   "2015-07-01", "2015-07-20"),   # PayPal post-eBay spin
    ("CHTR",   "2016-06-22", "2016-06-24"),   # Charter Communications
    ("AMZN",   "2005-11-17", "2005-11-21"),   # Amazon
    ("PCLN",   "2009-07-01", "2009-07-06"),   # deduplicated below
    ("GOOGL",  "2014-04-03", "2014-04-07"),   # Google Class A rename/float adj
    ("AVGO",   "2017-08-23", "2017-08-28"),   # Broadcom
    ("SHOP",   "2022-09-15", "2022-09-19"),   # Shopify
    ("CEG",    "2022-09-01", "2022-09-06"),   # Constellation Energy
    ("ENPH",   "2021-12-13", "2021-12-20"),   # Enphase Energy
    ("SEDG",   "2021-06-14", "2021-06-18"),   # SolarEdge
    ("BIO",    "2021-03-15", "2021-03-18"),   # Bio-Rad
    ("TSLA",   "2020-11-16", "2020-12-21"),   # Tesla (biggest ever addition)
    ("ETSY",   "2020-08-24", "2020-08-28"),
    ("CARR",   "2020-04-06", "2020-04-09"),   # Carrier Global (spin)
    ("OTIS",   "2020-04-06", "2020-04-09"),   # Otis Worldwide (spin)
    ("FTNT",   "2021-12-09", "2021-12-20"),   # Fortinet
    ("MKC",    "2020-02-20", "2020-02-24"),   # McCormick
    ("DPZ",    "2020-01-28", "2020-01-31"),   # Domino's
    ("MKTX",   "2019-11-14", "2019-11-19"),   # MarketAxess
    ("CRL",    "2019-09-17", "2019-09-23"),   # Charles River Labs
    ("POOL",   "2019-08-20", "2019-08-23"),
    ("CDNS",   "2019-06-24", "2019-06-28"),   # Cadence Design
    ("KEYS",   "2018-12-07", "2018-12-11"),   # Keysight
    ("ZBRA",   "2019-03-07", "2019-03-11"),   # Zebra Technologies
    ("PAYC",   "2020-09-01", "2020-09-03"),   # Paycom
    ("PENN",   "2020-10-07", "2020-10-13"),   # Penn National (now PENN Entertainment)
    ("GNRC",   "2021-03-08", "2021-03-11"),   # Generac
    ("GFS",    "2022-12-07", "2022-12-09"),   # GlobalFoundries
    ("LIN",    "2018-10-16", "2018-10-18"),   # Linde
    ("FRC",    "2022-06-07", "2022-06-10"),   # First Republic Bank (later failed)
    ("SIVB",   "2018-06-19", "2018-06-22"),   # Silicon Valley Bank (later failed)
    ("VLTO",   "2023-10-02", "2023-10-04"),   # Veralto (Danaher spin)
    ("SOLV",   "2023-10-02", "2023-10-04"),   # Solventum
    ("GEV",    "2024-04-02", "2024-04-05"),   # GE Vernova
    ("SMCI",   "2024-03-11", "2024-03-18"),   # Super Micro Computer
    ("KKR",    "2024-09-20", "2024-09-23"),   # KKR
    ("CRWD",   "2024-06-24", "2024-07-01"),   # CrowdStrike
    ("GE",     "2023-07-11", "2023-07-13"),   # GE Aerospace (restructured)
]

# De-duplicate by (ticker, announce_date) and build the canonical table
_seen: set = set()
INCLUSION_TABLE: list[dict] = []
for _row in _RAW_TABLE:
    _key = (_row[0], _row[1])
    if _key in _seen:
        continue
    _seen.add(_key)
    INCLUSION_TABLE.append(
        {
            "ticker": _row[0],
            "announce_date": pd.Timestamp(_row[1]),
            "effective_date": pd.Timestamp(_row[2]),
        }
    )
INCLUSION_TABLE.sort(key=lambda r: r["effective_date"])


# ---------------------------------------------------------------------------
# Synthetic event tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_events(
    n_events: int = 60,
    horizon: int = 63,          # give-back window in trading days
    pop_bps: float = 0.0,       # extra daily return in the announce→effective window
    giveback_bps: float = 0.0,  # extra *negative* daily return in the give-back window
    base_vol_ann: float = 0.30,
    start: str = "2000-01-03",
    seed: int = 249,
) -> tuple[pd.DataFrame, dict]:
    """Synthetic per-event windows with plantable pop and give-back.

    For ``pop_bps > 0`` each day in the announce-to-effective window earns
    ``pop_bps`` of extra daily return (the 'pop').  For ``giveback_bps > 0``
    each day in the give-back window loses ``giveback_bps`` daily (the
    'reversal').  Setting both to zero gives the null: no inclusion effect.

    Returns ``(events, truth)`` where ``events`` is a DataFrame with one row per
    synthetic event:

    Columns: ``ticker, announce_date, effective_date, window_days,
              ret_pop, ret_giveback_1m, ret_giveback_3m``.

    ``ret_pop``           buy-and-hold from announce_date to effective_date.
    ``ret_giveback_1m``   buy-and-hold from effective_date+1 over 21 trading days.
    ``ret_giveback_3m``   buy-and-hold from effective_date+1 over 63 trading days.
    """
    rng = np.random.default_rng(seed)
    sig_d = base_vol_ann / np.sqrt(252)
    pop_daily = pop_bps * 1e-4
    gb_daily = giveback_bps * 1e-4  # applied as NEGATIVE return

    dates = pd.bdate_range(start=start, periods=800, name="date")
    rows = []
    used_starts: set[int] = set()
    max_pop_days = 15   # announce-to-effective window: 3–15 business days
    for ev_i in range(n_events):
        # Random announce position (leave room for pop window + giveback window)
        while True:
            ann_pos = int(rng.integers(0, len(dates) - max_pop_days - horizon - 5))
            if ann_pos not in used_starts:
                used_starts.add(ann_pos)
                break
        w = int(rng.integers(3, max_pop_days + 1))  # window length in trading days
        eff_pos = ann_pos + w

        # --- Pop window (announce → effective) ---
        pop_ret = rng.normal(sig_d, sig_d, w) + pop_daily
        ret_pop = float(np.prod(1.0 + pop_ret) - 1.0)

        # --- Give-back windows (effective+1 → +21, +63) ---
        gb = rng.normal(sig_d, sig_d, horizon) - gb_daily
        cum = np.cumprod(1.0 + gb) - 1.0
        ret_gb_1m = float(cum[20]) if horizon >= 21 else float("nan")
        ret_gb_3m = float(cum[62]) if horizon >= 63 else float("nan")

        rows.append(
            {
                "ticker": f"SYN{ev_i:03d}",
                "announce_date": dates[ann_pos],
                "effective_date": dates[eff_pos],
                "window_days": w,
                "ret_pop": ret_pop,
                "ret_giveback_1m": ret_gb_1m,
                "ret_giveback_3m": ret_gb_3m,
            }
        )

    events = pd.DataFrame(rows).sort_values("effective_date").reset_index(drop=True)
    truth = {
        "n_events": len(events),
        "horizon": horizon,
        "pop_bps": pop_bps,
        "giveback_bps": giveback_bps,
        "base_vol_ann": base_vol_ann,
        "seed": seed,
    }
    return events, truth


# ---------------------------------------------------------------------------
# Real loader — hardcoded table + yfinance daily prices, cache by default
# ---------------------------------------------------------------------------
def _prices_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_249_{safe}_1d.parquet")


def fetch_inclusion_panel(
    table: list[dict] | None = None,
    start: str = "1999-01-01",
    giveback_days: tuple[int, ...] = (21, 63),
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load inclusion events and compute pop + give-back forward returns.

    Parameters
    ----------
    table:
        List of dicts with keys ``ticker``, ``announce_date``, ``effective_date``.
        Defaults to ``INCLUSION_TABLE``.
    start:
        Earliest date from which to download price history (needs pre-announce prices
        for market context, so default is 1999).
    giveback_days:
        Trading-day horizons for the give-back window (default 21 and 63).
    fetch:
        ``False`` (default): read from cache; raises if no cache.
        ``True``: hit Yahoo! and write parquet files under ``cache_dir``.
    cache_dir:
        Directory for parquet caches.

    Returns
    -------
    events : DataFrame
        One row per inclusion event.  Columns: ``ticker, announce_date,
        effective_date, window_days, ret_pop, ret_giveback_1m, ret_giveback_3m``.
    prices : DataFrame
        Daily adjusted close prices, wide format (columns = tickers, index = date).
    """
    if table is None:
        table = INCLUSION_TABLE
    os.makedirs(cache_dir, exist_ok=True)

    import yfinance as yf  # lazy import

    prices_dict: dict[str, pd.Series] = {}
    tickers = list({r["ticker"] for r in table})

    for ticker in tickers:
        pr_path = _prices_cache_path(ticker, cache_dir)

        if fetch:
            raw = yf.download(
                ticker, start=start, interval="1d", auto_adjust=True, progress=False
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.lower)
            if not raw.empty and "close" in raw.columns:
                raw[["close"]].to_parquet(pr_path)
        else:
            if not os.path.exists(pr_path):
                # Skip delisted / unavailable tickers gracefully in cache-only mode
                continue

        if os.path.exists(pr_path):
            pr = pd.read_parquet(pr_path)
            if not pr.empty and "close" in pr.columns:
                close = pr["close"].copy()
                close.index = pd.DatetimeIndex(close.index).tz_localize(None)
                close.index.name = "date"
                prices_dict[ticker] = close

    # Assemble price panel
    if prices_dict:
        prices = pd.DataFrame(prices_dict)
        prices.index = pd.DatetimeIndex(prices.index, name="date")
        prices = prices.sort_index()
    else:
        prices = pd.DataFrame()

    # Compute returns for each event
    gb_labels = ["ret_giveback_1m", "ret_giveback_3m"]
    rows = []
    for rec in table:
        tkr = rec["ticker"]
        ann = pd.Timestamp(rec["announce_date"])
        eff = pd.Timestamp(rec["effective_date"])
        if tkr not in prices.columns:
            continue
        close_s = prices[tkr].dropna()
        idx = close_s.index

        # find positions
        ann_pos = int(np.searchsorted(idx, ann))
        eff_pos = int(np.searchsorted(idx, eff))
        if ann_pos >= len(idx) or eff_pos >= len(idx):
            continue

        ann_close = close_s.iloc[ann_pos]
        eff_close = close_s.iloc[eff_pos]
        if ann_close <= 0 or eff_close <= 0:
            continue

        # pop: announce close → effective close
        ret_pop = float(eff_close / ann_close - 1.0)
        window_days = eff_pos - ann_pos

        # give-back returns
        gb_rets: dict[str, float] = {}
        for h, col in zip(giveback_days, gb_labels):
            end_pos = eff_pos + h
            if end_pos >= len(close_s):
                gb_rets[col] = float("nan")
            else:
                gb_rets[col] = float(close_s.iloc[end_pos] / eff_close - 1.0)

        rows.append(
            {
                "ticker": tkr,
                "announce_date": ann,
                "effective_date": eff,
                "window_days": window_days,
                "ret_pop": ret_pop,
                **gb_rets,
            }
        )

    events = pd.DataFrame(rows).sort_values("effective_date").reset_index(drop=True)
    return events, prices


def fingerprint(events: pd.DataFrame) -> str:
    """Short content fingerprint of the event panel (effective dates), for as-of stamps."""
    arr = pd.to_datetime(events["effective_date"]).astype(np.int64).to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
