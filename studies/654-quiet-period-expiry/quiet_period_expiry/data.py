"""Data layer for Study 654 — Quiet-Period-Expiry.

The claim (Bradley, Jordan & Ritter 2003, *The Quiet Period Goes Out With a Bang*, JF):
FINRA/NASD Rule 2711 bars the IPO's managing/co-managing underwriters from publishing
research on the stock for **25 calendar days** after the offering. The moment that
lockup on *opinions* (not shares — that's the separate 180-day lockup, study
[319-lockup-expiry](../319-lockup-expiry/)) lifts, the syndicate's own analysts pile in
with initiations that are almost all Buy/Overweight (the desk that just sold you the
deal is not about to badmouth it) — and BJR (2003) find a statistically significant
pop right around the expiration.

Two tapes, one shape — the same event-time abnormal-return panel design as sibling
[319-lockup-expiry](../319-lockup-expiry/), just centered on trading day ~25 (the
quiet-period proxy) instead of trading day ~125 (the 180-calendar-day lockup):

* **Real tape.** A hardcoded basket of ~65 real US underwritten IPOs, 2015→2025 (ticker +
  first trading day). For each, daily adjusted closes from yfinance (no key) plus SPY as
  the market-adjustment benchmark, cached under the study's own ``_cache/``. Abnormal
  return on trading day *t* since listing = stock return − SPY return on that calendar
  date (a raw market-adjusted, beta = 1 control — the same convention as 319, stated as a
  decision, not a detail: fitting per-IPO betas on a few weeks of post-listing history is
  noisier than assuming beta ≈ 1).
* **Synthetic world.** A deterministic, seeded panel of abnormal-return paths with a
  TUNABLE planted pop (``pop_bps``) landing in the [20..30] trading-day window.
  ``pop_bps = 0`` is the null — the machinery must not manufacture a pop from noise.

**Day-count convention (stated as a decision).** "Day *t*" = *t* trading sessions after
the stock's first trading day (day 0 = the first close, matching the day-0 convention in
siblings 219/319/623 — NOT the offer price). BJR's 25-*calendar*-day quiet period lands
close to trading day ~18, but analyst initiations cluster in the days *after* expiration
(compliance sign-off, scheduling) rather than exactly on it — so, per the brief, the
window tested is trading days **[20..30]**, bracketing the plain trading-day-25 proxy
used throughout this module, with a "buy day 22, sell day 27" timer as the literal
retail-facing version of the trade.

Pure numpy + pandas + stdlib on the offline path. ``fetch_event_panel(fetch=True)``
(network) runs once to build the cache; the notebooks' offline cells never touch it.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-10)
BENCHMARK = "SPY"

# Event-time window under test (trading days since the IPO's first close) — the
# quiet-period-expiry proxy, per the brief.
QUIET_TD = 25
WINDOW_LO, WINDOW_HI = 20, 30
# The retail-facing timer: buy the close of day 22, sell the close of day 27.
TIMER_BUY, TIMER_SELL = 22, 27
# A same-width [3..13] placebo window, early in the post-IPO life and away from the
# quiet-period-expiry window, used as the within-sample robustness comparison.
PLACEBO_LO, PLACEBO_HI = 3, 13
# How many trading days of panel to build per IPO (buffer past WINDOW_HI/TIMER_SELL).
MAX_TD = 40
# Era split for the "has it faded?" question — the sample's approximate calendar
# midpoint, fixed here (before any stats were run), not tuned to the result.
ERA_SPLIT = "2021-06-01"

# --------------------------------------------------------------------------- #
# Hardcoded basket of real, underwritten US IPOs, 2015 -> 2025 (first trading day).
# Direct listings and SPAC mergers are EXCLUDED by construction: FINRA Rule 2711's
# 25-day quiet period is a restriction on the *managing underwriters* of a firm-
# commitment offering, so it does not bind the same way on a direct listing (no
# underwriter) or SPAC de-SPAC (no traditional book-build). Sources: company IPO
# prospectuses / exchange listing notices / public IPO trackers (Renaissance Capital,
# NASDAQ/NYSE newsroom). Only tickers still queryable on yfinance for their post-IPO
# window enter the panel (survivorship on data availability, named on the Signal axis
# below and in docs/results.md — it can only THIN the basket, not manufacture a pop).
# --------------------------------------------------------------------------- #
IPO_BASKET: list[dict] = [
    {"ticker": "SHOP", "ipo_date": "2015-05-21"},
    {"ticker": "ETSY", "ipo_date": "2015-04-16"},
    {"ticker": "GDDY", "ipo_date": "2015-04-01"},
    {"ticker": "BOX",  "ipo_date": "2015-01-23"},
    {"ticker": "FIT",  "ipo_date": "2015-06-18"},
    {"ticker": "TWLO", "ipo_date": "2016-06-23"},
    {"ticker": "NTNX", "ipo_date": "2016-09-30"},
    {"ticker": "COUP", "ipo_date": "2016-10-06"},
    {"ticker": "SNAP", "ipo_date": "2017-03-02"},
    {"ticker": "OKTA", "ipo_date": "2017-04-07"},
    {"ticker": "MULE", "ipo_date": "2017-03-17"},
    {"ticker": "YEXT", "ipo_date": "2017-04-13"},
    {"ticker": "ROKU", "ipo_date": "2017-09-28"},
    {"ticker": "SFIX", "ipo_date": "2017-11-17"},
    {"ticker": "DBX",  "ipo_date": "2018-03-23"},
    {"ticker": "DOCU", "ipo_date": "2018-04-27"},
    {"ticker": "ZS",   "ipo_date": "2018-03-16"},
    {"ticker": "PAGS", "ipo_date": "2018-01-24"},
    {"ticker": "LYFT", "ipo_date": "2019-03-29"},
    {"ticker": "PINS", "ipo_date": "2019-04-18"},
    {"ticker": "ZM",   "ipo_date": "2019-04-18"},
    {"ticker": "UBER", "ipo_date": "2019-05-10"},
    {"ticker": "CRWD", "ipo_date": "2019-06-12"},
    {"ticker": "FVRR", "ipo_date": "2019-06-13"},
    {"ticker": "CHWY", "ipo_date": "2019-06-14"},
    {"ticker": "DDOG", "ipo_date": "2019-09-19"},
    {"ticker": "PD",   "ipo_date": "2019-04-11"},
    {"ticker": "BYND", "ipo_date": "2019-05-02"},
    {"ticker": "NET",  "ipo_date": "2019-09-13"},
    {"ticker": "SNOW", "ipo_date": "2020-09-16"},
    {"ticker": "U",    "ipo_date": "2020-09-18"},
    {"ticker": "ZI",   "ipo_date": "2020-06-04"},
    {"ticker": "LMND", "ipo_date": "2020-07-02"},
    {"ticker": "BIGC", "ipo_date": "2020-08-05"},
    {"ticker": "DASH", "ipo_date": "2020-12-09"},
    {"ticker": "ABNB", "ipo_date": "2020-12-10"},
    {"ticker": "RKT",  "ipo_date": "2020-08-06"},
    {"ticker": "GDRX", "ipo_date": "2020-09-23"},
    {"ticker": "RIVN", "ipo_date": "2021-11-10"},
    {"ticker": "AFRM", "ipo_date": "2021-01-13"},
    {"ticker": "PATH", "ipo_date": "2021-04-21"},
    {"ticker": "GLBE", "ipo_date": "2021-05-12"},
    {"ticker": "BROS", "ipo_date": "2021-09-15"},
    {"ticker": "DNUT", "ipo_date": "2021-07-01"},
    {"ticker": "CPNG", "ipo_date": "2021-03-11"},
    {"ticker": "MNDY", "ipo_date": "2021-06-10"},
    {"ticker": "CFLT", "ipo_date": "2021-06-24"},
    {"ticker": "MQ",   "ipo_date": "2021-06-09"},
    {"ticker": "HOOD", "ipo_date": "2021-07-29"},
    {"ticker": "TPG",  "ipo_date": "2022-01-13"},
    {"ticker": "BLCO", "ipo_date": "2022-05-06"},
    {"ticker": "CRBG", "ipo_date": "2022-09-14"},
    {"ticker": "MBLY", "ipo_date": "2022-10-26"},
    {"ticker": "CART", "ipo_date": "2023-09-19"},
    {"ticker": "KVYO", "ipo_date": "2023-09-20"},
    {"ticker": "ARM",  "ipo_date": "2023-09-14"},
    {"ticker": "BIRK", "ipo_date": "2023-10-11"},
    {"ticker": "RDDT", "ipo_date": "2024-03-21"},
    {"ticker": "ALAB", "ipo_date": "2024-03-20"},
    {"ticker": "AS",   "ipo_date": "2024-02-01"},
    {"ticker": "IBTA", "ipo_date": "2024-04-18"},
    {"ticker": "RBRK", "ipo_date": "2024-04-25"},
    {"ticker": "TEM",  "ipo_date": "2024-06-14"},
    {"ticker": "OS",   "ipo_date": "2024-07-25"},
    {"ticker": "SARO", "ipo_date": "2024-10-04"},
    {"ticker": "CRCL", "ipo_date": "2025-06-05"},
]


def basket_frame() -> pd.DataFrame:
    """The hardcoded IPO basket as a DataFrame (ticker, ipo_date), sorted."""
    rows = [{"ticker": r["ticker"], "ipo_date": pd.Timestamp(r["ipo_date"])}
            for r in IPO_BASKET]
    return pd.DataFrame(rows).sort_values("ipo_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real loader — yfinance stock + SPY, cache-first
# --------------------------------------------------------------------------- #
def _price_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_654_{safe}_1d.parquet")


def _load_close(ticker: str, cache_dir: str, fetch: bool, start: str) -> pd.Series | None:
    """Daily adjusted close for one ticker; cache-first. None if unavailable."""
    path = _price_cache_path(ticker, cache_dir)
    if fetch:
        import yfinance as yf  # lazy: only when going to the network

        try:
            raw = yf.download(ticker, start=start, interval="1d",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=str.lower)
            if not raw.empty and "close" in raw.columns:
                raw[["close"]].to_parquet(path)
            else:
                pd.DataFrame(columns=["close"]).to_parquet(path)
        except Exception:
            pd.DataFrame(columns=["close"]).to_parquet(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached prices for {ticker} at {path}. "
            f"Call fetch_event_panel(fetch=True) once to populate the cache."
        )
    pr = pd.read_parquet(path)
    if pr.empty or "close" not in pr.columns:
        return None
    s = pr["close"].copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    s = s.sort_index().dropna()
    s.name = ticker
    return s


def have_real(cache_dir: str = CACHE_DIR) -> bool:
    """True iff every basket ticker + the benchmark already has a cache file."""
    tickers = list(basket_frame()["ticker"]) + [BENCHMARK]
    return all(os.path.exists(_price_cache_path(t, cache_dir)) for t in tickers)


def fetch(cache_dir: str = CACHE_DIR) -> None:
    """Populate the cache once (network). Thin wrapper so ``verify.py`` matches siblings."""
    fetch_event_panel(fetch=True, cache_dir=cache_dir)


def fetch_event_panel(
    fetch: bool = False,
    cache_dir: str = CACHE_DIR,
    max_td: int = MAX_TD,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """Build the real event-time panel for the hardcoded IPO basket.

    For each IPO: locate the first trading day on/after ``ipo_date`` (day 0), take the
    next ``max_td`` sessions, compute the daily stock return and the SPY return on the
    SAME calendar dates (reindexed — no assumption the two tapes share a positional
    index), and abnormal return = stock − SPY (beta = 1 market adjustment, the
    convention shared with sibling 319-lockup-expiry).

    With ``fetch=False`` (default) this is cache-only — no network. ``fetch=True``
    populates the parquet cache.

    Returns
    -------
    panel : DataFrame, long format
        One row per (ticker, t) for t = 0..max_td. Columns: ticker, ipo_date, t, close,
        ret, abn_ret. t = 0 carries the first close (ret/abn_ret are NaN — no prior bar).
        IPOs with insufficient post-listing history are skipped (named — this can only
        THIN the basket, never manufacture a result).
    """
    os.makedirs(cache_dir, exist_ok=True)
    basket = basket_frame()

    bench_start = (basket["ipo_date"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    bench = _load_close(benchmark, cache_dir, fetch, bench_start)
    if bench is None:
        raise RuntimeError(f"benchmark {benchmark} has no cached close data")
    bench_ret = bench.pct_change()

    rows = []
    for _, r in basket.iterrows():
        tkr, ipo_dt = r["ticker"], r["ipo_date"]
        start = (ipo_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        s = _load_close(tkr, cache_dir, fetch, start)
        if s is None or len(s) < 5:
            continue
        pos0 = int(np.searchsorted(s.index, ipo_dt))
        if pos0 >= len(s) or pos0 + max_td >= len(s):
            continue  # not enough post-listing history yet (or ticker died early)

        stock_ret = s.pct_change()
        aligned_bench = bench_ret.reindex(s.index)
        abn = stock_ret - aligned_bench

        sl_close = s.iloc[pos0: pos0 + max_td + 1]
        sl_ret = stock_ret.iloc[pos0: pos0 + max_td + 1]
        sl_abn = abn.iloc[pos0: pos0 + max_td + 1]
        if len(sl_close) != max_td + 1:
            continue
        for t in range(max_td + 1):
            rows.append({
                "ticker": tkr, "ipo_date": ipo_dt, "t": t,
                "close": float(sl_close.iloc[t]),
                "ret": float(sl_ret.iloc[t]) if t > 0 else float("nan"),
                "abn_ret": float(sl_abn.iloc[t]) if t > 0 else float("nan"),
            })

    return pd.DataFrame(rows)


def fingerprint(panel: pd.DataFrame) -> str:
    """Short content fingerprint of the event panel (abnormal returns), for the as-of stamp."""
    arr = np.ascontiguousarray(
        panel.sort_values(["ticker", "t"])["abn_ret"].fillna(0.0).to_numpy(dtype=float)
    )
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted quiet-period-expiry pop (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_events: int = 65,
    pop_bps: float = 0.0,
    max_td: int = MAX_TD,
    abn_vol_ann: float = 0.55,
    seed: int = 654,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible panel of event-time abnormal-return paths, quiet-period shape.

    Each synthetic IPO gets i.i.d.-noise daily abnormal returns for t = 1..``max_td``,
    plus a one-shot planted pop of ``pop_bps`` spread evenly across the
    [``WINDOW_LO``..``WINDOW_HI``] window (the believers' story: the initiations land
    over the ~2 trading weeks after quiet-period expiry, not all on one bar).
    ``pop_bps = 0`` is a pure null — the detector must NOT manufacture a window effect.

    Returns
    -------
    panel : DataFrame, long format (ticker, t, abn_ret) — same shape as the real panel.
    truth : dict of the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sig_d = abn_vol_ann / np.sqrt(252)
    win_width = WINDOW_HI - WINDOW_LO + 1
    per_day_bps = pop_bps / win_width

    rows = []
    for ev in range(n_events):
        abn = rng.normal(0.0, sig_d, max_td + 1)
        abn[0] = np.nan
        for t in range(WINDOW_LO, WINDOW_HI + 1):
            abn[t] += per_day_bps * 1e-4
        for t in range(max_td + 1):
            rows.append({"ticker": f"SYN{ev:03d}", "t": t,
                         "abn_ret": float(abn[t]) if t > 0 else float("nan")})

    panel = pd.DataFrame(rows)
    truth = {"n_events": n_events, "pop_bps": pop_bps, "max_td": max_td,
             "abn_vol_ann": abn_vol_ann, "seed": seed}
    return panel, truth
