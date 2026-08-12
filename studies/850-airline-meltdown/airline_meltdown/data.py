"""Data layer for Study 850 — Airline Operational Meltdown.

The claim under test, steelmanned: **a very public operational meltdown — a multi-day
grounding, a mass-cancellation collapse, a viral PR disaster — inflicts a reputational
shock that dents the implicated airline's stock**, both in the days around the event
(an abnormal drop) and as a slow bleed over the following month (a reputational drift),
rather than fading within a session or two as the market shrugs it off.

Three ingredients, all offline-friendly once cached.

* **The meltdown table, hardcoded.** ``EVENTS`` is a curated table of **10 famous
  airline / Boeing operational meltdowns, 2016 -> 2024**, each tagged with the single
  **implicated ticker** whose reputation actually took the hit. No free,
  machine-readable "operational-meltdown index" exists, so — exactly like the sibling
  studies that hand-build an event calendar (``707-plane-crash-effect``'s
  ``DISASTERS``, ``313-geopolitical-shock``'s ``SHOCK_TABLE``) — this is a hand-built
  table of the meltdowns any reasonable person would call "the airline's front-page
  disaster that week", each row carrying a one-line public-record source note. The
  meltdown date is the first calendar date the collapse was unambiguously public; the
  event-study code snaps it forward to the first NYSE session on/after it (the single
  documented execution lag).

* **Real tape.** Daily total-return adjusted closes (yfinance, ``auto_adjust=True``, no
  key) for the market benchmark **SPY** and the five implicated equities that are still
  listed — **LUV** (Southwest), **DAL** (Delta), **UAL** (United), **AAL** (American)
  and **BA** (Boeing). **SAVE (Spirit Airlines) is NOT fetchable:** Spirit filed
  Chapter 11 in Nov-2024, its common stock was cancelled on the 2025 reorganisation,
  and the ``SAVE`` symbol is fully delisted from Yahoo Finance (no history returned).
  The Spirit-2021 meltdown therefore has **no price coverage** and is dropped from the
  real-tape event study — named honestly here and in ``docs/results.md``, never
  silently zero-filled. Each cached CSV lives under this study's OWN ``_cache/``.

* **Synthetic world.** A deterministic, seeded market + multi-stock panel (each stock
  ``= alpha + beta * market + idiosyncratic noise``) on which a set of pseudo-meltdown
  ``(ticker, date)`` events is sprinkled; each event stamps a TUNABLE planted abnormal
  drop ``edge`` on the implicated stock on the event day (and a small persistent bleed
  over the following sessions). ``edge = 0`` is the null world — event days statistically
  identical to the rest, and the market-model event-study machinery must NOT manufacture
  a significant CAR from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network, retry up to 4x)
runs once to build the cache and is never imported by the notebooks' offline cells;
``load_prices()`` reads the cached CSVs directly (no yfinance import).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

BENCH = "SPY"
BENCH_CACHE = os.path.join(CACHE_DIR, "amd_spy.csv")

# The equities we actually pull. SAVE (Spirit) is intentionally NOT here — it is fully
# delisted and returns no history from yfinance (see module docstring); the Spirit
# meltdown row below is kept in EVENTS for the record but drops out of the real test.
STOCK_TICKERS = ("LUV", "DAL", "UAL", "AAL", "BA")
STOCK_CACHE = {t: os.path.join(CACHE_DIR, f"amd_{t.lower()}.csv") for t in STOCK_TICKERS}

START = "2014-01-01"        # gives every 2016+ event a full estimation window
AS_OF = "2026-06-30"        # last complete calendar month at publication

# --------------------------------------------------------------------------- #
# Hardcoded table of famous airline / Boeing operational meltdowns, 2016 -> 2024.
# Each row: (event_date, ticker, label, source_note). ``event_date`` is the first
# calendar date the meltdown was unambiguously public news; the event-study code snaps
# it to the first NYSE session on/after it via searchsorted — a Friday-evening or
# weekend collapse rolls forward to the next open, a weekday collapse lands same-day.
# That snap is the study's single documented execution lag (see strategy.py). Each row
# is cross-referenced against contemporary wire coverage / regulator action; this is a
# hand-built calendar of front-page operational disasters, not an exhaustive register.
# --------------------------------------------------------------------------- #
EVENTS: list[tuple[str, str, str, str]] = [
    ("2016-08-08", "DAL", "Delta global IT / power-outage grounding (~2,300 flights cancelled Aug 8-10)",
     "Delta datacenter power failure grounded the fleet worldwide; multi-day cancellations, ~$150M pretax hit disclosed (Delta 8-K / contemporary AP & Reuters coverage, Aug 2016)."),
    ("2017-01-30", "DAL", "Delta systemwide IT outage / ground stop (~300 cancellations)",
     "Automation systems failed the evening of Jan 29, 2017, triggering a US ground stop and ~280+ cancellations into Jan 30 (Delta statement / Reuters, Jan 2017)."),
    ("2017-04-10", "UAL", "United Express 3411 forcible passenger removal (David Dao) viral PR crisis",
     "Passenger dragged bloodied off an overbooked UA Express flight Apr 9, 2017; cellphone video went globally viral Apr 10-11, sparking boycott calls (contemporary global press; United 8-K/CEO statements, Apr 2017)."),
    ("2019-03-11", "BA", "Boeing 737 MAX worldwide grounding wave (after Ethiopian 302, Mar 10)",
     "Second fatal 737 MAX crash (ET302) Mar 10, 2019; regulators worldwide grounded the MAX over Mar 11-13, halting deliveries (FAA emergency order; Boeing 8-K, Mar 2019)."),
    ("2021-08-02", "SAVE", "Spirit Airlines operational meltdown (mass cancellations Aug 1-8)",
     "Weather + IT + crew-scheduling cascade cancelled ~2,800 Spirit flights over Aug 1-8, 2021 (Spirit statements / AP & CNBC coverage). NOTE: SAVE delisted — no price coverage."),
    ("2021-10-11", "LUV", "Southwest October 2021 meltdown (~2,000 cancellations Oct 8-11)",
     "Southwest cancelled ~2,000 flights over the Oct 8-11 weekend citing ATC + weather + staffing; ~$75M Q4 impact disclosed (Southwest 8-K / Reuters, Oct 2021)."),
    ("2021-11-01", "AAL", "American Airlines Halloween meltdown (~1,900 cancellations Oct 29-Nov 1)",
     "American cancelled ~1,900 flights over the Oct 29-Nov 1 weekend on winds + staffing shortfalls (American statements / AP & CNBC, late Oct 2021)."),
    ("2022-12-27", "LUV", "Southwest Christmas 2022 collapse (~16,700 cancellations Dec 21-30)",
     "Crew-scheduling system collapse after winter storm Elliott cancelled ~16,700 Southwest flights Dec 21-30, 2022; ~$800M pretax hit, DOT probe (Southwest 8-K; DOT; contemporary coverage). First session after the Dec 26 market holiday is Dec 27."),
    ("2024-01-08", "BA", "Boeing 737 MAX 9 door-plug blowout (Alaska 1282, Jan 5) & grounding",
     "A cabin door-plug blew out mid-flight on Alaska 1282 (a 737 MAX 9) Fri Jan 5, 2024; FAA grounded the MAX 9 Jan 6 (NTSB; FAA emergency AD; Boeing 8-K, Jan 2024). First session after is Mon Jan 8."),
    ("2024-07-19", "DAL", "Delta / CrowdStrike outage meltdown (~7,000 cancellations Jul 19-24)",
     "A faulty CrowdStrike update crashed Windows systems worldwide Fri Jul 19, 2024; Delta alone cancelled ~7,000 flights through Jul 24, drawing a DOT investigation (Delta statements; DOT; contemporary coverage)."),
]


def events_table() -> pd.DataFrame:
    """The curated meltdown table as a frame: ``date`` (Timestamp), ``ticker``,
    ``label``, ``source``. Sorted by date."""
    df = pd.DataFrame(EVENTS, columns=["date", "ticker", "label", "source"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def coverable_events() -> pd.DataFrame:
    """The subset of the meltdown table whose implicated ticker is a fetchable equity
    (i.e. everything except the delisted-SAVE Spirit row) — the events the real-tape
    event study can actually score."""
    df = events_table()
    return df[df["ticker"].isin(STOCK_TICKERS)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _download_one(ticker: str, start: str, end: str, retries: int = 4) -> pd.DataFrame:
    """Download one ticker's total-return closes, retrying up to ``retries`` times."""
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                             progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df[["Close"]].dropna()
        except Exception as e:  # pragma: no cover - network path
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    if last_err is not None:  # pragma: no cover - network path
        raise last_err
    return pd.DataFrame(columns=["Close"])


def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download SPY + the five listed implicated equities; cache each as CSV.

    Network; run once. ``auto_adjust=True`` so closes fold in splits and dividends
    (total-return), and the event-study abnormal returns below are plain ``pct_change``
    on the cached close. SAVE is deliberately not fetched (delisted, no history)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    spy = _download_one(BENCH, start, end)
    spy.to_csv(BENCH_CACHE)
    for t in STOCK_TICKERS:
        _download_one(t, start, end).to_csv(STOCK_CACHE[t])


def have_real() -> bool:
    return os.path.exists(BENCH_CACHE) and all(
        os.path.exists(p) for p in STOCK_CACHE.values()
    )


def load_prices(asof: str = AS_OF) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Cached ``(spy_close, {ticker: close})`` series, sliced to ``[START, asof]``.

    Reads the CSVs directly — OFFLINE, no yfinance import."""
    spy = pd.read_csv(BENCH_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    spy = spy.loc[(spy.index >= START) & (spy.index <= asof)]
    stocks: dict[str, pd.Series] = {}
    for t, path in STOCK_CACHE.items():
        s = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["Close"]
        s = s.loc[(s.index >= START) & (s.index <= asof)]
        if not s.empty:
            stocks[t] = s
    return spy, stocks


# --------------------------------------------------------------------------- #
# Synthetic world — planted event drop on the implicated stock (positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(
    edge: float = 0.0,
    seed: int = 850,
    n_days: int = 2600,
    n_stocks: int = 6,
    n_events: int = 10,
    daily_vol: float = 0.011,
    idio_vol: float = 0.016,
    beta_lo: float = 0.9,
    beta_hi: float = 1.4,
    bleed_days: int = 21,
    start: str = "2014-01-02",
) -> tuple[pd.Series, dict[str, pd.Series], pd.DataFrame]:
    """A reproducible market + multi-stock tape with a TUNABLE planted meltdown drop.

    The market is an i.i.d.-normal random walk (std ``daily_vol``). Each stock ``i`` is
    a one-factor model ``r_i = alpha_i + beta_i * r_mkt + idio`` with ``beta_i`` spread
    over ``[beta_lo, beta_hi]`` (airlines are high-beta) and ``idio ~ N(0, idio_vol)``.
    ``n_events`` pseudo-meltdown ``(stock, date)`` pairs are scheduled well away from
    the edges; on each event day the implicated stock takes an extra ``-edge`` abnormal
    return, followed by a small persistent bleed (``-edge/4`` spread evenly across the
    next ``bleed_days`` sessions — the "reputational drift"). ``edge = 0`` is the null:
    every stock is a clean factor model and the event study must find nothing.

    Returns ``(mkt_close, {ticker: close}, events_df)``. Business-day index, span ~10
    years — far below the pandas ns-timestamp horizon. Tickers are ``SYN0..SYN{n-1}``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    tickers = [f"SYN{i}" for i in range(n_stocks)]

    mkt_ret = rng.normal(0.0003, daily_vol, n_days)
    betas = np.linspace(beta_lo, beta_hi, n_stocks)
    alphas = rng.normal(0.0, 0.0002, n_stocks)
    stock_ret = np.empty((n_stocks, n_days))
    for i in range(n_stocks):
        stock_ret[i] = alphas[i] + betas[i] * mkt_ret + rng.normal(0.0, idio_vol, n_days)

    # schedule events: pick n_events distinct dates well inside the tape, each on a
    # stock chosen round-robin so every name gets hit.
    margin = 300  # leave room for the estimation window before the earliest event
    pool = np.arange(margin, n_days - bleed_days - 5)
    locs = np.sort(rng.choice(pool, size=min(n_events, pool.size), replace=False))
    ev_stocks = [i % n_stocks for i in range(len(locs))]

    per_bleed = (edge / 4.0) / max(bleed_days, 1)
    ev_rows = []
    for loc, si in zip(locs, ev_stocks):
        stock_ret[si, loc] -= edge                       # the planted event-day drop
        for k in range(1, bleed_days + 1):               # small persistent bleed
            if loc + k < n_days:
                stock_ret[si, loc + k] -= per_bleed
        ev_rows.append({"date": idx[loc], "ticker": tickers[si]})

    mkt_close = pd.Series(100.0 * np.cumprod(1.0 + mkt_ret), index=idx)
    stocks = {
        tickers[i]: pd.Series(100.0 * np.cumprod(1.0 + stock_ret[i]), index=idx)
        for i in range(n_stocks)
    }
    events_df = pd.DataFrame(ev_rows).sort_values("date").reset_index(drop=True)
    return mkt_close, stocks, events_df
