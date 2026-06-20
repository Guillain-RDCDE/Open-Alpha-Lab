"""Data layer for Study 313 (Geopolitical-Shock).

Three things live here:

1. ``SHOCK_TABLE`` — a *curated, hardcoded* table of major geopolitical shocks
   (wars, invasions, terror attacks, assassinations, missile crises). The Caldara-
   Iacoviello Geopolitical Risk (GPR) index would be the data-driven way to pick
   events, but it is network-blocked here, so we use a hand-built table of the dates
   any reasonable person would call "the market woke up to a war/attack". Each row
   carries the *trade date* — the first NYSE session on which the news could be acted
   on (an attack on a Saturday is mapped to the following Monday).

2. ``synthetic_daily`` — a *deterministic, offline* daily SPY-like tape with a
   planted "shock response" knob. ``shock_drift`` plants a clean post-event drift on a
   set of planted event dates (the positive control); ``shock_drift=0`` plants nothing
   (the null). A test can then assert the event-study machinery recovers the planted
   drift only when it is there.

3. ``load_real`` — a cache-first loader for the real SPY daily tape (via the shared
   ``_cache`` parquet, or ``quantlab.data`` / ``yfinance`` on an explicit cache miss),
   with a graceful fallback so the offline core and tests never touch the network.

One execution-lag convention runs through the whole study: a shock that becomes
actionable at the **close of the trade date** earns the return of the *next* session
(``t+1``), applied with a single ``shift`` in ``strategy.py``. The dates in
``SHOCK_TABLE`` are already the first tradable session, so no second lag is hidden here.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# The curated shock table — the GPR index is network-blocked, so we hardcode.
# ---------------------------------------------------------------------------
# Each entry: (trade_date, label). ``trade_date`` is the first NYSE session on which
# the shock could be acted on at the close (weekend/holiday events roll forward to the
# next open session). These are the events a believer in "geopolitics moves markets"
# would point at — the strongest version of the claim, not strawmen.
_SHOCKS: list[tuple[str, str]] = [
    ("1990-08-02", "Iraq invades Kuwait"),
    ("1991-01-17", "Gulf War air campaign begins"),
    ("1993-02-26", "World Trade Center bombing"),
    ("1998-08-07", "US embassy bombings (Kenya/Tanzania)"),
    ("1998-08-20", "US cruise-missile strikes (Sudan/Afghanistan)"),
    ("2001-09-17", "9/11 attacks (first session after reopen)"),
    ("2001-10-08", "US invasion of Afghanistan begins"),
    ("2002-10-14", "DC sniper / pre-Iraq tension"),
    ("2003-03-20", "US invasion of Iraq begins"),
    ("2004-03-11", "Madrid train bombings"),
    ("2005-07-07", "London 7/7 bombings"),
    ("2008-08-08", "Russia-Georgia war"),
    ("2011-03-21", "NATO intervention in Libya"),
    ("2011-05-02", "Bin Laden killed"),
    ("2013-08-30", "Syria chemical-weapons crisis"),
    ("2014-03-03", "Russia annexes Crimea (markets react)"),
    ("2014-07-17", "MH17 shot down over Ukraine"),
    ("2015-11-16", "Paris attacks (first session after)"),
    ("2017-04-07", "US strikes Syria (Shayrat)"),
    ("2017-08-09", "North Korea 'fire and fury' missile crisis"),
    ("2018-04-16", "US-led strikes on Syria"),
    ("2019-09-16", "Saudi Aramco drone attack"),
    ("2020-01-03", "US kills Soleimani (Iran tension)"),
    ("2021-08-16", "Fall of Kabul"),
    ("2022-02-24", "Russia invades Ukraine"),
    ("2023-10-09", "Israel-Hamas war (first session after Oct 7)"),
    ("2024-04-15", "Iran-Israel direct strikes"),
    ("2025-06-13", "Israel-Iran escalation"),
]


def shock_table() -> pd.DataFrame:
    """The curated shock table as a frame: ``trade_date`` (Timestamp) + ``label``."""
    df = pd.DataFrame(_SHOCKS, columns=["trade_date", "label"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4000,
    shock_drift: float = 0.0,
    n_shocks: int = 24,
    daily_vol: float = 0.011,
    drift_days: int = 5,
    start: str = "2000-01-03",
    seed: int = 313,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible daily SPY-like tape with a *known* post-shock drift planted.

    A random walk in log returns (i.i.d. normal of std ``daily_vol``) on top of which
    ``n_shocks`` event dates are sprinkled. On each event date the next ``drift_days``
    sessions get an extra ``shock_drift / drift_days`` of log return each — so the total
    planted abnormal return over the ``drift_days`` window after an event is
    ``shock_drift`` (in log terms).

    - ``shock_drift = 0`` → the null: events carry no information; the event study must
      find nothing distinguishable from the placebo distribution.
    - ``shock_drift > 0`` → a planted recovery/relief rally the machinery must recover.
    - ``shock_drift < 0`` → a planted post-shock sell-off.

    Returns ``(bars, events, truth)``. ``events`` has the same shape as ``shock_table``
    (``trade_date`` + ``label``), pointing at the *planted* event sessions, so the same
    event-study code runs on synthetic and real tapes unchanged.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    log_ret = rng.normal(0.0, daily_vol, n_days)

    # Plant events well away from the edges so the +/- window always fits.
    margin = 30
    pool = np.arange(margin, n_days - margin)
    event_locs = np.sort(rng.choice(pool, size=min(n_shocks, pool.size), replace=False))

    per_day = shock_drift / max(drift_days, 1)
    for loc in event_locs:
        for k in range(1, drift_days + 1):
            if loc + k < n_days:
                log_ret[loc + k] += per_day

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(40_000_000, 120_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    events = pd.DataFrame(
        {
            "trade_date": bars.index[event_locs],
            "label": [f"planted shock {i+1}" for i in range(len(event_locs))],
        }
    )
    truth = {
        "shock_drift": shock_drift,
        "drift_days": drift_days,
        "n_shocks": int(len(event_locs)),
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, events, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first, graceful fallback
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1990-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    Resolution order:
      1. this study's own ``_cache/bars_<ticker>_daily.parquet`` (gitignored, populated
         once with ``fetch=True``);
      2. the shared ``quantlab.data`` cache, if importable and it has the ticker cached;
      3. (only with ``fetch=True``) the network via ``yfinance``, then cached locally.

    Raises ``FileNotFoundError`` when nothing is cached and ``fetch`` is False, so the
    offline core and tests never silently reach for the network.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
        return _clean(bars)

    # 2) shared quantlab cache, best-effort.
    try:
        import quantlab.data as qd  # noqa

        shared = qd.CACHE_DIR / f"{ticker}_split_only.parquet"
        if shared.exists():
            raw = pd.read_parquet(shared)
            cols = {c.lower(): c for c in raw.columns}
            keep = [cols[c] for c in ("open", "high", "low", "close", "volume") if c in cols]
            bars = raw[keep].copy()
            bars.columns = [c.lower() for c in bars.columns]
            bars.index.name = "date"
            return _clean(bars)
    except Exception:
        pass

    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        ticker, start=start, interval="1d", auto_adjust=True, progress=False
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    bars.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    bars.to_parquet(path)
    return _clean(bars)


def _clean(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
