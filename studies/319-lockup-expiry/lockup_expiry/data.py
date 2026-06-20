"""Data layer for Study 319 (Lockup-Expiry).

Two tapes, one shape — a per-event *abnormal-return path* indexed by event-time
(trading days relative to the lock-up expiry, ``tau``), where the abnormal return is the
IPO stock's daily return minus its market benchmark's daily return (a synthetic control).

- ``synthetic_panel`` — a *deterministic, offline* generator. Each synthetic IPO gets a
  market-relative (abnormal) return path with i.i.d. noise; a ``sag_bps`` knob plants a
  one-shot negative abnormal return on the lock-up-expiry bar (``tau = 0``) and an optional
  ``pre_drift_bps`` slow leak in the run-up window. ``sag_bps = 0, pre_drift_bps = 0`` is a
  pure null (the abnormal path is noise around zero). This is the study's positive control
  *and* its null in one knob.
- ``fetch_event_panel`` — the real loader. A hardcoded basket of recent US IPOs with their
  listing dates; for each it pulls the stock and a market benchmark (SPY) from yfinance,
  computes daily abnormal returns (stock − benchmark), and aligns them in event-time around
  the lock-up-expiry bar (listing + ``LOCKUP_TD`` trading days). Cache-first: ``fetch=False``
  reads parquet from ``_cache/`` and never touches the network.

Design decisions (house style — state them as decisions):
- **Lock-up convention.** The classic agreement is 180 *calendar* days. We approximate it
  as ``LOCKUP_TD = 125`` trading days from the first close (≈ 180 calendar days). The
  expiry date is *calendar-known in advance* (it is in the S-1), so the event is
  **anticipatable** — a key reason any effect is likely small/arbitraged.
- **Abnormal return = stock − SPY.** A market-model alpha/beta would be cleaner, but a
  raw market-adjusted (beta = 1) return is the standard simple event-study control and
  avoids fitting per-event betas on short post-IPO histories. Stated as a limitation.
- **No look-ahead.** The expiry bar is fixed by the calendar; event-time alignment uses
  only realised prices. Costs and execution lag live in ``strategy.py``.
- **Survivorship.** The basket only contains tickers still queryable on yfinance — named
  on the Signal axis, not buried. A current-membership basket can bias the sag either way.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# 180 calendar days ≈ 125 trading days. The lock-up-expiry bar sits at this offset from
# the first close.
LOCKUP_TD = 125

# Event-time window half-width (trading days each side of the expiry bar).
WINDOW = 20

# The benchmark used as the synthetic control (market-adjusted abnormal returns).
BENCHMARK = "SPY"


# ---------------------------------------------------------------------------
# Hardcoded basket of recent US IPOs (listing date + first-close date).
# Only tickers still queryable on yfinance (survivorship bias NAMED).
# Listing dates from public records / exchange notices.
# ---------------------------------------------------------------------------
IPO_BASKET: list[dict] = [
    # 2019 wave
    {"ticker": "LYFT", "listing": "2019-03-29"},
    {"ticker": "UBER", "listing": "2019-05-10"},
    {"ticker": "ZM",   "listing": "2019-04-18"},   # Zoom Video
    {"ticker": "PINS", "listing": "2019-04-18"},   # Pinterest
    {"ticker": "CRWD", "listing": "2019-06-12"},   # CrowdStrike
    {"ticker": "DDOG", "listing": "2019-09-19"},   # Datadog
    {"ticker": "WORK", "listing": "2019-06-20"},   # Slack (direct listing; delisted 2021)
    {"ticker": "FVRR", "listing": "2019-06-13"},   # Fiverr
    # 2020 wave
    {"ticker": "SNOW", "listing": "2020-09-16"},   # Snowflake
    {"ticker": "DASH", "listing": "2020-12-09"},   # DoorDash
    {"ticker": "ABNB", "listing": "2020-12-10"},   # Airbnb
    {"ticker": "U",    "listing": "2020-09-18"},   # Unity
    {"ticker": "PLTR", "listing": "2020-09-30"},   # Palantir (direct listing)
    {"ticker": "ZI",   "listing": "2020-06-04"},   # ZoomInfo
    {"ticker": "LMND", "listing": "2020-07-02"},   # Lemonade
    {"ticker": "BIGC", "listing": "2020-08-05"},   # BigCommerce
    # 2021 wave
    {"ticker": "COIN", "listing": "2021-04-14"},   # Coinbase (direct listing)
    {"ticker": "RBLX", "listing": "2021-03-10"},   # Roblox (direct listing)
    {"ticker": "HOOD", "listing": "2021-07-29"},   # Robinhood
    {"ticker": "RIVN", "listing": "2021-11-10"},   # Rivian
    {"ticker": "SOFI", "listing": "2021-06-01"},   # SoFi (via SPAC)
    {"ticker": "AFRM", "listing": "2021-01-13"},   # Affirm
    {"ticker": "BROS", "listing": "2021-09-15"},   # Dutch Bros
    {"ticker": "DNUT", "listing": "2021-07-01"},   # Krispy Kreme
    {"ticker": "PATH", "listing": "2021-04-21"},   # UiPath
    {"ticker": "GLBE", "listing": "2021-05-12"},   # Global-e
    # 2023-2024 wave
    {"ticker": "ARM",  "listing": "2023-09-14"},   # Arm Holdings
    {"ticker": "CART", "listing": "2023-09-19"},   # Instacart (Maplebear)
    {"ticker": "KVYO", "listing": "2023-09-20"},   # Klaviyo
    {"ticker": "BIRK", "listing": "2023-10-11"},   # Birkenstock
    {"ticker": "RDDT", "listing": "2024-03-21"},   # Reddit
    {"ticker": "ALAB", "listing": "2024-03-20"},   # Astera Labs
]

# Tickers that may have delisted / renamed — yfinance may still return history; skip if not.
DELISTED = {"WORK"}


def basket_frame() -> pd.DataFrame:
    """The hardcoded IPO basket as a DataFrame (ticker, listing date)."""
    rows = [
        {"ticker": r["ticker"], "listing": pd.Timestamp(r["listing"])}
        for r in IPO_BASKET
    ]
    return pd.DataFrame(rows).sort_values("listing").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_events: int = 30,
    sag_bps: float = -150.0,
    pre_drift_bps: float = 0.0,
    window: int = WINDOW,
    abn_vol_ann: float = 0.45,
    seed: int = 319,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible panel of event-time *abnormal-return* paths around a lock-up expiry.

    Each event is one IPO. The abnormal return (stock − market) on each event-time bar
    ``tau`` (from ``-window`` to ``+window``) is i.i.d. normal noise, plus:

    - ``sag_bps`` planted on the expiry bar (``tau = 0``). Negative = a one-shot price sag
      when insider supply unlocks (the believers' story).
    - ``pre_drift_bps`` added to each bar in the run-up window (``tau < 0``) — a slow leak
      as the market front-runs the known unlock.

    ``sag_bps = 0`` and ``pre_drift_bps = 0`` gives a pure null: abnormal returns are noise
    around zero, so an honest harness must *fail* to find a sag here.

    Returns
    -------
    panel : DataFrame
        Long format, one row per (event, tau). Columns: event, tau, abn_ret (fractional).
    truth : dict
        The planted parameters.
    """
    rng = np.random.default_rng(seed)
    sig_d = abn_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    taus = np.arange(-window, window + 1)

    rows = []
    for ev in range(n_events):
        abn = rng.normal(0.0, sig_d, taus.size)
        # plant the one-shot sag on the expiry bar
        abn[taus == 0] += sag_bps * 1e-4
        # plant the optional pre-expiry leak
        abn[taus < 0] += pre_drift_bps * 1e-4
        for k, tau in enumerate(taus):
            rows.append({"event": f"SYN{ev:03d}", "tau": int(tau), "abn_ret": float(abn[k])})

    panel = pd.DataFrame(rows)
    truth = {
        "n_events": n_events,
        "sag_bps": sag_bps,
        "pre_drift_bps": pre_drift_bps,
        "window": window,
        "abn_vol_ann": abn_vol_ann,
        "seed": seed,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real loader — yfinance stock & benchmark, cache-first
# ---------------------------------------------------------------------------
def _price_cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_319_{safe}_1d.parquet")


def _load_close(ticker: str, cache_dir: str, fetch: bool, start: str) -> pd.Series | None:
    """Daily adjusted close for one ticker; cache-first. None if unavailable."""
    path = _price_cache_path(ticker, cache_dir)
    if fetch:
        import yfinance as yf  # lazy: only when going to the network

        try:
            raw = yf.download(
                ticker, start=start, interval="1d", auto_adjust=True, progress=False
            )
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


def fetch_event_panel(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    window: int = WINDOW,
    lockup_td: int = LOCKUP_TD,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """Build the real event-time abnormal-return panel for the hardcoded IPO basket.

    For each IPO: pull stock + benchmark daily closes, compute daily simple returns,
    abnormal return = stock_ret − benchmark_ret (a market-adjusted synthetic control),
    locate the lock-up-expiry bar (``lockup_td`` trading days after the first close), and
    extract the ``[-window, +window]`` event-time slice.

    With ``fetch=False`` (default) this is cache-only — no network — so the test-suite and
    the reproducible core run fully offline. ``fetch=True`` populates the parquet cache.

    Returns
    -------
    panel : DataFrame
        Long format: event (ticker), tau, abn_ret. Events with insufficient history are
        skipped (the basket is large enough to retain meaningful n).
    """
    os.makedirs(cache_dir, exist_ok=True)
    basket = basket_frame()

    # Benchmark once (shared across events).
    bench_start = (basket["listing"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    bench = _load_close(benchmark, cache_dir, fetch, bench_start)
    if bench is None:
        raise RuntimeError(f"benchmark {benchmark} has no cached close data")
    bench_ret = bench.pct_change()

    rows = []
    for _, r in basket.iterrows():
        tkr = r["ticker"]
        start = (r["listing"] - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        s = _load_close(tkr, cache_dir, fetch, start)
        if s is None or len(s) < lockup_td + window + 5:
            continue
        stock_ret = s.pct_change()

        # First close = first available bar on/after the listing date.
        listing = pd.Timestamp(r["listing"])
        pos0 = int(np.searchsorted(s.index, listing))
        if pos0 >= len(s):
            continue
        expiry_pos = pos0 + lockup_td
        if expiry_pos - window < 1 or expiry_pos + window >= len(s):
            continue

        # Align benchmark returns to the stock's dates, compute abnormal returns.
        aligned_bench = bench_ret.reindex(s.index)
        abn = (stock_ret - aligned_bench)

        sl = abn.iloc[expiry_pos - window: expiry_pos + window + 1].to_numpy(dtype=float)
        taus = np.arange(-window, window + 1)
        if sl.size != taus.size or not np.isfinite(sl).all():
            continue
        for tau, a in zip(taus, sl):
            rows.append({"event": tkr, "tau": int(tau), "abn_ret": float(a)})

    panel = pd.DataFrame(rows)
    return panel


def fingerprint(panel: pd.DataFrame) -> str:
    """Short content fingerprint of an abnormal-return panel (for the as-of stamp)."""
    arr = np.ascontiguousarray(
        panel.sort_values(["event", "tau"])["abn_ret"].to_numpy(dtype=float)
    )
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
