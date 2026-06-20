"""Data layer for Study 334 (ARK-Innovation).

ARKK is the canonical *retail-hype / innovation-momentum* vehicle: a high-beta,
high-volatility growth ETF whose **fund flows famously chased its own performance**
(money piled in after the 2020 melt-up and bailed into the 2021-22 collapse). That
flow pattern is the heart of this study — it drives a wedge between the fund's
*time-weighted* return (what a buy-and-hold investor earned) and the
*dollar-weighted* return (what the average dollar that walked in the door actually
earned). The gap is the "behaviour penalty".

Two tapes, one shape (a tz-naive daily price frame plus, for the synthetic tape, a
monthly *flow* series):

- ``synthetic_hype_cycle`` — a *deterministic, offline* generator that plants a known
  hype cycle: a momentum/AR(1) price process whose investor flows chase trailing
  returns with a lag, so the dollar-weighted return is *engineered* to lag the
  time-weighted one by a known amount. A ``chase`` knob dials the flow-chasing from 0
  (flows independent of returns → no gap, the null) up (performance-chasing → a
  negative behaviour gap, the positive control). This is the study's null **and**
  positive control in one bottle.
- ``fetch_prices`` — the real Yahoo! daily tape (``yfinance``) for ARKK (and QQQ /
  XLK as innovation-beta benchmarks), cache-only by default so the test-suite and the
  reproducible core never touch the network.

No look-ahead is baked in here — that discipline lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (the hype cycle in a bottle)
# ---------------------------------------------------------------------------
def synthetic_hype_cycle(
    n_months: int = 180,
    chase: float = 0.0,
    bust: float = 0.0,
    grow: float = 0.012,
    boom: float = 0.045,
    monthly_vol: float = 0.05,
    start: str = "2014-10-31",
    seed: int = 334,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly price + investor-flow tape with a *known* behaviour gap.

    Two knobs make the null and the positive control:

    - ``bust`` (the *price shape*). At ``bust = 0`` the price is a **steady grower** with
      monthly drift ``grow`` — a fund that just goes up, where flat flows earn essentially
      the buy-and-hold return (the **null**: behaviour gap ≈ 0). Turn ``bust`` up and a
      deterministic **boom-bust hump** is mixed in (``bust * cos(pi*t/peak)``): the price
      ramps to a peak ~2/3 through the sample, then round-trips back down — the ARKK shape.
    - ``chase`` (the *flow shape*). At ``chase = 0`` flows are a flat baseline. Turn it up
      and net subscriptions chase the trailing 12-month return convexly, so a flood of
      money arrives **near the peak** — buying high — with a one-month lag.

    The **positive control** (``bust > 0`` *and* ``chase > 0``) plants a large gap between
    the *time-weighted* return (buy-and-hold) and the *dollar-weighted* return (what the
    average dollar earned): money piled in at the top rides the bust down. The **null**
    (``bust = 0``, ``chase = 0``) plants no gap. This is the study's null and positive
    control in one bottle::

        r_t   = grow + bust * cos(pi*t/peak) + eps_t,   eps_t ~ N(0, monthly_vol)
        flow_t = base_flow + chase * max(trailing_12m_return_{t-1}, 0)^2 + small noise

    Flows are clipped non-negative so the cashflow stream is *conventional* (one sign
    change) and the money-weighted IRR has a single well-defined root.

    Returns ``(frame, truth)`` where ``frame`` has a monthly DatetimeIndex and columns
    ``price`` (the NAV path, base 100), ``ret`` (monthly simple return) and ``flow`` (net
    investor flow that month, in dollars). A decorative *monthly* index is built with
    ``period_range`` (never a huge ``date_range``) so the synthetic span can never overflow
    a ns-timestamp.
    """
    rng = np.random.default_rng(seed)
    # period_range -> to_timestamp keeps us safely inside ns-timestamp range.
    months = pd.period_range(start=start, periods=n_months, freq="M").to_timestamp(how="end")
    months = months.normalize()

    t = np.arange(n_months, dtype=float)
    peak = 0.62 * n_months
    # Steady growth + an optional boom-bust hump (the cosine is +bust early, 0 at the peak,
    # then negative — a deterministic round-trip on top of the trend).
    drift = grow + bust * np.cos(np.pi * t / peak)
    log_ret = drift + rng.normal(0.0, monthly_vol, n_months)

    price = 100.0 * np.exp(np.cumsum(log_ret))
    simple_ret = np.empty(n_months)
    simple_ret[0] = price[0] / 100.0 - 1.0
    simple_ret[1:] = price[1:] / price[:-1] - 1.0

    # Flows chase the trailing *12-month* return — highest right at the peak, so chasers
    # pile in near the top — with a one-month lag (you see the run-up, then you buy).
    trail12 = pd.Series(simple_ret).rolling(12).sum().to_numpy()
    flow = np.zeros(n_months)
    base_flow = 1.0  # baseline net subscription (arbitrary dollar units)
    for i in range(n_months):
        signal = trail12[i - 1] if i >= 1 and np.isfinite(trail12[i - 1]) else 0.0
        # Convex (squared) response, clipped at zero: a year of strong gains brings a
        # disproportionate flood of money — concentrated near the peak — while net flows
        # stay non-negative, so the cashflow stream is "conventional" (one sign change) and
        # the money-weighted IRR has a single well-defined root.
        s = max(signal, 0.0)
        chased = chase * s * s
        flow[i] = base_flow + chased + rng.normal(0.0, 0.05)

    frame = pd.DataFrame(
        {"price": price, "ret": simple_ret, "flow": flow},
        index=pd.DatetimeIndex(months, name="date"),
    )
    truth = {
        "chase": chase,
        "bust": bust,
        "grow": grow,
        "monthly_vol": monthly_vol,
        "n_months": n_months,
        "seed": seed,
        "peak_month": int(round(peak)),
    }
    return frame, truth


def synthetic_daily(
    n_days: int = 2500,
    momentum: float = 0.04,
    daily_vol: float = 0.022,
    drift: float = 0.0003,
    start: str = "2014-10-31",
    seed: int = 334,
) -> pd.DataFrame:
    """A reproducible *daily* close tape with mild momentum — the offline stand-in for
    ARKK's daily series, used by the trend / mean-reversion tests.

    Log returns are an AR(1): ``r_t = drift + momentum * r_{t-1} + eps_t``. With
    ``momentum > 0`` a trend-following overlay can harvest the persistence; with
    ``momentum = 0`` it cannot (the null). Returns a single-column ``close`` frame.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)
    log_ret = np.empty(n_days)
    prev = 0.0
    for i in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        r = drift + momentum * prev + eps
        log_ret[i] = r
        prev = r
    close = 50.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"close": close}, index=pd.DatetimeIndex(sessions, name="date"))


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_daily.parquet")


def fetch_prices(
    ticker: str,
    start: str = "2014-10-31",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily total-return (auto-adjusted) close for ``ticker``; cache-only unless
    ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). ARKK launched 2014-10-31, so that is the natural start.
    Auto-adjusted closes fold dividends back in (ARKK pays some), so these are
    *total-return* proxies — labelled as such everywhere they are used.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        px = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        px = raw.rename(columns=str.lower)[["close"]]
        px.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        px.to_parquet(path)

    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    return px


def fetch_flows(
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame | None:
    """Optional real ARKK shares-outstanding / AUM proxy for dollar-weighting, if a
    cached file exists.

    Real ARKK share-count history is not on Yahoo!, so we keep this cache-only and
    optional: when no ``flows_ARKK.parquet`` is present we return ``None`` and the
    money-weighted analysis falls back to the synthetic flow tape (clearly bannered).
    Expected columns: ``shares`` (shares outstanding) on a monthly DatetimeIndex.
    """
    path = os.path.join(cache_dir, "flows_ARKK.parquet")
    if not os.path.exists(path):
        return None
    flows = pd.read_parquet(path)
    if flows.index.tz is not None:
        flows.index = flows.index.tz_localize(None)
    return flows


def fingerprint(frame: pd.DataFrame, col: str | None = None) -> str:
    """A short content fingerprint of a tape, for the as-of stamp."""
    if col is None:
        col = "close" if "close" in frame.columns else frame.columns[0]
    h = hashlib.sha1(np.ascontiguousarray(frame[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]
