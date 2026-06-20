"""Data layer for Study 346 (Multiple-Testing).

This is a *methodology* study, so the data layer supplies a **battery of effects** — a
panel of candidate "anomalies" each of which is either genuinely real or pure noise — in
two flavours, with the *same shape*: a (T x M) matrix of per-effect daily excess returns
plus a ``truth`` mask naming which columns carry a planted edge.

Two tapes, one shape (a (T x M) return matrix + a truth mask):

- ``synthetic_battery`` — a *deterministic, offline* generator. Two knobs decide the
  experiment: ``n_true`` (how many of the ``m_effects`` columns carry a real positive edge)
  and ``edge`` (its strength). At ``n_true = 0`` *every* effect is a pure null (i.i.d.
  noise, mean zero) — so every |*t*| > 2 "winner" is a false discovery, and counting them
  is the whole point (the **null**). At ``n_true > 0`` a known subset carries a planted
  edge a faithful procedure must recover while *not* drowning in false positives (the
  **positive control**). Returns share a common market factor so the effects are weakly
  correlated, the realistic case the corrections were designed for.

- ``load_real`` — a real battery of **calendar / seasonality effects** computed from a
  single daily price series (default SPY total-return) via the shared ``quantlab.data``
  loader, cache-first parquet with graceful offline fallback. Each "effect" is a
  long-the-effect-window / flat-otherwise excess-of-cash return series (e.g. *turn of
  month*, *Monday*, *day-of-week dummies*, *month-of-year dummies*, *FOMC week*, …) — the
  exact kind of folklore this desk tears down one study at a time, here run as a *family*
  so the multiple-testing reckoning is unavoidable.

One execution lag: every effect's return is the *next-day* return earned by a position
formed on a *calendar-known* rule. Calendar membership is known in advance, so no signal
shift is needed; the lag lives in the forward-return alignment, applied once.
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
# Synthetic battery — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_battery(
    n_days: int = 2520,
    m_effects: int = 100,
    n_true: int = 0,
    edge: float = 0.004,
    daily_vol: float = 0.012,
    market_load: float = 0.3,
    seed: int = 346,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible battery of ``m_effects`` candidate-anomaly daily return series.

    Each column is the excess-of-cash daily return of one "effect": a position that is on
    a calendar-determined fraction of the days and flat otherwise. The first ``n_true``
    columns (``e000``..) carry a genuine planted edge of ``edge`` per active day; the rest
    are pure nulls (mean-zero noise). All effects share a common market factor (scaled by
    ``market_load``) so they are weakly *correlated* — the regime FWER/FDR corrections are
    built for.

    - ``n_true = 0`` → every effect is a pure null. Any |*t*| > 2 "winner" is a false
      positive; the expected count is ``m_effects * P(|t|>2) ~ 0.045 * m_effects``. This is
      the **null** the demo lives on.
    - ``n_true > 0`` → a known subset (the truth mask) carries a real edge a faithful
      procedure must recover (high *power*) without flooding the report with the nulls'
      false positives. This is the **positive control**.

    Returns ``(returns, truth, meta)`` where ``returns`` is (n_days x m_effects) daily
    excess returns, ``truth`` is a boolean Series (True = a real planted effect), and
    ``meta`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)

    # A common market factor every effect loads on (so columns are weakly correlated).
    mkt = rng.normal(0.0, daily_vol, n_days)

    # Each effect is "on" a calendar-determined ~fraction of days. We assign each effect a
    # distinct active-day mask (deterministic, by a rotating period) so the design mirrors
    # real calendar dummies that fire on a slice of the tape.
    cols = [f"e{i:03d}" for i in range(m_effects)]
    ret = np.zeros((n_days, m_effects))
    active_frac = np.empty(m_effects)
    for j in range(m_effects):
        period = 5 + (j % 18)                     # cycle lengths 5..22 days
        phase = j % period
        on = ((np.arange(n_days) % period) == phase)
        active_frac[j] = on.mean()
        idio = rng.normal(0.0, daily_vol, n_days)
        base = market_load * mkt + idio
        # Planted edge on the first n_true effects, applied only on active days.
        mu = edge if j < n_true else 0.0
        ret[:, j] = np.where(on, base + mu, 0.0)

    # Decorative business-day labels via a SAFE range (no ns-Timestamp overflow).
    idx = pd.date_range("2005-01-03", periods=n_days, freq="B")
    df = pd.DataFrame(ret, index=pd.DatetimeIndex(idx, name="date"), columns=cols)
    truth = pd.Series(np.arange(m_effects) < n_true, index=cols, name="is_real")

    meta = {
        "n_days": n_days,
        "m_effects": m_effects,
        "n_true": n_true,
        "edge": edge,
        "daily_vol": daily_vol,
        "market_load": market_load,
        "seed": seed,
        "active_frac": active_frac.tolist(),
    }
    return df, truth, meta


# ---------------------------------------------------------------------------
# Real battery — calendar / seasonality effects on a single tape, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, ticker: str = "SPY") -> str:
    return os.path.join(cache_dir, f"battery_{ticker}.parquet")


# The named calendar effects we sweep on the real tape. Each name maps to a boolean
# "active day" mask built purely from the calendar (look-ahead-free). This is the family
# of folklore the desk usually tears down one at a time.
CALENDAR_EFFECTS = (
    ["mon", "tue", "wed", "thu", "fri"]
    + [f"mo_{m:02d}" for m in range(1, 13)]        # month-of-year dummies
    + ["turn_of_month", "first_half", "second_half",
       "first_5_days", "last_5_days", "mid_month",
       "quarter_end", "month_start", "monday_after_opex",
       "santa_rally", "sell_in_may", "summer", "q4",
       "jan", "dec", "odd_day", "even_day", "first_tue",
       "third_fri", "pre_weekend", "post_weekend"]
)


def _effect_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Boolean active-day masks for every named calendar effect (look-ahead-free)."""
    dow = index.dayofweek.to_numpy()              # 0=Mon..4=Fri
    dom = index.day.to_numpy()
    mon = index.month.to_numpy()
    # Days-from-month-end via the period boundary.
    days_in_month = index.days_in_month.to_numpy()
    dom_from_end = days_in_month - dom

    masks: dict[str, np.ndarray] = {}
    for d, name in enumerate(["mon", "tue", "wed", "thu", "fri"]):
        masks[name] = dow == d
    for m in range(1, 13):
        masks[f"mo_{m:02d}"] = mon == m
    masks["turn_of_month"] = (dom >= 28) | (dom <= 3)
    masks["first_half"] = dom <= 15
    masks["second_half"] = dom > 15
    masks["first_5_days"] = dom <= 5
    masks["last_5_days"] = dom_from_end <= 4
    masks["mid_month"] = (dom >= 13) & (dom <= 17)
    masks["quarter_end"] = np.isin(mon, [3, 6, 9, 12]) & (dom_from_end <= 4)
    masks["month_start"] = dom <= 2
    masks["monday_after_opex"] = (dow == 0) & (dom >= 15) & (dom <= 21)
    masks["santa_rally"] = (mon == 12) & (dom >= 24)
    masks["sell_in_may"] = np.isin(mon, [5, 6, 7, 8, 9, 10])
    masks["summer"] = np.isin(mon, [6, 7, 8])
    masks["q4"] = np.isin(mon, [10, 11, 12])
    masks["jan"] = mon == 1
    masks["dec"] = mon == 12
    masks["odd_day"] = (dom % 2) == 1
    masks["even_day"] = (dom % 2) == 0
    masks["first_tue"] = (dow == 1) & (dom <= 7)
    masks["third_fri"] = (dow == 4) & (dom >= 15) & (dom <= 21)
    masks["pre_weekend"] = dow == 4
    masks["post_weekend"] = dow == 0
    return masks


def load_real(
    ticker: str = "SPY",
    start: str = "1995-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Real battery of calendar effects → the SAME ``(returns, truth)`` shape.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract) so the reproducible core and tests stay deterministic.

    Each column is a *long-on-the-effect-day / flat-otherwise* excess-of-cash daily return
    series for one named calendar effect. One execution lag: the effect's day-membership is
    *calendar-known*, and the return it earns is the *same-day* close-to-close return on the
    days the rule is on — the lag is the close-of-(t-1) decision earning day-t's return,
    encoded once here. No survivorship issue (a single liquid ETF), so no opt-in guard.

    The real ``truth`` mask is all-False: on real data we do not *know* which calendar
    effects are real, which is the whole predicament — the demo shows how the corrections
    differ in what they let through, and the honest verdict is the methodology one.

    Returns ``(returns, truth)`` — an (T x M) excess-return matrix and the (all-False)
    truth Series.
    """
    cpath = _cache_path(cache_dir, ticker)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached tape at {cpath}. "
                f"Call load_real(fetch=True) once to populate (needs network)."
            )
        px = pd.read_parquet(cpath)["close"]
    else:
        px = _fetch_prices(ticker, start)
        os.makedirs(cache_dir, exist_ok=True)
        px.to_frame("close").to_parquet(cpath)

    px = px.dropna()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    # Drop the in-progress final month so a stamped run never includes a partial bar
    # (the month-of-year dummies would otherwise carry a half-month of January, etc.).
    last_full_month_end = (px.index.max().to_period("M").to_timestamp())
    px = px[px.index < last_full_month_end]
    ret = px.pct_change().dropna()

    masks = _effect_masks(ret.index)
    cols = {}
    for name in CALENDAR_EFFECTS:
        on = masks[name]
        # Excess-of-cash: on active days earn the day's return, else 0 (in cash). With
        # rf≈0 at daily scale this is the effect's contribution; all effects share the
        # same fully-invested-on-active-days convention, so they race like-for-like.
        cols[name] = pd.Series(np.where(on, ret.to_numpy(), 0.0), index=ret.index)
    df = pd.DataFrame(cols)
    truth = pd.Series(False, index=df.columns, name="is_real")
    return df, truth


def _fetch_prices(ticker: str, start: str) -> pd.Series:
    """Daily total-return closes via the shared loader, falling back to yfinance."""
    try:
        from quantlab import data as qld

        bars = qld.load(ticker, start=start, mode="total_return")
        s = bars["close"] if "close" in bars else bars.iloc[:, 0]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()
    except Exception:
        pass

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna()


def fingerprint(returns: pd.DataFrame) -> str:
    """A short content fingerprint of the battery, for the as-of stamp."""
    arr = np.ascontiguousarray(returns.to_numpy().ravel())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
