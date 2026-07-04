"""Strategy / stats layer for Study 630 — Late Filers (Form NT).

The tested rule: on the day a company files an NT 10-K / NT 10-Q ("we cannot file on time"),
SHORT the stock at the next day's close and hold for 60 trading days; measure the
market-adjusted (vs SPY) cumulative abnormal return (CAR).

Inference toolkit (the desk's shared discipline):

* **Event CARs** — per event, the sum of daily (stock − SPY) returns over the 60 trading days
  *after* the entry close. Exactly ONE execution lag: the filing lands on day ``d0`` (often
  after hours); we trade at the close of ``d0 + 1`` and the first measured return is the next
  day's. The un-tradable announcement move (``d0-1`` close → entry close) is reported
  separately and never counted in the tradable CAR.
* **Calendar-time portfolio (primary test)** — every day, the equal-weight mean market-adjusted
  return of all names inside their post-NT window; HAC/Newey-West *t* on that daily series.
  Event windows overlap and cluster around filing deadlines, so the naive cross-sectional *t*
  over-counts; the calendar-time *t* is the honest one (Fama 1998, Mitchell-Stafford 2000).
* **Matched self-control** — pseudo-events on the SAME tickers 252 trading days *before* each
  real NT (skipping pseudo-dates that collide with another NT by the same firm); Welch *t* of
  event CARs vs pseudo CARs answers "or do these firms just always drift down?".
* **Random-dates placebo** — same tickers, uniform random dates, mean CAR **averaged over
  ≥ 20 seeds** (single-seed baselines are banned by the house rules).
* **Third axis** — repeat offenders (a prior NT within ~15 months) vs first offenses, Welch *t*.

Pure numpy + pandas; offline once the caches exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 60          # trading days measured after entry
LAG = 1              # entry at the close of d0 + LAG (the one documented execution lag)
MIN_COV = 0.80       # required share of non-missing daily returns inside the window
MIN_PRICE = 1.0      # entry-price floor ($): sub-$1 pinks are unshortable AND carry the worst
                     # Yahoo adjustment glitches; stated in the open, no silent caps
MAX_DAILY = 1.0      # glitch guard: drop events whose window contains a |daily move| > 100%
                     # (reverse-split artifacts / undtradeable penny pumps corrupt an arithmetic CAR)


# --------------------------------------------------------------------------------------- #
# Small stats helpers
# --------------------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 10) -> float:
    """Newey-West (HAC) t-statistic of mean(x) != 0 with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else np.nan


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch t of mean(a) − mean(b); returns (t, mean_a, mean_b)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    return float(t), float(a.mean()), float(b.mean())


def _t_mean(x: np.ndarray) -> tuple[float, float, int]:
    """(plain cross-sectional t, mean, n) — reported but NOT the primary test."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    return float(t), float(x.mean()), n


# --------------------------------------------------------------------------------------- #
# Event CARs
# --------------------------------------------------------------------------------------- #
def build_event_cars(events: pd.DataFrame, prices: pd.DataFrame,
                     window: int = WINDOW, lag: int = LAG,
                     min_cov: float = MIN_COV, spy: str = "SPY",
                     min_price: float = MIN_PRICE,
                     max_daily: float = MAX_DAILY) -> pd.DataFrame:
    """Per-event market-adjusted CAR over the ``window`` trading days after the entry close.

    Returns one row per usable event: ``ticker, form, date_filed, repeat, car, reaction, pos``
    with ``car``/``reaction`` in decimal (0.01 = +1%). Screens (all stated, none silent):
    ticker must have a price at entry >= ``min_price`` (sub-$1 pinks are unshortable and carry
    the worst Yahoo adjustment glitches), >= ``min_cov`` coverage inside the window, and no
    |daily stock move| > ``max_daily`` inside it (reverse-split artifacts / penny pumps make an
    arithmetic CAR meaningless). Dropped counts are reported by the caller.
    """
    rets = prices.pct_change()
    mkt = rets[spy].to_numpy()
    dates = prices.index
    out = []
    for ev in events.itertuples(index=False):
        tk = ev.ticker
        if tk not in rets.columns or tk == spy:
            continue
        d0 = int(dates.searchsorted(pd.Timestamp(ev.date_filed)))
        entry = d0 + lag
        end = entry + window
        if d0 >= len(dates) or end >= len(dates):
            continue
        r = rets[tk].to_numpy()
        px_entry = prices[tk].to_numpy()[entry]
        if not np.isfinite(px_entry) or px_entry < min_price:
            continue
        w = r[entry + 1: end + 1] - mkt[entry + 1: end + 1]
        ok = np.isfinite(w)
        if ok.sum() < min_cov * window:
            continue
        raw = r[entry + 1: end + 1]
        if np.nanmax(np.abs(raw[np.isfinite(raw)])) > max_daily:
            continue
        car = float(np.nansum(w[ok]))
        # announcement move: close before the filing day -> entry close (NOT tradable)
        ra = r[d0: entry + 1] - mkt[d0: entry + 1]
        raw_a = r[d0: entry + 1]
        ok_a = np.isfinite(raw_a)
        if ok_a.any() and np.max(np.abs(raw_a[ok_a])) <= max_daily:
            reaction = float(np.nansum(ra[np.isfinite(ra)]))
        else:
            reaction = np.nan
        out.append({"ticker": tk, "form": ev.form, "date_filed": pd.Timestamp(ev.date_filed),
                    "repeat": bool(ev.repeat), "car": car, "reaction": reaction, "pos": entry})
    return pd.DataFrame(out)


def cross_sectional_stats(cars: pd.DataFrame) -> dict:
    t, m, n = _t_mean(cars["car"].to_numpy())
    tr, mr, _ = _t_mean(cars["reaction"].to_numpy())
    return {"n": n, "car_bps": m * 1e4, "t_naive": t,
            "reaction_bps": mr * 1e4, "t_reaction": tr}


# --------------------------------------------------------------------------------------- #
# Calendar-time portfolio (the primary test)
# --------------------------------------------------------------------------------------- #
def calendar_time_portfolio(cars: pd.DataFrame, prices: pd.DataFrame,
                            window: int = WINDOW, spy: str = "SPY",
                            hac_lags: int = 10) -> dict:
    """EW daily market-adjusted return of all names inside their post-NT window; HAC t.

    Uses the events that survived :func:`build_event_cars` (same entry positions), so the
    two tests see the same panel. Days with no active event are skipped (not zero-filled).
    """
    rets = prices.pct_change()
    mkt = rets[spy].to_numpy()
    n_days = len(prices.index)
    sums = np.zeros(n_days)
    cnts = np.zeros(n_days)
    for ev in cars.itertuples(index=False):
        r = rets[ev.ticker].to_numpy()
        lo, hi = ev.pos + 1, min(ev.pos + window, n_days - 1)
        seg = r[lo: hi + 1] - mkt[lo: hi + 1]
        ok = np.isfinite(seg)
        sums[lo: hi + 1][ok] += seg[ok]
        cnts[lo: hi + 1][ok] += 1
    active = cnts > 0
    daily = sums[active] / cnts[active]
    t = hac_t(daily, lags=hac_lags)
    mean_bps = float(daily.mean() * 1e4)
    ann = float((1 + daily.mean()) ** 252 - 1) * 100
    return {"n_days": int(active.sum()), "mean_bps_day": mean_bps, "hac_t": t,
            "ann_pct": ann, "avg_names": float(cnts[active].mean())}


# --------------------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------------------- #
def pseudo_events(events: pd.DataFrame, prices: pd.DataFrame,
                  shift_td: int = 252, collision_days: int = 120) -> pd.DataFrame:
    """Matched self-control: same ticker, ``shift_td`` trading days BEFORE each real NT.

    Pseudo-dates that fall within ``collision_days`` calendar days of any other NT by the
    same firm are dropped, so the control window is genuinely NT-free for that name.
    """
    dates = prices.index
    by_cik: dict = {}
    for ev in events.itertuples(index=False):
        by_cik.setdefault(ev.cik, []).append(pd.Timestamp(ev.date_filed))
    rows = []
    for ev in events.itertuples(index=False):
        d0 = int(dates.searchsorted(pd.Timestamp(ev.date_filed)))
        p = d0 - shift_td
        if p < 1:
            continue
        pdate = dates[p]
        clash = any(abs((pdate - other).days) <= collision_days
                    for other in by_cik[ev.cik] if other != pd.Timestamp(ev.date_filed))
        if clash:
            continue
        rows.append({"cik": ev.cik, "ticker": ev.ticker, "company": ev.company,
                     "form": ev.form, "date_filed": pdate, "repeat": False})
    return pd.DataFrame(rows)


def random_dates_placebo(events: pd.DataFrame, prices: pd.DataFrame,
                         n_seeds: int = 25, window: int = WINDOW, lag: int = LAG,
                         base_seed: int = 630) -> dict:
    """Same tickers, uniform random dates; mean CAR averaged over ``n_seeds`` seeds (>= 20).

    The house rule: any RANDOM baseline is averaged over >= 20 seeds. Reports the across-seed
    mean and std of the mean CAR, and the count of seeds whose mean CAR is below the observed.
    """
    dates = prices.index
    last_ok = len(dates) - (lag + window + 2)
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        fake = events.copy()
        fake["date_filed"] = dates[rng.integers(260, last_ok, size=len(events))]
        cars = build_event_cars(fake, prices, window=window, lag=lag)
        if len(cars):
            means.append(cars["car"].mean())
    means = np.array(means, dtype=float)
    return {"n_seeds": len(means), "mean_bps": float(means.mean() * 1e4),
            "std_bps": float(means.std(ddof=1) * 1e4)}


# --------------------------------------------------------------------------------------- #
# Costs — the short side pays the spread twice AND the borrow
# --------------------------------------------------------------------------------------- #
def short_net(car_bps: float, cost_bps_oneway: float, borrow_ann_pct: float,
              window: int = WINDOW) -> float:
    """Net bps captured by the short: −CAR − 2×one-way cost − borrow held for the window."""
    borrow_bps = borrow_ann_pct * 100.0 * (window / 252.0)
    return -car_bps - 2.0 * cost_bps_oneway - borrow_bps
