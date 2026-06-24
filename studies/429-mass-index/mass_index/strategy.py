"""Strategy + honest arbiters for Study 429 — Mass Index (Donald Dorsey, 1992).

Dorsey's **Mass Index** is built to detect a *reversal* from the **range** of price,
not its direction. The recipe (the "Reversal Bulge"):

    range_t   = high_t - low_t                          (the day's high-low range)
    ema1_t    = EMA_9(range)                            (a 9-day EMA of the range)
    ema2_t    = EMA_9(ema1)                             (a 9-day EMA of *that*)
    ratio_t   = ema1_t / ema2_t                         (single/double EMA ratio)
    MI_t      = sum of the last 25 ratios               (the Mass Index)

The folk **reversal bulge**: when the Mass Index rises **above 27** and then falls
back **below 26.5**, a trend reversal is supposedly imminent. The believer's recipe is
"the bulge fires → a reversal is coming → fade the prevailing trend / brace for a turn."

We test that two complementary ways:

* **Event study.** Treat each completed bulge (the 27→26.5 trigger) as an event and
  measure the **forward H-day return** after it, against the **unconditional base rate**
  over the same tape. A one-sample *t* of the post-bulge returns vs zero, a *t* of the
  bulge sample vs the non-bulge sample, and a **label-shuffle placebo** (draw the same
  number of random trigger dates, recompute the forward return) — the honest null for a
  rare-event signal. A *reversal* claim is directional, so we also test the **trend-aware
  fade**: forward return *signed against the prevailing trend* (a real reversal makes
  "fade the trend after a bulge" pay).

* **Timing rule.** Turn the bulge into a daily **fade** rule on SPY: when a bulge has
  just fired, take the trend-fading position for H days; otherwise hold. Race it **NET**
  of one-way costs × NAV turnover, **excess-of-cash vs excess-of-cash**, against
  buy-and-hold, with a one-day execution lag.

A deterministic synthetic positive control (`data.synthetic_panel`) plants a real
post-bulge reversal so the harness proves it *can* bank the effect when it exists.

No look-ahead: the index is formed on ranges up to bar *t*; the bulge trigger is known at
the close of *t*, the position/forward window starts at *t+1* (one shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0
HORIZONS = (5, 10, 20, 40)          # forward-return horizons (trading days)


# --------------------------------------------------------------------------- #
# The Mass Index
# --------------------------------------------------------------------------- #
def mass_index(bars: pd.DataFrame, ema: int = 9, sum_window: int = 25) -> pd.Series:
    """Dorsey's Mass Index: rolling sum of the single/double EMA-of-range ratio.

    ``range = high - low``; ``ema1 = EMA_ema(range)``; ``ema2 = EMA_ema(ema1)``;
    ``MI = rolling_sum(ema1 / ema2, sum_window)``. Indexed identically to ``bars``;
    the first ``ema + sum_window`` bars are warm-up (NaN until the sum window fills).
    """
    rng = (bars["high"] - bars["low"]).astype(float)
    ema1 = rng.ewm(span=ema, adjust=False).mean()
    ema2 = ema1.ewm(span=ema, adjust=False).mean()
    ratio = ema1 / ema2.replace(0.0, np.nan)
    mi = ratio.rolling(sum_window, min_periods=sum_window).sum()
    return mi.rename("mass_index")


def bulge_signal(bars: pd.DataFrame, hi: float = 27.0, lo: float = 26.5,
                 ema: int = 9, sum_window: int = 25) -> pd.Series:
    """Boolean series: True on the bar where a **reversal bulge** completes.

    The bulge completes the first day the Mass Index, having previously risen **above**
    ``hi`` (27), falls back **below** ``lo`` (26.5). This is the discrete trigger Dorsey
    describes; it is known at the close of that bar. Returns a 0/1-flag Series aligned to
    ``bars`` (True only on the completing bar, not for the whole bulge).
    """
    mi = mass_index(bars, ema=ema, sum_window=sum_window)
    above = mi > hi
    # "has been above hi at some point since the last completion" — track the armed state
    armed = False
    flag = np.zeros(len(mi), dtype=bool)
    vals = mi.to_numpy()
    for i in range(len(vals)):
        v = vals[i]
        if not np.isfinite(v):
            continue
        if v > hi:
            armed = True
        elif armed and v < lo:
            flag[i] = True
            armed = False
    return pd.Series(flag, index=bars.index, name="bulge")


# --------------------------------------------------------------------------- #
# Trend context (a reversal is defined relative to a prevailing trend)
# --------------------------------------------------------------------------- #
def trend_sign(bars: pd.DataFrame, lookback: int = 25) -> pd.Series:
    """+1 if the close is above its ``lookback``-day SMA (up-trend), else −1.

    A "reversal" only means something against a prevailing trend; the fade trade after a
    bulge is "bet the trend flips," i.e. take ``-trend_sign`` exposure.
    """
    sma = bars["close"].rolling(lookback, min_periods=lookback).mean()
    s = np.where(bars["close"] > sma, 1.0, -1.0)
    s = pd.Series(s, index=bars.index, dtype=float)
    s[sma.isna()] = 0.0
    return s.rename("trend")


# --------------------------------------------------------------------------- #
# Event study — forward returns after a bulge vs the base rate
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizon: int, lag: int = 1) -> np.ndarray:
    """Forward ``horizon``-day simple return entered ``lag`` days after each bar.

    Entry at close[t+lag], exit at close[t+lag+horizon]. Aligned to ``bars`` index;
    NaN where the window overruns the tape. (Per-bar; the event study masks this with
    the bulge / non-bulge flags.)
    """
    c = bars["close"].to_numpy(dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(n):
        e = t + lag
        x = e + horizon
        if e >= 0 and x < n:
            out[t] = c[x] / c[e] - 1.0
    return out


def event_study(bars: pd.DataFrame, horizon: int, lag: int = 1, fade: bool = False,
                hi: float = 27.0, lo: float = 26.5) -> dict:
    """Forward returns after a bulge vs the unconditional base rate.

    ``fade=False`` → raw forward return (the "what happens next" picture).
    ``fade=True``  → forward return **signed against the prevailing trend** at the bulge
                     (``-trend_sign`` × forward return): a genuine reversal makes this
                     positive, a trend *continuation* makes it negative.

    Returns the bulge-sample mean, the non-bulge (base-rate) mean, a one-sample *t* of the
    bulge sample vs 0, a Welch *t* of bulge vs non-bulge, and the bulge win-rate.
    """
    fr = forward_returns(bars, horizon, lag=lag)
    bulge = bulge_signal(bars, hi=hi, lo=lo).to_numpy()
    if fade:
        tr = trend_sign(bars).to_numpy()
        fr = -tr * fr            # fade the prevailing trend
    finite = np.isfinite(fr)
    ev = fr[bulge & finite]
    base = fr[(~bulge) & finite]
    return {
        "horizon": horizon, "n_bulge": int(ev.size), "n_base": int(base.size),
        "bulge_mean": float(ev.mean()) if ev.size else float("nan"),
        "base_mean": float(base.mean()) if base.size else float("nan"),
        "t_vs_zero": ttest_vs_zero(ev),
        "t_vs_base": welch_t(ev, base),
        "win": float((ev > 0).mean()) if ev.size else float("nan"),
        "sample": ev, "base": base,
    }


def placebo_pvalue(bars: pd.DataFrame, horizon: int, lag: int = 1, fade: bool = False,
                   hi: float = 27.0, lo: float = 26.5, n_draws: int = 5000,
                   seed: int = 429) -> dict:
    """Label-shuffle placebo for the post-bulge mean forward return.

    Draw the **same number** of random "trigger" dates as there are real bulges, compute
    the mean forward return over each random set, and ask how often a random set's |mean|
    is at least as large as the observed |bulge mean| (a two-sided rare-event null). For
    a *fade* test we use the signed-against-trend forward return throughout.

    Returns ``{obs, p_value, draws}``.
    """
    fr = forward_returns(bars, horizon, lag=lag)
    if fade:
        tr = trend_sign(bars).to_numpy()
        fr = -tr * fr
    bulge = bulge_signal(bars, hi=hi, lo=lo).to_numpy()
    finite = np.isfinite(fr)
    pool = np.where(finite)[0]
    fr_pool = fr[pool]
    k = int((bulge & finite).sum())
    obs = float(fr[bulge & finite].mean()) if k else float("nan")
    if k == 0 or pool.size == 0:
        return {"obs": obs, "p_value": float("nan"), "draws": np.array([])}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(fr_pool.size, size=k, replace=False)
        draws[i] = fr_pool[pick].mean()
    p = float((np.abs(draws) >= abs(obs)).mean())
    return {"obs": obs, "p_value": p, "draws": draws}


# --------------------------------------------------------------------------- #
# Inference helpers (shared shape across the desk)
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    s = np.asarray(sample, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    se = s.std(ddof=1) / np.sqrt(s.size)
    return float(s.mean() / se) if se > 0 else float("nan")


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) − mean(base) (unequal variance)."""
    a = np.asarray(sample, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(base, dtype=float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def sharpe(excess: np.ndarray) -> float:
    """Annualised Sharpe of a daily *excess* return stream."""
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))


def hac_t(diff: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``diff`` (per-day strat−benchmark) vs 0."""
    r = np.asarray(diff, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def cagr(excess: np.ndarray) -> float:
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / ANN
    return float(growth ** (1.0 / yrs) - 1.0) if growth > 0 and yrs > 0 else float("nan")


def max_drawdown(excess: np.ndarray) -> float:
    r = np.asarray(excess, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# Timing rule — fade the trend for H days after each bulge, NET vs buy-and-hold
# --------------------------------------------------------------------------- #
def fade_position(bars: pd.DataFrame, hold: int = 20, hi: float = 27.0, lo: float = 26.5,
                  long_short: bool = False) -> pd.Series:
    """Desired exposure (as known at close *t*) for the post-bulge fade rule.

    On the bar a bulge completes, take the **fade** exposure for the next ``hold`` days:
    ``-trend_sign`` (bet the prevailing trend reverses). With ``long_short=False`` the
    fade only acts when it points *down* a current up-trend → flat (a long/flat rule that
    steps aside after an up-trend bulge); with ``long_short=True`` it takes the full
    ``-trend`` (−1/+1) exposure. Outside any active fade window the rule is fully long (1).

    The one-day execution lag is applied later in :func:`book_returns` (one shift).
    """
    bulge = bulge_signal(bars, hi=hi, lo=lo).to_numpy()
    tr = trend_sign(bars).to_numpy()
    n = len(bulge)
    pos = np.ones(n, dtype=float)          # default: fully invested (long)
    active = 0
    fade_dir = 0.0
    for t in range(n):
        if bulge[t]:
            active = hold
            fade_dir = -tr[t]              # fade the trend at the bulge
        if active > 0:
            if long_short:
                pos[t] = fade_dir
            else:
                pos[t] = 1.0 if fade_dir > 0 else 0.0   # long or step-aside
            active -= 1
    return pd.Series(pos, index=bars.index, dtype=float)


def book_returns(bars: pd.DataFrame, position: pd.Series, cost_bps: float = 1.0,
                 borrow_bps_ann: float = 50.0, rf_ann: float = 0.0) -> pd.DataFrame:
    """Daily P&L of a timed book, net of costs, **excess of cash**, one-day lag.

    The position known at close *t* (``position`` row *t*) earns the asset return of *t+1*
    — a single ``shift(1)``. Costs: ``cost_bps`` one-way × |Δposition| (turnover as one-way
    × NAV) on the day the position changes; any short exposure pays ``borrow_bps_ann``
    annualised. Returns are excess of the risk-free rate (excess-vs-excess Sharpe race).
    """
    asset_ret = bars["close"].pct_change().fillna(0.0)
    rf_daily = rf_ann / ANN
    asset_excess = asset_ret - rf_daily

    held = position.shift(1).fillna(0.0)              # one-day execution lag
    turnover = held.diff().abs().fillna(held.abs())   # one-way turnover × NAV
    cost = turnover * (cost_bps / 1e4)
    borrow = np.clip(-held, 0.0, None) * (borrow_bps_ann / 1e4) / ANN
    strat_excess = held * asset_excess - cost - borrow
    return pd.DataFrame(
        {"asset_excess": asset_excess, "strat_excess": strat_excess, "pos": held},
        index=bars.index,
    )


def run_experiment(bars: pd.DataFrame, hold: int = 20, long_short: bool = False,
                   cost_bps: float = 1.0, hi: float = 27.0, lo: float = 26.5) -> dict:
    """Full timing race for one tape: post-bulge fade long/flat vs buy-and-hold.

    Returns the fade rule's net annualised Sharpe (excess-of-cash), CAGR, max drawdown,
    time-in-market, annual turnover, the buy-and-hold Sharpe, and the HAC t of the daily
    (fade − buy-hold) excess-return difference. Also reports the number of bulges fired.
    """
    pos = fade_position(bars, hold=hold, hi=hi, lo=lo, long_short=long_short)
    fb = book_returns(bars, pos, cost_bps=cost_bps)
    asset = fb["asset_excess"].to_numpy(dtype=float)
    fade = fb["strat_excess"].to_numpy(dtype=float)
    held = fb["pos"].to_numpy(dtype=float)
    turn = np.abs(np.diff(held, prepend=0.0)).sum()
    yrs = len(bars) / ANN
    n_bulge = int(bulge_signal(bars, hi=hi, lo=lo).sum())
    return {
        "n_days": int(len(bars)), "n_bulge": n_bulge,
        "fade_sharpe": sharpe(fade), "fade_cagr": cagr(fade), "fade_mdd": max_drawdown(fade),
        "fade_tim": float((held != 0).mean()),
        "fade_turn_yr": float(turn / yrs) if yrs > 0 else float("nan"),
        "asset_sharpe": sharpe(asset), "asset_cagr": cagr(asset),
        "asset_mdd": max_drawdown(asset),
        "t_vs_hold": hac_t(fade - asset),
    }
