"""Strategy + inference for Study 667 — Negative Volume Index (Fosback).

The claim, as Norman Fosback stated it in *Stock Market Logic* (1976): "smart money"
trades quietly, on **low-volume days**, while the crowd chases headlines on high-volume
days. Cumulate the index's return only on days volume FALLS versus the prior day (freeze
it otherwise) — the **Negative Volume Index (NVI)** — and Fosback's own back-test claimed
that whenever NVI sits above its 1-year (255-session) EMA, "the odds favor a bull market"
with **96% reliability**.

    NVI[0]   = 1000
    NVI[t]   = NVI[t-1] * (1 + r[t])   if volume[t] < volume[t-1]      (quiet day: track price)
             = NVI[t-1]                otherwise                       (loud day: freeze)

Measurements:

* **The headline replication — Fosback's own framing.** NVI's state at each calendar
  year's close vs the FOLLOWING year's realized return: P(year up | NVI>EMA at the prior
  year-end) with a Wilson interval, benchmarked against the **unconditional** P(year up) —
  the base-rate check the folklore never runs. A label-shuffle placebo (20,000 draws)
  asks how much of the gap survives once the calendar's own bull-year base rate is priced
  in.
* **A higher-power daily cross-check.** Forward 21/63/252-trading-day returns split by
  regime (NVI>EMA vs NVI<EMA) over ~19,000 daily observations — Welch *t* and an
  overlap-robust Newey-West *t* (lag = horizon) give the real-tape statistic the desk's
  REAL bar (*t* ≥ 2) is actually judged against.
* **Third axis — the costed timer.** A long/flat rule (long when NVI>EMA, cash
  otherwise) on SPY, net of one-way costs × NAV, one execution lag (position formed on
  the close of *t* earns *t+1*'s return) — active spread vs buy-and-hold, HAC *t*, a
  circular-shift permutation placebo, and a cost sweep.
* **Synthetic positive control** — a deterministic tape where falling volume genuinely
  transmits a next-day drift (tunable knob `edge`); the detector must stay silent at
  `edge=0` across ≥10 seeds and light up on a planted effect.

The decisive number is the REAL-tape *t* on the daily/HAC cross-check; the Fosback
replication is graded on whether its 96%-style number survives the base-rate and placebo
checks, not on the raw headline percentage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# NVI construction
# --------------------------------------------------------------------------- #
def nvi(close: pd.Series, volume: pd.Series, base: float = 1000.0) -> pd.Series:
    """Fosback's Negative Volume Index: cumulate returns only on falling-volume days."""
    ret = close.pct_change()
    fell = volume.diff() < 0
    out = np.empty(len(close))
    out[0] = base
    r = ret.to_numpy()
    f = fell.to_numpy()
    for t in range(1, len(close)):
        out[t] = out[t - 1] * (1.0 + r[t]) if f[t] else out[t - 1]
    return pd.Series(out, index=close.index, name="nvi")


def nvi_ema(nvi_series: pd.Series, span: int = 255) -> pd.Series:
    """Fosback's "1-year" EMA of NVI (255 US trading sessions)."""
    return nvi_series.ewm(span=span, adjust=False).mean()


def regime(nvi_series: pd.Series, ema_series: pd.Series, min_periods: int = 255) -> pd.Series:
    """'Bull per Fosback' flag: NVI above its EMA; NaN before the EMA has enough history."""
    out = (nvi_series > ema_series).astype(object)
    out.iloc[:min_periods] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d.

    ``b`` is exactly the treated-minus-rest mean difference; with overlapping forward
    returns, ``lags`` should be set to roughly the return horizon (Hodrick-style).
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~np.isnan(y) & ~np.isnan(d)
    y, d = y[keep], d[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Forward returns, one documented execution lag: regime known at close t predicts
# the return from close t+1 to close t+1+horizon (position formed on t, entered t+1).
# --------------------------------------------------------------------------- #
def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    fwd = close.shift(-1 - horizon) / close.shift(-1) - 1.0
    fwd.index = close.index
    return fwd


def horizon_split(close: pd.Series, reg: pd.Series, horizon: int, nw_lags: int | None = None) -> dict:
    """Forward-return regime split at one horizon: means, Welch t, HAC t, hit rates."""
    fwd = forward_return(close, horizon)
    r = reg.astype(float)
    mask = r.notna() & fwd.notna()
    fwd_v, r_v = fwd[mask].to_numpy(), r[mask].to_numpy().astype(bool)
    a, b = fwd_v[r_v], fwd_v[~r_v]
    k_a, k_b = int((a > 0).sum()), int((b > 0).sum())
    lo_a, hi_a = wilson_interval(k_a, len(a))
    lo_b, hi_b = wilson_interval(k_b, len(b))
    k_u = int((fwd_v > 0).sum())
    lo_u, hi_u = wilson_interval(k_u, len(fwd_v))
    lags = nw_lags if nw_lags is not None else horizon
    return {
        "horizon": horizon, "n_on": len(a), "n_off": len(b),
        "mean_on": float(a.mean()), "mean_off": float(b.mean()),
        "mean_all": float(fwd_v.mean()),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(fwd_v, r_v.astype(float), lags=lags),
        "hit_on": k_a / len(a), "hit_on_ci": (lo_a, hi_a),
        "hit_off": k_b / len(b), "hit_off_ci": (lo_b, hi_b),
        "hit_all": k_u / len(fwd_v), "hit_all_ci": (lo_u, hi_u),
    }


# --------------------------------------------------------------------------- #
# The headline replication — Fosback's own annual framing
# --------------------------------------------------------------------------- #
def annual_bull_test(close: pd.Series, reg: pd.Series) -> dict:
    """Calendar-year replication of Fosback's claim.

    NVI/EMA state read at each year's FINAL close; that year's regime predicts the
    FOLLOWING calendar year's realized return (close[year-end] -> close[next year-end]) —
    a scheduled, zero-incremental-look-ahead entry at a fully-known state, the same
    convention the desk uses for public-calendar entries. "Bull year" = realized return
    > 0, mirroring Fosback's own annual bull/bear framing.
    """
    df = pd.DataFrame({"close": close, "reg": reg}).dropna()
    year_end = df.groupby(df.index.year).tail(1)
    year_end = year_end[year_end.index.month == 12]   # drop any partial (in-progress) year
    years = year_end.index.year.to_numpy()
    px = year_end["close"].to_numpy()
    rg = year_end["reg"].to_numpy().astype(bool)
    yr_ret = px[1:] / px[:-1] - 1.0
    rg_prior = rg[:-1]                      # regime at the END of the PRIOR year
    years_out = years[1:]
    bull = yr_ret > 0

    k_on, n_on = int((bull & rg_prior).sum()), int(rg_prior.sum())
    k_off, n_off = int((bull & ~rg_prior).sum()), int((~rg_prior).sum())
    k_all, n_all = int(bull.sum()), len(bull)
    lo_on, hi_on = wilson_interval(k_on, n_on)
    lo_off, hi_off = wilson_interval(k_off, n_off)
    lo_all, hi_all = wilson_interval(k_all, n_all)

    return {
        "n_years": n_all, "lo_year": int(years_out.min()), "hi_year": int(years_out.max()),
        "p_on": k_on / n_on if n_on else float("nan"), "n_on": n_on,
        "p_on_ci": (lo_on, hi_on),
        "p_off": k_off / n_off if n_off else float("nan"), "n_off": n_off,
        "p_off_ci": (lo_off, hi_off),
        "p_all": k_all / n_all, "p_all_ci": (lo_all, hi_all),
        "gap_on_vs_all": (k_on / n_on if n_on else float("nan")) - k_all / n_all,
        "years": years_out, "returns": yr_ret, "regime": rg_prior,
    }


def annual_placebo(years_returns: np.ndarray, regime_flags: np.ndarray,
                   n_draws: int = 20_000, seed: int = 667) -> dict:
    """Label-shuffle placebo on the annual test: is the P(on)-vs-base-rate gap real?

    Shuffles the year-end regime labels relative to the (fixed) sequence of yearly
    returns, ``n_draws`` times, and recomputes P(bull | shuffled ON) each draw. p-value
    = share of shuffles whose |gap vs the unconditional base rate| is >= the observed
    gap — a two-sided test of whether the observed conditioning beats a same-duty-cycle
    random label.
    """
    bull = years_returns > 0
    base = float(bull.mean())
    obs_p_on = float(bull[regime_flags].mean()) if regime_flags.sum() else float("nan")
    obs_gap = obs_p_on - base
    n = len(bull)
    k_on = int(regime_flags.sum())
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_draws)
    idx = np.arange(n)
    for i in range(n_draws):
        chosen = rng.choice(idx, size=k_on, replace=False)
        gaps[i] = bull[chosen].mean() - base
    pval = float((np.abs(gaps) >= abs(obs_gap)).mean())
    return {"base_rate": base, "obs_p_on": obs_p_on, "obs_gap": obs_gap,
            "placebo_gap_mean": float(gaps.mean()), "placebo_gap_sd": float(gaps.std(ddof=1)),
            "p_value": pval, "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Third axis — the costed long/flat timer (SPY)
# --------------------------------------------------------------------------- #
def backtest(close: pd.Series, position: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
    """Daily backtest, one execution lag, one-way costs x NAV on every position change.

    Convention: the desired position formed on the close of t (from that day's NVI/EMA
    state) is held for the return of t+1 (a single shift). Long-only (0/1); no borrow.
    """
    asset_ret = close.pct_change().fillna(0.0)
    held = position.shift(1).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * (cost_bps * 1e-4)
    strat_gross = held * asset_ret
    strat_net = strat_gross - cost
    return pd.DataFrame({
        "asset_ret": asset_ret, "held": held, "turnover": turnover,
        "strat_gross": strat_gross, "strat_net": strat_net, "bh": asset_ret,
    }, index=close.index)


def annual_sharpe(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size == 0:
        return float("nan")
    total = float(np.prod(1.0 + d))
    yrs = d.size / TRADING_DAYS
    return total ** (1.0 / yrs) - 1.0 if yrs > 0 and total > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + d)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West HAC one-sample t-stat for the mean of a daily series being != 0."""
    r = pd.Series(daily).dropna().to_numpy(dtype=float)
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


def permutation_pvalue(asset_ret: pd.Series, position: pd.Series,
                       n_perm: int = 2000, seed: int = 667) -> dict:
    """Circular-shift placebo: is the NVI *timing* real, or just its net long exposure?

    Circularly shifts the realised (lagged) position path against returns, preserving
    its turnover and time-in-market but destroying any real alignment. Statistic: mean
    daily gross active spread (strategy - buy&hold).
    """
    rng = np.random.default_rng(seed)
    a = asset_ret.fillna(0.0).to_numpy(dtype=float)
    held = position.shift(1).fillna(0.0).to_numpy(dtype=float)
    n = a.size

    def _spread(h):
        return float((h * a - a).mean())

    obs = _spread(held)
    placebo = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(1, n))
        placebo[i] = _spread(np.roll(held, shift))
    pval = float((placebo >= obs).mean())
    return {"observed_spread_bps": obs * 1e4, "placebo_mean_bps": float(placebo.mean()) * 1e4,
            "p_value": pval, "n_perm": n_perm}


def summarize(bt: pd.DataFrame) -> dict:
    net = bt["strat_net"]
    spread = bt["strat_net"] - bt["bh"]
    pos_changes = int((bt["held"].diff().abs() > 1e-9).sum())
    yrs = len(bt) / TRADING_DAYS
    return {
        "n_days": int(len(bt)),
        "sharpe_net": annual_sharpe(net), "bh_sharpe": annual_sharpe(bt["bh"]),
        "cagr_net": cagr(net), "bh_cagr": cagr(bt["bh"]),
        "maxdd_net": max_drawdown(net), "bh_maxdd": max_drawdown(bt["bh"]),
        "mean_spread_bps": float(spread.mean() * 1e4),
        "spread_t": hac_tstat(spread),
        "n_switches": pos_changes,
        "switches_per_yr": pos_changes / yrs if yrs > 0 else float("nan"),
        "time_in_market": float((bt["held"].abs() > 1e-9).mean()),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(tape: pd.DataFrame, horizon: int = 21) -> dict:
    """Run the headline daily forward-return regime split on a synthetic tape."""
    n = nvi(tape["Close"], tape["Volume"])
    e = nvi_ema(n)
    reg = regime(n, e)
    return horizon_split(tape["Close"], reg, horizon)
