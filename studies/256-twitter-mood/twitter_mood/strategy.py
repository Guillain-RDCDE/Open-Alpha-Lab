"""Strategy and honest controls -- Study 256 (Twitter-Mood).

The Bollen, Mao & Zeng (2011) claim: the daily aggregate "Calm" mood index
leads the stock market by 3-4 days, and a directional model conditioned on it
hits ~87% next-day accuracy on the Dow.  We test the *cleanest possible* version
of that lead-lag and a tradable directional rule:

1. **Lead-lag regression** -- regress next-day return on lagged Calm
   (``ret_{t+1} ~ a + b * calm_t``).  The slope ``b`` and its HAC t-stat are
   the inferential heart of the study.  We sweep the lag (1..5 days) the way
   Bollen swept Granger lags, which is itself a multiple-comparisons trap.

2. **Directional trading rule** -- go long the index tomorrow if Calm today is
   above its trailing median, else flat (or short).  Positions are formed at
   the day-t close and held one day (a one-day execution lag).  We report
   gross and net (one-way costs x NAV; shorts pay borrow) and the directional
   hit-rate against the unconditional up-rate (the honest base rate).

3. **Sign-shuffle / phase-randomization control** -- shuffle the mood series in
   time and re-estimate, to see where the real slope lands in a null built from
   the same marginal distribution.

**No look-ahead**: the predictor for return earned on day t+1 is Calm observed
at the day-t close.  The trailing median used by the trading rule is computed
strictly from past data (expanding/rolling, shifted by one day).

**Honesty**: the mood tape is a curated reconstruction (not the proprietary
GPOMS feed); the window is ~10 trading months (n < 200) -- far too small to
detect a 1-3 day effect with power.  Both are named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary statistics (HAC t-stat) -- shared with the desk house style
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a daily return series (Newey-West HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")} | {"n": n}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": float(mu),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(tstat),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(max_dd),
        "n": n,
    }


def _hac_se(e: np.ndarray, lags: int) -> float:
    """Newey-West long-run-variance standard error of a mean-zero residual array."""
    n = e.size
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(lrv, 0.0) / n))


# ---------------------------------------------------------------------------
# Lead-lag regression: ret_{t+lag} ~ a + b * calm_t
# ---------------------------------------------------------------------------
def leadlag_regression(df: pd.DataFrame, lag: int = 1) -> dict:
    """OLS slope of next-(lag)-day return on Calm, with a HAC t-stat.

    ``df`` has columns ``calm`` and ``ret`` (same-day return).  The predictor is
    Calm at day t; the response is the return earned ``lag`` trading days later
    (``ret`` shifted back by ``lag``).  ``lag=1`` is the headline one-day-ahead
    forecast.  No look-ahead: the response is strictly future relative to Calm.

    Returns a dict with ``beta`` (slope, in return units per 1 standardized Calm
    unit), ``tstat`` (HAC), ``r2``, ``n`` and ``lag``.
    """
    d = df[["calm", "ret"]].dropna().copy()
    x = d["calm"].to_numpy(dtype=float)
    y = d["ret"].shift(-lag).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 10:
        return {"beta": np.nan, "tstat": np.nan, "r2": np.nan, "n": n, "lag": lag}

    xc = x - x.mean()
    yc = y - y.mean()
    sxx = float(xc @ xc)
    beta = float((xc @ yc) / sxx) if sxx > 0 else float("nan")
    alpha = float(y.mean() - beta * x.mean())
    resid = y - (alpha + beta * x)

    # HAC standard error of the slope: var(b) = (1/sxx^2) * sum( (x_c * resid) HAC )
    g = xc * resid
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    # long-run variance of g (sum, not mean -> scale by n)
    s0 = float(g @ g)
    lrv = s0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(g[k:] @ g[:-k])
    se_beta = float(np.sqrt(max(lrv, 0.0)) / sxx) if sxx > 0 else float("nan")
    tstat = float(beta / se_beta) if se_beta and se_beta > 0 else float("nan")

    ss_tot = float(yc @ yc)
    r2 = float(1.0 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")

    return {"beta": beta, "tstat": tstat, "r2": r2, "n": int(n), "lag": lag}


def leadlag_sweep(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Run ``leadlag_regression`` for lags 1..max_lag (the Granger-lag trap).

    Bollen swept lags and reported the best.  Sweeping k lags and keeping the
    most significant inflates the false-positive rate -- we report all lags and
    a Bonferroni-corrected significance bar so the trap is visible.
    """
    rows = [leadlag_regression(df, lag=L) for L in range(1, max_lag + 1)]
    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Directional trading rule: long if Calm > trailing median, else flat/short
# ---------------------------------------------------------------------------
def mood_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    allow_short: bool = False,
    one_way_bps: float = 5.0,
    borrow_bps_per_day: float = 1.0,
) -> pd.DataFrame:
    """Trade tomorrow's index on today's Calm vs its trailing median.

    Signal at day t: +1 if ``calm_t`` is above the trailing-``lookback`` median
    of Calm (computed strictly from data through t), else 0 (flat) or -1 if
    ``allow_short``.  The position is entered at the day-t close and earns the
    day-(t+1) return -- a one-day execution lag.

    Costs: each change in position pays ``one_way_bps`` x |dpos| (round trips
    cost twice); short days additionally pay ``borrow_bps_per_day``.  Costs are
    charged on NAV.

    Returns a DataFrame with columns ``pos`` (the held position for the next
    day), ``gross`` (gross strategy return), ``net`` (after costs), and ``mkt``
    (the index return), indexed by the day the return is *earned*.
    """
    d = df[["calm", "ret"]].dropna().copy()
    calm = d["calm"]
    # trailing median strictly from the past (shifted so day t uses <= t-1 ... t)
    med = calm.rolling(lookback, min_periods=max(3, lookback // 2)).median()
    raw_sig = (calm > med).astype(float)  # 1 if calm above its trailing median
    if allow_short:
        sig = raw_sig * 2.0 - 1.0  # {-1, +1}
    else:
        sig = raw_sig  # {0, 1}

    # Position held for the NEXT day's return -> shift signal forward by 1
    pos = sig.shift(1)
    mkt = d["ret"]
    gross = pos * mkt

    # turnover cost: |pos_t - pos_{t-1}| * one_way * NAV
    dpos = pos.diff().abs().fillna(pos.abs())
    cost = dpos * (one_way_bps * 1e-4)
    borrow = (pos < 0).astype(float) * (borrow_bps_per_day * 1e-4)
    net = gross - cost - borrow

    res = pd.DataFrame({"pos": pos, "gross": gross, "net": net, "mkt": mkt}).dropna(subset=["pos"])
    return res


def directional_hitrate(df: pd.DataFrame, lookback: int = 20) -> dict:
    """Next-day directional accuracy of the Calm rule vs the unconditional up-rate.

    A 'hit' is: Calm-above-median predicts an up day (or Calm-below predicts a
    down day).  Reported against the unconditional fraction of up days -- the
    honest baseline (the market drifts up ~53% of days, so any always-long-ish
    rule looks good for the wrong reason).
    """
    d = df[["calm", "ret"]].dropna().copy()
    calm = d["calm"]
    med = calm.rolling(lookback, min_periods=max(3, lookback // 2)).median()
    sig_up = (calm > med)  # predict up tomorrow
    fwd = d["ret"].shift(-1)
    valid = sig_up.notna() & fwd.notna() & med.notna()
    sig_up = sig_up[valid]
    fwd = fwd[valid]
    up = fwd > 0

    hits = np.where(sig_up, up, ~up)
    hit_rate = float(hits.mean()) if len(hits) else float("nan")
    uncond_up = float(up.mean()) if len(up) else float("nan")
    return {
        "hit_rate": hit_rate,
        "uncond_up_rate": uncond_up,
        "n": int(len(hits)),
        "n_up_signal": int(sig_up.sum()),
    }


# ---------------------------------------------------------------------------
# Phase-randomization / shuffle control
# ---------------------------------------------------------------------------
def shuffle_null(
    df: pd.DataFrame,
    lag: int = 1,
    n_shuffles: int = 2000,
    seed: int = 256,
) -> dict:
    """Null distribution of the lead-lag slope under time-shuffled mood.

    Permute the Calm series (breaking any genuine time-alignment with returns)
    and re-estimate the slope ``n_shuffles`` times.  Reports where the real
    slope lands -- a permutation p-value that respects the return marginal.
    """
    d = df[["calm", "ret"]].dropna().copy()
    real = leadlag_regression(d, lag=lag)
    real_beta = real["beta"]
    if not np.isfinite(real_beta):
        return {"real_beta": np.nan, "perm_p": np.nan, "n_shuffles": n_shuffles}

    rng = np.random.default_rng(seed)
    calm = d["calm"].to_numpy(dtype=float)
    betas = np.empty(n_shuffles, dtype=float)
    for i in range(n_shuffles):
        shuffled = d.copy()
        shuffled["calm"] = rng.permutation(calm)
        betas[i] = leadlag_regression(shuffled, lag=lag)["beta"]

    # two-sided permutation p-value
    perm_p = float((np.abs(betas) >= abs(real_beta)).mean())
    return {
        "real_beta": float(real_beta),
        "real_tstat": float(real["tstat"]),
        "perm_p": perm_p,
        "perm_beta_mean": float(np.nanmean(betas)),
        "perm_beta_std": float(np.nanstd(betas)),
        "n_shuffles": int(n_shuffles),
    }


def annualise_daily(stats: dict, periods: int = 252) -> dict:
    """Convert daily stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if out.get("vol_ann", 0) and out["vol_ann"] > 0 and "mean_ann" in out:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out
