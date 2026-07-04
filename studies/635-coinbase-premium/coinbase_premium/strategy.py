"""Strategy + inference for Study 635 — Coinbase Premium.

The claim: **when US institutions buy, Coinbase trades rich to Binance — and the
premium is said to lead the market.** We measure it three complementary ways:

* **The series itself.** Daily premium ``cb/bn − 1`` in bps: level, persistence,
  and the year-by-year regime narrative (the 2020-21 "institutional bid").

* **Predictive regressions.** Forward *h*-day BTC log return (h = 1..7) on today's
  premium **level** and on its **1-day change**, with **Newey-West (HAC) t** on the
  slope (lags cover the overlap: ``h + 2``). This is the claim as stated — the
  premium *leads* the market.

* **A tradable timing overlay.** Long BTC for day *t+1* iff the trailing 63-day
  z-score of the premium is positive at the close of day *t* (past-only window —
  no look-ahead; **exactly one execution lag**: signal read at close *t*, position
  earns the close-*t* → close-*t+1* return). Costs are one-way bps × traded NAV on
  every position change; long-only, flat otherwise (no shorts, no borrow). The
  honest benchmark is an **exposure-matched random-timing baseline averaged over
  ≥ 20 seeds** (single-seed baselines are banned) plus exposure-matched buy & hold.

Returns are **Coinbase BTC-USD close-to-close** (the leg a US account actually
trades), price-only (spot BTC pays nothing), rf taken as 0 for Sharpe on BOTH the
overlay and buy & hold (same treatment, labeled). The decisive number is the HAC
slope *t* on the real tape, and whether the overlay clears its random twin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


# --------------------------------------------------------------------------- #
# Series construction
# --------------------------------------------------------------------------- #
def premium_bps(df: pd.DataFrame) -> pd.Series:
    """Daily Coinbase-vs-Binance premium in basis points: (cb/bn − 1) × 1e4."""
    return (df["cb"] / df["bn"] - 1.0) * 1e4


def log_returns(df: pd.DataFrame, leg: str = "cb") -> pd.Series:
    """Daily close-to-close log returns of one leg (default: the tradable USD leg)."""
    return np.log(df[leg]).diff()


def trailing_z(x: pd.Series, window: int = 63) -> pd.Series:
    """Past-only trailing z-score: (x − rolling mean) / rolling sd, both over the
    PRIOR ``window`` days *including today's print* (known at today's close)."""
    mu = x.rolling(window, min_periods=window // 2).mean()
    sd = x.rolling(window, min_periods=window // 2).std(ddof=1)
    return (x - mu) / sd


def yearly_table(prem: pd.Series) -> pd.DataFrame:
    """Year-by-year premium regime: mean, sd, share of positive days."""
    g = prem.groupby(prem.index.year)
    return pd.DataFrame({
        "mean_bps": g.mean(), "sd_bps": g.std(ddof=1),
        "pos_share": g.apply(lambda s: float((s > 0).mean())),
        "n": g.size(),
    })


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 5) -> float:
    """Newey-West t of mean(x) vs 0 with Bartlett kernel (autocorrelation-robust)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(e[lag:] @ e[:-lag]) / n
    se = np.sqrt(max(s, 1e-30) / n)
    return float(x.mean() / se)


def hac_slope(x: np.ndarray, y: np.ndarray, lags: int = 5) -> dict:
    """OLS slope of ``y`` on ``x`` (with intercept) with a Newey-West (Bartlett) t.

    The long-run variance is taken on the score ``u_t = (x_t − x̄) e_t``; the slope
    standard error is ``sqrt(n·S) / Σ(x−x̄)²`` — the univariate HAC sandwich.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 30:
        return {"beta": float("nan"), "t": float("nan"), "corr": float("nan"), "n": n}
    xd = x - x.mean()
    sxx = float(xd @ xd)
    beta = float(xd @ y) / sxx
    alpha = y.mean() - beta * x.mean()
    e = y - alpha - beta * x
    u = xd * e
    s = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    se = np.sqrt(max(n * s, 1e-30)) / sxx
    corr = float(np.corrcoef(x, y)[0, 1])
    return {"beta": beta, "t": float(beta / se), "corr": corr, "n": n}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) − mean(b), unequal variances."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Predictive regressions — the claim as stated
# --------------------------------------------------------------------------- #
def forward_return(ret: pd.Series, h: int) -> pd.Series:
    """Forward h-day log return starting the NEXT day: sum of ret[t+1 .. t+h]."""
    return ret.shift(-1).rolling(h).sum().shift(-(h - 1))


def predictive_table(df: pd.DataFrame, horizons=(1, 2, 3, 5, 7),
                     signal: str = "level", winsor: float | None = None) -> pd.DataFrame:
    """HAC slope t of forward h-day return on the premium level or its 1d change.

    ``beta`` is reported per +1 standard deviation of the signal (bps of forward
    return), so magnitudes are comparable across signals. ``winsor`` optionally
    clips the signal at the (q, 1−q) quantiles — the USDT-depeg robustness leg.
    NW lags = h + 2 (covers the overlap of h-day forward windows).
    """
    prem = premium_bps(df)
    x = prem if signal == "level" else prem.diff()
    if winsor is not None:
        lo, hi = x.quantile(winsor), x.quantile(1.0 - winsor)
        x = x.clip(lo, hi)
    ret = log_returns(df)
    rows = []
    sd = float(x.std(ddof=1))
    for h in horizons:
        y = forward_return(ret, h)
        r = hac_slope(x.to_numpy(), y.to_numpy(), lags=h + 2)
        rows.append({"h": h, "beta_bps_per_sd": r["beta"] * sd * 1e4,
                     "hac_t": r["t"], "corr": r["corr"], "n": r["n"]})
    return pd.DataFrame(rows).set_index("h")


# --------------------------------------------------------------------------- #
# Tradable timing overlay
# --------------------------------------------------------------------------- #
def overlay(df: pd.DataFrame, z_window: int = 63, cost_bps: float = 10.0) -> dict:
    """Long-only premium-timing overlay vs buy & hold, with one clean lag.

    Position for day t+1 = 1 if trailing z(premium) at close t > 0 else 0. Each
    position CHANGE pays ``cost_bps`` one-way × traded NAV. Also reports the
    exposure-matched buy & hold (exposure × B&H mean — what a coin-flip with the
    same average market time would earn in expectation) and the timing alpha vs it.
    """
    prem = premium_bps(df)
    z = trailing_z(prem, window=z_window)
    ret = log_returns(df)
    pos = (z > 0).astype(float)                 # known at close t
    strat = (pos.shift(1) * ret).dropna()       # earns t -> t+1: ONE lag
    pos_al = pos.shift(1).reindex(strat.index)
    trades = pos.diff().abs().shift(1).reindex(strat.index).fillna(0.0)
    cost = trades * (cost_bps / 1e4)
    net = strat - cost
    bh = ret.reindex(strat.index)
    expo = float(pos_al.mean())
    return {
        "n_days": int(len(strat)), "exposure": expo,
        "gross_bps_d": float(strat.mean() * 1e4), "net_bps_d": float(net.mean() * 1e4),
        "gross_t": hac_t(strat.to_numpy()), "net_t": hac_t(net.to_numpy()),
        "net_ann_pct": float(np.expm1(net.mean() * DAYS_PER_YEAR) * 100),
        "net_sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)),
        "bh_bps_d": float(bh.mean() * 1e4),
        "bh_ann_pct": float(np.expm1(bh.mean() * DAYS_PER_YEAR) * 100),
        "bh_sharpe": float(bh.mean() / bh.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)),
        "alpha_bps_d": float(net.mean() * 1e4 - expo * bh.mean() * 1e4),
        "turnover_per_yr": float(trades.mean() * DAYS_PER_YEAR),
        "net": net, "gross": strat, "bh": bh, "pos": pos_al,
    }


def timing_alpha_t(df: pd.DataFrame, z_window: int = 63, cost_bps: float = 10.0) -> float:
    """HAC t of the daily ACTIVE return: net overlay − exposure-matched B&H.

    The active series is ``net_t − exposure × ret_t`` — the day-by-day surplus over
    a static position holding the same average market time. Its HAC t is the honest
    'is the timing itself worth anything' statistic.
    """
    ov = overlay(df, z_window=z_window, cost_bps=cost_bps)
    active = ov["net"] - ov["exposure"] * ov["bh"]
    return hac_t(active.to_numpy())


def random_baseline(df: pd.DataFrame, exposure: float, cost_bps: float = 10.0,
                    n_seeds: int = 20, base_seed: int = 635) -> dict:
    """Exposure-matched random-timing twin, averaged over ``n_seeds`` seeds.

    Each seed draws i.i.d. Bernoulli(exposure) daily positions, applies the same
    one-day lag and the same per-change costs, and records the net mean and HAC t.
    The report is the ACROSS-SEED mean ± sd — never a single seed.
    """
    ret = log_returns(df).dropna()
    means, ts = [], []
    for k in range(n_seeds):
        rng = np.random.default_rng(base_seed + k)
        pos = pd.Series(rng.random(len(ret) + 1) < exposure,
                        index=[ret.index[0] - pd.Timedelta(days=1), *ret.index],
                        dtype=float)
        strat = (pos.shift(1).reindex(ret.index) * ret)
        trades = pos.diff().abs().shift(1).reindex(ret.index).fillna(0.0)
        net = (strat - trades * (cost_bps / 1e4)).dropna()
        means.append(float(net.mean() * 1e4))
        ts.append(hac_t(net.to_numpy()))
    return {"mean_bps_d": float(np.mean(means)), "sd_bps_d": float(np.std(means, ddof=1)),
            "mean_hac_t": float(np.mean(ts)), "n_seeds": n_seeds}


# --------------------------------------------------------------------------- #
# Sub-periods — the third axis
# --------------------------------------------------------------------------- #
def slice_df(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    return out.copy()


def era_summary(df: pd.DataFrame, h: int = 1, z_window: int = 63,
                cost_bps: float = 10.0) -> dict:
    """One era's card: premium level, h-day predictive HAC t, overlay net t + alpha."""
    prem = premium_bps(df)
    pt = predictive_table(df, horizons=(h,), signal="level")
    ov = overlay(df, z_window=z_window, cost_bps=cost_bps)
    active = ov["net"] - ov["exposure"] * ov["bh"]
    return {
        "n": int(len(prem)), "prem_mean_bps": float(prem.mean()),
        "prem_sd_bps": float(prem.std(ddof=1)),
        "beta_bps_per_sd": float(pt["beta_bps_per_sd"].iloc[0]),
        "pred_t": float(pt["hac_t"].iloc[0]),
        "net_bps_d": ov["net_bps_d"], "net_t": ov["net_t"],
        "alpha_bps_d": ov["alpha_bps_d"], "alpha_t": hac_t(active.to_numpy()),
        "active": active,
    }
