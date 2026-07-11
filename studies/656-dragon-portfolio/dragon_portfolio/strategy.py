"""Strategy + inference for Study 656 — Dragon Portfolio.

The claim (Chris Cole, Artemis Capital, *"The Allegory of the Hawk and Serpent"*,
2020): a **100-year all-weather** mix — equities + long bonds + gold + commodity
trend + LONG volatility — survives both the inflationary decades (1970s-style) and
the deflationary ones (2008-style) that break a plain 60/40 book, because it holds
one sleeve built to *profit* from each regime rather than merely diversify across them.
The headline published weights (widely quoted from the paper): **24% equities / 18%
fixed income / 19% gold / 18% commodity trend / 21% long volatility**.

We proxy the five sleeves on liquid ETFs (SPY / TLT / GLD / a 12-month trend overlay
on DBC / VXX for the long-vol sleeve — named as a crude, decaying stand-in, see
``data.py``), rebalance monthly, and compare two variants:

* **Dragon-lite** (4 sleeves, equities/bonds/gold/trend, weights renormalised to 100%
  without the vol sleeve) — the longest window (2007→2026), because it needs no VXX
  and so lives through the 2008 deflationary crash.
* **Dragon-full** (5 sleeves, Cole's own weights) — the honest version, but its window
  is capped at VXX's 2009-01-30 inception: **2009→2026 sees 2020 and 2022, never 2008.**

Both race against **60/40 (SPY/TLT)** and an **All-Weather-lite** equal-weight
(25/25/25/25 SPY/TLT/GLD/DBC, buy-and-hold — a simpler, *non-risk-parity* cousin of
Study 68's inverse-vol engine, named as such) and **100% SPY**.

Execution: the only forecast in the whole engine is the trend flag — DBC's trailing
12-month total return, evaluated at the **close of the prior calendar month**, decides
the trend sleeve's position for the *entire following month* (long DBC or parked in
SHY). That is the study's single documented execution lag; it is calendar-known and
involves no other look-ahead. Every other sleeve's weight is an exogenous constant reset
at each month's first trading day — no forecast, no lag needed. Costs are one-way ×
NAV × turnover, charged once per rebalancing leg; no shorts anywhere, so no borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Cole's published Dragon weights (equity / bonds / gold / trend / long-vol)
DRAGON_FULL = {"SPY": 0.24, "TLT": 0.18, "GLD": 0.19, "TREND": 0.18, "VXX": 0.21}
# Same four non-vol sleeves, weights renormalised to sum to 1.0 (0.79 -> 1.0)
DRAGON_LITE = {"SPY": 0.24 / 0.79, "TLT": 0.18 / 0.79, "GLD": 0.19 / 0.79,
               "TREND": 0.18 / 0.79}
SIXTY_FORTY = {"SPY": 0.60, "TLT": 0.40}
ALL_WEATHER_LITE = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "DBC": 0.25}


# --------------------------------------------------------------------------- #
# Returns and the commodity-trend overlay
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns per column (independent NaN handling — columns keep
    their own inception, no premature dropna across the whole frame)."""
    return prices.pct_change()


def trend_flag(dbc_price: pd.Series, lookback_months: int = 12) -> pd.Series:
    """Month-end trailing ``lookback_months`` DBC total return -> a period-indexed
    0/1 flag. The flag computed at month *m*'s close governs the position held
    during month *m+1* (the single documented execution lag, applied below)."""
    me = dbc_price.resample("ME").last()
    ret = me.pct_change(lookback_months)
    flag = (ret > 0).astype(float)
    flag.index = flag.index.to_period("M")
    return flag


def commodity_trend_sleeve(dbc_ret: pd.Series, cash_ret: pd.Series,
                            dbc_price: pd.Series, lookback_months: int = 12) -> pd.Series:
    """Daily return of the trend-timed DBC sleeve: long DBC when the trailing
    ``lookback_months`` return (known at the PRIOR month's close) is positive,
    otherwise parked in SHY (``cash_ret``). Pre-warm-up months (no 12-month DBC
    history yet) default to flat/cash — conservative, named."""
    flags = trend_flag(dbc_price, lookback_months)
    pos = flags.shift(1)                     # month m's flag decides month m+1's position
    idx = dbc_ret.index.intersection(cash_ret.index)
    periods = pd.PeriodIndex(idx, freq="M")
    pos_daily = pos.reindex(periods)
    pos_daily = pos_daily.fillna(0.0).to_numpy()
    d = dbc_ret.reindex(idx).to_numpy()
    c = cash_ret.reindex(idx).to_numpy()
    out = np.where(pos_daily >= 0.5, d, c)
    return pd.Series(out, index=idx, name="TREND").dropna()


def build_returns(prices: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """Daily returns for every raw column PLUS the derived TREND sleeve."""
    ret = to_returns(prices)
    trend = commodity_trend_sleeve(ret["DBC"].dropna(), ret["SHY"].dropna(),
                                    prices["DBC"].dropna(), lookback_months)
    out = ret.copy()
    out["TREND"] = trend
    return out


# --------------------------------------------------------------------------- #
# Fixed-weight blended portfolio, monthly rebalance
# --------------------------------------------------------------------------- #
def blended_portfolio(returns: pd.DataFrame, weights: dict[str, float],
                       cost_bps: float = 5.0, start: str | None = None,
                       end: str | None = None) -> pd.Series:
    """Daily net return of a fixed-weight blend, reset to ``weights`` on the first
    trading day of every calendar month. Weights drift with returns between resets;
    each reset charges one-way turnover x ``cost_bps`` x NAV (long-only, no borrow)."""
    cols = list(weights.keys())
    df = returns[cols].copy()
    if start:
        df = df.loc[df.index >= start]
    if end:
        df = df.loc[df.index <= end]
    df = df.dropna(how="any")
    idx = df.index
    R = df.to_numpy()
    w_target = np.array([weights[c] for c in cols], dtype=float)
    n = R.shape[0]

    marks = idx.to_series().groupby([idx.year, idx.month]).head(1).index
    rebal_set = set(marks)

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        turn = 0.0
        if idx[t] in rebal_set:
            turn = float(np.abs(w_target - w).sum())
            w = w_target.copy()
        port_ret = float(w @ R[t]) - turn * cost
        out[t] = port_ret
        w = w * (1.0 + R[t])
        s = w.sum()
        if s > 0:
            w = w / s
    return pd.Series(out, index=idx, name="portfolio")


def spy_only(returns: pd.DataFrame, start: str | None = None, end: str | None = None
             ) -> pd.Series:
    s = returns["SPY"].dropna()
    if start:
        s = s.loc[s.index >= start]
    if end:
        s = s.loc[s.index <= end]
    return s.rename("SPY")


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _worst_12m(daily: pd.Series) -> float:
    equity = (1.0 + daily).cumprod()
    roll = equity / equity.shift(TRADING_DAYS) - 1.0
    return float(roll.min()) if roll.notna().any() else float("nan")


def portfolio_stats(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """Headline annualised stats. ``rf`` (SHY daily returns) gives excess-of-cash
    Sharpe so every arm is compared on a like-for-like risk-adjusted basis."""
    net = net.astype(float).dropna()
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    exsd = ex.std(ddof=1)
    sharpe = float(ex.mean() / exsd * np.sqrt(TRADING_DAYS)) if exsd > 0 else float("nan")
    return {
        "net": net, "equity": equity, "n": n, "years": years,
        "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "worst_12m": _worst_12m(net),
        "start": net.index[0], "end": net.index[-1],
    }


def window_return(daily: pd.Series, start: str, end: str) -> float:
    """Cumulative total return of a daily-return series over [start, end]."""
    seg = daily.loc[(daily.index >= start) & (daily.index <= end)]
    if len(seg) == 0:
        return float("nan")
    return float((1.0 + seg).prod() - 1.0)


def calendar_year_return(daily: pd.Series, year: int) -> float:
    seg = daily.loc[daily.index.year == year]
    if len(seg) == 0:
        return float("nan")
    return float((1.0 + seg).prod() - 1.0)


# --------------------------------------------------------------------------- #
# Inference: Newey-West HAC t on monthly return differences + block bootstrap
# --------------------------------------------------------------------------- #
def _monthly_returns(daily: pd.Series) -> pd.Series:
    return (1.0 + daily).resample("ME").prod() - 1.0


def newey_west_mean_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett) t of the sample mean of ``x`` being zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return float("nan")
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_diff_monthly(a_daily: pd.Series, b_daily: pd.Series) -> dict:
    """HAC t-stat for mean(monthly return of A − monthly return of B), Newey-West."""
    ma, mb = _monthly_returns(a_daily), _monthly_returns(b_daily)
    common = ma.index.intersection(mb.index)
    diff = (ma.loc[common] - mb.loc[common]).to_numpy(dtype=float)
    return {"n_months": len(diff), "mean_diff_monthly": float(np.nanmean(diff)),
            "t": newey_west_mean_t(diff)}


def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, rf: pd.Series | None = None,
                           block: int = 21, n_boot: int = 2000, seed: int = 656) -> dict:
    """Circular block-bootstrap CI for the Sharpe DIFFERENCE (a − b), daily blocks."""
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f

    def _sharpe(r: np.ndarray) -> float:
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    n = len(ra)
    point = _sharpe(ra) - _sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _sharpe(ra[sel]), _sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            wins += 1
    lo, hi = float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))
    return {"point": float(point), "ci95": (lo, hi), "frac_a_wins": wins / n_boot,
            "n": n, "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Synthetic control — faithful-engine / power check on the crisis-hedge channel
# --------------------------------------------------------------------------- #
def synthetic_crisis_test(frame: pd.DataFrame, truth: dict) -> dict:
    """Isolate the crisis-alpha CHANNEL itself: does the Dragon-weighted TREND+VOL
    sub-sleeve (0.18*TREND + 0.21*VOL) pay off MORE in crisis months than in normal
    months (Welch t, crisis vs non-crisis)? This is deliberately NOT "Dragon vs
    60/40" — a lower equity weight alone would beat 60/40 in any stock crash even
    with zero genuine crisis alpha, which would make the null fire on a pure
    beta-mix artefact rather than testing the hedge mechanism. Isolating the two
    "hedge" sleeves' own crisis-vs-normal gap tests the mechanism directly: the null
    (hedge_strength=0) must not fire; a planted hedge must."""
    rng = np.random.default_rng(truth["seed"])
    is_crisis = rng.random(truth["n_months"]) < truth["crisis_p"]

    hedge = 0.18 * frame["TREND"].to_numpy() + 0.21 * frame["VOL"].to_numpy()
    a, b = hedge[is_crisis], hedge[~is_crisis]
    t = welch_t(a, b)
    return {"n_crisis": int(is_crisis.sum()),
            "hedge_crisis_mean": float(a.mean()) if is_crisis.any() else float("nan"),
            "hedge_normal_mean": float(b.mean()) if (~is_crisis).any() else float("nan"),
            "welch_t": t}
