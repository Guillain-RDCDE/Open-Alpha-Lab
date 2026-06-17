"""Strategy and honest controls -- Study 237 (Residual-Momentum).

The Blitz, Huij & Martens (2011) recipe: each month, estimate each stock's
CAPM beta via a rolling 36-month OLS of its monthly excess return on the
market excess return (SPY used as market proxy). Compute the *residual*
return (actual minus beta * market) for each of the trailing ``formation``
months (typically 11, skipping 1 most-recent month). Sort stocks by their
trailing cumulative residual return; long the top quintile (residual winners)
and short (or compare against) the bottom quintile (residual losers).

The claim: residual momentum captures the same cross-sectional signal as raw
momentum but with lower systematic risk loading, fewer momentum crashes
(which occur when beta-driven names reverse in unison), and lower beta of
the long-short itself.

Three honest comparisons:

1. **Residual vs Raw momentum** -- does the residual signal add anything
   beyond the standard 12-1 price momentum?
2. **Beta of the L-S portfolio** -- is it lower for residual momentum as
   claimed by Blitz et al.?
3. **Drawdown in crash months** -- 2009-03 and 2020-03 are notorious
   momentum crash events; does residual momentum cushion them?

**No look-ahead**: rolling beta uses data t-beta_window to t-1.
Formation window is t-formation-skip to t-1-skip (skip=1 by default).
**Survivorship bias**: real tape is current large-cap basket backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Beta estimation: rolling OLS
# ---------------------------------------------------------------------------
def rolling_betas(
    prices: pd.DataFrame,
    market: pd.Series,
    beta_window: int = 36,
    min_obs: int = 24,
) -> pd.DataFrame:
    """Rolling CAPM beta for each stock vs the market (SPY).

    At each month t, regresses the stock's monthly return on SPY's monthly
    return using the prior ``beta_window`` months. Returns a (month x ticker)
    DataFrame of estimated betas; NaN when fewer than ``min_obs`` valid
    observations are available.

    Parameters
    ----------
    prices : month x ticker price panel.
    market : month Series of market (SPY) prices.
    beta_window : rolling window in months for the OLS.
    min_obs : minimum valid observations to estimate a beta.
    """
    stock_ret = prices.pct_change()
    mkt_ret = market.pct_change()

    betas_dict: dict = {}
    for t_pos in range(beta_window, len(prices)):
        t = prices.index[t_pos]
        wnd_stocks = stock_ret.iloc[t_pos - beta_window:t_pos]  # (beta_window, n)
        wnd_mkt = mkt_ret.iloc[t_pos - beta_window:t_pos]       # (beta_window,)

        row: dict = {}
        for col in wnd_stocks.columns:
            y = wnd_stocks[col].to_numpy(dtype=float)
            x = wnd_mkt.to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < min_obs:
                row[col] = np.nan
            else:
                xm = x[mask]
                ym = y[mask]
                cov_xy = np.cov(xm, ym)[0, 1]
                var_x = np.var(xm)
                row[col] = cov_xy / var_x if var_x > 1e-12 else np.nan
        betas_dict[t] = row

    beta_df = pd.DataFrame(betas_dict).T
    beta_df.index = pd.DatetimeIndex(beta_df.index)
    return beta_df


# ---------------------------------------------------------------------------
# Residual return construction
# ---------------------------------------------------------------------------
def residual_signal(
    prices: pd.DataFrame,
    market: pd.Series,
    beta_df: pd.DataFrame,
    formation: int = 11,
    skip: int = 1,
) -> pd.DataFrame:
    """Trailing cumulative residual return, lagged by ``skip`` months.

    At month t the signal is the sum of residual monthly returns from
    t - formation - skip to t - 1 - skip::

        resid_{i,s} = ret_{i,s} - beta_{i,t} * mkt_s

    where beta is the rolling estimate from ``rolling_betas`` (using data up
    to t-1 -- no look-ahead). The formation window default (11 months, skip 1)
    mirrors standard 12-1 momentum convention.

    Returns a (month x ticker) DataFrame of cumulative residual returns.
    """
    stock_ret = prices.pct_change()
    mkt_ret = market.pct_change()

    signal_dict: dict = {}
    for t_pos in range(len(prices)):
        t = prices.index[t_pos]
        if t not in beta_df.index:
            continue
        betas_t = beta_df.loc[t]

        # Formation window: t - formation - skip to t - 1 - skip (inclusive)
        start = t_pos - formation - skip
        end = t_pos - skip
        if start < 0:
            continue

        wnd_stocks = stock_ret.iloc[start:end]   # (formation, n_tickers)
        wnd_mkt = mkt_ret.iloc[start:end]        # (formation,)

        row: dict = {}
        for col in wnd_stocks.columns:
            if col not in betas_t.index or np.isnan(betas_t[col]):
                row[col] = np.nan
                continue
            beta = betas_t[col]
            stock_r = wnd_stocks[col].to_numpy(dtype=float)
            mkt_r = wnd_mkt.to_numpy(dtype=float)
            valid = np.isfinite(stock_r) & np.isfinite(mkt_r)
            if valid.sum() < formation // 2:
                row[col] = np.nan
            else:
                resid = stock_r[valid] - beta * mkt_r[valid]
                row[col] = float(resid.sum())
        signal_dict[t] = row

    sig_df = pd.DataFrame(signal_dict).T
    sig_df.index = pd.DatetimeIndex(sig_df.index)
    return sig_df


def raw_signal(
    prices: pd.DataFrame,
    formation: int = 11,
    skip: int = 1,
) -> pd.DataFrame:
    """Trailing cumulative raw return for comparison (standard 12-1 momentum).

    Returns a (month x ticker) DataFrame of cumulative price returns over
    the ``formation`` months ending at t - ``skip``.
    """
    log_prices = np.log(prices.replace(0, np.nan))
    signal = log_prices.shift(skip) - log_prices.shift(skip + formation)
    signal.index.name = "date"
    return signal


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """Monthly winner-minus-loser spread (residual or raw momentum direction).

    For each month t, stocks are sorted by signal.loc[t] into quintiles.
    The top ``q`` fraction (highest residual signal = *winners*) is the long
    leg; the bottom ``q`` fraction (lowest signal = *losers*) is the short.

    Returns a DataFrame with columns:
    - ``winner`` : equal-weight mean return of the top quintile
    - ``loser``  : equal-weight mean return of the bottom quintile
    - ``market`` : equal-weight mean return of all ranked stocks
    - ``spread`` : winner - loser (positive = momentum)
    - ``n_total``, ``n_winner``, ``n_loser``
    """
    fwd = prices.pct_change(periods=1).shift(-1)  # month t -> t+1

    rows: list[dict] = []
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        s = s.loc[common]
        f = f.loc[common]

        lo_thresh = s.quantile(q)
        hi_thresh = s.quantile(1.0 - q)
        winner_mask = s >= hi_thresh
        loser_mask = s <= lo_thresh

        winner_ret = float(f[winner_mask].mean())
        loser_ret = float(f[loser_mask].mean())
        market_ret = float(f.mean())

        rows.append({
            "date": t,
            "winner": winner_ret,
            "loser": loser_ret,
            "market": market_ret,
            "spread": winner_ret - loser_ret,
            "n_total": int(len(common)),
            "n_winner": int(winner_mask.sum()),
            "n_loser": int(loser_mask.sum()),
        })

    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


# ---------------------------------------------------------------------------
# Portfolio beta (vs market)
# ---------------------------------------------------------------------------
def portfolio_beta(
    port_ret: pd.Series,
    mkt_ret: pd.Series,
) -> float:
    """OLS beta of portfolio return on market return."""
    common = port_ret.index.intersection(mkt_ret.index)
    y = port_ret.loc[common].to_numpy(dtype=float)
    x = mkt_ret.loc[common].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 10:
        return np.nan
    xm, ym = x[mask], y[mask]
    cov = np.cov(xm, ym)[0, 1]
    var = np.var(xm)
    return float(cov / var) if var > 1e-12 else np.nan


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(annual_ret: pd.Series) -> dict:
    """Headline statistics with Newey-West HAC t-stat."""
    r = pd.Series(annual_ret).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat
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


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and "vol_ann" in out and out["vol_ann"] > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


def turnover_cost_drag(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.20,
    one_way_bps: float = 10.0,
    min_stocks: int = 10,
) -> pd.Series:
    """Monthly cost drag from winner-quintile portfolio turnover.

    Returns a Series of monthly cost drag (fraction; 0.001 = 10 bps).
    """
    rows: list[dict] = []
    prev_winners: set = set()

    for t in signal.index:
        s = signal.loc[t].dropna()
        f = prices.loc[t].dropna() if t in prices.index else pd.Series(dtype=float)
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_winners = set()
            continue
        s = s.loc[common]
        hi_thresh = s.quantile(1.0 - q)
        curr_winners = set(s[s >= hi_thresh].index.tolist())
        if prev_winners:
            retained = prev_winners & curr_winners
            turnover = 1.0 - len(retained) / len(curr_winners) if curr_winners else 1.0
        else:
            turnover = 1.0
        drag = turnover * 2.0 * one_way_bps * 1e-4
        rows.append({"date": t, "turnover": turnover, "drag": drag})
        prev_winners = curr_winners

    if not rows:
        return pd.Series(dtype=float, name="drag")
    df = pd.DataFrame(rows).set_index("date")
    return df["drag"].rename("drag")
