"""The strategy and its honest controls — Study 174 (Bitcoin-Rainbow).

The Bitcoin Rainbow Chart (popularised by Blockchaincenter.net, ~2014) works as follows:

1. Fit OLS: log(price) ~ alpha + beta * log(days_since_genesis) on the **entire** available
   history.
2. Place coloured bands at k*sigma_residual offsets from the regression line (k = ±1,2,3,4).
3. Interpret: buy in the cold (low) bands, sell in the hot (high) bands.

The chart looks *prescient* because step 1 uses the whole history — a perfect look-ahead
bias.  The sell-to-rainbow signal in 2021 was generated using 12 years of future data that
the "analyst" of 2021 did not have.

This module implements:

- ``fit_rainbow_insample`` — the look-ahead-biased full-history fit.  Produces the familiar
  beautiful chart.  All trading signals derived from this are cheating.
- ``fit_rainbow_walkforward`` — an expanding-window walk-forward fit: at each date, we only
  use data up to that date to compute the regression and bands.  This is the honest version.
- ``rainbow_signal`` — the strategy: go long when price is below the buy-band threshold
  (``BUY_THRESHOLD_SIGMA``), flat otherwise; optionally short above the sell-band.
- ``buy_and_hold`` — the benchmark: invested at the first available bar, never exits.
- ``summarize`` — HAC t-stat on per-day strategy returns vs the benchmark, the decisive
  inference-bar number.

No look-ahead in the walk-forward version: the regression at date *t* uses only data
up to and including *t* to produce the signal used at *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BUY_THRESHOLD_SIGMA, SELL_THRESHOLD_SIGMA


# ---------------------------------------------------------------------------
# In-sample (cheating) fit — the rainbow as popularly presented
# ---------------------------------------------------------------------------
def fit_rainbow_insample(df: pd.DataFrame) -> pd.DataFrame:
    """Fit log(price) ~ alpha + beta*log(days) on the *whole* tape.

    Returns ``df`` with added columns:
        ``trend``      — the fitted log-price at each date
        ``sigma``      — the residual standard deviation (constant, full-sample)
        ``band_sigma`` — (log_price − trend) / sigma  (how many σ from the line)

    Every signal derived from this is look-ahead-biased: the 2013 bands already
    "know" that BTC will reach $69k in 2021.
    """
    log_px = df["log_close"].to_numpy(dtype=float)
    ld = df["log_days"].to_numpy(dtype=float)
    mask = np.isfinite(log_px) & np.isfinite(ld)

    alpha, beta, sigma = _fit_ols_core(log_px[mask], ld[mask])
    trend = alpha + beta * ld
    resid = log_px - trend
    band_sigma = resid / sigma if sigma > 0 else np.zeros_like(resid)

    out = df.copy()
    out["trend_insample"] = trend
    out["sigma_insample"] = sigma
    out["band_sigma_insample"] = band_sigma
    return out


# ---------------------------------------------------------------------------
# Walk-forward (honest) fit — expanding window, no future data
# ---------------------------------------------------------------------------
def fit_rainbow_walkforward(
    df: pd.DataFrame,
    min_obs: int = 252,
) -> pd.DataFrame:
    """Fit the rainbow regression with an expanding window — the honest version.

    At each date *t*, only data ``[0, t]`` is used to estimate alpha, beta, and
    sigma.  The resulting ``band_sigma_wf`` at date *t* is known *at the close of
    t*, so trading signals use it for entry on *t+1* (no look-ahead).

    Parameters
    ----------
    min_obs : int
        Minimum number of observations before issuing any signal.  Defaults to one
        year (252 trading days) — a rainbow chart with 90 days of history is worse
        than useless.
    """
    log_px = df["log_close"].to_numpy(dtype=float)
    ld = df["log_days"].to_numpy(dtype=float)
    n = len(log_px)

    trend_wf = np.full(n, np.nan)
    sigma_wf = np.full(n, np.nan)
    band_sigma_wf = np.full(n, np.nan)

    for t in range(min_obs, n):
        lp_t = log_px[:t + 1]
        ld_t = ld[:t + 1]
        mask = np.isfinite(lp_t) & np.isfinite(ld_t)
        if mask.sum() < min_obs:
            continue
        alpha, beta, sigma = _fit_ols_core(lp_t[mask], ld_t[mask])
        tr = alpha + beta * ld[t]
        trend_wf[t] = tr
        sigma_wf[t] = sigma
        if sigma > 0:
            band_sigma_wf[t] = (log_px[t] - tr) / sigma

    out = df.copy()
    out["trend_wf"] = trend_wf
    out["sigma_wf"] = sigma_wf
    out["band_sigma_wf"] = band_sigma_wf
    return out


def _fit_ols_core(
    log_price: np.ndarray, log_days: np.ndarray
) -> tuple[float, float, float]:
    """OLS: log_price ~ 1 + log_days.  Returns (alpha, beta, sigma_resid)."""
    n = len(log_price)
    if n < 3:
        return 0.0, 0.0, 1.0
    X = np.column_stack([np.ones(n), log_days])
    params, _, _, _ = np.linalg.lstsq(X, log_price, rcond=None)
    alpha, beta = float(params[0]), float(params[1])
    resid = log_price - (alpha + beta * log_days)
    sigma = float(resid.std(ddof=2)) if n > 2 else 1.0
    return alpha, beta, max(sigma, 1e-6)


# ---------------------------------------------------------------------------
# Trading signals
# ---------------------------------------------------------------------------
def rainbow_signal(
    df_fit: pd.DataFrame,
    band_col: str,
    buy_threshold: float = BUY_THRESHOLD_SIGMA,
    sell_threshold: float = SELL_THRESHOLD_SIGMA,
    allow_short: bool = False,
) -> pd.Series:
    """Binary ±1/0 position signal derived from the rainbow band column.

    Entry logic (all look-ahead-free when ``band_col`` is the walk-forward column):
        - Long (+1)  when ``band_sigma < buy_threshold``  (price in the cold/cheap zone)
        - Flat (0)   when ``buy_threshold <= band_sigma < sell_threshold``
        - Short (-1) when ``band_sigma >= sell_threshold`` — only when ``allow_short``
          is True (default False, since shorting BTC during a bull run is suicidal)

    The signal at date *t* is derived from the fit at *t*, and the position is
    entered at the *open* of *t+1* — handled by ``backtest`` shifting by 1.

    Returns a Series aligned on ``df_fit.index``.
    """
    bs = df_fit[band_col]
    pos = pd.Series(0, index=df_fit.index, name="position", dtype=float)
    pos[bs < buy_threshold] = 1.0
    if allow_short:
        pos[bs >= sell_threshold] = -1.0
    return pos


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------
def backtest(
    df: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Daily strategy returns from a position signal.

    ``position`` is the *signal* formed on the close of day *t*; it is shifted
    forward by one day so it applies to the *next* day's return — no look-ahead.

    Each *change* in the position incurs a round-trip cost of ``cost_bps`` basis
    points (10 bps is a reasonable BTC trading cost on a liquid exchange like
    Coinbase or Binance; slippage is higher for a large retail size but 10 bps is
    conservative for small size).

    Returns a DataFrame with columns:
        ``r_market``   — daily log return of BTC
        ``position``   — position used (shifted, as traded)
        ``r_gross``    — strategy gross daily return (position × r_market)
        ``r_net``      — strategy net return after costs
        ``cum_log_net``— cumulative log-return of the strategy
        ``cum_log_bh`` — cumulative log-return of buy-and-hold
    """
    log_ret = df["log_close"].diff()  # daily log returns

    # Shift position by 1: signal on day t → position taken on day t+1.
    pos_traded = position.shift(1).fillna(0.0)
    turnover = pos_traded.diff().abs().fillna(0.0)
    cost = turnover * cost_bps * 1e-4

    r_gross = pos_traded * log_ret
    r_net = r_gross - cost

    out = pd.DataFrame(
        {
            "r_market": log_ret,
            "position": pos_traded,
            "r_gross": r_gross,
            "r_net": r_net,
            "cum_log_net": r_net.cumsum(),
            "cum_log_bh": log_ret.cumsum(),
        },
        index=df.index,
    )
    return out


# ---------------------------------------------------------------------------
# Performance summary & inference
# ---------------------------------------------------------------------------
def summarize(bt: pd.DataFrame, col: str = "r_net") -> dict:
    """Headline per-day statistics for one backtest.

    Returns trade count, win-rate, mean daily return (bps), annualised Sharpe,
    and a HAC t-stat (Newey-West) on the mean — the inference-bar number.
    Also returns the annualised geometric return vs buy-and-hold.

    The effective n is the number of non-NaN observations in ``col``.
    """
    r = bt[col].dropna().to_numpy(dtype=float)
    r_bh = bt["r_market"].dropna().to_numpy(dtype=float)
    n = r.size

    # Annualised geometric returns.
    ann_strat = float(np.exp(r.sum() * (252.0 / n)) - 1.0) if n > 0 else float("nan")
    ann_bh = float(np.exp(r_bh.sum() * (252.0 / len(r_bh))) - 1.0) if len(r_bh) > 0 else float("nan")

    out: dict = {
        "n_days": int(n),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "ann_ret_pct": float(ann_strat * 100.0) if not np.isnan(ann_strat) else float("nan"),
        "ann_bh_pct": float(ann_bh * 100.0) if not np.isnan(ann_bh) else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": float("nan"),
        "time_in_market_pct": float((bt["position"].dropna() != 0).mean() * 100.0),
    }
    if n > 10:
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def insample_vs_walkforward_comparison(df: pd.DataFrame) -> dict:
    """Run both in-sample and walk-forward versions and report headline metrics.

    This is the core of the overfitting expose: the in-sample version looks great
    (all signals are "right" because they were fitted on the whole history); the
    walk-forward version reveals the true out-of-sample performance.
    """
    # In-sample fit.
    df_is = fit_rainbow_insample(df)
    sig_is = rainbow_signal(df_is, "band_sigma_insample")
    bt_is = backtest(df_is, sig_is, cost_bps=0.0)
    s_is = summarize(bt_is, "r_gross")

    # Walk-forward fit.
    df_wf = fit_rainbow_walkforward(df)
    sig_wf = rainbow_signal(df_wf, "band_sigma_wf")
    bt_wf = backtest(df_wf, sig_wf, cost_bps=0.0)
    s_wf = summarize(bt_wf, "r_gross")

    # Buy-and-hold baseline.
    bh_r = df["log_close"].diff().dropna().to_numpy(dtype=float)
    n_bh = len(bh_r)
    ann_bh = float(np.exp(bh_r.sum() * (252.0 / n_bh)) - 1.0) * 100.0 if n_bh > 0 else float("nan")

    return {
        "insample": s_is,
        "walkforward": s_wf,
        "bh_ann_pct": ann_bh,
        "df_is": df_is,
        "df_wf": df_wf,
        "bt_is": bt_is,
        "bt_wf": bt_wf,
        "sig_is": sig_is,
        "sig_wf": sig_wf,
    }
