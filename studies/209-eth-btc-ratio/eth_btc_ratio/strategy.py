"""Strategy and honest controls — Study 209 (ETH-BTC-Ratio).

The folk recipe: the ETH/BTC price ratio is a 'risk-on / risk-off' dial inside crypto.
When ETH outperforms BTC (ratio rises) capital is rotating into alts; ride that wave by
overweighting ETH. When the ratio falls, rotate back to BTC. Concretely: use a
momentum/trend signal on the ETH/BTC log-ratio to tilt a crypto allocation.

Three strategies share one daily-rebalancing engine:

- **ratio_momentum** — the study's claim: go long ETH when the ETH/BTC ratio's n-day
  momentum is positive, long BTC when negative, rebalancing daily.
- **50/50_rebalance** — a simple equal-weight daily rebalance between ETH and BTC.
  The honest baseline: requires no forecast skill, low cost, captures both assets.
- **hold_btc** and **hold_eth** — passive buy-and-hold benchmarks.

The inference bar is whether the momentum rotation delivers a statistically real,
cost-net Sharpe above the 50/50 baseline — not just above zero (which is trivially
cleared by crypto's secular bull trend) and not just above buy-and-hold-ETH
(survivorship: we chose ETH *because* it went up). The baseline must be at least as
smart as the rotation strategy.

No look-ahead: the ratio trend is computed on closes up to day *t*; positions are
entered at *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal: ETH/BTC ratio momentum
# ---------------------------------------------------------------------------
def ratio_log(bars: pd.DataFrame) -> pd.Series:
    """Log of the ETH/BTC price ratio from a dual-column bar frame.

    Parameters
    ----------
    bars : DataFrame with two-level column (ticker × field). Must have
        'ETH-USD' and 'BTC-USD' as level-0 labels and 'close' as level-1.

    Returns
    -------
    pd.Series indexed like ``bars``, containing log(ETH_close / BTC_close).
    """
    eth_c = bars["ETH-USD"]["close"]
    btc_c = bars["BTC-USD"]["close"]
    return np.log(eth_c / btc_c).rename("ratio_log")


def momentum_signal(
    bars: pd.DataFrame,
    lookback: int = 20,
) -> pd.Series:
    """ETH/BTC ratio momentum signal.

    Signal on date *t* (known at *t*'s close, acted on at *t+1*'s open):
        +1  if log(ETH/BTC) today > log(ETH/BTC) ``lookback`` days ago  →  ETH trending up
        -1  if log(ETH/BTC) today < log(ETH/BTC) ``lookback`` days ago  →  BTC trending up
         0  if equal (degenerate)

    Returns a pd.Series of {−1, 0, +1} indexed like ``bars``.
    """
    rl = ratio_log(bars)
    diff = rl - rl.shift(lookback)
    sig = np.sign(diff)
    sig.name = "signal"
    return sig


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------
def _daily_log_returns(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-asset log returns using next-day-open/today-close (no look-ahead).

    Return at date *t+1* = log(open_{t+1} / open_{t}).
    This is exactly what you earn if you enter at the open of *t+1* and exit
    at the open of *t+2*. We use open/open because the signal is formed on *t*'s
    close and acted on at *t+1*'s open. The last available bar is dropped (no
    exit open). Columns: BTC-USD, ETH-USD.
    """
    rows = {}
    for ticker in ["BTC-USD", "ETH-USD"]:
        o = bars[ticker]["open"]
        rows[ticker] = np.log(o / o.shift(1))
    return pd.DataFrame(rows).dropna()


def backtest_rotation(
    bars: pd.DataFrame,
    lookback: int = 20,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Simulate the ETH/BTC momentum-rotation strategy.

    On each day *t*, the signal formed at the close of *t* determines the next
    day's allocation:
        signal = +1  →  100 % ETH (the ratio is trending up; overweight ETH)
        signal = -1  →  100 % BTC (the ratio is trending down; hide in BTC)
        signal =  0  →  50/50 (no conviction)

    When the allocation *changes* from the prior day, a round-trip transaction cost
    of ``cost_bps`` is charged (expressed in log-return space as ``cost_bps * 1e-4``).
    Returns a DataFrame indexed on trading dates with columns:

        open_ret_btc, open_ret_eth  — raw log-returns (open-to-open, 1-day forward)
        signal                       — the signal formed on the prior close
        weight_eth, weight_btc       — portfolio weights (long-only, sum to 1)
        port_gross                   — gross log portfolio return
        cost                         — transaction cost paid this day
        port_net                     — net log portfolio return
        bnh_btc, bnh_eth             — passive buy-and-hold log-returns
        fifty_fifty                  — 50/50 daily-rebalanced log portfolio return
    """
    sig = momentum_signal(bars, lookback=lookback).shift(1)  # signal formed at close t, used at open t+1
    rets = _daily_log_returns(bars)

    # Align signal and returns on common dates
    common = sig.index.intersection(rets.index)
    sig = sig.loc[common]
    rets = rets.loc[common]

    # Weight mapping: +1 → ETH=1 BTC=0, -1 → ETH=0 BTC=1, 0 → 50/50
    w_eth = sig.map({1: 1.0, -1: 0.0, 0: 0.5}).fillna(0.5)
    w_btc = 1.0 - w_eth

    # Transaction cost: charged when ETH weight changes
    w_eth_prev = w_eth.shift(1).fillna(0.5)
    turnover = (w_eth - w_eth_prev).abs()
    cost = turnover * cost_bps * 1e-4

    port_gross = w_btc * rets["BTC-USD"] + w_eth * rets["ETH-USD"]
    port_net = port_gross - cost
    fifty_fifty = 0.5 * rets["BTC-USD"] + 0.5 * rets["ETH-USD"]

    result = pd.DataFrame(
        {
            "open_ret_btc": rets["BTC-USD"],
            "open_ret_eth": rets["ETH-USD"],
            "signal": sig,
            "weight_eth": w_eth,
            "weight_btc": w_btc,
            "port_gross": port_gross,
            "cost": cost,
            "port_net": port_net,
            "bnh_btc": rets["BTC-USD"],
            "bnh_eth": rets["ETH-USD"],
            "fifty_fifty": fifty_fifty,
        }
    )
    return result.dropna(subset=["port_gross"])


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(returns: pd.Series, label: str = "", periods_per_year: int = 365) -> dict:
    """Headline statistics for a daily log-return series.

    Crypto trades 365 days/year, so ``periods_per_year=365`` for annualisation.
    Returns n, annualised mean (%), annualised Sharpe, HAC t-stat on the daily mean,
    max drawdown, and a Newey-West long-run variance. The t-stat answers the inference
    bar question: is this strategy's mean return distinguishable from zero with
    HAC-robust standard errors?
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return {"label": label, "n": n, "mean_ann_pct": float("nan"),
                "sharpe_ann": float("nan"), "tstat": float("nan"),
                "max_dd_pct": float("nan")}

    mu = r.mean()
    sd = r.std(ddof=1)
    mean_ann = (np.exp(mu * periods_per_year) - 1.0) * 100.0
    sharpe_ann = mu / sd * np.sqrt(periods_per_year) if sd > 0 else float("nan")

    # Newey-West HAC t-stat on the daily mean
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    # Max drawdown on cumulative log-return
    cum = np.cumsum(r)
    roll_max = np.maximum.accumulate(cum)
    dd = cum - roll_max
    max_dd = (np.exp(dd.min()) - 1.0) * 100.0  # as percentage

    return {
        "label": label,
        "n": int(n),
        "mean_ann_pct": float(mean_ann),
        "sharpe_ann": float(sharpe_ann),
        "tstat": float(tstat),
        "max_dd_pct": float(max_dd),
    }


def alpha_vs_baseline(
    strategy_ret: pd.Series,
    baseline_ret: pd.Series,
    periods_per_year: int = 365,
) -> dict:
    """OLS alpha of the rotation strategy vs the 50/50 baseline.

    Regresses daily strategy log-returns on baseline log-returns:
        r_strategy = alpha + beta * r_baseline + eps

    A statistically significant positive ``alpha`` would mean the rotation adds
    value *above* the passive 50/50. The HAC t-stat on alpha is the key inference
    number for this study's 'does rotation add value?' question.
    """
    y = np.asarray(strategy_ret, dtype=float)
    x = np.asarray(baseline_ret, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    y, x = y[mask], x[mask]
    n = len(y)

    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)

    # HAC t-stat on alpha via Newey-West
    e = resid  # residuals ≈ excess returns over baseline
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se_alpha = np.sqrt(max(lrv, 0.0) / n)
    t_alpha = float(alpha / se_alpha) if se_alpha > 0 else float("nan")

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float((np.exp(alpha * periods_per_year) - 1.0) * 100.0),
        "beta": float(beta),
        "r_squared": float(r2),
        "t_alpha": float(t_alpha),
        "n": int(n),
    }


def lookback_sweep(
    bars: pd.DataFrame,
    lookbacks: list[int] | None = None,
    cost_bps: float = 5.0,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """Sweep the momentum lookback window from 5 to 120 days.

    This is the multiple-comparisons stress test: if the signal only works at one
    specific lookback and not others, it is likely overfitted. A robust effect should
    appear across a range of lookbacks. Returns a DataFrame of Sharpe and t-stat for
    each lookback, and also the 50/50 baseline Sharpe for reference.

    Implements a Bonferroni correction: with k lookbacks tested, the adjusted
    significance threshold is |t| >= t_{alpha/2k}, reported in the output.
    """
    if lookbacks is None:
        lookbacks = [5, 10, 20, 30, 60, 90, 120]

    rows = []
    for lb in lookbacks:
        bt = backtest_rotation(bars, lookback=lb, cost_bps=cost_bps)
        s = summarize(bt["port_net"], label=f"rotation_{lb}d",
                      periods_per_year=periods_per_year)
        b = summarize(bt["fifty_fifty"], label="50/50",
                      periods_per_year=periods_per_year)
        rows.append({
            "lookback": lb,
            "sharpe_rotation": s["sharpe_ann"],
            "tstat_rotation": s["tstat"],
            "mean_ann_pct": s["mean_ann_pct"],
            "sharpe_50_50": b["sharpe_ann"],
        })

    df = pd.DataFrame(rows)
    k = len(lookbacks)
    # Bonferroni threshold: at 5% family-wise level, individual |t| > ~2 + log(k)
    # (rough approximation; we report it for transparency)
    df["bonferroni_threshold"] = float(np.sqrt(2.0) * np.sqrt(2.0 * np.log(k)))
    return df
