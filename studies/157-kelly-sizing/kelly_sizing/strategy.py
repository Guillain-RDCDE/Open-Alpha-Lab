"""Strategy and honest controls — Study 157 (Kelly-Sizing).

The folk recipe: size positions using the **Kelly criterion** — the growth-optimal
leverage ``f* = mu / sigma^2`` (continuous) or ``f* = (p*(b+1) - 1) / b`` (binary)
— and compare it against simpler fixed-fractional sizes (full exposure, 50% exposure)
on the *same* underlying return stream.

Four sizing strategies share one engine (:func:`run_backtest`):

- **Full-Kelly** — leverage = estimated Kelly fraction from the in-sample window.
  If estimated < 0, stays at 0 (no short). Caps at 2.0x to avoid blow-up on
  estimation error.
- **Half-Kelly** — leverage = Kelly / 2. The practitioner's favourite: half the
  geometric growth, a quarter the variance; dramatically smaller drawdowns.
- **Full-exposure** — leverage = 1.0 always (buy-and-hold the asset).
- **Half-exposure** — leverage = 0.5 always.

The **control / fair baseline** is full buy-and-hold (full-exposure), which is what
a passive investor does. Kelly's claim is that *growth-optimal sizing beats
full-exposure in geometric return while managing drawdowns*. We test that claim with
a strict walk-forward to avoid look-ahead: the Kelly fraction is estimated on the
``train_years`` preceding each rebalance; positions are held for ``rebal_freq`` days.

No look-ahead: Kelly estimates use returns up to and including day *t*; positions
are set from *t+1* onward. Broker costs (``cost_bps`` per round-trip on leverage
changes) are charged on every rebalance.

Key references:
    Kelly (1956), Thorp (1971 / 2006), MacLean, Thorp & Ziemba (2010).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MAX_KELLY_CAP = 2.0  # hard cap: never lever more than 2x even if Kelly says so


# ---------------------------------------------------------------------------
# Kelly fraction estimators
# ---------------------------------------------------------------------------
def kelly_fraction_continuous(
    returns: pd.Series,
    clip_negative: bool = True,
) -> float:
    """Continuous-time Kelly fraction: mu / sigma^2.

    For a return stream with daily mean ``mu`` and daily variance ``sigma^2``,
    the growth-optimal leverage is ``f* = mu / sigma^2``. This is the log-return
    maximiser for a geometric Brownian motion. The estimate is noisy — this is the
    core problem of Kelly sizing in practice (Estimation Risk).

    With ``clip_negative=True`` (default) negative fractions are floored at 0
    (no short positions). The result is also capped at ``MAX_KELLY_CAP`` (2.0x).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return 0.0
    mu = r.mean()
    var = r.var(ddof=1)
    if var <= 0:
        return 0.0
    f = mu / var
    if clip_negative:
        f = max(f, 0.0)
    return float(min(f, MAX_KELLY_CAP))


def kelly_fraction_half(returns: pd.Series) -> float:
    """Half-Kelly: full Kelly divided by 2 (the 'fractional Kelly' family)."""
    return kelly_fraction_continuous(returns) / 2.0


# ---------------------------------------------------------------------------
# Walk-forward backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    returns: pd.Series,
    sizing_fn,
    train_years: float = 5.0,
    rebal_freq: int = 21,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Walk-forward backtest for a leverage sizing rule.

    The ``sizing_fn`` maps a Series of past returns to a scalar leverage. It is
    called at every rebalance date using only returns strictly *before* that date
    (no look-ahead). The leverage is clipped to [0, MAX_KELLY_CAP].

    Each rebalance period the strategy:
    1. Estimates leverage ``f`` from the lookback window (length ``train_years * 252``).
    2. Applies daily returns ``r_t`` as ``port_ret_t = f * r_t`` during the holding
       period (i.e., simple leverage of a single asset — no compounding within a period).
    3. Charges cost_bps / 10000 on the absolute change in leverage at each rebalance.

    The result is a daily portfolio-return Series (``ret_port``) plus a column
    for the daily leverage ``leverage_t`` used each day.

    Columns: ``date``, ``ret_port``, ``ret_asset``, ``leverage``.
    """
    r = returns.dropna().copy()
    train_n = int(round(train_years * TRADING_DAYS))
    n = len(r)

    dates = r.index
    port_rets = np.zeros(n)
    leverages = np.zeros(n)
    prev_lev = np.nan

    # Rebalance dates: every rebal_freq steps, after the training window is full.
    for i in range(n):
        if i < train_n:
            # Still in the training window — no position
            leverages[i] = 0.0
            port_rets[i] = 0.0
            prev_lev = 0.0
            continue
        # At each rebalance date (or first eligible date), re-estimate leverage.
        if (i - train_n) % rebal_freq == 0:
            past_ret = r.iloc[i - train_n: i]
            new_lev = float(np.clip(sizing_fn(past_ret), 0.0, MAX_KELLY_CAP))
            # Charge cost on leverage change (round-trip on |delta f| * notional)
            if prev_lev is not None and np.isfinite(prev_lev):
                cost = abs(new_lev - prev_lev) * cost_bps * 1e-4
            else:
                cost = 0.0
            lev = new_lev
            prev_lev = lev
        # Daily portfolio return = leverage * asset return
        leverages[i] = lev if "lev" in dir() else 0.0
        port_rets[i] = (lev if "lev" in dir() else 0.0) * r.iloc[i] - (cost if (i - train_n) % rebal_freq == 0 else 0.0)

    result = pd.DataFrame(
        {"ret_port": port_rets, "ret_asset": r.values, "leverage": leverages},
        index=dates,
    )
    return result


def run_all_strategies(
    returns: pd.Series,
    train_years: float = 5.0,
    rebal_freq: int = 21,
    cost_bps: float = 1.0,
) -> dict[str, pd.DataFrame]:
    """Run full-Kelly, half-Kelly, full-exposure, and half-exposure on the same tape.

    Returns a dict of strategy-name -> backtest DataFrame (columns: ret_port,
    ret_asset, leverage). All four strategies see the same returns and incur the
    same rebalancing cost structure.
    """
    # Fixed-fraction sizing functions (lambda ignores the returns argument)
    full_exp = lambda r: 1.0
    half_exp = lambda r: 0.5

    return {
        "full-Kelly": run_backtest(returns, kelly_fraction_continuous,
                                   train_years, rebal_freq, cost_bps),
        "half-Kelly": run_backtest(returns, kelly_fraction_half,
                                   train_years, rebal_freq, cost_bps),
        "full-exposure": run_backtest(returns, full_exp,
                                      train_years, rebal_freq, cost_bps),
        "half-exposure": run_backtest(returns, half_exp,
                                      train_years, rebal_freq, cost_bps),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def portfolio_stats(bt: pd.DataFrame) -> dict:
    """Headline statistics for one backtest frame.

    Returns annualised geometric return, annualised volatility, Sharpe ratio,
    maximum drawdown, and a HAC t-stat on the daily mean excess return over zero
    (using Newey-West with the standard lag rule).
    """
    r = bt["ret_port"].astype(float).dropna()
    # Only look at the period where we actually had a position (leverage > 0)
    r_active = r[bt["leverage"] > 0]
    if len(r_active) < 2:
        return {k: float("nan") for k in (
            "n_days", "ann_geo_ret", "ann_vol", "sharpe", "max_drawdown",
            "mean_lev", "tstat"
        )}
    n = len(r_active)
    # Geometric (log) return annualised
    log_r = np.log1p(r_active)
    ann_geo = float(np.exp(log_r.mean() * TRADING_DAYS) - 1.0)
    ann_vol = float(r_active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(ann_geo / ann_vol) if ann_vol > 0 else float("nan")

    # Max drawdown on the cumulative wealth curve (active period only)
    cum = (1.0 + r_active).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min())

    # HAC t-stat (Newey-West)
    x = r_active.to_numpy()
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = float(np.sqrt(max(lrv, 0.0) / n))
    tstat = float(mu / se) if se > 0 else float("nan")

    mean_lev = float(bt.loc[bt["leverage"] > 0, "leverage"].mean())

    return {
        "n_days": int(n),
        "ann_geo_ret": ann_geo,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "mean_lev": mean_lev,
        "tstat": tstat,
    }


def compare_strategies(returns: pd.Series, **kwargs) -> pd.DataFrame:
    """Run all four strategies and return a comparison table of headline stats.

    Extra kwargs are forwarded to :func:`run_all_strategies`.
    """
    strategies = run_all_strategies(returns, **kwargs)
    rows = {name: portfolio_stats(bt) for name, bt in strategies.items()}
    return pd.DataFrame(rows).T


def kelly_sensitivity(
    returns: pd.Series,
    train_years: float = 5.0,
) -> pd.DataFrame:
    """Sweep the Kelly fraction from 0 to 2x and return growth / vol / drawdown.

    Computes *in-sample* metrics (no walk-forward) across a grid of fixed
    leverage fractions. Used to visualise the 'Kelly parabola' — the famous
    inverted-U of geometric growth vs leverage, peaking at the true Kelly
    fraction and turning negative above 2x Kelly (the ruin zone).

    This is a *theoretical* curve from the real tape's empirical moments,
    not a backtest.
    """
    r = np.asarray(returns.dropna(), dtype=float)
    mu = r.mean()
    var = r.var(ddof=1)

    fractions = np.linspace(0.0, 2.5, 51)
    rows = []
    for f in fractions:
        port = f * r
        log_g = np.log1p(port)
        geo = float(np.exp(log_g.mean() * TRADING_DAYS) - 1.0)
        vol = float(port.std(ddof=1) * np.sqrt(TRADING_DAYS))
        cum = np.cumprod(1.0 + port)
        roll_max = np.maximum.accumulate(cum)
        dd = float(np.min(cum / roll_max - 1.0))
        rows.append({
            "fraction": float(f),
            "ann_geo_ret": geo,
            "ann_vol": vol,
            "max_drawdown": dd,
        })
    return pd.DataFrame(rows)


def kelly_estimation_noise(
    returns: pd.Series,
    train_years_grid: tuple = (1, 2, 5, 10, 20),
    seed: int = 157,
) -> pd.DataFrame:
    """How noisy is the Kelly estimate for different lookback windows?

    Bootstraps the Kelly fraction from sub-samples of length ``train_years * 252``
    to show the estimation-error distribution. A wider dispersion means the
    Kelly fraction is a guess, not a precision instrument.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns.dropna(), dtype=float)
    rows = []
    for years in train_years_grid:
        n = int(years * TRADING_DAYS)
        if n > len(r):
            continue
        estimates = []
        for _ in range(500):
            idx = rng.integers(0, len(r), n)
            sub = pd.Series(r[idx])
            estimates.append(kelly_fraction_continuous(sub))
        arr = np.array(estimates)
        rows.append({
            "train_years": int(years),
            "n_obs": n,
            "mean_kelly": float(arr.mean()),
            "std_kelly": float(arr.std()),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        })
    return pd.DataFrame(rows)
