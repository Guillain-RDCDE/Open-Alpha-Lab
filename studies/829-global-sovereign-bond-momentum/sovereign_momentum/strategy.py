"""Strategy + inference for Study 829 — Global Sovereign-Bond Momentum.

The claim (Moskowitz, Ooi & Pedersen 2012, *"Time Series Momentum"*, applied to global
government bonds): a bond market's **own trailing trend predicts its own next return**.
Measure each sovereign-bond ETF's **12-minus-1-month total return** — the cumulative
total return over the past twelve months, *excluding the most recent month* — and take a
position equal to the **sign** of that trend: long the up-trending markets, short the
down-trending ones (long-short), or own only the up-trending ones (long-only timing).

This is distinct from:

* [795-corporate-bond-momentum](../../795-corporate-bond-momentum/) — a **cross-sectional**
  momentum sort within a universe of **corporate** bonds (rank and long the past winners
  vs the past losers). This study is **time-series** (each market vs its *own* sign, not a
  cross-sectional rank) on **foreign / global sovereigns**.
* [518-time-series-momentum](../../518-time-series-momentum/) — the **general** TSMOM
  factor across broad asset classes (equity indices, commodities, currencies, bonds
  pooled). This study isolates the **global-sovereign-bond** sleeve on its own tradable
  ETF tape and costs it as a standalone strategy.
* [662-em-local-bonds](../../662-em-local-bonds/) — an **EM-local-currency carry / level**
  study, not a **trend** signal.
* [247-bond-seasonality](../../247-bond-seasonality/) — a **calendar** effect in bonds, not
  a **momentum / trend** signal.

Method:

* **Month-end total-return levels.** ``auto_adjust=True`` levels resampled to month-end.
* **12-1 momentum.** For each market, ``mom_t = level_{t-1} / level_{t-12} − 1`` — the
  twelve-month total return ending one month ago (the most recent month is skipped, the
  standard 12-1 convention). Vectorised as a ratio of shifted price frames.
* **Point-in-time positions.** The sign of the momentum **known at the close of month
  ``t−1``** (one ``shift``) sets the position held over month ``t``. Long-short: ``+1`` on
  positive trend, ``−1`` on negative. Long-only: ``+1`` on positive, ``0`` otherwise.
* **Inference.** Newey-West (HAC) *t* on the monthly strategy return; a one-sample *t*
  cross-checks; a block-rotation placebo breaks the trend→forward link; a costed backtest
  charges one-way turnover per rebalance leg plus borrow on the short book and reports the
  net Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Returns + the 12-1 momentum signal
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple month-over-month total return per market (index=month, columns=ticker)."""
    return prices.sort_index().pct_change()


def momentum_12_1(prices: pd.DataFrame, lookback: int = 12, gap: int = 1) -> pd.DataFrame:
    """The 12-minus-1-month total-return momentum, per market.

    ``mom_t = level_{t-gap} / level_{t-lookback} − 1``. With the defaults this is the
    twelve-month total return **ending one month ago** — the most recent month (``gap``)
    is skipped, the standard 12-1 momentum. The value on row ``t`` uses only prices
    through ``t−gap``; :func:`tsmom_returns` applies a further one-month execution
    ``shift`` so a month-``t`` position is formed on information known at ``t−1``.
    """
    p = prices.sort_index()
    return p.shift(gap) / p.shift(lookback) - 1.0


def positions(prices: pd.DataFrame, lookback: int = 12, gap: int = 1,
              long_only: bool = False) -> pd.DataFrame:
    """Point-in-time positions: sign of the (execution-lagged) 12-1 momentum.

    ``pos_t`` is held over month ``t`` and is formed from the momentum known at the close
    of month ``t−1`` (one ``shift`` — the single documented execution lag). Long-short
    returns ``+1 / −1`` on the sign of the trend; long-only returns ``+1 / 0``.
    """
    mom = momentum_12_1(prices, lookback, gap).shift(1)  # known at close t-1, hold t
    sign = np.sign(mom)
    if long_only:
        sign = (mom > 0).astype(float)
        sign = sign.where(mom.notna())
    return sign


# --------------------------------------------------------------------------- #
# The time-series-momentum backtest -> monthly strategy return
# --------------------------------------------------------------------------- #
def tsmom_returns(prices: pd.DataFrame, lookback: int = 12, gap: int = 1,
                  long_only: bool = False) -> pd.DataFrame:
    """Equal-weight time-series-momentum monthly return across the available markets.

    On each month ``t`` every market that has a valid (lagged) 12-1 momentum contributes
    ``pos_{i,t} · r_{i,t}`` where ``pos`` is the sign of its trend known at ``t−1`` and
    ``r`` is its month-``t`` total return. The strategy return is the **equal-weight mean
    across the active markets** (``nanmean`` over columns, so a market that is not yet
    listed simply does not participate). Returns a frame with columns ``[ret, n_active,
    n_long, n_short]`` indexed by month.
    """
    r = monthly_returns(prices)
    pos = positions(prices, lookback, gap, long_only)
    P = pos.to_numpy(dtype=float)
    R = r.to_numpy(dtype=float)
    active = ~np.isnan(P) & ~np.isnan(R)
    n_active = active.sum(axis=1)
    contrib = np.where(active, P * R, 0.0)
    strat = np.divide(contrib.sum(axis=1), n_active,
                      out=np.full(len(n_active), np.nan), where=n_active > 0)
    n_long = ((P > 0) & active).sum(axis=1)
    n_short = ((P < 0) & active).sum(axis=1)
    out = pd.DataFrame(
        {"ret": strat, "n_active": n_active, "n_long": n_long, "n_short": n_short},
        index=prices.sort_index().index,
    )
    return out[out["n_active"] > 0]


# --------------------------------------------------------------------------- #
# Inference primitives (shared house code — see study 803)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
ANN = 12  # months per year


def _sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(ANN)) if s > 0 else float("nan")


def tsmom_stats(bt: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Headline statistics of a time-series-momentum backtest frame (``bt.ret``)."""
    x = bt["ret"].to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    return {
        "n_months": int(len(x)),
        "mean_bps": float(np.mean(x) * 1e4),
        "ann_pct": float(np.mean(x) * ANN * 100),
        "vol_ann_pct": float(np.std(x, ddof=1) * np.sqrt(ANN) * 100) if len(x) > 1 else float("nan"),
        "sharpe": _sharpe(x),
        "t_nw": newey_west_t(x, nw_lags),
        "t_1s": one_sample_t(x),
        "hit_rate": float(np.mean(x > 0)) if len(x) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the trend real, or a lucky alignment given the return series?
# --------------------------------------------------------------------------- #
def placebo_pvalue(prices: pd.DataFrame, lookback: int = 12, gap: int = 1,
                   long_only: bool = False, n_perm: int = 2000,
                   min_shift: int = 6, seed: int = 829) -> dict:
    """Block-rotation placebo for the mean strategy return.

    Keep the point-in-time positions but **circularly rotate the return matrix in time**
    by a random shift (>= ``min_shift`` months) before scoring — this breaks the
    trend→forward-return link while preserving each market's own return autocorrelation
    and cross-sectional structure. p = share of rotated worlds whose mean strategy return
    is >= the observed (right-tail test on the long-trend book)."""
    r = monthly_returns(prices)
    pos = positions(prices, lookback, gap, long_only)
    P = pos.to_numpy(dtype=float)
    R = r.to_numpy(dtype=float)
    active = ~np.isnan(P) & ~np.isnan(R)
    Pm = np.where(active, P, 0.0)
    cnt = active.sum(axis=1)
    valid = cnt > 0

    def _score(Rmat: np.ndarray) -> float:
        contrib = np.where(active, Pm * np.where(np.isnan(Rmat), 0.0, Rmat), 0.0)
        row = contrib.sum(axis=1)
        strat = np.divide(row, cnt, out=np.full_like(row, np.nan), where=valid)
        return float(np.nanmean(strat))

    obs = _score(R)
    rng = np.random.default_rng(seed)
    T = R.shape[0]
    draws = np.empty(n_perm)
    for j in range(n_perm):
        sh = int(rng.integers(min_shift, T - min_shift))
        draws[j] = _score(np.roll(R, sh, axis=0))
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(np.mean(draws) * 1e4),
        "placebo_sd_bps": float(np.std(draws, ddof=1) * 1e4),
        "p_value": float((draws >= obs).mean()),
        "n_draws": int(n_perm),
        "draws_bps": draws * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed backtest
# --------------------------------------------------------------------------- #
def timer_stats(prices: pd.DataFrame, lookback: int = 12, gap: int = 1,
                long_only: bool = False, cost_bps: float = 10.0,
                borrow_bps_yr: float = 75.0) -> dict:
    """Cost the time-series-momentum book.

    Turnover: at each month-end the position vector rotates as trends flip; we charge a
    one-way ``cost_bps`` on the traded fraction of NAV, i.e. ``cost_bps × Σ_i |Δpos_i| /
    n_active`` per rebalance (each market carries ``1/n_active`` of the book). The short
    leg pays ``borrow_bps_yr`` annualised on the short-weighted NAV. Returns gross/net
    monthly means, the net Sharpe, and a net *t*.
    """
    r = monthly_returns(prices)
    pos = positions(prices, lookback, gap, long_only)
    P = pos.to_numpy(dtype=float)
    R = r.to_numpy(dtype=float)
    active = ~np.isnan(P) & ~np.isnan(R)
    cnt = active.sum(axis=1).astype(float)
    valid = cnt > 0

    Pw = np.where(active, P, 0.0)
    contrib = Pw * np.where(np.isnan(R), 0.0, R)
    gross = np.divide(contrib.sum(axis=1), cnt, out=np.full(len(cnt), np.nan), where=valid)

    # turnover: change in the (equal-weight) position weights month over month
    W = np.divide(Pw, cnt[:, None], out=np.zeros_like(Pw), where=valid[:, None])
    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1]))))
    turnover = dW.sum(axis=1)
    trade_cost = turnover * (cost_bps / 1e4)

    short_w = np.divide(np.where(Pw < 0, -Pw, 0.0).sum(axis=1), cnt,
                        out=np.full(len(cnt), np.nan), where=valid)
    borrow_cost = np.where(np.isnan(short_w), 0.0, short_w) * (borrow_bps_yr / 1e4) / ANN

    net = gross - trade_cost - borrow_cost
    net = net[valid]
    gross_v = gross[valid]
    return {
        "n_months": int(valid.sum()),
        "gross_bps": float(np.nanmean(gross_v) * 1e4),
        "net_bps": float(np.nanmean(net) * 1e4),
        "cost_bps_per_mo": float(np.nanmean((trade_cost + borrow_cost)[valid]) * 1e4),
        "avg_turnover": float(np.nanmean(turnover[valid])),
        "gross_sharpe": _sharpe(gross_v),
        "net_sharpe": _sharpe(net),
        "ann_net_pct": float(np.nanmean(net) * ANN * 100),
        "t_net": one_sample_t(net),
        "t_net_nw": newey_west_t(net, 6),
    }


# --------------------------------------------------------------------------- #
# Robustness — sub-period cut
# --------------------------------------------------------------------------- #
def subperiod_sweep(prices: pd.DataFrame, edges, lookback: int = 12, gap: int = 1,
                    long_only: bool = False) -> pd.DataFrame:
    """Mean strategy return + NW t within date sub-periods (``edges`` = (label, lo, hi))."""
    bt = tsmom_returns(prices, lookback, gap, long_only)
    rows = []
    for lab, lo, hi in edges:
        sub = bt[(bt.index >= pd.Timestamp(lo)) & (bt.index < pd.Timestamp(hi))]
        s = tsmom_stats(sub)
        rows.append((lab, s["mean_bps"], s["t_nw"], s["sharpe"], s["n_months"]))
    return pd.DataFrame(rows, columns=["period", "mean_bps", "t_nw", "sharpe", "n"]).set_index("period")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, lookback: int = 12, gap: int = 1,
                     long_only: bool = False) -> dict:
    """Run the headline TSMOM stats on a synthetic monthly price panel."""
    bt = tsmom_returns(prices, lookback, gap, long_only)
    s = tsmom_stats(bt)
    return {"mean_bps": s["mean_bps"], "t_nw": s["t_nw"],
            "sharpe": s["sharpe"], "n_months": s["n_months"]}


def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 25, base_seed: int = 829,
                     long_only: bool = False) -> dict:
    """Average the TSMOM NW *t* and Sharpe over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over >= 20 seeds so
    no single lucky RNG seed can manufacture significance.
    """
    ts, sh = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel = data_mod.synthetic_panel(edge=edge, seed=s)
        d = synthetic_detect(panel, long_only=long_only)
        ts.append(d["t_nw"]); sh.append(d["sharpe"])
    return {"mean_t": float(np.nanmean(ts)), "mean_sharpe": float(np.nanmean(sh)),
            "fire_frac": float(np.mean(np.abs(ts) >= 2.0))}
