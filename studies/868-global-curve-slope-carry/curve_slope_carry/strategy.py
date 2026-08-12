"""Strategy + inference for Study 868 — Global Curve-Slope Carry.

The claim (fixed-income *roll + carry* / curve-carry — Koijen, Moskowitz, Pedersen &
Vrugt 2018, *"Carry"*): a **steep** yield curve pays a duration holder to hold. Across
bond markets a duration investor should prefer the **high-carry / steep-curve** sleeves
and avoid the **flat / low-carry** ones. We take a **cross-sectional** book: each month
rank the markets by a **carry proxy** and go **long the high-carry half, short the
low-carry half** (dollar-neutral, equal-weight within each leg), rebalanced monthly, then
cost it and benchmark against equal-weight buy-and-hold.

This is distinct from:

* [829-global-sovereign-bond-momentum](../../829-global-sovereign-bond-momentum/) — a
  **time-series momentum / trend** signal (each market signed by its own 12-1 trend). This
  study is **cross-sectional carry** (rank markets by a level-of-yield proxy, long high vs
  short low), a different signal measured over a long window, not a fast trend.
* [826-treasury-duration-bab](../../826-treasury-duration-bab/) — a **US-only**
  beta-neutral betting-against-beta book across the Treasury maturity ladder. This study
  is a **carry** sort **across US + international** markets, not a levered low-beta book.
* [380-curve-roll-down](../../380-curve-roll-down/) — a **single-curve** roll-down timer on
  one market. This study is the **cross-market** carry sort.
* [660-carry-everywhere](../../660-carry-everywhere/) — the **cross-asset-class** carry
  factor (FX/commodity/equity/bond pooled). This study isolates the **sovereign-bond
  curve-carry** sleeve on its own tradable ETF tape and costs it standalone.

Method:

* **Month-end total-return levels.** ``auto_adjust=True`` levels resampled to month-end.
* **Carry proxy.** ``realized_yield`` = the annualised mean monthly total return over a
  long trailing ``window`` (default 36 months) — a slow-moving estimate of the yield/roll
  a sleeve pays, with transient price trends averaged out. ``carry_signal`` divides that by
  the sleeve's published **effective duration** to give a *yield-to-duration* score (carry
  per unit of rate risk). A price-only proxy — the honesty caveat travels on the Signal axis.
* **Cross-sectional book.** Each month rank the markets by the carry proxy known at the
  close of ``t−1`` (one ``shift``): **long** the above-median (high-carry) markets, **short**
  the below-median (low-carry) markets, equal-weight within each leg, dollar-neutral. The
  book return is ``mean(long forward returns) − mean(short forward returns)``.
* **Inference.** Newey-West (HAC) *t* on the monthly book return; a one-sample *t*
  cross-checks; a column-permutation placebo breaks the carry → forward-return link; a
  costed backtest charges one-way turnover per rebalance leg plus borrow on the short book
  and reports the net Sharpe; every headline is set against equal-weight buy-and-hold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

ANN = 12            # months per year
DEFAULT_WINDOW = 36  # trailing months for the realized-yield carry proxy


# --------------------------------------------------------------------------- #
# Returns + the carry proxy
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple month-over-month total return per market (index=month, columns=ticker)."""
    return prices.sort_index().pct_change()


def realized_yield(prices: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Trailing realized-carry proxy: annualised mean monthly total return over ``window``.

    ``ry_t = mean(r_{t-window+1..t}) × 12`` per market. A **long** window is deliberate —
    transient price moves average out and the slow structural income/roll component
    dominates, which is the piece a curve-carry sort is meant to capture. The value on row
    ``t`` uses returns through ``t``; :func:`carry_book` applies a further one-month
    ``shift`` so a month-``t`` position is formed on information known at ``t−1``.
    """
    r = monthly_returns(prices)
    min_p = max(6, window // 2)
    return r.rolling(window, min_periods=min_p).mean() * ANN


def carry_signal(prices: pd.DataFrame, window: int = DEFAULT_WINDOW,
                 durations: dict | pd.Series | None = None) -> pd.DataFrame:
    """The carry score per market.

    Without ``durations`` this is the raw trailing realized yield (a level-of-carry rank).
    With ``durations`` (ticker → effective duration) it is **yield-to-duration**,
    ``realized_yield / duration`` — carry per unit of rate risk, the ratio the steep-curve
    argument rewards. Dividing by a per-column constant re-ranks the cross-section (it is
    not a no-op)."""
    y = realized_yield(prices, window)
    if durations is None:
        return y
    d = pd.Series(durations, dtype=float).reindex(y.columns)
    return y.div(d, axis=1)


# --------------------------------------------------------------------------- #
# The cross-sectional carry backtest -> monthly book return
# --------------------------------------------------------------------------- #
def carry_book(prices: pd.DataFrame, window: int = DEFAULT_WINDOW,
               durations: dict | pd.Series | None = None) -> pd.DataFrame:
    """Dollar-neutral cross-sectional carry book: long high-carry, short low-carry.

    On each month ``t`` the markets with a valid (lagged) carry proxy are split at their
    **cross-sectional median**: the above-median markets form the long leg (equal weight
    ``+1/n_long``), the below-median markets the short leg (equal weight ``−1/n_short``).
    The book return is ``mean(long r_t) − mean(short r_t)`` — dollar-neutral. Rows with
    fewer than two valid markets, or with an empty leg, are dropped. Returns a frame with
    columns ``[ret, long_ret, short_ret, n_active, n_long, n_short]`` indexed by month;
    the signed weight matrix is stored in ``.attrs['W']`` for the costed timer. Fully
    vectorised across dates.
    """
    r = monthly_returns(prices)
    sig = carry_signal(prices, window, durations).shift(1)   # known at close t-1, hold t
    S = sig.to_numpy(dtype=float)
    R = r.to_numpy(dtype=float)
    idx = prices.sort_index().index

    valid = ~np.isnan(S) & ~np.isnan(R)
    Sm = np.where(valid, S, np.nan)
    n_active = valid.sum(axis=1)

    with warnings.catch_warnings():                # all-NaN rows -> NaN median, guarded below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(Sm, axis=1)

    long_mask = valid & (Sm > med[:, None])
    short_mask = valid & (Sm < med[:, None])
    n_long = long_mask.sum(axis=1)
    n_short = short_mask.sum(axis=1)
    good = (n_active >= 2) & (n_long > 0) & (n_short > 0)

    Rz = np.where(valid, R, 0.0)
    nl = np.where(n_long > 0, n_long, 1)
    ns = np.where(n_short > 0, n_short, 1)
    long_ret = (long_mask * Rz).sum(axis=1) / nl
    short_ret = (short_mask * Rz).sum(axis=1) / ns
    book = long_ret - short_ret

    # signed dollar-neutral weights per asset (for turnover): +1/n_long long, -1/n_short short
    W = long_mask / nl[:, None] - short_mask / ns[:, None]

    out = pd.DataFrame(
        {"ret": book, "long_ret": long_ret, "short_ret": short_ret,
         "n_active": n_active, "n_long": n_long, "n_short": n_short},
        index=idx,
    )
    out = out[good]
    out.attrs["W"] = pd.DataFrame(W, index=idx, columns=prices.columns)[good]
    return out


def benchmark_returns(prices: pd.DataFrame) -> np.ndarray:
    """Naive equal-weight buy-and-hold monthly return across the available markets."""
    r = monthly_returns(prices)
    return r.mean(axis=1).dropna().to_numpy(dtype=float)


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
def _sharpe(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(ANN)) if s > 0 else float("nan")


def carry_stats(bt: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Headline statistics of a cross-sectional carry backtest frame (``bt.ret``)."""
    x = bt["ret"].to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    return {
        "n_months": int(len(x)),
        "mean_bps": float(np.mean(x) * 1e4) if len(x) else float("nan"),
        "ann_pct": float(np.mean(x) * ANN * 100) if len(x) else float("nan"),
        "vol_ann_pct": float(np.std(x, ddof=1) * np.sqrt(ANN) * 100) if len(x) > 1 else float("nan"),
        "sharpe": _sharpe(x),
        "t_nw": newey_west_t(x, nw_lags),
        "t_1s": one_sample_t(x),
        "hit_rate": float(np.mean(x > 0)) if len(x) else float("nan"),
        "long_bps": float(np.nanmean(bt["long_ret"].to_numpy()) * 1e4) if len(bt) else float("nan"),
        "short_bps": float(np.nanmean(bt["short_ret"].to_numpy()) * 1e4) if len(bt) else float("nan"),
    }


def benchmark_stats(prices: pd.DataFrame, nw_lags: int = 6) -> dict:
    """Naive equal-weight buy-and-hold headline — the yardstick every carry book must beat."""
    bh = benchmark_returns(prices)
    return {
        "n_months": int(len(bh)),
        "mean_bps": float(np.mean(bh) * 1e4) if len(bh) else float("nan"),
        "ann_pct": float(np.mean(bh) * ANN * 100) if len(bh) else float("nan"),
        "sharpe": _sharpe(bh),
        "t_nw": newey_west_t(bh, nw_lags),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the carry sort real, or a lucky alignment of the ranks?
# --------------------------------------------------------------------------- #
def placebo_pvalue(prices: pd.DataFrame, window: int = DEFAULT_WINDOW,
                   durations: dict | pd.Series | None = None, n_perm: int = 2000,
                   seed: int = 868) -> dict:
    """Column-permutation placebo for the mean carry-book return.

    Keep the carry-rank long/short weights fixed but **permute which market's forward
    return feeds each rank** every draw — this breaks the carry → forward-return link while
    preserving each month's cross-sectional return distribution. p = share of permuted
    worlds whose mean book return is >= the observed (right-tail test — the claim predicts a
    positive high-minus-low carry spread)."""
    bt = carry_book(prices, window, durations)
    W = bt.attrs["W"].to_numpy(dtype=float)         # (m, ncol) signed weights on good rows
    r = monthly_returns(prices).reindex(bt.index)
    R = r.to_numpy(dtype=float)
    R = np.where(np.isnan(R), 0.0, R)
    m, ncol = R.shape

    obs = float(np.mean((W * R).sum(axis=1)))
    rng = np.random.default_rng(seed)
    row_ar = np.arange(m)[:, None]
    draws = np.empty(n_perm)
    for j in range(n_perm):
        perm = rng.permutation(ncol)
        Rp = R[row_ar, perm[None, :]]
        draws[j] = float(np.mean((W * Rp).sum(axis=1)))
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
def timer_stats(prices: pd.DataFrame, window: int = DEFAULT_WINDOW,
                durations: dict | pd.Series | None = None, cost_bps: float = 10.0,
                borrow_bps_yr: float = 75.0) -> dict:
    """Cost the cross-sectional carry book.

    Turnover: at each month-end the dollar-neutral weight vector rotates as carry ranks
    change; we charge a one-way ``cost_bps`` on the traded fraction of NAV, i.e.
    ``cost_bps × Σ_i |Δw_i|`` per rebalance. The short leg pays ``borrow_bps_yr`` annualised
    on the short-weighted NAV (≈ 1 for a dollar-neutral book). Returns gross/net monthly
    means, the net Sharpe, and a net *t*.
    """
    bt = carry_book(prices, window, durations)
    if bt.empty:
        return {"n_months": 0, "gross_bps": float("nan"), "net_bps": float("nan")}
    gross = bt["ret"].to_numpy(dtype=float)
    W = bt.attrs["W"].to_numpy(dtype=float)

    dW = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1]))))
    turnover = dW.sum(axis=1)
    trade_cost = turnover * (cost_bps / 1e4)

    short_w = np.where(W < 0, -W, 0.0).sum(axis=1)          # ~1.0 for dollar-neutral book
    borrow_cost = short_w * (borrow_bps_yr / 1e4) / ANN

    net = gross - trade_cost - borrow_cost
    return {
        "n_months": int(len(net)),
        "gross_bps": float(np.nanmean(gross) * 1e4),
        "net_bps": float(np.nanmean(net) * 1e4),
        "cost_bps_per_mo": float(np.nanmean(trade_cost + borrow_cost) * 1e4),
        "avg_turnover": float(np.nanmean(turnover)),
        "gross_sharpe": _sharpe(gross),
        "net_sharpe": _sharpe(net),
        "ann_net_pct": float(np.nanmean(net) * ANN * 100),
        "t_net": one_sample_t(net),
        "t_net_nw": newey_west_t(net, 6),
    }


# --------------------------------------------------------------------------- #
# Robustness — sub-period cut + window sweep
# --------------------------------------------------------------------------- #
def subperiod_sweep(prices: pd.DataFrame, edges, window: int = DEFAULT_WINDOW,
                    durations: dict | pd.Series | None = None) -> pd.DataFrame:
    """Mean book return + NW t within date sub-periods (``edges`` = (label, lo, hi))."""
    bt = carry_book(prices, window, durations)
    rows = []
    for lab, lo, hi in edges:
        sub = bt[(bt.index >= pd.Timestamp(lo)) & (bt.index < pd.Timestamp(hi))]
        s = carry_stats(sub)
        rows.append((lab, s["mean_bps"], s["t_nw"], s["sharpe"], s["n_months"]))
    return pd.DataFrame(rows, columns=["period", "mean_bps", "t_nw", "sharpe", "n"]).set_index("period")


def window_sweep(prices: pd.DataFrame, windows, durations: dict | pd.Series | None = None) -> pd.DataFrame:
    """Mean book return + NW t across carry-formation windows (robustness in the proxy)."""
    rows = []
    for w in windows:
        s = carry_stats(carry_book(prices, window=w, durations=durations))
        rows.append((w, s["mean_bps"], s["t_nw"], s["sharpe"], s["n_months"]))
    return pd.DataFrame(rows, columns=["window", "mean_bps", "t_nw", "sharpe", "n"]).set_index("window")


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, window: int = DEFAULT_WINDOW,
                     durations: dict | pd.Series | None = None) -> dict:
    """Run the headline carry stats on a synthetic monthly price panel."""
    s = carry_stats(carry_book(prices, window, durations))
    return {"mean_bps": s["mean_bps"], "t_nw": s["t_nw"],
            "sharpe": s["sharpe"], "n_months": s["n_months"]}


def synthetic_mean_t(data_mod, edge: float, n_seeds: int = 25, base_seed: int = 868,
                     window: int = DEFAULT_WINDOW) -> dict:
    """Average the carry NW *t* and Sharpe over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the statistic over >= 20 seeds so no
    single lucky RNG seed can manufacture significance. Synthetic uses the raw realized-yield
    carry sort (no durations) — it proves the ranking / long-short / cost machinery is
    unbiased, never a real-tape stamp.
    """
    ts, sh = [], []
    for s in range(base_seed, base_seed + n_seeds):
        panel = data_mod.synthetic_panel(edge=edge, seed=s)
        d = synthetic_detect(panel, window=window, durations=None)
        ts.append(d["t_nw"]); sh.append(d["sharpe"])
    return {"mean_t": float(np.nanmean(ts)), "mean_sharpe": float(np.nanmean(sh)),
            "fire_frac": float(np.mean(np.abs(ts) >= 2.0))}
