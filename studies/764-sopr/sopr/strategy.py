"""Strategy and honest controls -- Study 764 (SOPR).

The folk thesis (Renato Shirakashi, 2019; Glassnode): SOPR is a *momentum /
regime* gauge of Bitcoin.  When SOPR is above 1 the coins changing hands are, on
aggregate, being sold *in profit* -- read as a healthy uptrend (be long).  When
SOPR is below 1 they are sold *at a loss* -- read as capitulation / downtrend
(step aside).  The chart-lore adds that in bull markets SOPR bounces off 1 as
support, in bear markets it caps at 1 as resistance.  We formalise it three ways
and pit each against the only honest benchmark for a single trending asset:
**buy-and-hold BTC**.

1. **Predictive regression.** Regress next-month BTC log-return on this month's
   SOPR *stretch* ``log(sopr)`` (centred at 1).  A genuine momentum indicator
   needs a *positive* slope with a HAC *t* whose magnitude clears 2.  We also run
   a *horse race* that adds price's own one-month momentum as a control.

2. **Regime timing rule.** Be long BTC for month t+1 whenever this month's SOPR
   is at or above the threshold (``sopr >= thresh``, default the folk value 1.0),
   step to cash otherwise.  Compare net-of-cost equity to **buy-and-hold BTC**.

3. **Band states.** Map SOPR to {greed, neutral, capitulation} and report the
   average *next-month* return earned from each state -- the cleanest read on
   whether SOPR > 1 actually marks the good months and SOPR < 1 the bad ones.

We also run a **placebo**: shuffle the SOPR signal in time (many seeds) and show
the real rule's edge over buy-and-hold sits inside the placebo distribution.

**No look-ahead:** the signal at month-end t uses only data through t and is
applied to the return *earned during t+1* (one-month execution lag, baked into
``forward_returns``).

**Single-survivor caveat (named on the Signal axis):** BTC went up ~150x over
the aligned sample and SOPR is computed from BTC's own on-chain spending; the
regime rule is fitted to a handful of cycle turns (n is tiny).  The decisive
test is whether the rule beats buy-and-hold net of costs (it does not).

**Costs:** one-way ``cost_bps`` charged on every change in position
(|delta position|) on the NAV.  The timing rule here is long/flat (no borrow);
returns are price-only (BTC pays no yield), labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def sopr_stretch(df: pd.DataFrame, center: float = 1.0) -> pd.Series:
    """Log SOPR stretch ``log(sopr / center)``, known at month-end t.

    ``center`` is the profit/loss boundary (SOPR = 1).  A positive stretch =
    coins moving in profit (folk "greed / bullish"); negative = coins moving at
    a loss (folk "capitulation").
    """
    z = np.log(df["sopr"].astype(float) / float(center))
    z.name = "sopr_stretch"
    return z


def price_momentum(df: pd.DataFrame, lookback: int = 1) -> pd.Series:
    """Log-growth of BTC price over ``lookback`` months (the control regressor)."""
    log_p = np.log(df["price"].astype(float))
    mom = log_p.diff(lookback)
    mom.name = "price_mom"
    return mom


def forward_returns(df: pd.DataFrame) -> pd.Series:
    """Simple BTC return earned *during month t+1*, aligned to the signal at t.

    ``forward_returns.loc[t]`` is the return from close(t) to close(t+1).  This
    is what a position taken at month-end t (one-month execution lag) earns.
    """
    ret = df["price"].astype(float).pct_change().shift(-1)
    ret.name = "fwd_ret"
    return ret


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def predictive_regression(
    df: pd.DataFrame,
    center: float = 1.0,
    add_price_control: bool = False,
) -> dict:
    """Regress next-month BTC log-return on SOPR stretch (HAC t-stats).

    Model::

        r_{t+1} = a + b * sopr_stretch_t  [+ c * price_mom_t] + e_{t+1}

    A momentum indicator predicts ``b > 0`` (SOPR > 1 -> higher forward return).
    Returns a dict with the slope(s), HAC *t*-stats, R^2 and n.  When
    ``add_price_control`` is True, BTC's own one-month momentum is added so we
    can see whether SOPR adds anything beyond price trend.
    """
    z = sopr_stretch(df, center=center)
    fwd = forward_returns(df)
    fwd_log = np.log1p(fwd)

    cols = {"sopr_stretch": z}
    if add_price_control:
        cols["price_mom"] = price_momentum(df, lookback=1)
    X = pd.DataFrame(cols)
    data = pd.concat([fwd_log.rename("y"), X], axis=1).dropna()
    if len(data) < 12:
        return {"n": int(len(data)), "slope_sopr": np.nan, "t_sopr": np.nan,
                "slope_price": np.nan, "t_price": np.nan, "r2": np.nan}

    y = data["y"].to_numpy()
    reg_cols = list(X.columns)
    Xmat = np.column_stack([np.ones(len(data))] + [data[c].to_numpy() for c in reg_cols])
    n, k = Xmat.shape

    xtx_inv = np.linalg.pinv(Xmat.T @ Xmat)
    beta = xtx_inv @ Xmat.T @ y
    resid = y - Xmat @ beta

    # Newey-West HAC covariance
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (Xmat * resid[:, None]).T @ (Xmat * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = Xmat * resid[:, None]
        gamma = u[L:].T @ u[:-L]
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = {"n": int(n), "r2": float(r2),
           "slope_sopr": float(beta[1]),
           "t_sopr": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
           "slope_price": np.nan, "t_price": np.nan}
    if add_price_control:
        out["slope_price"] = float(beta[2])
        out["t_price"] = float(beta[2] / se[2]) if se[2] > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Band classification
# ---------------------------------------------------------------------------
def sopr_state(
    df: pd.DataFrame,
    high: float = 1.02,
    low: float = 0.98,
) -> pd.Series:
    """Map SOPR to a 3-state band: +1 greed, 0 neutral, -1 capitulation.

    ``sopr >= high`` -> greed (+1, coins deep in profit); ``sopr <= low`` ->
    capitulation (-1, coins deep in loss); otherwise neutral (0).  Known at
    month-end t.
    """
    m = df["sopr"].astype(float)
    state = pd.Series(0, index=df.index, dtype=float)
    state[m >= high] = 1.0
    state[m <= low] = -1.0
    state.name = "sopr_state"
    return state


def state_forward_stats(
    df: pd.DataFrame,
    high: float = 1.02,
    low: float = 0.98,
) -> pd.DataFrame:
    """Average *next-month* BTC return conditioned on the SOPR band.

    Returns a DataFrame indexed by band label with columns ``mean`` (mean
    forward simple return), ``n`` (count), ``hit`` (fraction positive).  This is
    the cleanest read on whether 'greed' (SOPR > 1) months precede gains and
    'capitulation' (SOPR < 1) months precede losses -- or the reverse.
    """
    state = sopr_state(df, high=high, low=low)
    fwd = forward_returns(df)
    d = pd.concat([state.rename("state"), fwd.rename("fwd")], axis=1).dropna()
    labels = {1.0: "greed", 0.0: "neutral", -1.0: "capitulation"}
    rows = []
    for val, lab in labels.items():
        sub = d.loc[d["state"] == val, "fwd"]
        rows.append({
            "band": lab,
            "mean": float(sub.mean()) if len(sub) else np.nan,
            "n": int(len(sub)),
            "hit": float((sub > 0).mean()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows).set_index("band")


# ---------------------------------------------------------------------------
# Regime timing rule
# ---------------------------------------------------------------------------
def timing_signal(
    df: pd.DataFrame,
    thresh: float = 1.0,
) -> pd.Series:
    """Regime long/flat exposure for month t+1, known at month-end t.

    Long (1) when ``sopr >= thresh`` (coins in profit, folk "uptrend"); flat (0)
    when ``sopr < thresh`` (coins at a loss, folk "capitulation").  This is the
    literal ">1 / <1" rule.  Long-only -- it never shorts, it just sidesteps the
    loss regime.
    """
    m = df["sopr"].astype(float)
    pos = pd.Series(0.0, index=df.index, dtype=float)
    pos[m >= thresh] = 1.0
    pos.name = "position"
    return pos


def backtest_timing(
    df: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 30.0,
) -> pd.DataFrame:
    """Net-of-cost monthly returns of a long/flat timing rule on BTC.

    ``position.loc[t]`` is the exposure (0 or 1) held during month t+1.  The
    gross return is ``position * forward_return``.  A flip in position costs
    ``cost_bps`` one-way on the traded notional (|delta position|).  BTC is
    long-only here -- no borrow.  Price-only returns (BTC pays no yield).

    Returns a DataFrame with ``["pos", "fwd_ret", "gross", "cost", "net",
    "bh"]`` where ``bh`` is the buy-and-hold BTC return earned in the same month.
    """
    fwd = forward_returns(df)
    pos = position.reindex(df.index)
    idx = pos.dropna().index.intersection(fwd.dropna().index)
    pos = pos.loc[idx]
    fwd = fwd.loc[idx]

    gross = pos * fwd
    dpos = pos.diff().abs().fillna(pos.abs())   # first trade = entering position
    cost = dpos * (cost_bps * 1e-4)
    net = gross - cost

    out = pd.DataFrame({
        "pos": pos,
        "fwd_ret": fwd,
        "gross": gross,
        "cost": cost,
        "net": net,
        "bh": fwd,                # buy-and-hold benchmark earns the raw BTC return
    })
    return out


def time_in_market(position: pd.Series) -> float:
    """Fraction of months the timing rule is long (exposure share)."""
    p = position.dropna()
    return float((p > 0).mean()) if len(p) else float("nan")


def turnover(position: pd.Series) -> float:
    """Average per-month one-way turnover (|delta position|)."""
    p = position.dropna()
    if len(p) < 2:
        return float("nan")
    return float(p.diff().abs().dropna().mean())


# ---------------------------------------------------------------------------
# Placebo -- shuffle the signal in time
# ---------------------------------------------------------------------------
def placebo_edge(
    df: pd.DataFrame,
    n_shuffles: int = 2000,
    thresh: float = 1.0,
    cost_bps: float = 30.0,
    seed: int = 764,
) -> dict:
    """Distribution of the timing rule's edge-over-buy-and-hold under a placebo.

    The real ">1/<1" rule earns ``real_edge`` = mean(net timing - buy-and-hold)
    per month.  We destroy any real SOPR->return alignment by *shuffling the SOPR
    values in time* (``n_shuffles`` permutations), re-running the rule, and
    recording the mean edge each time.  If the real edge sits inside the placebo
    cloud, the rule has no genuine timing content -- its result is what a random
    long/flat schedule with the same in-market share would produce.

    Returns a dict with the real edge, the placebo mean/std, and a two-sided
    empirical p-value (fraction of |placebo edge| >= |real edge|).
    """
    rng = np.random.default_rng(seed)
    fwd = forward_returns(df).dropna()

    pos_real = timing_signal(df, thresh=thresh)
    bt = backtest_timing(df, pos_real, cost_bps=cost_bps)
    real_edge = float((bt["net"] - bt["bh"]).mean())

    sopr = df["sopr"].astype(float).to_numpy()
    edges = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(sopr)
        shuffled = pd.DataFrame({"sopr": perm, "price": df["price"].to_numpy()},
                                index=df.index)
        p = timing_signal(shuffled, thresh=thresh)
        b = backtest_timing(shuffled, p, cost_bps=cost_bps)
        edges[i] = float((b["net"] - b["bh"]).mean())

    pval = float((np.abs(edges) >= abs(real_edge)).mean())
    return {
        "real_edge_mo": real_edge,
        "placebo_mean_mo": float(edges.mean()),
        "placebo_std_mo": float(edges.std(ddof=1)),
        "p_value": pval,
        "n_shuffles": int(n_shuffles),
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return series (Newey-West HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

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
