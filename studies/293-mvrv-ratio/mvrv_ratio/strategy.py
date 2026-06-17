"""Strategy and honest controls -- Study 293 (MVRV-Ratio).

The folk thesis (Mahmudov & Puell, 2018): MVRV is a *contrarian* gauge of
Bitcoin valuation.  When MVRV is high (the market trades far above the
aggregate cost basis), the coin is over-heated and due to fall; when MVRV is
low (price below cost basis), it is under-valued and due to bounce.  We
formalise it three ways and pit each against the only honest benchmark for a
single trending asset: **buy-and-hold BTC**.

1. **Predictive regression.** Regress next-month BTC log-return on this month's
   MVRV *stretch* ``log(mvrv)``.  A genuine contrarian indicator needs a
   *negative* slope with a HAC *t* whose magnitude clears 2.  We also run a
   *horse race* that adds price's own one-month momentum as a control.

2. **Contrarian timing rule.** Stay long BTC by default, but step to cash for
   month t+1 whenever this month's MVRV is in the "over-heated" zone
   (``mvrv > high``).  Optionally re-enter aggressively in the "under-valued"
   zone (``mvrv < low``).  Compare net-of-cost equity to **buy-and-hold BTC**.

3. **Band states.** Map MVRV to {over-heated, neutral, under-valued} and report
   the average *next-month* return earned from each state -- the cleanest read
   on whether the bands actually mark tops and bottoms.

**No look-ahead:** the signal at month-end t uses only data through t and is
applied to the return *earned during t+1* (one-month execution lag, baked into
``forward_returns``).

**Single-survivor caveat (named on the Signal axis):** BTC went up ~1000x and
MVRV is mechanically derived from BTC's own price path; the contrarian rule is
fitted to a handful of cycle turns (n is tiny).  The decisive test is whether
the rule beats buy-and-hold net of costs (it does not).

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
def mvrv_stretch(df: pd.DataFrame, center: float = 1.8) -> pd.Series:
    """Log MVRV stretch ``log(mvrv / center)``, known at month-end t.

    ``center`` is a fair-value anchor (the long-run MVRV mid-band ~1.8).  A
    positive stretch = over-heated; negative = under-valued.
    """
    z = np.log(df["mvrv"].astype(float) / float(center))
    z.name = "mvrv_stretch"
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
    center: float = 1.8,
    add_price_control: bool = False,
) -> dict:
    """Regress next-month BTC log-return on MVRV stretch (HAC t-stats).

    Model::

        r_{t+1} = a + b * mvrv_stretch_t  [+ c * price_mom_t] + e_{t+1}

    A contrarian indicator predicts ``b < 0`` (high MVRV -> low forward return).
    Returns a dict with the slope(s), HAC *t*-stats, R^2 and n.  When
    ``add_price_control`` is True, BTC's own one-month momentum is added so we
    can see whether MVRV adds anything beyond price trend.
    """
    z = mvrv_stretch(df, center=center)
    fwd = forward_returns(df)
    fwd_log = np.log1p(fwd)

    cols = {"mvrv_stretch": z}
    if add_price_control:
        cols["price_mom"] = price_momentum(df, lookback=1)
    X = pd.DataFrame(cols)
    data = pd.concat([fwd_log.rename("y"), X], axis=1).dropna()
    if len(data) < 12:
        return {"n": int(len(data)), "slope_mvrv": np.nan, "t_mvrv": np.nan,
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
           "slope_mvrv": float(beta[1]),
           "t_mvrv": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
           "slope_price": np.nan, "t_price": np.nan}
    if add_price_control:
        out["slope_price"] = float(beta[2])
        out["t_price"] = float(beta[2] / se[2]) if se[2] > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Band classification
# ---------------------------------------------------------------------------
def mvrv_state(
    df: pd.DataFrame,
    high: float = 3.5,
    low: float = 1.0,
) -> pd.Series:
    """Map MVRV to a 3-state band: +1 under-valued, 0 neutral, -1 over-heated.

    ``mvrv >= high`` -> over-heated (-1, contrarian "sell"); ``mvrv <= low`` ->
    under-valued (+1, contrarian "buy"); otherwise neutral (0).  Known at
    month-end t.
    """
    m = df["mvrv"].astype(float)
    state = pd.Series(0, index=df.index, dtype=float)
    state[m >= high] = -1.0
    state[m <= low] = 1.0
    state.name = "mvrv_state"
    return state


def state_forward_stats(
    df: pd.DataFrame,
    high: float = 3.5,
    low: float = 1.0,
) -> pd.DataFrame:
    """Average *next-month* BTC return conditioned on the MVRV band.

    Returns a DataFrame indexed by band label with columns ``mean`` (mean
    forward simple return), ``n`` (count), ``hit`` (fraction positive).  This is
    the cleanest read on whether 'over-heated' months precede falls and
    'under-valued' months precede rallies.
    """
    state = mvrv_state(df, high=high, low=low)
    fwd = forward_returns(df)
    d = pd.concat([state.rename("state"), fwd.rename("fwd")], axis=1).dropna()
    labels = {-1.0: "over-heated", 0.0: "neutral", 1.0: "under-valued"}
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
# Contrarian timing rule
# ---------------------------------------------------------------------------
def timing_signal(
    df: pd.DataFrame,
    high: float = 3.5,
    low: float = 1.0,
) -> pd.Series:
    """Contrarian long/flat exposure for month t+1, known at month-end t.

    Default long (1); step to flat (0) when ``mvrv >= high`` (over-heated).
    The under-valued zone (``mvrv <= low``) is also long (1) -- the rule never
    shorts, it just sidesteps the over-heated zone.  This is the charitable
    reading of "sell the top" without a borrow leg.
    """
    m = df["mvrv"].astype(float)
    pos = pd.Series(1.0, index=df.index, dtype=float)
    pos[m >= high] = 0.0
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
