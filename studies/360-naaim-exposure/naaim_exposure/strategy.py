"""Strategy and honest controls -- Study 360 (NAAIM-Exposure).

The contrarian NAAIM rule: when active managers are **all-in** (high exposure),
lean *out* of the market; when they have fled to **cash** (low exposure), lean
*in*. We implement and stress-test three flavours of the timing rule on weekly SPY
total returns:

1. **Regime sort** -- split weeks into low / mid / high exposure terciles by the
   *prior* week's NAAIM Number; compare the next-week mean return across regimes.
   The contrarian prediction: low-exposure weeks earn MORE than high-exposure weeks.

2. **Predictive regression** -- regress next-week return on the (standardised)
   prior exposure; report the slope, its Newey-West (HAC) t-stat, and the R^2. A
   real contrarian effect needs a *negative* slope with HAC ``|t| >= 2``.

3. **Contrarian timing overlay** -- a long/flat (or long/short) rule that is long
   SPY when exposure is below a threshold and flat (or short) when above, charged
   one-way costs, pinned head-to-head against **buy-and-hold** total return.

Honesty rails baked in:

- **One-week execution lag**: the reading at week *t* trades week *t+1* (the panel
  is already aligned, so ``ret`` at row t is the forward return -- one shift, once).
- **One-way costs x NAV**: the overlay pays ``one_way_bps`` each time it changes
  state; a short week additionally pays a borrow haircut.
- **Total return on both legs**: timing and buy-and-hold use the same SPY
  total-return index, so the dividend treatment is identical and the comparison is
  excess-of-the-same-benchmark (no price-vs-total-return mismatch).
- **Vintage**: NAAIM's published series is the current vintage, not strictly
  point-in-time; any edge is an upper bound (named on the Signal axis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS = 52


# ---------------------------------------------------------------------------
# Summary statistics with a Newey-West (HAC) t-stat
# ---------------------------------------------------------------------------
def _hac_se_mean(r: np.ndarray) -> float:
    """Newey-West HAC standard error of the sample mean of ``r``."""
    n = r.size
    e = r - r.mean()
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return float(np.sqrt(max(lrv, 0.0) / n))


def summarize(series: pd.Series) -> dict:
    """Headline stats for a weekly return series with a Newey-West HAC t-stat."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat",
                                    "hit_rate", "max_drawdown")} | {"n": n}
    arr = r.to_numpy()
    mu = float(arr.mean())
    std = float(arr.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")
    se = _hac_se_mean(arr)
    tstat = float(mu / se) if se > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {"mean": mu, "vol": std, "sharpe": sr, "tstat": tstat,
            "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": n}


def annualise(stats: dict, periods: int = WEEKS) -> dict:
    """Annualise weekly mean / vol / Sharpe."""
    out = dict(stats)
    if np.isfinite(out.get("mean", np.nan)):
        out["mean_ann"] = float(out["mean"] * periods)
    if np.isfinite(out.get("vol", np.nan)):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if out.get("vol_ann", 0.0) > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


# ---------------------------------------------------------------------------
# 1. Regime sort: next-week return by prior-exposure tercile
# ---------------------------------------------------------------------------
def regime_returns(panel: pd.DataFrame, q: float = 1.0 / 3.0) -> pd.DataFrame:
    """Bucket weeks by the *prior* (already-observed) NAAIM exposure tercile.

    ``panel`` must have columns ``naaim`` (observed at week t) and ``ret`` (return
    earned during week t+1). No look-ahead: tercile thresholds use only ``naaim``,
    and each row's ``ret`` is already the forward return. ``low`` = managers in cash
    (the contrarian *buy* zone), ``high`` = managers all-in (the *sell* zone).
    """
    df = panel.dropna(subset=["naaim", "ret"]).copy()
    lo = df["naaim"].quantile(q)
    hi = df["naaim"].quantile(1.0 - q)
    regime = pd.Series("mid", index=df.index)
    regime[df["naaim"] <= lo] = "low"
    regime[df["naaim"] >= hi] = "high"
    df["regime"] = regime
    return df[["naaim", "ret", "regime"]]


def regime_summary(panel: pd.DataFrame, q: float = 1.0 / 3.0) -> pd.DataFrame:
    """Mean forward return + HAC t-stat per exposure regime (annualised mean too).

    Contrarian prediction: low-exposure weeks earn MORE next week than high-exposure
    weeks (the ``low - high`` gap is positive).
    """
    reg = regime_returns(panel, q=q)
    rows = []
    for name in ("low", "mid", "high"):
        s = summarize(reg.loc[reg["regime"] == name, "ret"])
        mu = s.get("mean", np.nan)
        rows.append({"regime": name, "n": s["n"], "mean_wk": mu,
                     "mean_ann": mu * WEEKS if np.isfinite(mu) else np.nan,
                     "tstat": s.get("tstat", np.nan)})
    return pd.DataFrame(rows).set_index("regime")


def regime_spread(panel: pd.DataFrame, q: float = 1.0 / 3.0) -> pd.Series:
    """Realised return of a self-financing long-panic / short-euphoria rule.

    +ret in low-exposure weeks, -ret in high-exposure weeks, 0 in mid. The HAC
    t-stat of this series is the headline contrarian test.
    """
    reg = regime_returns(panel, q=q)
    out = pd.Series(0.0, index=reg.index, name="contrarian")
    out[reg["regime"] == "low"] = reg.loc[reg["regime"] == "low", "ret"]
    out[reg["regime"] == "high"] = -reg.loc[reg["regime"] == "high", "ret"]
    return out


# ---------------------------------------------------------------------------
# 2. Predictive regression: next-week return on prior exposure
# ---------------------------------------------------------------------------
def predictive_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on standardised prior exposure, HAC t on the slope.

    Returns ``alpha``, ``beta`` (return per +1 sd of exposure), ``tstat`` (HAC t on
    beta), ``r2``, ``n``. A contrarian effect => ``beta < 0`` with ``|tstat| >= 2``.
    """
    df = panel.dropna(subset=["naaim", "ret"]).copy()
    x = df["naaim"].to_numpy(dtype=float)
    y = df["ret"].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return {"alpha": np.nan, "beta": np.nan, "tstat": np.nan, "r2": np.nan, "n": n}

    xm, xs = x.mean(), x.std(ddof=0)
    xz = (x - xm) / xs if xs > 0 else x - xm
    X = np.column_stack([np.ones(n), xz])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef

    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma = u[k:].T @ u[:-k]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    tstat = float(beta / se_beta) if se_beta and se_beta > 0 else float("nan")

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": alpha, "beta": beta, "tstat": tstat, "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# 3. Contrarian timing overlay vs buy-and-hold
# ---------------------------------------------------------------------------
def timing_overlay(
    panel: pd.DataFrame,
    buy_below: float | None = None,
    sell_above: float | None = None,
    allow_short: bool = False,
    one_way_bps: float = 5.0,
    borrow_bps_ann: float = 100.0,
) -> pd.DataFrame:
    """Long/flat (or long/short) contrarian overlay vs buy-and-hold, net of costs.

    Position for week t+1 is set by the exposure observed at week t:
      - naaim <= ``buy_below``  -> long  (+1)   (managers in cash: lean in)
      - naaim >= ``sell_above`` -> flat (0) or short (-1) if ``allow_short``
      - otherwise                -> long (+1)   (default-long contrarian)

    Thresholds default to the in-sample 33rd / 67th exposure percentiles. Costs:
    ``one_way_bps`` x NAV per state change; a short week pays ``borrow_bps_ann/52``
    borrow. Columns: ``pos``, ``ret``, ``gross``, ``cost``, ``net``, ``bh``.
    """
    df = panel.dropna(subset=["naaim", "ret"]).copy()
    if buy_below is None:
        buy_below = df["naaim"].quantile(1.0 / 3.0)
    if sell_above is None:
        sell_above = df["naaim"].quantile(2.0 / 3.0)

    pos = pd.Series(1.0, index=df.index)
    pos[df["naaim"] >= sell_above] = -1.0 if allow_short else 0.0
    pos[df["naaim"] <= buy_below] = 1.0

    gross = pos * df["ret"]
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * one_way_bps * 1e-4
    borrow = (pos < 0).astype(float) * (borrow_bps_ann / WEEKS) * 1e-4
    net = gross - cost - borrow
    return pd.DataFrame({"pos": pos, "ret": df["ret"], "gross": gross,
                         "cost": cost + borrow, "net": net, "bh": df["ret"]})
