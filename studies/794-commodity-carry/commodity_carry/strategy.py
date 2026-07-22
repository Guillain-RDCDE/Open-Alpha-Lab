"""Strategy + inference for Study 794 — Commodity Carry.

The claim: **backwardated commodities (positive roll yield / carry) out-earn contangoed
ones, cross-sectionally.** We isolate the CROSS-SECTIONAL carry premium (Gorton-
Rouwenhorst 2006; Erb-Harvey 2006; Koijen et al. 2018) using an ex-ante carry signal
read from the *real* futures term structure and investable ETF returns.

Design (proxy — see docs/references.md for the limitation):

* **Carry signal (known at t).** For each commodity, monthly annualized log roll yield
  ``carry_t = ln(C1 / C2) * 12`` from the EIA near-month curve settle on the last
  session of month ``t``. ``C1 > C2`` (backwardation) => positive carry; ``C1 < C2``
  (contango) => negative. A robustness variant uses the C1-vs-C4 slope.
* **Outcome (t -> t+1).** The investable front-month ETF total return over the NEXT
  month (USO for WTI, UNG for gas). One documented execution lag: the curve settle is
  public after the close of month ``t``; the position is entered at the FIRST session of
  month ``t+1`` and held to month ``t+1``'s last session.
* **Cross-sectional sort.** Each month, rank the (two) commodities by carry, go long the
  more-backwardated ETF and short the more-contangoed one, equal dollar notional — the
  literal two-name cross-sectional carry portfolio.
* **Pooled panel test (the workhorse given a two-name cross-section).** Pool
  ``(carry_t, ret_{t+1})`` across both commodities, cross-sectionally demean BOTH within
  each month (removing the common energy move), and Newey-West-regress demeaned return on
  demeaned carry — the pure cross-sectional carry->return slope.
* **Roll-drag proxy (USO vs USL).** The front-minus-laddered monthly gap on the *same*
  crude directly measures realized roll: in backwardation the front (USO) should beat the
  12-month ladder (USL). Regress that gap on WTI carry (NW t) as a mechanism check.
* **Costed timer.** The monthly-rebalanced cross-sectional long-short book, costs one-way
  bps x turnover x NAV per leg, short leg pays borrow.

The decisive number is the pooled cross-sectional carry->return NW t on the REAL tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_slope(x: np.ndarray, y: np.ndarray, lags: int = 6) -> tuple[float, float]:
    """HAC (Newey-West, Bartlett) t and slope of the OLS regression y = a + b*x."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    keep = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return (float(beta[1] / se) if se > 0 else float("nan")), float(beta[1])


def newey_west_mean_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC t of mean(x) vs 0 (regress x on a constant)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    S = float(u @ u)
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        S += 2.0 * w * float(u[l:] @ u[:-l])
    se = np.sqrt(S) / n
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def sharpe(x: np.ndarray, periods: int = MONTHS_PER_YEAR) -> float:
    """Annualized Sharpe of a per-period return series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Build the monthly carry/return panel from the real tape
# --------------------------------------------------------------------------- #
def carry_series(curve: pd.DataFrame, prefix: str, near: int = 1, far: int = 2,
                 ) -> pd.Series:
    """Monthly (month-end) annualized log roll yield ln(C_near / C_far) * (12 / gap).

    ``gap`` = far - near contract months. Positive => backwardation. Sampled on the last
    available settle of each calendar month (point-in-time; no look-ahead)."""
    c_near = curve[f"{prefix}{near}"].dropna()
    c_far = curve[f"{prefix}{far}"].dropna()
    both = pd.DataFrame({"n": c_near, "f": c_far}).dropna()
    ann = MONTHS_PER_YEAR / (far - near)
    roll = np.log(both["n"] / both["f"]) * ann
    # last settle of each calendar month
    return roll.groupby(roll.index.to_period("M")).last()


def etf_monthly_returns(etfs: pd.DataFrame, ticker: str) -> pd.Series:
    """Month-over-month total return of an ETF, indexed by calendar month (period)."""
    px = etfs[ticker].dropna()
    m_last = px.groupby(px.index.to_period("M")).last()
    return m_last.pct_change()


def build_panel(curve: pd.DataFrame, etfs: pd.DataFrame, commodities: dict,
                near: int = 1, far: int = 2) -> pd.DataFrame:
    """Long monthly panel: for each (month, commodity) the carry known at month t and the
    front-ETF return realized over t -> t+1 (the one-month execution lag).

    Columns: ``month`` (Period), ``commodity``, ``carry`` (known at t),
    ``ret`` (t->t+1 forward return), plus ``front`` / ``ladder`` returns for the proxy.
    """
    recs = []
    for name, spec in commodities.items():
        carry = carry_series(curve, spec["prefix"], near, far)
        front = etf_monthly_returns(etfs, spec["front_etf"])
        ladder = (etf_monthly_returns(etfs, spec["ladder_etf"])
                  if spec["ladder_etf"] in etfs.columns else pd.Series(dtype=float))
        # forward return: carry at month m predicts return over m+1
        fwd = front.shift(-1)
        fwd_lad = ladder.shift(-1) if len(ladder) else None
        idx = carry.index.intersection(fwd.dropna().index)
        for m in idx:
            recs.append({
                "month": m, "commodity": name,
                "carry": float(carry.loc[m]),
                "ret": float(fwd.loc[m]),
                "ladder_ret": (float(fwd_lad.loc[m])
                               if fwd_lad is not None and m in fwd_lad.index
                               and not np.isnan(fwd_lad.loc[m]) else np.nan),
            })
    panel = pd.DataFrame(recs).sort_values(["month", "commodity"]).reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------- #
# The pooled cross-sectional carry -> return test (the workhorse)
# --------------------------------------------------------------------------- #
def _cross_demean(panel: pd.DataFrame, col: str) -> pd.Series:
    """Subtract, within each month, the cross-sectional mean of ``col`` (removes the
    common energy move so only the cross-sectional carry tilt is left)."""
    return panel[col] - panel.groupby("month")[col].transform("mean")


def pooled_carry_test(panel: pd.DataFrame, lags: int = 6) -> dict:
    """Cross-sectionally demean carry and return within each month, then NW-regress
    demeaned return on demeaned carry. The slope is the cross-sectional carry premium
    per unit of annualized roll yield; the NW t is the decisive real-tape number."""
    p = panel.dropna(subset=["carry", "ret"]).copy()
    # keep only months with both commodities present (a real cross-section)
    counts = p.groupby("month")["commodity"].transform("count")
    p = p[counts >= 2]
    dcarry = _cross_demean(p, "carry").to_numpy()
    dret = _cross_demean(p, "ret").to_numpy()
    t, b = newey_west_slope(dcarry, dret, lags=lags)
    # correlation as a scale-free companion
    keep = ~np.isnan(dcarry) & ~np.isnan(dret)
    corr = (float(np.corrcoef(dcarry[keep], dret[keep])[0, 1])
            if keep.sum() > 2 and dcarry[keep].std() > 0 else float("nan"))
    return {"n_obs": int(keep.sum()), "n_months": int(p["month"].nunique()),
            "slope": b, "nw_t": t, "corr": corr}


# --------------------------------------------------------------------------- #
# The cross-sectional long-short carry portfolio + costed timer
# --------------------------------------------------------------------------- #
def cross_sectional_portfolio(panel: pd.DataFrame) -> pd.Series:
    """Monthly return of the equal-dollar long-short: +1 the highest-carry commodity,
    -1 the lowest, per month (two-name => long one / short the other). Gross, no cost."""
    rets = {}
    for m, g in panel.dropna(subset=["carry", "ret"]).groupby("month"):
        if g["commodity"].nunique() < 2:
            continue
        hi = g.loc[g["carry"].idxmax()]
        lo = g.loc[g["carry"].idxmin()]
        rets[m] = 0.5 * (hi["ret"] - lo["ret"])   # 50/50 gross-notional long-short
    return pd.Series(rets).sort_index()


def timer_stats(panel: pd.DataFrame, cost_bps: float = 10.0,
                borrow_bps_yr: float = 100.0) -> dict:
    """Cross-sectional long-short book, monthly rebalanced. Costs = one-way ``cost_bps``
    x turnover x NAV per leg (full 100% turnover assumed each month on the two-name book),
    short leg pays ``borrow_bps_yr`` / 12 per month. Gross/net mean, NW t, Sharpe."""
    gross = cross_sectional_portfolio(panel)
    if len(gross) == 0:
        return {"n": 0}
    # two legs (one long, one short), full turnover each month:
    monthly_cost = 2.0 * (cost_bps / 1e4) * 2.0        # 2 legs x round-trip proxy
    monthly_borrow = (borrow_bps_yr / 1e4) / MONTHS_PER_YEAR
    net = gross - monthly_cost - monthly_borrow
    return {
        "n": len(gross),
        "gross_mean_pct": float(gross.mean() * 100),
        "net_mean_pct": float(net.mean() * 100),
        "gross_ann_pct": float(gross.mean() * MONTHS_PER_YEAR * 100),
        "net_ann_pct": float(net.mean() * MONTHS_PER_YEAR * 100),
        "gross_t": newey_west_mean_t(gross.to_numpy()),
        "net_t": newey_west_mean_t(net.to_numpy()),
        "gross_sharpe": sharpe(gross.to_numpy()),
        "net_sharpe": sharpe(net.to_numpy()),
        "worst_pct": float(net.min() * 100),
        "series": net,
    }


# --------------------------------------------------------------------------- #
# Roll-drag mechanism proxy — USO vs USL
# --------------------------------------------------------------------------- #
def roll_drag_proxy(panel: pd.DataFrame, commodity: str = "WTI", lags: int = 6) -> dict:
    """Front-minus-laddered monthly return gap (e.g. USO - USL) regressed on that
    commodity's carry. In backwardation (carry>0) the front should beat the ladder =>
    a positive slope validates the roll mechanism directly on the investable tape."""
    g = panel[panel["commodity"] == commodity].dropna(subset=["carry", "ret", "ladder_ret"])
    gap = (g["ret"] - g["ladder_ret"]).to_numpy()
    carry = g["carry"].to_numpy()
    t, b = newey_west_slope(carry, gap, lags=lags)
    # sign agreement: does carry>0 line up with front>ladder?
    agree = float(np.mean((carry > 0) == (gap > 0))) if len(gap) else float("nan")
    return {"n": len(gap), "slope": b, "nw_t": t, "sign_agree": agree,
            "mean_gap_backwardation_pct": (float(gap[carry > 0].mean() * 100)
                                           if (carry > 0).sum() else float("nan")),
            "mean_gap_contango_pct": (float(gap[carry < 0].mean() * 100)
                                      if (carry < 0).sum() else float("nan"))}


# --------------------------------------------------------------------------- #
# Conditional split — backwardation vs contango front-ETF returns (per commodity)
# --------------------------------------------------------------------------- #
def conditional_split(panel: pd.DataFrame) -> dict:
    """Pooling both commodities: mean forward front-ETF return when the commodity is
    backwardated (carry>0) vs contangoed (carry<0), and the Welch t between them."""
    p = panel.dropna(subset=["carry", "ret"])
    back = p.loc[p["carry"] > 0, "ret"].to_numpy()
    cont = p.loc[p["carry"] < 0, "ret"].to_numpy()
    return {"n_back": len(back), "n_cont": len(cont),
            "back_pct": float(back.mean() * 100) if len(back) else float("nan"),
            "cont_pct": float(cont.mean() * 100) if len(cont) else float("nan"),
            "welch_t": welch_t(back, cont),
            "hit": int((back > 0).sum()), "hit_n": len(back)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(syn_panel: pd.DataFrame) -> dict:
    """Run the pooled cross-sectional carry->return test on a synthetic panel.

    The synthetic frame already carries contemporaneous (carry, ret) — the planted
    premium is contemporaneous — so we regress directly (no forward shift)."""
    p = syn_panel.rename(columns={"month": "month"}).copy()
    dcarry = (p["carry"] - p.groupby("month")["carry"].transform("mean")).to_numpy()
    dret = (p["ret"] - p.groupby("month")["ret"].transform("mean")).to_numpy()
    t, b = newey_west_slope(dcarry, dret, lags=6)
    return {"n": len(p), "slope": b, "nw_t": t}
