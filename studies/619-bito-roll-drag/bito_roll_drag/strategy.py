"""Rules + statistics for Study 619 — BITO Roll Drag.

The tested object is the **daily total-return gap** ``r_BITO - r_ref`` (ref = spot bitcoin
sampled on BITO's trading days, or IBIT once it lists). The claim is that this gap has a
significantly **negative** mean — a toll — driven by the monthly futures roll (contango carry)
plus the fee. Inference:

* the mean daily gap gets a **Newey-West (Bartlett) HAC t** with ~1 trading month of lags
  (basis regimes persist, so an i.i.d. t would overstate certainty);
* group splits (roll-window vs ordinary days; contango vs backwardation days) use **Welch t**;
* the contango/backwardation classification uses the **prior day's** CME front-month basis
  (``fut/spot - 1`` at t-1) — the study's single execution lag, documented here and in results.

The dollar-neutral expression (long IBIT / short BITO) charges borrow on the short leg and
one-way costs on daily rebalancing turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Core series
# --------------------------------------------------------------------------- #
def daily_gap(frame: pd.DataFrame, col: str = "bito", ref: str = "spot") -> pd.Series:
    """Daily simple-return gap ``r_col - r_ref`` on their common dates (total-return legs)."""
    sub = frame[[col, ref]].dropna()
    r = sub.pct_change().dropna()
    return (r[col] - r[ref]).rename(f"{col}-{ref}")


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray, lags: int = 21) -> dict:
    """Newey-West (Bartlett-kernel) HAC t-stat of the mean of ``x``.

    Returns mean, HAC SE, t = mean/SE and n. Two-sided by convention; the drag claim expects
    a negative t.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "n": n}
    m = x.mean()
    e = x - m
    var = (e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        var += 2.0 * w * (e[k:] @ e[:-k]) / n
    se = np.sqrt(var / n)
    return {"mean": float(m), "se": float(se),
            "t": float(m / se) if se > 0 else float("nan"), "n": n}


def welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch two-sample t for mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    return {"diff": float(a.mean() - b.mean()),
            "t": float((a.mean() - b.mean()) / se) if se > 0 else float("nan"),
            "n_a": len(a), "n_b": len(b)}


# --------------------------------------------------------------------------- #
# Headline: the drag
# --------------------------------------------------------------------------- #
def summarize_drag(frame: pd.DataFrame, col: str = "bito", ref: str = "spot",
                   lags: int = 21) -> dict:
    """Annualized drag of ``col`` vs ``ref`` with a HAC t, plus the cumulative shortfall."""
    g = daily_gap(frame, col=col, ref=ref)
    h = hac_t(g.values, lags=lags)
    sub = frame[[col, ref]].dropna()
    tr_col = sub[col].iloc[-1] / sub[col].iloc[0] - 1.0
    tr_ref = sub[ref].iloc[-1] / sub[ref].iloc[0] - 1.0
    years = (sub.index[-1] - sub.index[0]).days / 365.25
    return {
        "n_days": h["n"], "years": float(years),
        "start": str(sub.index[0].date()), "end": str(sub.index[-1].date()),
        "gap_bps_day": h["mean"] * 1e4,
        "drag_ann_pct": h["mean"] * TRADING_DAYS * 100.0,
        "hac_t": h["t"],
        "tr_col_pct": tr_col * 100.0, "tr_ref_pct": tr_ref * 100.0,
        "shortfall_pp": (tr_col - tr_ref) * 100.0,
        "wealth_ratio": (1.0 + tr_col) / (1.0 + tr_ref),
    }


def monthly_gap(frame: pd.DataFrame, col: str = "bito", ref: str = "spot",
                lags: int = 3) -> dict:
    """Complete-calendar-month compounded gap (bps/month) with a HAC t, and the hit rate."""
    sub = frame[[col, ref]].dropna()
    r = sub.pct_change().dropna()
    per = r.index.to_period("M")
    # keep only complete months (the aligned frame is already as-of'd to a month end)
    m_col = (1.0 + r[col]).groupby(per).prod() - 1.0
    m_ref = (1.0 + r[ref]).groupby(per).prod() - 1.0
    # drop the first month if partial (inception mid-month)
    if sub.index[0].day > 5:
        m_col, m_ref = m_col.iloc[1:], m_ref.iloc[1:]
    gap = m_col - m_ref
    h = hac_t(gap.values, lags=lags)
    return {
        "n_months": int(len(gap)),
        "gap_bps_month": h["mean"] * 1e4,
        "hac_t": h["t"],
        "neg_share": float((gap < 0).mean()),
        "series": gap,
    }


# --------------------------------------------------------------------------- #
# Roll-window attribution
# --------------------------------------------------------------------------- #
def roll_attribution(frame: pd.DataFrame, flags: np.ndarray,
                     col: str = "bito", ref: str = "spot") -> dict:
    """Mean daily gap inside vs outside the expiry-week roll window (Welch t).

    Also reports each bucket's share of the total cumulative log shortfall vs its share of
    days — if the toll were paid *at the roll*, the window's shortfall share would dwarf its
    day share; basis-convergence carry predicts shares roughly proportional.
    """
    g = daily_gap(frame, col=col, ref=ref)
    f = pd.Series(flags, index=frame.index).reindex(g.index).fillna(False).values.astype(bool)
    inside, outside = g.values[f], g.values[~f]
    w = welch_t(inside, outside)
    lg = np.log1p(frame[col].pct_change()) - np.log1p(frame[ref].pct_change())
    lg = lg.reindex(g.index)
    total = lg.sum()
    share_short = float(lg.values[f].sum() / total) if total != 0 else float("nan")
    return {
        "in_bps_day": float(np.mean(inside) * 1e4), "n_in": int(f.sum()),
        "out_bps_day": float(np.mean(outside) * 1e4), "n_out": int((~f).sum()),
        "welch_t": w["t"],
        "share_days": float(f.mean()),
        "share_shortfall": share_short,
    }


# --------------------------------------------------------------------------- #
# Contango / backwardation split (single execution lag: prior-day basis)
# --------------------------------------------------------------------------- #
def basis_series(frame: pd.DataFrame) -> pd.Series:
    """CME front-month basis ``fut/spot - 1`` (both sampled at the same date)."""
    sub = frame[["fut", "spot"]].dropna()
    return (sub["fut"] / sub["spot"] - 1.0).rename("basis")


def regime_split(frame: pd.DataFrame, col: str = "bito", ref: str = "spot") -> dict:
    """Daily gap conditional on the PRIOR day's basis sign (lag 1 — no look-ahead)."""
    g = daily_gap(frame, col=col, ref=ref)
    b = basis_series(frame).shift(1).reindex(g.index)
    ok = b.notna()
    g, b = g[ok], b[ok]
    contango = b.values > 0
    w = welch_t(g.values[contango], g.values[~contango])
    return {
        "share_contango": float(contango.mean()),
        "n_contango": int(contango.sum()), "n_backward": int((~contango).sum()),
        "contango_bps_day": float(g.values[contango].mean() * 1e4),
        "contango_ann_pct": float(g.values[contango].mean() * TRADING_DAYS * 100.0),
        "backward_bps_day": float(g.values[~contango].mean() * 1e4),
        "backward_ann_pct": float(g.values[~contango].mean() * TRADING_DAYS * 100.0),
        "welch_t": w["t"],
        "hac_t_contango": hac_t(g.values[contango])["t"],
        "median_basis_bps": float(np.median(b.values) * 1e4),
    }


# --------------------------------------------------------------------------- #
# Tradability: the dollar-neutral substitution spread
# --------------------------------------------------------------------------- #
def spread_trade(frame: pd.DataFrame, borrow_ann: float = 0.02,
                 cost_bps: float = 2.0, lags: int = 21) -> dict:
    """Long IBIT / short BITO, dollar-neutral, rebalanced daily (IBIT era only).

    Net = (r_ibit - r_bito) - borrow/252 (the short pays borrow on full notional) -
    one-way ``cost_bps`` x daily rebalancing turnover (|r_ibit| + |r_bito| relative drift,
    both legs traded back to equal notional). Positions are entered at the prior close
    (always-on book — the single lag is trivial here) .
    """
    sub = frame[["ibit", "bito"]].dropna()
    r = sub.pct_change().dropna()
    gross = r["ibit"] - r["bito"]
    turnover = (r["ibit"].abs() + r["bito"].abs())
    net = gross - borrow_ann / TRADING_DAYS - cost_bps * 1e-4 * turnover
    h = hac_t(net.values, lags=lags)
    sd = net.std(ddof=1)
    return {
        "n_days": h["n"],
        "gross_ann_pct": float(gross.mean() * TRADING_DAYS * 100.0),
        "net_ann_pct": float(net.mean() * TRADING_DAYS * 100.0),
        "net_hac_t": h["t"],
        "sharpe": float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
        "worst_day_pct": float(net.min() * 100.0),
        "vol_ann_pct": float(sd * np.sqrt(TRADING_DAYS) * 100.0),
    }
