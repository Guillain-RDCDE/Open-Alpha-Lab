"""Strategy + inference for Study 621 — Share-Class Spreads.

The claim under test: **BRK.B can never cost more than 1/1500th of BRK.A** — the one-way A→B
conversion (any A holder can mint 1,500 B and sell them) caps the B premium, while nothing
converts B back to A, so B *can* trade at a discount. **GOOG vs GOOGL has no bound at all** —
no conversion bridge exists, so the voting spread floats free. We measure:

    * **Bound enforcement.** The distribution of the BRK relative-value gap
      ``d = 1500·B/A − 1`` (negative = B cheap): share of days above +0/+10/+50/+100 bps, the
      up-vs-down tail asymmetry, and the same numbers for the unbounded GOOGL−GOOG spread.
      Close prints are non-synchronous (an A share trades a few hundred times a day), so
      a few bps of apparent "premium" is measurement noise — the bound is judged at the
      50–100 bps level, not at zero.
    * **The structural B discount.** Mean gap with a **Newey-West (HAC) t** (daily gaps are
      heavily autocorrelated), the full distribution, era slices, and AR(1) half-lives of both
      pairs' gaps.
    * **Tradability.** (a) A long-cheap/short-rich z-score pairs trade on the BRK gap and
      (b) an A-holder **class-switch overlay** (hold B when it is ≥ thr cheap, back to A at
      parity). Exactly **one execution lag**: signal at close *t*, order filled at close *t+1*,
      first return accrues *t+1 → t+2* (``fill="next"``). The diagnostic ``fill="same"``
      variant — a fill at close *t*, the very prints that define the signal — is reported only
      to show where the paper profit lives; it is NOT the headline. Costs are one-way × NAV per
      leg (a class switch = sell one class + buy the other = 2 legs); the pairs short leg pays
      borrow.

The decisive numbers are the HAC t of the mean BRK gap (the bound + structural discount) and
the honest-lag net t of the two trading rules (the tradability verdict).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Series construction
# --------------------------------------------------------------------------- #
def gap_series(prices: pd.DataFrame, a: str, b: str, parity: float,
               start: str | None = None, end: str | None = None) -> pd.Series:
    """Relative-value gap ``parity·B/A − 1`` (negative = the junior class B is cheap)."""
    px = prices[[a, b]].dropna()
    if start is not None:
        px = px[px.index >= pd.Timestamp(start)]
    if end is not None:
        px = px[px.index <= pd.Timestamp(end)]
    return (parity * px[b] / px[a] - 1.0).rename("gap")


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x, lags: int | None = None) -> tuple[float, int, int]:
    """Newey-West (Bartlett) t of the mean of ``x`` against zero.

    Default bandwidth: the standard ``floor(4·(n/100)^(2/9))`` rule. Returns (t, lags, n).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return float("nan"), 0, n
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    xc = x - x.mean()
    v = float(xc @ xc) / n
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        v += 2.0 * w * float(xc[:-L] @ xc[L:]) / n
    return float(x.mean() / np.sqrt(v / n)), lags, n


def dist_stats(gap: pd.Series) -> dict:
    """Distribution of a gap series, in bps, plus the HAC t of its mean."""
    g = gap.dropna()
    t, lags, n = hac_t(g.values)
    return {
        "n": n, "mean_bps": g.mean() * 1e4, "median_bps": g.median() * 1e4,
        "std_bps": g.std() * 1e4,
        "p1_bps": g.quantile(0.01) * 1e4, "p99_bps": g.quantile(0.99) * 1e4,
        "min_bps": g.min() * 1e4, "max_bps": g.max() * 1e4,
        "hac_t": t, "hac_lags": lags,
    }


def violation_stats(gap: pd.Series, thresholds_bps=(0.0, 10.0, 50.0, 100.0)) -> dict:
    """Share/count of days the junior class trades ABOVE parity by more than each threshold,
    plus the ±50 bps tail asymmetry (down-share / up-share) — the one-way-bound signature."""
    g = gap.dropna()
    out = {"n": len(g)}
    for thr in thresholds_bps:
        m = g > thr * 1e-4
        out[f"above_{thr:.0f}bps_share"] = float(m.mean())
        out[f"above_{thr:.0f}bps_days"] = int(m.sum())
    up = float((g > 0.005).mean())
    dn = float((g < -0.005).mean())
    out["up50_share"], out["dn50_share"] = up, dn
    out["asym_ratio"] = dn / up if up > 0 else float("inf")
    return out


def half_life(gap: pd.Series) -> tuple[float, float]:
    """AR(1) fit of the gap; returns (rho, half-life in trading days)."""
    g = gap.dropna()
    x0 = g.shift(1).dropna()
    x1 = g.loc[x0.index]
    rho = float(np.polyfit(x0.values, x1.values, 1)[0])
    hl = float(np.log(0.5) / np.log(rho)) if 0.0 < rho < 1.0 else float("nan")
    return rho, hl


# --------------------------------------------------------------------------- #
# Tradability — A-holder class-switch overlay
# --------------------------------------------------------------------------- #
def _positions_from_gap(gap: pd.Series, thr: float, exit_level: float = 0.0) -> pd.Series:
    """Stateful switch rule: 1 = hold the junior class B (entered when gap < −thr),
    0 = hold A (re-entered when the gap has closed back to ``exit_level``)."""
    sig = pd.Series(np.nan, index=gap.index)
    sig[gap < -thr] = 1.0
    sig[gap >= exit_level] = 0.0
    return sig.ffill().fillna(0.0)


def switch_overlay(prices: pd.DataFrame, a: str, b: str, parity: float,
                   thr: float = 0.01, cost_bps: float = 5.0,
                   fill: str = "next") -> dict:
    """A-holder overlay: hold B while it is ≥ ``thr`` cheap, hold A otherwise; excess vs
    buy-and-hold A, net of ``cost_bps`` one-way × 2 legs per switch.

    ``fill="next"`` (headline): signal at close *t*, filled at close *t+1* — one clean
    execution lag. ``fill="same"`` (diagnostic only): filled at close *t*, the signal's own
    prints — the optimistic convention that manufactures the paper profit.
    """
    px = prices[[a, b]].dropna()
    gap = parity * px[b] / px[a] - 1.0
    ra, rb = px[a].pct_change(), px[b].pct_change()
    pos = _positions_from_gap(gap, thr)
    shift = {"same": 1, "next": 2}[fill]
    held = pos.shift(shift).fillna(0.0)
    strat = held * rb + (1.0 - held) * ra
    switches = pos.shift(shift - 1).fillna(0.0).diff().abs()
    cost = switches * 2.0 * cost_bps * 1e-4        # sell one class + buy the other = 2 legs
    excess = (strat - cost - ra).dropna()
    t, lags, n = hac_t(excess.values)
    return {
        "thr_pct": thr * 100.0, "fill": fill, "cost_bps": cost_bps,
        "net_ann_pct": excess.mean() * TRADING_DAYS * 100.0,
        "hac_t": t, "n": n, "switches": int(switches.sum()),
        "time_in_b_pct": float((held == 1.0).mean()) * 100.0,
        "excess": excess,
    }


# --------------------------------------------------------------------------- #
# Tradability — long-cheap / short-rich z-score pairs trade
# --------------------------------------------------------------------------- #
def pairs_trade(prices: pd.DataFrame, a: str, b: str, parity: float,
                entry_z: float = 1.5, exit_z: float = 0.25, window: int = 252,
                cost_bps: float = 5.0, borrow_bps_yr: float = 50.0,
                fill: str = "next") -> dict:
    """Dollar-neutral: long the cheap class / short the rich one when the gap's rolling
    z-score exceeds ``entry_z``, flat when it re-enters ``exit_z``. One-way costs per leg on
    turnover; the short leg pays ``borrow_bps_yr``. Same fill conventions as the overlay."""
    px = prices[[a, b]].dropna()
    gap = parity * px[b] / px[a] - 1.0
    ra, rb = px[a].pct_change(), px[b].pct_change()
    z = (gap - gap.rolling(window).mean()) / gap.rolling(window).std()
    raw = pd.Series(np.nan, index=gap.index)
    raw[z < -entry_z] = 1.0        # B cheap: long B / short A
    raw[z > entry_z] = -1.0        # B rich: short B / long A
    raw[z.abs() < exit_z] = 0.0
    pos = raw.ffill().fillna(0.0)
    shift = {"same": 1, "next": 2}[fill]
    held = pos.shift(shift).fillna(0.0)
    rel = rb - ra                                   # long-B-short-A relative return
    gross = held * rel
    turn = pos.shift(shift - 1).fillna(0.0).diff().abs()
    cost = turn * 2.0 * cost_bps * 1e-4             # 2 legs per unit of position change
    borrow = held.abs() * (borrow_bps_yr * 1e-4) / TRADING_DAYS
    net = (gross - cost - borrow).dropna()
    tg, _, _ = hac_t(gross.dropna().values)
    tn, lags, n = hac_t(net.values)
    return {
        "fill": fill, "cost_bps": cost_bps,
        "gross_ann_pct": gross.dropna().mean() * TRADING_DAYS * 100.0, "gross_t": tg,
        "net_ann_pct": net.mean() * TRADING_DAYS * 100.0, "net_t": tn,
        "in_market_pct": float((held != 0.0).mean()) * 100.0, "n": n,
    }


# --------------------------------------------------------------------------- #
# Synthetic detector (positive control)
# --------------------------------------------------------------------------- #
def bound_detector(prices: pd.DataFrame, parity: float) -> dict:
    """The study's detector run on a two-class tape: mean-gap HAC t + tail asymmetry."""
    gap = gap_series(prices, "A", "B", parity)
    ds = dist_stats(gap)
    vs = violation_stats(gap)
    return {"mean_bps": ds["mean_bps"], "hac_t": ds["hac_t"],
            "up50_share": vs["up50_share"], "dn50_share": vs["dn50_share"],
            "asym_ratio": vs["asym_ratio"]}
