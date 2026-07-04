"""Event study + inference for Study 631 — Restatement Shock (Item 4.02).

The claim: an 8-K that says "**do not rely** on our previously issued financials"
(Item 4.02) keeps hurting for months — the market *underreacts* to accounting bombs.
We measure it as a classic market-adjusted event study:

* **Announcement window [0, +1].** Log return of the stock over the filing day and the
  next trading day (many 8-Ks hit after the close), minus the same for SPY. This is the
  *bomb* itself.
* **Drift window [+2, +64].** Market-adjusted log return from the close of day **+1**
  (the single, documented execution lag: you read the 8-K, you can trade at the next
  close) to the close of day **+64** — 63 trading days ≈ 3 calendar months. If the market
  underreacts, this is negative on average.

Inference:

* a **one-sample t** of the per-event CAR against zero (each event = one observation);
* the same t on **1%/99% winsorized** CARs (microcap tails are violent);
* a **calendar-month clustered** t — average the CARs of all events filed in the same
  month, then run a **Newey-West (HAC, 3 lags)** t across the monthly means. Drift windows
  overlap in calendar time, so plain per-event t's overstate independence; the clustered
  HAC t is the honest headline statistic.
* the **small-vs-large split** (third axis): median split on pre-event dollar volume
  (a limits-to-arbitrage proxy), Welch t of the CAR difference.

Tradability: short each event at the close of day +1, cover at day +64, hedged with SPY;
one-way costs x NAV on the stock round trip, plus **borrow** at an annual rate x 63/252.
Shorts pay borrow — hard-to-borrow microcaps pay a lot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN_LO, ANN_HI = 0, 1          # announcement window, trading days relative to filing
DRIFT_START, DRIFT_DAYS = 2, 63  # drift = close(+1) -> close(+64)
PRE_REQ = 60                   # trading days of pre-event history required
DVOL_WIN = (65, 6)             # pre-event dollar-volume window [t0-65, t0-6]


# --------------------------------------------------------------------------- #
# Event table
# --------------------------------------------------------------------------- #
def build_event_table(events: pd.DataFrame, close: pd.DataFrame,
                      dvol: pd.DataFrame, spy: pd.Series,
                      drift_days: int = DRIFT_DAYS) -> pd.DataFrame:
    """One row per usable event: announcement AR, drift CAR, pre-event dollar volume.

    An event is usable iff its ticker has a price series, at least ``PRE_REQ`` prior
    trading days, and a *complete* post window (close at +1 and at +1+drift_days) — so a
    stamped run never contains a truncated drift. Endpoint closes are forward-filled up to
    5 days to ride over sparse-quote gaps.
    """
    close = close.sort_index()
    spy = spy.sort_index()
    cal = spy.index
    cf = close.reindex(cal).ffill(limit=5)
    df = dvol.reindex(cal).ffill(limit=5)
    lspy = np.log(spy.to_numpy(dtype=float))
    pos_of = pd.Series(np.arange(len(cal)), index=cal)

    rows = []
    for _, e in events.iterrows():
        tk = str(e["ticker"]).replace(".", "-")
        if not tk or tk not in cf.columns:
            continue
        d = pd.Timestamp(e["file_date"])
        loc = pos_of.index.searchsorted(d)            # first trading day >= filing date
        if loc < PRE_REQ + 10 or loc + 1 + drift_days >= len(cal):
            continue
        px = cf[tk].to_numpy(dtype=float)
        p_pre, p_ann = px[loc - 1], px[loc + ANN_HI]
        p_in, p_out = px[loc + 1], px[loc + 1 + drift_days]
        if not np.all(np.isfinite([p_pre, p_ann, p_in, p_out])):
            continue
        if np.isnan(px[max(0, loc - PRE_REQ):loc]).mean() > 0.5:
            continue
        ar0 = (np.log(p_ann / p_pre)
               - (lspy[loc + ANN_HI] - lspy[loc - 1]))
        car = (np.log(p_out / p_in)
               - (lspy[loc + 1 + drift_days] - lspy[loc + 1]))
        dv = df[tk].to_numpy(dtype=float)[loc - DVOL_WIN[0]: loc - DVOL_WIN[1]]
        dv_med = float(np.nanmedian(dv)) if np.isfinite(dv).any() else np.nan
        rows.append({"ticker": tk, "date": cal[loc], "ar0": float(ar0),
                     "car": float(car), "dvol": dv_med, "p_in": float(p_in),
                     "month": cal[loc].to_period("M")})
    return pd.DataFrame(rows)


def chronic_decay_placebo(events: pd.DataFrame, close: pd.DataFrame,
                          dvol: pd.DataFrame, spy: pd.Series,
                          drift_days: int = DRIFT_DAYS, lag: int = 252) -> dict:
    """THE adversarial control: is the drift event-specific, or just what these stocks do?

    Firms that file a 4.02 are disproportionately chronically-bleeding microcaps (dilution
    machines melting vs SPY in *any* window). For each usable event, measure the SAME stock's
    market-adjusted CAR over a same-length window entered ``lag`` trading days *before* the
    event entry (a no-event placebo window ending ~9 months pre-confession). The
    **paired difference** (event CAR − placebo CAR) is the event-specific drift; only IT can
    certify "the market underreacts to the confession". Returns paired means, the paired
    one-sample t, winsorized t, and the month-clustered HAC t of the difference.
    """
    tab = build_event_table(events, close, dvol, spy, drift_days=drift_days)
    close = close.sort_index()
    cal = spy.sort_index().index
    cf = close.reindex(cal).ffill(limit=5)
    lspy = np.log(spy.sort_index().to_numpy(dtype=float))
    pos_of = pd.Series(np.arange(len(cal)), index=cal)
    rows = []
    for _, r in tab.iterrows():
        loc = int(pos_of[r["date"]])
        a, b = loc + 1 - lag, loc + 1 - lag + drift_days
        if a < 0:
            continue
        px = cf[r["ticker"]].to_numpy(dtype=float)
        if not (np.isfinite(px[a]) and np.isfinite(px[b])):
            continue
        plc = float(np.log(px[b] / px[a]) - (lspy[b] - lspy[a]))
        rows.append({"ticker": r["ticker"], "month": r["month"],
                     "car": r["car"], "plc": plc, "diff": r["car"] - plc})
    d = pd.DataFrame(rows)
    return {
        "n_pairs": int(len(d)),
        "event_pct": float(d["car"].mean() * 100), "event_t": ttest_vs_zero(d["car"]),
        "placebo_pct": float(d["plc"].mean() * 100), "placebo_t": ttest_vs_zero(d["plc"]),
        "diff_pct": float(d["diff"].mean() * 100), "diff_t": ttest_vs_zero(d["diff"]),
        "diff_t_wins": ttest_vs_zero(winsorize(d["diff"].to_numpy())),
        "diff_t_hac": clustered_hac_t(d.rename(columns={"diff": "x"}), "x")["t_hac"],
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def winsorize(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanquantile(x, [p, 1 - p])
    return np.clip(x, lo, hi)


def nw_tstat(x: np.ndarray, lags: int = 3) -> float:
    """Newey-West (HAC) t of mean(x) vs 0 — for serially-correlated series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def clustered_hac_t(tab: pd.DataFrame, col: str = "car", lags: int = 3) -> dict:
    """Collapse events to calendar-month means, then HAC t across months.

    Drift windows of events filed in the same month overlap almost completely (shared
    market/sector shocks survive the SPY adjustment only partly), and 3-month windows of
    *adjacent* months still overlap — hence cluster first, then Newey-West with ``lags``
    monthly lags. This is the headline statistic for the drift claim.
    """
    g = tab.groupby("month")[col].mean()
    return {"n_months": int(len(g)), "mean": float(g.mean()),
            "t_hac": nw_tstat(g.to_numpy(), lags=lags)}


def summarize(tab: pd.DataFrame) -> dict:
    """Headline numbers: announcement drop + drift, raw / winsorized / clustered-HAC."""
    ar0, car = tab["ar0"].to_numpy(), tab["car"].to_numpy()
    cl_ar0 = clustered_hac_t(tab, "ar0")
    cl_car = clustered_hac_t(tab, "car")
    return {
        "n_events": int(len(tab)),
        "ar0_mean_pct": float(np.mean(ar0) * 100), "ar0_t": ttest_vs_zero(ar0),
        "ar0_median_pct": float(np.median(ar0) * 100),
        "ar0_t_wins": ttest_vs_zero(winsorize(ar0)),
        "ar0_t_hac": cl_ar0["t_hac"],
        "car_mean_pct": float(np.mean(car) * 100), "car_t": ttest_vs_zero(car),
        "car_median_pct": float(np.median(car) * 100),
        "car_t_wins": ttest_vs_zero(winsorize(car)),
        "car_t_hac": cl_car["t_hac"], "n_months": cl_car["n_months"],
        "car_neg_share": float((car < 0).mean()),
    }


def size_split(tab: pd.DataFrame) -> dict:
    """Third axis: is the drift concentrated in small names (limits-to-arbitrage)?

    Median split on pre-event dollar volume. Welch t of (small CAR - large CAR), plus
    each half's own one-sample and clustered-HAC t.
    """
    t = tab.dropna(subset=["dvol"])
    med = t["dvol"].median()
    small, large = t[t["dvol"] <= med], t[t["dvol"] > med]
    return {
        "median_dvol_musd": float(med / 1e6),
        "n_small": int(len(small)), "n_large": int(len(large)),
        "small_car_pct": float(small["car"].mean() * 100),
        "large_car_pct": float(large["car"].mean() * 100),
        "small_t": ttest_vs_zero(small["car"].to_numpy()),
        "large_t": ttest_vs_zero(large["car"].to_numpy()),
        "small_t_hac": clustered_hac_t(small)["t_hac"],
        "large_t_hac": clustered_hac_t(large)["t_hac"],
        "welch_t_diff": welch_t(small["car"].to_numpy(), large["car"].to_numpy()),
    }


def horizon_curve(events: pd.DataFrame, close: pd.DataFrame, dvol: pd.DataFrame,
                  spy: pd.Series, horizons=(5, 21, 42, 63, 126)) -> pd.DataFrame:
    """Mean drift CAR + t at several horizons (trading days after entry at close +1)."""
    rows = []
    for h in horizons:
        tab = build_event_table(events, close, dvol, spy, drift_days=h)
        car = tab["car"].to_numpy()
        rows.append({"days": h, "n": len(tab),
                     "car_pct": float(np.mean(car) * 100),
                     "t": ttest_vs_zero(car),
                     "t_hac": clustered_hac_t(tab)["t_hac"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tradability — short the confession, hedged with SPY
# --------------------------------------------------------------------------- #
def short_overlay(tab: pd.DataFrame, cost_bps: float = 20.0,
                  borrow_ann_pct: float = 3.0,
                  drift_days: int = DRIFT_DAYS) -> dict:
    """Per-event P&L of shorting at close +1, covering at close +1+drift_days, SPY-hedged.

    Gross per event = **-CAR** (market-adjusted, so the SPY hedge is built in). Costs:
    one round trip on the stock at ``cost_bps`` one-way x 2 legs, one round trip on the
    SPY hedge at 1 bp x 2 legs, plus borrow at ``borrow_ann_pct``/yr x holding period.
    Returns mean gross/net per event and the net one-sample t.
    """
    car = tab["car"].to_numpy()
    gross = -car
    hold_frac = drift_days / 252.0
    cost = 2 * cost_bps / 1e4 + 2 * 1.0 / 1e4 + borrow_ann_pct / 100.0 * hold_frac
    net = gross - cost
    return {"cost_bps": cost_bps, "borrow_ann_pct": borrow_ann_pct,
            "gross_pct": float(gross.mean() * 100),
            "drag_pct": float(cost * 100),
            "net_pct": float(net.mean() * 100),
            "net_t": ttest_vs_zero(net),
            "net_t_hac": clustered_hac_t(tab.assign(net=net), "net")["t_hac"]}
