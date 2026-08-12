"""Tested rules + inference for Study 874 — IPO-Price Anchoring.

THE CLAIM: investors anchor on an IPO's **offer price**. Two consequences:

* **Anchoring pull** — the forward return is *negatively* related to the current
  **gap-from-offer** ``g = log(price / offer)``: names stretched far above the offer get
  pulled back down, names below get pulled back up (slow reversion toward the round anchor).
  Tested as a monthly **Fama-MacBeth** cross-sectional slope of next-month **market-adjusted**
  return on the gap, averaged across months with a Newey-West (HAC) *t*. Anchoring ⇒ a
  robustly **negative** average slope.
* **Below-offer drag** — crossing *below* the offer is a persistent drag. Tested as a monthly
  **below-offer minus above-offer** equal-weight basket spread of forward market-adjusted
  returns, with a Newey-West *t*. Loss-aversion lore ⇒ a **negative** spread (below-offer
  names keep lagging).

Both tests first average *across names within each calendar month*, then run a time-series
HAC *t* on the resulting monthly series — this is deliberate: recent-IPO names are dominated by
a single 2021 cohort that rose and crashed together, so their cross-section is heavily
time-correlated; collapsing each month to one number and running the *t* over months is the
honest way to keep that common variation from masquerading as independent observations.

**Abnormal return.** Every name-month forward return is measured *market-adjusted* — the name's
next-month total return minus ``SPY``'s — so a generic "IPOs fell in 2022" tape (the subject of
[623-ipo-long-run-underperformance](../../623-ipo-long-run-underperformance/)) is netted out;
what remains is whatever is specific to the **offer-price anchor**.

**Execution & costs (documented once).** The gap/below flag is measured at the close of month
``t`` and predicts month ``t+1`` — a one-month execution lag by construction, zero look-ahead.
The tradable expression of "below-offer is a drag" is SHORT the below-offer basket / LONG the
above-offer basket (or LONG SPY); the short leg pays borrow, and the monthly re-hedge pays
one-way cost × NAV on each leg. Low N (~45 names, one dominant cohort) ⇒ low power ⇒ the honest
prior is None; the synthetic control only proves the machinery is unbiased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0
MIN_NAMES = 5           # a month needs at least this many live names to enter a stat


# --------------------------------------------------------------------------- #
# Build the name-month event panel from the daily-close tape
# --------------------------------------------------------------------------- #
def build_panel(prices: pd.DataFrame, ipo_tbl: pd.DataFrame, bench: str = "SPY",
                asof: str = "2026-06-30") -> dict:
    """Turn the daily-close tape + curated anchor table into the ``Panel`` shape.

    For every curated name we resample its adjusted close to **month-end**, form the monthly
    total return, and — only from its first full trading month onward — record:

    * ``gap[m, n]``     = ``log(price_end_of_month / offer)`` (distance from the anchor);
    * ``below[m, n]``   = ``price_end_of_month < offer``;
    * ``fwd_abn[m, n]`` = the name's month-``m+1`` total return **minus** ``bench``'s
      (market-adjusted forward return; the value known at ``m`` predicts ``m+1``).

    Returns ``{months, names, gap, fwd_abn, below}`` with ``(M, N)`` arrays — the same shape
    :func:`ipo_anchor.data.synthetic_panel` emits, so every downstream stat is shared.
    Vectorised: monthly resample + column algebra, no per-date Python loop over the tape.
    """
    px = prices[prices.index <= pd.Timestamp(asof)].sort_index()
    names = [t for t in ipo_tbl.index if t in px.columns]
    monthly = px[names + [bench]].resample("ME").last()
    ret = monthly.pct_change()                       # monthly total returns
    bench_ret = ret[bench]

    months = monthly.index
    M, N = len(months), len(names)
    gap = np.full((M, N), np.nan)
    fwd = np.full((M, N), np.nan)
    below = np.zeros((M, N), dtype=bool)

    offers = ipo_tbl.loc[names, "offer"].to_numpy(dtype=float)
    first = pd.to_datetime(ipo_tbl.loc[names, "first_trade"]).to_numpy()
    price_mat = monthly[names].to_numpy(dtype=float)
    ret_mat = ret[names].to_numpy(dtype=float)
    bench_mat = bench_ret.to_numpy(dtype=float)
    month_ts = months.to_numpy()

    for j in range(N):
        # first FULL month strictly after the listing month (a clean month of trading)
        live = month_ts > (first[j] + np.timedelta64(31, "D"))
        p = price_mat[:, j]
        g = np.log(p / offers[j])
        g = np.where(live & np.isfinite(p) & (p > 0), g, np.nan)
        gap[:, j] = g
        below[:, j] = np.where(np.isfinite(g), p < offers[j], False)
        # forward market-adjusted return: shift the abnormal next-month return back to m
        abn = ret_mat[:, j] - bench_mat
        fwd_next = np.roll(abn, -1)
        fwd_next[-1] = np.nan
        fwd[:, j] = np.where(np.isfinite(g), fwd_next, np.nan)

    return {"months": months, "names": names, "gap": gap, "fwd_abn": fwd, "below": below}


def panel_coverage(P: dict) -> dict:
    """Descriptive coverage of a panel: names, months, live name-months, below-offer share."""
    g = P["gap"]
    live = np.isfinite(g) & np.isfinite(P["fwd_abn"])
    per_month = live.sum(axis=1)
    active = per_month[per_month >= MIN_NAMES]
    below_share = float(P["below"][live].mean()) if live.any() else float("nan")
    return {
        "n_names": len(P["names"]),
        "n_months": int(len(P["months"])),
        "n_active_months": int(len(active)),
        "n_obs": int(live.sum()),
        "avg_names_per_active_month": float(active.mean()) if len(active) else float("nan"),
        "below_offer_share": below_share,
    }


# --------------------------------------------------------------------------- #
# Inference primitives (shared with the desk's canon)
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
    u = x - x.mean()
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Test 1 — the anchoring pull (monthly Fama-MacBeth cross-sectional slope)
# --------------------------------------------------------------------------- #
def fm_slopes(P: dict, min_names: int = MIN_NAMES) -> pd.Series:
    """Monthly cross-sectional OLS slope of forward market-adjusted return on the gap.

    On each month with at least ``min_names`` live names, regress ``fwd_abn`` on ``gap``
    (plus intercept) across names and keep the slope. Anchoring pull ⇒ a **negative** slope
    (stretched-above names under-earn next month). Returns the slope series indexed by month.
    """
    G, Y = P["gap"], P["fwd_abn"]
    months = P["months"]
    out_vals, out_idx = [], []
    for i in range(len(months)):
        x = G[i]; y = Y[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_names:
            continue
        xv = x[m]; yv = y[m]
        xc = xv - xv.mean()
        denom = float(xc @ xc)
        if denom <= 0:
            continue
        slope = float(xc @ (yv - yv.mean()) / denom)
        out_vals.append(slope); out_idx.append(months[i])
    return pd.Series(out_vals, index=pd.Index(out_idx, name="month"), name="fm_slope")


def anchoring_stats(P: dict, lags: int = 6, min_names: int = MIN_NAMES) -> dict:
    """Average Fama-MacBeth anchoring slope with a Newey-West *t*.

    ``slope_bps_per_10pct`` re-expresses the slope as the forward abnormal return (bps/mo) for
    a name trading 10% (``0.1`` in log-gap) above its offer — the sign the claim cares about.
    """
    s = fm_slopes(P, min_names)
    arr = s.to_numpy(dtype=float)
    mean = float(np.nanmean(arr)) if len(arr) else float("nan")
    return {
        "n_months": int(len(s)),
        "mean_slope": mean,
        "slope_bps_per_10pct": mean * 0.10 * 1e4,
        "t_nw": newey_west_t(arr, lags),
        "t_1s": one_sample_t(arr),
        "share_negative": float((arr < 0).mean()) if len(arr) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Test 2 — the below-offer drag (below vs above basket spread)
# --------------------------------------------------------------------------- #
def below_offer_spreads(P: dict, min_side: int = 3) -> pd.DataFrame:
    """Monthly below-offer minus above-offer equal-weight forward-abnormal-return spread.

    Each month split the live names by the below-offer flag; ``below`` = mean forward abnormal
    return of the below-offer names, ``above`` = mean of the above-offer names, ``spread =
    below - above``. A month enters only if BOTH sides have at least ``min_side`` names (so the
    spread is a genuine contrast, not one basket vs a near-empty other). Drag ⇒ negative spread.
    """
    B = P["below"]; Y = P["fwd_abn"]; G = P["gap"]
    months = P["months"]
    rows_sp, rows_lo, rows_hi, rows_n, idx = [], [], [], [], []
    for i in range(len(months)):
        y = Y[i]; live = np.isfinite(G[i]) & np.isfinite(y)
        if live.sum() < 2 * min_side:
            continue
        below_m = live & B[i]
        above_m = live & ~B[i]
        if below_m.sum() < min_side or above_m.sum() < min_side:
            continue
        lo = float(y[below_m].mean())   # below-offer basket
        hi = float(y[above_m].mean())   # above-offer basket
        rows_sp.append(lo - hi); rows_lo.append(lo); rows_hi.append(hi)
        rows_n.append(int(live.sum())); idx.append(months[i])
    return pd.DataFrame(
        {"spread": rows_sp, "below": rows_lo, "above": rows_hi, "n": rows_n},
        index=pd.Index(idx, name="month"),
    )


def below_offer_stats(spreads: pd.DataFrame, lags: int = 6) -> dict:
    """Mean below-minus-above spread (bps/mo), annualised, HAC *t*, plus each leg."""
    sp = spreads["spread"].to_numpy(dtype=float)
    return {
        "n_months": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4) if len(sp) else float("nan"),
        "t_nw": newey_west_t(sp, lags),
        "t_1s": one_sample_t(sp),
        "below_bps": float(np.nanmean(spreads["below"].to_numpy()) * 1e4) if len(sp) else float("nan"),
        "above_bps": float(np.nanmean(spreads["above"].to_numpy()) * 1e4) if len(sp) else float("nan"),
        "welch_t": welch_t(spreads["below"].to_numpy(), spreads["above"].to_numpy()),
        "ann_pct": (np.expm1(np.log1p(np.nanmean(sp)) * MONTHS)) * 100.0 if len(sp) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the FM slope a lucky alignment of gap and forward return?
# --------------------------------------------------------------------------- #
def placebo_pvalue(P: dict, n_seeds: int = 20, n_draws_per_seed: int = 50,
                   base_seed: int = 874, min_names: int = MIN_NAMES) -> dict:
    """Shuffle the gap→forward-return link WITHIN each month, recompute the mean FM slope.

    Each month's cross-sectional gap vector is permuted across names (the month's own gap and
    forward-return distributions are preserved; only their pairing is broken). p_left = share
    of permuted worlds whose mean slope is ``<= observed`` — the anchoring-pull direction is
    negative, so a real pull sits in the LEFT tail of the permutation null.
    """
    G, Y = P["gap"], P["fwd_abn"]
    months = P["months"]
    # pre-extract each qualifying month's finite (gap, fwd) vectors
    xs, ys = [], []
    for i in range(len(months)):
        x = G[i]; y = Y[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < min_names:
            continue
        xs.append(x[m]); ys.append(y[m])
    obs = float(fm_slopes(P, min_names).mean()) if xs else float("nan")

    def _mean_slope(perm_xs):
        slopes = []
        for xv, yv in zip(perm_xs, ys):
            xc = xv - xv.mean()
            denom = float(xc @ xc)
            if denom <= 0:
                continue
            slopes.append(float(xc @ (yv - yv.mean()) / denom))
        return float(np.mean(slopes)) if slopes else float("nan")

    draws = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed)
        for _ in range(n_draws_per_seed):
            perm_xs = [rng.permutation(xv) for xv in xs]
            draws.append(_mean_slope(perm_xs))
    draws = np.asarray([d for d in draws if np.isfinite(d)])
    return {
        "obs_slope": obs,
        "placebo_mean": float(draws.mean()) if len(draws) else float("nan"),
        "placebo_sd": float(draws.std(ddof=1)) if len(draws) > 1 else float("nan"),
        "p_left": float((draws <= obs).mean()) if len(draws) else float("nan"),
        "p_two_sided": float((np.abs(draws) >= abs(obs)).mean()) if len(draws) else float("nan"),
        "n_draws": int(len(draws)),
    }


# --------------------------------------------------------------------------- #
# Robustness — sub-era split of the below-offer spread
# --------------------------------------------------------------------------- #
def era_split(spreads: pd.DataFrame, cut: str, lags: int = 6) -> dict:
    """Below-offer spread stats on the two halves either side of ``cut`` (a month string)."""
    idx = spreads.index
    if isinstance(idx, pd.PeriodIndex):
        mask = idx < pd.Period(cut, freq="M")
    else:
        mask = idx < pd.Timestamp(cut)
    early = below_offer_stats(spreads[mask], lags)
    late = below_offer_stats(spreads[~mask], lags)
    return {"cut": cut, "early": early, "late": late}


# --------------------------------------------------------------------------- #
# The costed timer — SHORT below-offer / LONG above-offer, net of friction
# --------------------------------------------------------------------------- #
def timer_stats(spreads: pd.DataFrame, cost_bps: float = 10.0,
                borrow_ann_pct: float = 3.0, lags: int = 6) -> dict:
    """Cost the tradable expression of the drag: SHORT below-offer / LONG above-offer.

    The claim (below-offer is a drag) is traded by shorting the below-offer basket and going
    long the above-offer basket — i.e. the strategy return is ``-(below - above) = above -
    below``. Newly-listed / recently-crossed names make the basket turn over roughly monthly;
    we charge a conservative full round-trip (``2 × one-way × NAV``, both legs) per month plus
    borrow on the short (below-offer) leg. Small-cap recent-IPO borrow is dear, so 3%/yr is a
    floor. Returns gross/net bps/mo, net annualised %, net HAC *t*, worst drawdown.
    """
    d = -spreads["spread"].to_numpy(dtype=float)     # short below / long above
    d = d[~np.isnan(d)]
    n = len(d)
    round_trip = 2.0 * cost_bps / 1e4
    borrow_m = borrow_ann_pct / 100.0 / 12.0
    net = d - round_trip - borrow_m
    gross_mean = float(d.mean()) if n else float("nan")
    net_mean = float(net.mean()) if n else float("nan")
    return {
        "n_months": n,
        "cost_bps": cost_bps, "borrow_ann_pct": borrow_ann_pct,
        "gross_bps": gross_mean * 1e4,
        "gross_t": newey_west_t(d, lags),
        "net_bps": net_mean * 1e4,
        "net_t": newey_west_t(net, lags),
        "net_ann_pct": (np.expm1(np.log1p(net_mean) * MONTHS)) * 100.0 if n else float("nan"),
        "worst_drawdown_pct": _max_drawdown(net) * 100.0 if n else float("nan"),
    }


def _max_drawdown(monthly: np.ndarray) -> float:
    if len(monthly) == 0:
        return float("nan")
    nav = np.exp(np.cumsum(np.log1p(monthly)))
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(P: dict, lags: int = 6, min_names: int = MIN_NAMES) -> dict:
    """Run the anchoring-slope + below-offer-spread detectors on a synthetic panel."""
    a = anchoring_stats(P, lags, min_names)
    sp = below_offer_spreads(P, min_side=3)
    b = below_offer_stats(sp, lags)
    return {
        "mean_slope": a["mean_slope"], "anchor_t": a["t_nw"], "anchor_n": a["n_months"],
        "spread_bps": b["spread_bps"], "spread_t": b["t_nw"], "spread_n": b["n_months"],
    }


def synthetic_control(edge: float, n_seeds: int = 20, lags: int = 6,
                      base_seed: int = 874) -> dict:
    """Run the anchoring-slope detector on ``n_seeds`` synthetic worlds with planted ``edge``.

    Averaged over ≥ 20 seeds per desk law. With ``edge=0`` the |t| ≥ 2 rejection rate must sit
    near nominal 5%; with a large planted pull it must fire (a negative mean slope).
    """
    from . import data as _data
    slopes, ts = [], []
    for s in range(n_seeds):
        P = _data.synthetic_panel(edge=edge, seed=base_seed + s)
        r = synthetic_detect(P, lags)
        slopes.append(r["mean_slope"]); ts.append(r["anchor_t"])
    ts = np.asarray(ts)
    return {
        "edge": edge, "n_seeds": int(n_seeds),
        "mean_slope": float(np.mean(slopes)),
        "mean_t": float(np.nanmean(ts)),
        "reject_rate": float((np.abs(ts) >= 2.0).mean()),
    }
