"""Estimator + inference for Study 930 — the when-issued window of a spin-off.

A US spin-off has three distinct clocks, and the folklore attaches a different story to
each. This study measures all three on the *same* curated event table:

1. **The parent run-up** — from the public announcement of the separation to the day
   before the child trades regular way. The claim: the market re-rates the parent as the
   conglomerate discount is dismantled, so the parent beats the market *before* anything
   is actually distributed.
2. **The child's first sessions** — 5, 10 and 21 regular-way sessions. The claim
   (Greenblatt's forced-seller story, in its *short-horizon* form): index funds and
   mandate-bound holders dump the child in the first days, so the child is cheap and
   snaps back.
3. **The combined entity** — parent plus child in the distribution ratio, held from the
   first regular-way close. The claim: the sum of the parts, once separately priced,
   is worth more than the pre-spin whole.

**The one execution lag.** Every leg is entered at a *close* one full session after the
information is public, and exited at a close:

- parent leg: enter at the close of the first session **strictly after** the announcement
  date (announcements land pre-market, intraday or after the bell — acting at the next
  close is the only convention that is honest for all three); exit at the last close
  **strictly before** the child's first regular-way session, so no parent return ever
  crosses the distribution date and Yahoo's inconsistent spin-off adjustment never enters.
- child leg: enter at the close of the **first regular-way session** (so a full session of
  regular-way price discovery has already happened and the day-one pop is *excluded*),
  exit at the close ``h`` sessions later.
- combined leg: enter both names at the first regular-way close, exit 21 sessions later.

**Costs.** One-way ``cost_bps`` x NAV on the asset side, charged on entry and on exit
(a round trip = 2x). The benchmark side of every alpha is a **short** SPY position: it
pays its own round-trip cost and a **borrow** fee accrued over the holding period. All
alphas are therefore cash-neutral spreads, so excess-of-cash *is* the spread itself.

Because the child leg's raw alpha is **negative**, costs charged to that long position
make the headline *more* negative. That is the wrong direction for a headline, so the
stated effect size is the **gross** abnormal return and the net figure is quoted beside it
only to show that frictions cannot be what creates the effect. The costed number that
decides tradability is the mirror trade in ``short_child_economics``, where the frictions
land on the side you would actually pay them.

**Unit beta is an ASSUMPTION.** Every alpha is a 1:1 dollar spread, asset minus index —
no beta is estimated, and a freshly spun child is typically higher-beta and smaller-cap
than SPY. Over a five-session window the beta error is second-order next to the
idiosyncratic move, and the ``bench`` swap to IWM and MDY is the check on it, but the
estimator is a *dollar* hedge and never claims to be a risk-model alpha.

**Price-only vs total-return.** Every series is a Yahoo ``auto_adjust=True`` close, i.e.
**total return** (splits and cash dividends reinvested). The one thing it does not
capture is the spin-off distribution itself, which is exactly why no window crosses it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

CHILD_HORIZONS = (5, 10, 21)
COMBINED_WINDOW = 21


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk lineage)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain cross-sectional t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 2))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a hit rate — honest at the small n an event study has."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def bootstrap_mean_ci(x, n_boot: int = 5000, seed: int = 930, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the mean of a cross-section of event alphas.

    Events are resampled with replacement (i.i.d. across events — the cross-sectional
    analogue of a block bootstrap, since the windows are mostly non-overlapping).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = x[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((means < 0).mean()), "n_obs": n, "n_boot": int(n_boot)}


def block_bootstrap_mean_ci(x, block: int = 21, n_boot: int = 2000,
                            seed: int = 930, alpha: float = 0.05) -> dict:
    """Circular block bootstrap CI for the mean of a *daily* series (spread returns)."""
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[b] = r[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((means < 0).mean()), "n_obs": n, "block": block}


def family_wise_p(panel: pd.DataFrame, cols, n_boot: int = 5000, seed: int = 930) -> dict:
    """Westfall-Young max-|t| family-wise p across the study's PRE-SPECIFIED legs.

    Five legs were pre-specified (parent, child +5/+10/+21, parent+child). Quoting the
    best of five as if it were the only one tested is exactly the sin this desk audits for,
    so the family-wise error is priced here rather than argued away.

    Events are resampled jointly across legs after each leg is centred on its own mean, so
    the null is imposed while the (strong) cross-leg correlation — child +5, +10 and +21
    share their first five sessions — is preserved. Bonferroni would over-punish precisely
    because of that correlation; this does not.

    Returns the observed t of every leg, the single-test bootstrap p of the strongest leg,
    and the family-wise p of that same leg.
    """
    cols = list(cols)
    X = panel[cols].to_numpy(dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    n = len(X)
    if n < 3 or not cols:
        return {"n": n, "p_fwe": float("nan"), "p_single": float("nan")}
    t_obs = np.array([one_sample_t(X[:, j]) for j in range(X.shape[1])])
    best = int(np.argmax(np.abs(t_obs)))
    Xc = X - X.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    B = Xc[draws]
    sd = B.std(axis=1, ddof=1)
    tb = np.where(sd > 0, B.mean(axis=1) / np.where(sd > 0, sd, 1.0) * np.sqrt(n), 0.0)
    max_t = np.abs(tb).max(axis=1)
    t_best = float(abs(t_obs[best]))
    return {
        "n": int(n), "n_boot": int(n_boot), "n_legs": len(cols),
        "t_obs": {c: float(t) for c, t in zip(cols, t_obs)},
        "best_leg": cols[best], "t_best": float(t_obs[best]),
        "p_single": float((np.abs(tb[:, best]) >= t_best).mean()),
        "p_fwe": float((max_t >= t_best).mean()),
    }


def sharpe(daily, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(daily).astype(float).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #
def _pos_on_or_after(idx: pd.DatetimeIndex, d) -> int:
    return int(idx.searchsorted(pd.Timestamp(d), side="left"))


def _pos_strictly_after(idx: pd.DatetimeIndex, d) -> int:
    return int(idx.searchsorted(pd.Timestamp(d), side="right"))


def _ret_between(s: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    """Buy-and-hold return of ``s`` between two calendar dates, snapping to the tape.

    Used for the benchmark leg so it spans the *identical calendar window* as the asset
    leg even when the two series have slightly different trading calendars.
    """
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    i0 = _pos_on_or_after(s.index, d0)
    i1 = _pos_on_or_after(s.index, d1)
    if i0 >= len(s) or i1 >= len(s) or i1 <= i0:
        return float("nan")
    return float(s.iloc[i1] / s.iloc[i0] - 1.0)


# --------------------------------------------------------------------------- #
# The event panel — one row per spin-off, one column per leg
# --------------------------------------------------------------------------- #
def event_panel(
    prices: pd.DataFrame,
    events: list[dict],
    bench: str = "SPY",
    horizons: tuple[int, ...] = CHILD_HORIZONS,
    combined_window: int = COMBINED_WINDOW,
    cost_bps: float = 10.0,
    bench_cost_bps: float = 1.0,
    borrow_bps: float = 40.0,
    ann_entry_shift: int = 0,
    rw_shift: int = 0,
    ratio_mult: float = 1.0,
    equal_weight_combined: bool = False,
) -> pd.DataFrame:
    """Build the per-event alpha panel for all three legs.

    Parameters
    ----------
    prices:
        Daily total-return closes, one column per ticker (parents, children, benchmark).
    events:
        The event table (``data.SPINOFF_EVENTS`` or a synthetic one, same schema).
    cost_bps, bench_cost_bps:
        One-way costs (x NAV) on the asset and benchmark sides. Each leg is a round trip,
        so the charge is ``2 x`` each.
    borrow_bps:
        Annualised borrow on the **short benchmark** side of every alpha, accrued over the
        actual holding period. An ASSUMPTION (SPY is easy to borrow; 40 bp/yr is generous)
        — swept in ``cost_sweep``.
    ann_entry_shift:
        Shift the parent-leg entry by this many sessions. The announcement date is a
        hand-compiled PROXY; a +/- 5 sweep shows whether the run-up estimate depends on it.
    rw_shift:
        Shift the regular-way anchor by this many sessions. The first regular-way session
        is the other hand-compiled PROXY, and it is the one the child leg hangs on: a
        negative shift risks anchoring inside the when-issued period, a positive one starts
        the clock late. Swept in ``rw_shift_sweep``.
    ratio_mult, equal_weight_combined:
        Sensitivity knobs on the distribution ratio, which is the other non-tape input and
        touches only the combined leg.

    Returns a frame with one row per usable event and columns:
    ``parent, child, name, announce, regular_way, entry, exit, rw0, parent_days,
    r_parent, b_parent, a_parent_gross, a_parent_net, a_parent_per21,
    r_child_{h}, b_child_{h}, a_child_{h}_gross, a_child_{h}_net (for each horizon),
    w_child, r_comb, b_comb, a_comb_gross, a_comb_net``.
    """
    bs = prices[bench].dropna()
    cost = cost_bps * 1e-4
    bcost = bench_cost_bps * 1e-4
    borrow = borrow_bps * 1e-4

    rows = []
    for ev in events:
        p_tk, c_tk = ev["parent"], ev["child"]
        if p_tk not in prices.columns or c_tk not in prices.columns:
            continue
        ps = prices[p_tk].dropna()
        cs = prices[c_tk].dropna()
        if len(ps) < 30 or len(cs) < 30:
            continue

        # --- anchor: the first regular-way session actually on the child's tape ------
        i_rw = _pos_on_or_after(cs.index, ev["regular_way"]) + int(rw_shift)
        if not (0 <= i_rw < len(cs)):
            continue
        rw0 = cs.index[i_rw]

        row: dict = {
            "parent": p_tk, "child": c_tk, "name": ev.get("name", f"{c_tk} / {p_tk}"),
            "announce": pd.Timestamp(ev["announce"]), "regular_way": pd.Timestamp(ev["regular_way"]),
            "rw0": rw0, "ratio": float(ev.get("ratio", 1.0)) * ratio_mult,
        }

        # --- leg 1: parent, announcement -> day before regular way ------------------
        i_in = _pos_strictly_after(ps.index, ev["announce"]) + int(ann_entry_shift)
        i_out = _pos_on_or_after(ps.index, rw0) - 1
        if 0 <= i_in < i_out < len(ps):
            d_in, d_out = ps.index[i_in], ps.index[i_out]
            r_par = float(ps.iloc[i_out] / ps.iloc[i_in] - 1.0)
            b_par = _ret_between(bs, d_in, d_out)
            days = int(i_out - i_in)
            a_gross = r_par - b_par
            a_net = a_gross - 2 * cost - 2 * bcost - borrow * days / TRADING_DAYS
            row.update({
                "entry": d_in, "exit": d_out, "parent_days": days,
                "r_parent": r_par, "b_parent": b_par,
                "a_parent_gross": a_gross, "a_parent_net": a_net,
                "a_parent_per21": a_net * 21.0 / days if days > 0 else np.nan,
            })
        else:
            row.update({"entry": pd.NaT, "exit": pd.NaT, "parent_days": np.nan,
                        "r_parent": np.nan, "b_parent": np.nan,
                        "a_parent_gross": np.nan, "a_parent_net": np.nan,
                        "a_parent_per21": np.nan})

        # --- leg 2: child, first h regular-way sessions -----------------------------
        for h in horizons:
            i_end = i_rw + h
            if i_end < len(cs):
                d_end = cs.index[i_end]
                r_ch = float(cs.iloc[i_end] / cs.iloc[i_rw] - 1.0)
                b_ch = _ret_between(bs, rw0, d_end)
                a_g = r_ch - b_ch
                a_n = a_g - 2 * cost - 2 * bcost - borrow * h / TRADING_DAYS
            else:
                r_ch = b_ch = a_g = a_n = np.nan
            row[f"r_child_{h}"] = r_ch
            row[f"b_child_{h}"] = b_ch
            row[f"a_child_{h}_gross"] = a_g
            row[f"a_child_{h}_net"] = a_n

        # --- leg 3: parent + child combined, first `combined_window` sessions --------
        i_pw = _pos_on_or_after(ps.index, rw0)
        i_cw = i_rw + combined_window
        if i_pw + combined_window < len(ps) and i_cw < len(cs):
            p0 = float(ps.iloc[i_pw]); p1 = float(ps.iloc[i_pw + combined_window])
            c0 = float(cs.iloc[i_rw]); c1 = float(cs.iloc[i_cw])
            ratio = row["ratio"]
            if equal_weight_combined:
                w_c = 0.5
            else:
                denom = p0 + ratio * c0
                w_c = float(ratio * c0 / denom) if denom > 0 else np.nan
            r_comb = (1.0 - w_c) * (p1 / p0 - 1.0) + w_c * (c1 / c0 - 1.0)
            b_comb = _ret_between(bs, rw0, cs.index[i_cw])
            a_g = r_comb - b_comb
            a_n = a_g - 2 * cost - 2 * bcost - borrow * combined_window / TRADING_DAYS
            row.update({"w_child": w_c, "r_comb": r_comb, "b_comb": b_comb,
                        "a_comb_gross": a_g, "a_comb_net": a_n})
        else:
            row.update({"w_child": np.nan, "r_comb": np.nan, "b_comb": np.nan,
                        "a_comb_gross": np.nan, "a_comb_net": np.nan})

        rows.append(row)

    panel = pd.DataFrame(rows)
    if len(panel):
        panel = panel.sort_values("rw0").reset_index(drop=True)
    return panel


# --------------------------------------------------------------------------- #
# Leg statistics
# --------------------------------------------------------------------------- #
def leg_stats(panel: pd.DataFrame, col: str, seed: int = 930) -> dict:
    """Mean, median, hit rate, cross-sectional t, HAC t and bootstrap CI for one leg.

    The cross-sectional t treats events as independent draws. The HAC t is computed on the
    *date-ordered* alpha series with a small Bartlett lag, which is the honest guard for
    the handful of events whose windows overlap in calendar time (two GE spins, two XPO
    spins, and the 2023-10-02 cluster).
    """
    if col not in panel.columns:
        return {"n": 0}
    x = panel[col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    boot = bootstrap_mean_ci(x, seed=seed)
    return {
        "n": n,
        "mean_pct": float(x.mean() * 100),
        "median_pct": float(np.median(x) * 100),
        "sd_pct": float(x.std(ddof=1) * 100) if n > 1 else float("nan"),
        "hit_rate": k / n,
        "hit_lo": lo, "hit_hi": hi,
        "t": one_sample_t(x),
        "t_hac": newey_west_t(x, lags=2),
        "ci_low_pct": boot["ci_low"] * 100,
        "ci_high_pct": boot["ci_high"] * 100,
        "frac_negative": boot["frac_negative"],
        "min_pct": float(x.min() * 100),
        "max_pct": float(x.max() * 100),
    }


def leg_table(panel: pd.DataFrame, horizons=CHILD_HORIZONS, net: bool = True,
              seed: int = 930) -> pd.DataFrame:
    """All three legs side by side (gross or net), one row per leg."""
    suf = "net" if net else "gross"
    specs = [("parent (ann -> dist)", f"a_parent_{suf}")]
    specs += [(f"child +{h}d", f"a_child_{h}_{suf}") for h in horizons]
    specs += [("parent+child +21d", f"a_comb_{suf}")]
    rows = []
    for label, col in specs:
        s = leg_stats(panel, col, seed=seed)
        if not s.get("n"):
            continue
        s["leg"] = label
        rows.append(s)
    return pd.DataFrame(rows).set_index("leg")


# --------------------------------------------------------------------------- #
# The tradable sleeve — a daily, hedged, costed spread
# --------------------------------------------------------------------------- #
def hedged_sleeve(
    prices: pd.DataFrame,
    events: list[dict],
    window: int = 21,
    bench: str = "SPY",
    cash: str = "BIL",
    cost_bps: float = 10.0,
    bench_cost_bps: float = 1.0,
    borrow_bps: float = 40.0,
    trim_to_events: bool = True,
) -> pd.DataFrame:
    """Daily returns of an equal-weight sleeve of children inside their first ``window``.

    Each event contributes weight from the session **after** its first regular-way close
    (the entry is that close) through ``window`` sessions later; when several events
    overlap the sleeve is equal-weighted across them. The short benchmark leg carries
    notional equal to the sleeve's gross exposure, pays its own round-trip cost and
    accrues borrow daily.

    Returns a daily frame with ``r_long`` (the children), ``r_bench``, ``r_cash``,
    ``exposure``, ``r_spread`` (long minus short, net of all frictions — cash-neutral, so
    this *is* the excess-of-cash return) and ``r_sleeve`` (the long leg parked in cash when
    idle, net) so a reader can see both the spread and the fund-level experience.

    **The calendar is the study window, not the cache's.** With ``trim_to_events`` the frame
    is cut to the first regular-way anchor .. the last exit. The shared desk cache is
    written by many studies and one of them may hold decades of SPY pre-history; leaving it
    in would make the sleeve's "active fraction" a statement about how much unrelated data
    happens to sit on disk rather than about this strategy's duty cycle.
    """
    idx = prices[bench].dropna().index
    # ``fill_method=None`` explicitly: the pandas 2.x default padded across gaps and 3.x
    # does not, and this study is run on both.
    ret = prices.reindex(idx).pct_change(fill_method=None)
    rb = ret[bench].fillna(0.0)
    rc = ret[cash].fillna(0.0) if cash in prices.columns else pd.Series(0.0, index=idx)

    W = pd.DataFrame(0.0, index=idx, columns=[e["child"] for e in events])
    span: list[pd.Timestamp] = []
    for ev in events:
        c_tk = ev["child"]
        if c_tk not in prices.columns:
            continue
        cs = prices[c_tk].dropna()
        i_rw = _pos_on_or_after(cs.index, ev["regular_way"])
        if i_rw >= len(cs):
            continue
        i_end = min(i_rw + window, len(cs) - 1)
        if i_end <= i_rw:
            continue
        d0, d1 = cs.index[i_rw], cs.index[i_end]
        mask = (idx > d0) & (idx <= d1)
        W.loc[mask, c_tk] = W.loc[mask, c_tk] + 1.0
        span += [d0, d1]

    active = W.sum(axis=1)
    Wn = W.div(active.where(active > 0), axis=0).fillna(0.0)

    child_ret = ret[[c for c in W.columns if c in ret.columns]].fillna(0.0)
    r_long = (Wn[child_ret.columns] * child_ret).sum(axis=1)
    exposure = (active > 0).astype(float)

    turn = Wn.diff().abs().sum(axis=1).fillna(Wn.iloc[0].abs().sum())
    bturn = exposure.diff().abs().fillna(exposure.iloc[0])
    fric = turn * cost_bps * 1e-4 + bturn * bench_cost_bps * 1e-4
    borrow_daily = exposure * borrow_bps * 1e-4 / TRADING_DAYS

    r_spread = r_long - exposure * rb - fric - borrow_daily
    r_sleeve = r_long + (1.0 - exposure) * rc - turn * cost_bps * 1e-4

    out = pd.DataFrame({
        "r_long": r_long, "r_bench": rb, "r_cash": rc,
        "exposure": exposure, "r_spread": r_spread, "r_sleeve": r_sleeve,
    })
    if trim_to_events and span:
        out = out[(out.index >= min(span)) & (out.index <= max(span))]
    return out


def sleeve_stats(sl: pd.DataFrame, seed: int = 930) -> dict:
    """Headline stats for the hedged sleeve, on active days and on the full calendar."""
    act = sl[sl["exposure"] > 0]
    spread = act["r_spread"]
    full = sl["r_spread"]
    e_sleeve = (sl["r_sleeve"] - sl["r_cash"]).dropna()
    e_bench = (sl["r_bench"] - sl["r_cash"]).dropna()
    boot = block_bootstrap_mean_ci(spread, seed=seed)
    return {
        "n_active_days": int(len(act)),
        "n_days": int(len(sl)),
        "active_frac": float((sl["exposure"] > 0).mean()),
        "spread_mean_bps": float(spread.mean() * 1e4),
        "spread_t_hac": newey_west_t(spread.to_numpy()),
        "spread_sharpe_active": sharpe(spread),
        "spread_ci_lo_bps": boot["ci_low"] * 1e4,
        "spread_ci_hi_bps": boot["ci_high"] * 1e4,
        "spread_frac_neg": boot["frac_negative"],
        "spread_t_full": newey_west_t(full.to_numpy()),
        "sleeve_excess_sharpe": sharpe(e_sleeve),
        "bench_excess_sharpe": sharpe(e_bench),
        "t_sleeve_minus_bench": newey_west_t((e_sleeve - e_bench).to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Robustness — eras, cost sweep, proxy sweeps
# --------------------------------------------------------------------------- #
def era_cut(panel: pd.DataFrame, col: str, split: str = "2020-01-01", seed: int = 930) -> dict:
    """Split the event table by distribution date and re-run one leg on each half."""
    out = {}
    d = pd.Timestamp(split)
    for tag, sub in [("early", panel[panel["rw0"] < d]), ("late", panel[panel["rw0"] >= d])]:
        out[tag] = leg_stats(sub, col, seed=seed) if len(sub) else {"n": 0}
    return out


def cost_sweep(prices: pd.DataFrame, events: list[dict], col: str,
               cost_grid=(0.0, 5.0, 10.0, 25.0, 50.0),
               borrow_grid=(0.0, 40.0, 100.0), seed: int = 930) -> pd.DataFrame:
    """Sweep the one-way asset cost and the short-benchmark borrow for one leg."""
    rows = []
    for c in cost_grid:
        for b in borrow_grid:
            panel = event_panel(prices, events, cost_bps=c, borrow_bps=b)
            s = leg_stats(panel, col, seed=seed)
            rows.append({"cost_bps": c, "borrow_bps": b,
                         "mean_pct": s.get("mean_pct", np.nan),
                         "t": s.get("t", np.nan),
                         "hit_rate": s.get("hit_rate", np.nan)})
    return pd.DataFrame(rows)


def announce_shift_sweep(prices: pd.DataFrame, events: list[dict],
                         shifts=(-5, -2, 0, 2, 5), cost_bps: float = 10.0,
                         seed: int = 930) -> pd.DataFrame:
    """Sweep the parent-leg entry around the (PROXY) announcement date."""
    rows = []
    for s in shifts:
        panel = event_panel(prices, events, cost_bps=cost_bps, ann_entry_shift=int(s))
        st_ = leg_stats(panel, "a_parent_net", seed=seed)
        rows.append({"shift_sessions": int(s), "n": st_.get("n", 0),
                     "mean_pct": st_.get("mean_pct", np.nan),
                     "t": st_.get("t", np.nan),
                     "per21_mean_pct": leg_stats(panel, "a_parent_per21", seed=seed).get("mean_pct", np.nan)})
    return pd.DataFrame(rows)


def rw_shift_sweep(prices: pd.DataFrame, events: list[dict], col: str = "a_child_5_net",
                   shifts=(-3, -1, 0, 1, 3), cost_bps: float = 10.0,
                   seed: int = 930) -> pd.DataFrame:
    """Sweep the (PROXY) first-regular-way-session anchor that the child leg hangs on."""
    rows = []
    for s in shifts:
        panel = event_panel(prices, events, cost_bps=cost_bps, rw_shift=int(s))
        st_ = leg_stats(panel, col, seed=seed)
        rows.append({"rw_shift_sessions": int(s), "n": st_.get("n", 0),
                     "mean_pct": st_.get("mean_pct", np.nan),
                     "t": st_.get("t", np.nan),
                     "hit_rate": st_.get("hit_rate", np.nan)})
    return pd.DataFrame(rows).set_index("rw_shift_sessions")


def ratio_sweep(prices: pd.DataFrame, events: list[dict], cost_bps: float = 10.0,
                seed: int = 930) -> pd.DataFrame:
    """Sweep the (PROXY) distribution ratio, which touches only the combined leg."""
    rows = []
    for label, kw in [("ratio x0.5", dict(ratio_mult=0.5)),
                      ("ratio x1 (table)", dict(ratio_mult=1.0)),
                      ("ratio x2", dict(ratio_mult=2.0)),
                      ("equal weight", dict(equal_weight_combined=True))]:
        panel = event_panel(prices, events, cost_bps=cost_bps, **kw)
        s = leg_stats(panel, "a_comb_net", seed=seed)
        rows.append({"variant": label, "n": s.get("n", 0),
                     "mean_pct": s.get("mean_pct", np.nan), "t": s.get("t", np.nan),
                     "hit_rate": s.get("hit_rate", np.nan)})
    return pd.DataFrame(rows).set_index("variant")


def short_child_economics(
    panel: pd.DataFrame,
    horizon: int = 5,
    cost_bps: float = 10.0,
    bench_cost_bps: float = 1.0,
    borrow_grid_pct=(0.0, 5.0, 25.0, 50.0, 100.0),
    spy_borrow_bps: float = 40.0,
    seed: int = 930,
) -> pd.DataFrame:
    """The other side of a negative child alpha: **short the child, long SPY**.

    A negative alpha is only bankable if the short is actually financeable, and a
    freshly distributed spin-off is the textbook hard-to-borrow name: a small float, no
    settled lending supply, and index funds holding most of what exists. The annualised
    borrow rate is therefore a pure **ASSUMPTION** and is swept from 0% (free, fantasy)
    to 100%/yr (a genuinely special name). Per-trade P&L:

        -(r_child - r_bench) - 2 x cost_bps (child) - 2 x bench_cost_bps (SPY)
        - child_borrow x h/252

    Note the SPY leg is now **long**, so its borrow disappears; only the child pays.
    ``spy_borrow_bps`` is kept in the signature for symmetry with ``event_panel`` and is
    unused here (documented rather than silently dropped).
    """
    g = panel[f"a_child_{horizon}_gross"].to_numpy(dtype=float)
    g = g[np.isfinite(g)]
    rows = []
    for b in borrow_grid_pct:
        pnl = -g - 2 * cost_bps * 1e-4 - 2 * bench_cost_bps * 1e-4 - (b / 100.0) * horizon / TRADING_DAYS
        boot = bootstrap_mean_ci(pnl, seed=seed)
        k = int((pnl > 0).sum())
        lo, hi = wilson_interval(k, len(pnl))
        rows.append({
            "child_borrow_pct_yr": b,
            "n": len(pnl),
            "mean_pct": float(pnl.mean() * 100),
            "median_pct": float(np.median(pnl) * 100),
            "t": one_sample_t(pnl),
            "hit_rate": k / len(pnl),
            "hit_lo": lo, "hit_hi": hi,
            "ci_low_pct": boot["ci_low"] * 100,
            "ci_high_pct": boot["ci_high"] * 100,
        })
    return pd.DataFrame(rows).set_index("child_borrow_pct_yr")


def caar_path(prices: pd.DataFrame, events: list[dict], bench: str = "SPY",
              pre: int = 5, post: int = 21) -> pd.DataFrame:
    """Cumulative average abnormal return around the first regular-way close.

    Event time 0 is that close. Negative event time therefore sits inside the
    **when-issued** period (the child already trades, but the distribution has not
    settled); positive event time is regular way. Abnormal = child minus benchmark, both
    on the same calendar sessions; gross, since this is a shape diagnostic and not a P&L.
    """
    bs = prices[bench].dropna()
    rows = []
    for ev in events:
        c_tk = ev["child"]
        if c_tk not in prices.columns:
            continue
        cs = prices[c_tk].dropna()
        i_rw = _pos_on_or_after(cs.index, ev["regular_way"])
        if i_rw - pre < 0 or i_rw + post >= len(cs):
            continue
        base = float(cs.iloc[i_rw])
        d0 = cs.index[i_rw]
        j0 = _pos_on_or_after(bs.index, d0)
        if j0 >= len(bs):
            continue
        b_base = float(bs.iloc[j0])
        rec = {}
        for k in range(-pre, post + 1):
            d = cs.index[i_rw + k]
            j = _pos_on_or_after(bs.index, d)
            if j >= len(bs):
                rec[k] = np.nan
                continue
            rec[k] = float(cs.iloc[i_rw + k] / base) - float(bs.iloc[j] / b_base)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "caar_pct": df.mean(axis=0) * 100,
        "median_pct": df.median(axis=0) * 100,
        "n": df.notna().sum(axis=0),
    })
    out.index.name = "event_session"
    return out


def horizon_shape(panel: pd.DataFrame, horizons=(1, 2, 3, 5, 10, 21, 63),
                  net: bool = False, seed: int = 930) -> pd.DataFrame:
    """The child alpha as a function of holding horizon — an EXPLORATORY shape diagnostic.

    Only 5 / 10 / 21 sessions are pre-specified; the rest of the grid is searched *after*
    seeing the data, so its t-stats carry an unpriced multiple-comparison penalty and must
    never be quoted as the headline. It is here because the *shape* — where the trough
    sits and how fast it fills in — is what identifies the mechanism.
    """
    suf = "net" if net else "gross"
    rows = []
    for h in horizons:
        col = f"a_child_{h}_{suf}"
        if col not in panel.columns:
            continue
        s = leg_stats(panel, col, seed=seed)
        s["sessions"] = h
        rows.append(s)
    return pd.DataFrame(rows).set_index("sessions")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, events: list[dict], cost_bps: float = 0.0,
                     borrow_bps: float = 0.0, seed: int = 930) -> dict:
    """Run the estimator on a synthetic world and report what it recovered.

    On the planted world the parent run-up and the child's 21-session drift must both come
    back positive with a clear t; on the null both must sit near zero. This proves the
    windows, the lag and the alignment are unbiased — it never supports a real-tape stamp.
    """
    panel = event_panel(prices, events, cost_bps=cost_bps, bench_cost_bps=0.0,
                        borrow_bps=borrow_bps)
    par = leg_stats(panel, "a_parent_net", seed=seed)
    ch = leg_stats(panel, "a_child_21_net", seed=seed)
    return {
        "n_events": int(len(panel)),
        "parent_mean_pct": par.get("mean_pct", np.nan),
        "parent_t": par.get("t", np.nan),
        "child21_mean_pct": ch.get("mean_pct", np.nan),
        "child21_t": ch.get("t", np.nan),
        "child21_hit": ch.get("hit_rate", np.nan),
    }
