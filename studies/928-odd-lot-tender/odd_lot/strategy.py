"""Strategy + inference for Study 928 — Odd-Lot Priority.

The folk trade. An issuer announces a modified Dutch auction self-tender at a premium.
Almost every such offer carries an **odd-lot priority** clause: a holder of fewer than 100
shares who tenders *all* of them is bought **first and in full**, ahead of the proration
that hits everyone else. The folklore concludes: buy 99 shares in the open market, tender
them, collect the premium, walk away — a free arbitrage the round-lot world cannot touch.

What this file measures, and in what currency.

*The mechanical premium is an assumption; the price risk is tape.* The clearing price of a
tender is not in a price series, so the premium over the pre-announcement close is a
declared **PROXY** (``data.PREMIUM_DEFAULT``, swept across ``data.PREMIUM_GRID``). Every
*other* leg of the round trip is measured on the real tape:

- the **run-up** ``p_entry / p_pre − 1`` — the part of the premium the market has already
  taken by the time an outside buyer can act (the offer is public at the close of day
  ``t``; the odd lot is bought at the close of ``t+1`` — the study's one and only
  execution lag);
- the **offer-window drift** ``p_exp / p_entry − 1`` — what the shares do while you wait;
- the **post-expiry give-back** ``p_post / p_exp − 1`` — the only thing that makes odd-lot
  *priority* worth anything, because priority is only valuable when the tender price beats
  the price at which the un-filled stub can be sold.

Two decompositions carry the study:

    1 + r_odd  =  (1 + premium) / (1 + runup)          ["premium minus run-up"]
    r_odd − r_round  =  (1 − f) × [(p_clear − p_post) / p_entry + cost]
                                                       ["what priority is worth"]

where ``f`` is the round-lot holder's proration fill (a declared PROXY too; at ``f = 1``
the offer is under-subscribed and priority is worth exactly zero).

**Both identities carry the premium assumption, and so does every number derived from
them — including ``priority_value``.** ``p_clear`` is ``p_pre × (1 + premium)``, so "what
priority is worth" is the assumed premium net of the *measured* post-expiry price: it is a
statement about an assumption meeting the tape, not a tape fact, and its sign flips inside
the swept premium grid. ``premium_sweep`` reports it alongside the round trip precisely so
that this cannot be forgotten.

And one summary statistic that needs no *premium* assumption: the **breakeven premium** —
the clearing premium over the pre-announcement close at which the costed, market-hedged
odd-lot round trip merely matches SPY. It is free of the premium, but it is not free of
every assumption: it still inherits the pre-announcement **anchor** (which fixes
``p_pre``), the assumed **offer length** (which fixes the exit session) and the declared
**frictions**. Those are swept too, and the anchor is the widest of them.

Costs. One-way ``cost_bps`` × NAV on the market purchase; a flat broker corporate-action
fee (``tender_fee_usd``) on a position capped at 99 shares — on a ~$5,000 odd lot that
flat fee *is* the dominant cost, which is the whole tradability story; the market-hedged
variant shorts SPY and therefore pays **borrow** (swept). No other short leg exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Defaults mirrored from data.py so strategy can be used standalone.
ANCHOR_LAG_DEFAULT = 6      # trading days before commencement: the pre-announcement close
POST_DAYS_DEFAULT = 5       # trading days after expiry: the "post-expiry price"


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain cross-sectional t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int = 4) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    Applied to the **calendar-ordered** vector of event returns: self-tenders arrive in
    waves (a buyback boom is a market-wide event), so neighbouring events are not
    independent draws and the plain cross-sectional t overstates precision.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (float(mid - half), float(mid + half))


def block_bootstrap_mean_ci(x, n_boot: int = 4000, block: int = 5,
                            seed: int = 928, alpha: float = 0.05) -> dict:
    """Circular block bootstrap CI for the mean of a calendar-ordered event series.

    Blocks of ``block`` consecutive *events* keep the buyback-wave clustering intact.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": int(n),
            "block": int(block), "n_boot": int(n_boot)}


def annualised_sharpe(r, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(r).astype(float).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Per-event window extraction (pure tape — no assumption enters here)
# --------------------------------------------------------------------------- #
def event_row(
    stock: pd.Series,
    market: pd.Series,
    date: str,
    expiry_days: int = 21,
    anchor_lag: int = ANCHOR_LAG_DEFAULT,
    post_days: int = POST_DAYS_DEFAULT,
) -> dict:
    """Extract one event's price marks, or ``None`` if the window is not fully covered.

    ``t0`` is the first session on or after the SC TO-I commencement date. The marks:

    ==================  ==================================================
    ``p_pre``           close at ``t0 − anchor_lag`` (pre-leak reference)
    ``p_ann``           close at ``t0`` (the offer is public at this close)
    ``p_entry``         close at ``t0 + 1`` — **the one execution lag**
    ``p_exp``           close at ``t0 + expiry_days`` (assumed expiry)
    ``p_post``          close at ``t0 + expiry_days + post_days``
    ==================  ==================================================

    The market leg is marked on the *same* sessions. All closes are **total-return**
    (``auto_adjust=True``): a dividend paid inside the window is credited to the holder,
    which is what a round trip actually receives.
    """
    s = stock.dropna().sort_index()
    if len(s) == 0:
        return None
    ts = pd.Timestamp(date)
    pos = s.index.searchsorted(ts, side="left")
    if pos >= len(s):
        return None
    i0 = int(pos)
    i_last = i0 + expiry_days + post_days
    if i0 - anchor_lag < 0 or i_last >= len(s):
        return None

    dates = [s.index[i0 - anchor_lag], s.index[i0], s.index[i0 + 1],
             s.index[i0 + expiry_days], s.index[i_last]]
    m = market.dropna().sort_index().reindex(s.index).ffill()
    mv = [m.loc[d] for d in dates]
    if not np.all(np.isfinite(np.asarray(mv, dtype=float))):
        return None
    p_pre, p_ann, p_entry, p_exp, p_post = (float(s.iloc[i]) for i in
                                            (i0 - anchor_lag, i0, i0 + 1,
                                             i0 + expiry_days, i_last))
    m_pre, m_ann, m_entry, m_exp, m_post = (float(v) for v in mv)
    if min(p_pre, p_entry, p_exp, p_post) <= 0:
        return None

    return {
        "t0": s.index[i0], "entry_date": s.index[i0 + 1],
        "expiry_date": s.index[i0 + expiry_days], "post_date": s.index[i_last],
        "i0": i0, "expiry_days": int(expiry_days),
        "p_pre": p_pre, "p_ann": p_ann, "p_entry": p_entry,
        "p_exp": p_exp, "p_post": p_post,
        "m_pre": m_pre, "m_ann": m_ann, "m_entry": m_entry,
        "m_exp": m_exp, "m_post": m_post,
        # Pure-tape derived quantities
        "runup": p_entry / p_pre - 1.0,
        "runup_abn": (p_entry / p_pre) / (m_entry / m_pre) - 1.0,
        "ann_day": p_ann / s.iloc[i0 - 1] - 1.0,
        "drift": p_exp / p_entry - 1.0,
        "giveback": p_post / p_exp - 1.0,
        "perm": p_post / p_pre - 1.0,
        "perm_abn": (p_post / p_pre) / (m_post / m_pre) - 1.0,
        "mkt_entry_exp": m_exp / m_entry - 1.0,
        "mkt_entry_post": m_post / m_entry - 1.0,
    }


def build_event_table(
    panel: dict,
    events,
    market_key: str = "SPY",
    expiry_days: int = 21,
    anchor_lag: int = ANCHOR_LAG_DEFAULT,
    post_days: int = POST_DAYS_DEFAULT,
) -> pd.DataFrame:
    """Assemble the per-event tape table for every event whose window is fully covered.

    Rows are returned in calendar order of the *commencement* date, which is the order the
    HAC and block-bootstrap machinery assumes. Events whose issuer tape is missing (the
    survivorship drop-outs) or whose window runs off the end of the tape are skipped and
    counted by the caller.
    """
    market = panel[market_key]
    rows = []
    for tk, date, accession, label in events:
        if tk not in panel:
            continue
        row = event_row(panel[tk], market, date, expiry_days=expiry_days,
                        anchor_lag=anchor_lag, post_days=post_days)
        if row is None:
            continue
        row.update({"ticker": tk, "date": pd.Timestamp(date),
                    "accession": accession, "registrant": label})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    tbl = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    return tbl


# --------------------------------------------------------------------------- #
# The odd-lot round trip (this is where the declared assumptions enter)
# --------------------------------------------------------------------------- #
def odd_lot_returns(
    tbl: pd.DataFrame,
    premium: float = 0.13,
    proration: float = 0.35,
    cost_bps: float = 15.0,
    tender_fee_usd: float = 30.0,
    position_usd: float = 5000.0,
    borrow_bps: float = 40.0,
) -> pd.DataFrame:
    """Price the odd-lot and round-lot round trips on top of the tape table.

    ``premium`` and ``proration`` are declared PROXIES; ``cost_bps`` is charged one-way ×
    NAV on each market execution; ``tender_fee_usd`` is a flat broker corporate-action fee
    charged once per event on a ``position_usd`` position (this is why the odd lot's small
    size hurts); ``borrow_bps`` is the annual borrow paid on the short SPY leg of the
    market-hedged variant, pro-rated over the holding window.

    Columns added:

    - ``p_clear`` — the assumed clearing price ``p_pre × (1 + premium)``;
    - ``r_odd_gross`` / ``r_odd_net`` — the odd lot: bought at ``p_entry``, 100% filled at
      ``p_clear`` (that *is* the priority clause), net of buy cost and the flat fee;
    - ``r_odd_hedged`` — the same, minus the SPY return over the same sessions, minus the
      hedge's own round-trip cost and borrow: the market-neutral, fully costed number;
    - ``r_round_net`` — the round-lot holder: fraction ``proration`` filled at
      ``p_clear``, the stub marked and sold at ``p_post``;
    - ``priority_value`` — ``r_odd_net − r_round_net``, what the odd-lot clause is worth
      **under the assumed premium**: it is proportional to ``p_clear − p_post`` and so
      moves one-for-one with ``premium``, changing sign inside the swept grid;
    - ``breakeven_premium`` — the premium at which ``r_odd_hedged`` would be exactly zero
      (**no premium assumption enters this column**; the anchor, the offer length and the
      frictions still do).
    """
    if len(tbl) == 0:
        return tbl.copy()
    out = tbl.copy()
    c = cost_bps * 1e-4
    fee = tender_fee_usd / position_usd if position_usd > 0 else 0.0
    hold_years = out["expiry_days"].astype(float) / TRADING_DAYS
    borrow = borrow_bps * 1e-4 * hold_years

    out["p_clear"] = out["p_pre"] * (1.0 + premium)
    out["r_odd_gross"] = out["p_clear"] / out["p_entry"] - 1.0
    out["r_odd_net"] = out["r_odd_gross"] - c - fee
    # Market-hedged: short SPY over the same sessions (a round-trip cost + borrow).
    out["r_odd_hedged"] = out["r_odd_net"] - out["mkt_entry_exp"] - 2.0 * c - borrow

    # The round-lot holder: prorated fill, the stub sold after expiry (one more one-way
    # cost on the unsold portion) — and market-hedged over the longer entry->post window.
    stub = 1.0 - proration
    out["r_round_gross"] = (proration * out["p_clear"]
                            + stub * out["p_post"]) / out["p_entry"] - 1.0
    out["r_round_net"] = out["r_round_gross"] - c - fee - stub * c
    out["priority_value"] = out["r_odd_net"] - out["r_round_net"]

    # Breakeven premium: the clearing premium over p_pre at which r_odd_hedged == 0.
    out["breakeven_premium"] = (
        (out["p_entry"] / out["p_pre"])
        * (1.0 + out["mkt_entry_exp"] + c + fee + 2.0 * c + borrow) - 1.0
    )
    out["premium_margin"] = premium - out["breakeven_premium"]
    out["dollar_profit_odd"] = out["r_odd_hedged"] * position_usd
    return out


def summarise_events(ev: pd.DataFrame, col: str, block: int = 5,
                     seed: int = 928, lags: int = 4, boot_n: int = 4000) -> dict:
    """Mean, cross-sectional t, HAC t, block-bootstrap CI and hit rate for one column.

    ``boot_n = 0`` skips the bootstrap (the CI columns come back NaN) — used inside the
    sweeps, where only the point estimate and the HAC t are wanted.
    """
    x = pd.to_numeric(ev[col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"col": col, "n": 0}
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    boot = (block_bootstrap_mean_ci(x, n_boot=boot_n, block=block, seed=seed)
            if boot_n > 0 else
            {"ci_low": float("nan"), "ci_high": float("nan"),
             "frac_negative": float("nan")})
    return {
        "col": col, "n": n,
        "mean_pct": float(x.mean() * 100.0),
        "median_pct": float(np.median(x) * 100.0),
        "sd_pct": float(x.std(ddof=1) * 100.0) if n > 1 else float("nan"),
        "t_cross": one_sample_t(x),
        "t_hac": newey_west_t(x, lags=lags),
        "hit_rate": k / n, "hit_lo": lo, "hit_hi": hi,
        "ci_low_pct": boot["ci_low"] * 100.0, "ci_high_pct": boot["ci_high"] * 100.0,
        "frac_negative": boot["frac_negative"],
    }


def headline(
    tbl: pd.DataFrame,
    premium: float = 0.13,
    proration: float = 0.35,
    cost_bps: float = 15.0,
    tender_fee_usd: float = 30.0,
    position_usd: float = 5000.0,
    borrow_bps: float = 40.0,
    seed: int = 928,
    boot_n: int = 4000,
) -> dict:
    """The full headline: the tape legs, the priced round trips, and the breakeven."""
    ev = odd_lot_returns(tbl, premium=premium, proration=proration, cost_bps=cost_bps,
                         tender_fee_usd=tender_fee_usd, position_usd=position_usd,
                         borrow_bps=borrow_bps)
    if len(ev) == 0:
        return {"n_events": 0, "events": ev}
    legs = {c: summarise_events(ev, c, seed=seed, boot_n=boot_n)
            for c in ("runup", "runup_abn", "drift", "giveback", "perm", "perm_abn",
                      "r_odd_net", "r_odd_hedged", "r_round_net", "priority_value")}
    be = pd.to_numeric(ev["breakeven_premium"], errors="coerce").to_numpy(dtype=float)
    be = be[np.isfinite(be)]
    return {
        "n_events": int(len(ev)),
        "first": ev["date"].min(), "last": ev["date"].max(),
        "legs": legs,
        "breakeven_mean_pct": float(be.mean() * 100.0),
        "breakeven_median_pct": float(np.median(be) * 100.0),
        "breakeven_p75_pct": float(np.percentile(be, 75) * 100.0),
        "share_premium_below_breakeven": float((be > premium).mean()),
        "mean_dollar_profit": float(ev["dollar_profit_odd"].mean()),
        "events_per_year": float(len(ev) / max(
            (ev["date"].max() - ev["date"].min()).days / 365.25, 1e-9)),
        "premium": premium, "proration": proration, "cost_bps": cost_bps,
        "tender_fee_usd": tender_fee_usd, "position_usd": position_usd,
        "borrow_bps": borrow_bps,
        "events": ev,
    }


# --------------------------------------------------------------------------- #
# Sweeps over every declared assumption
# --------------------------------------------------------------------------- #
def premium_sweep(tbl: pd.DataFrame, grid=(0.05, 0.09, 0.13, 0.17, 0.21, 0.25),
                  **kw) -> pd.DataFrame:
    """The assumed clearing premium is the study's biggest lever — sweep it end to end.

    Both priced arms are swept, and that is the point: ``priority_value`` is **not** a
    tape fact either. ``p_clear`` is ``p_pre × (1 + premium)``, so the value of odd-lot
    priority — and the *t* attached to it — moves one-for-one with the assumption and
    changes sign inside the grid. Any sentence that calls the priority number "measured"
    without naming the premium is a sentence this column refutes.
    """
    kw.setdefault("boot_n", 0)
    rows = []
    for p in grid:
        h = headline(tbl, premium=p, **kw)
        s = h["legs"]["r_odd_hedged"]
        pv = h["legs"]["priority_value"]
        rows.append({"premium_pct": p * 100, "mean_pct": s["mean_pct"],
                     "t_hac": s["t_hac"], "t_cross": s["t_cross"],
                     "hit_rate": s["hit_rate"],
                     "ci_low_pct": s["ci_low_pct"], "ci_high_pct": s["ci_high_pct"],
                     "dollar": h["mean_dollar_profit"],
                     "priority_pct": pv["mean_pct"], "priority_t": pv["t_hac"]})
    return pd.DataFrame(rows)


def proration_sweep(tbl: pd.DataFrame, grid=(0.15, 0.25, 0.35, 0.50, 0.75, 1.00),
                    **kw) -> pd.DataFrame:
    """What odd-lot *priority* is worth as the round-lot fill varies (1.00 ⇒ nothing)."""
    kw.setdefault("boot_n", 0)
    rows = []
    for f in grid:
        h = headline(tbl, proration=f, **kw)
        s = h["legs"]["priority_value"]
        rows.append({"proration": f, "priority_pct": s["mean_pct"],
                     "t_hac": s["t_hac"], "hit_rate": s["hit_rate"],
                     "round_pct": h["legs"]["r_round_net"]["mean_pct"]})
    return pd.DataFrame(rows)


def cost_sweep(tbl: pd.DataFrame, fee_grid=(0.0, 15.0, 30.0, 50.0),
               bps_grid=(0.0, 15.0, 40.0, 100.0), **kw) -> pd.DataFrame:
    """Cost sweep: the flat broker corporate-action fee dominates on a 99-share lot.

    ``cost_bps`` is charged **three times** per round trip (one-way on the odd-lot
    purchase, plus the SPY hedge's round trip), so the bps axis bites harder than it
    looks. The grid runs to **100 bps** on purpose: a third of this event list is
    micro-cap (Socket Mobile, Optex, Mannatech, Bimini, IZEA, Wheeler), where the touch
    on 99 shares is a spread cost far wider than the 15 bps default, and quoting only a
    liquid-name cost would be a one-sided friction.
    """
    kw.setdefault("boot_n", 0)
    rows = []
    for fee in fee_grid:
        for bps in bps_grid:
            h = headline(tbl, tender_fee_usd=fee, cost_bps=bps, **kw)
            s = h["legs"]["r_odd_hedged"]
            rows.append({"fee_usd": fee, "cost_bps": bps, "mean_pct": s["mean_pct"],
                         "t_hac": s["t_hac"], "dollar": h["mean_dollar_profit"],
                         "breakeven_pct": h["breakeven_mean_pct"]})
    return pd.DataFrame(rows)


def capacity_sweep(tbl: pd.DataFrame, grid=(1000.0, 2500.0, 5000.0, 9900.0),
                   **kw) -> pd.DataFrame:
    """The capacity axis: a 99-share lot is worth 99 × the share price, nothing more.

    A flat corporate-action fee is a *percentage* cost that scales inversely with the lot,
    so a 99-share position in a $10 stock ($990) carries three times the drag of the same
    lot in a $30 stock. Sweeping the position size is the honest way to show that the
    odd-lot rule caps the trade's dollar profit no matter how good the percentage is.
    """
    kw.setdefault("boot_n", 0)
    rows = []
    for pos in grid:
        h = headline(tbl, position_usd=pos, **kw)
        s = h["legs"]["r_odd_hedged"]
        rows.append({"position_usd": pos, "mean_pct": s["mean_pct"], "t_hac": s["t_hac"],
                     "dollar_per_event": h["mean_dollar_profit"],
                     "dollar_per_year": h["mean_dollar_profit"] * h["events_per_year"]})
    return pd.DataFrame(rows)


def borrow_sweep(tbl: pd.DataFrame, grid=(0.0, 40.0, 100.0, 300.0), **kw) -> pd.DataFrame:
    """The hedged variant is short SPY — sweep the borrow it pays."""
    kw.setdefault("boot_n", 0)
    rows = []
    for b in grid:
        h = headline(tbl, borrow_bps=b, **kw)
        s = h["legs"]["r_odd_hedged"]
        rows.append({"borrow_bps": b, "mean_pct": s["mean_pct"], "t_hac": s["t_hac"]})
    return pd.DataFrame(rows)


def expiry_sweep(panel: dict, events, grid=(20, 25, 30, 40), market_key: str = "SPY",
                 **kw) -> pd.DataFrame:
    """The expiry date is assumed, not observed — sweep the assumed offer length."""
    kw.setdefault("boot_n", 0)
    rows = []
    for d in grid:
        tbl = build_event_table(panel, events, market_key=market_key, expiry_days=d)
        h = headline(tbl, **kw)
        if h["n_events"] == 0:
            continue
        rows.append({"expiry_days": d, "n": h["n_events"],
                     "runup_pct": h["legs"]["runup"]["mean_pct"],
                     "giveback_pct": h["legs"]["giveback"]["mean_pct"],
                     "hedged_pct": h["legs"]["r_odd_hedged"]["mean_pct"],
                     "t_hac": h["legs"]["r_odd_hedged"]["t_hac"],
                     "breakeven_pct": h["breakeven_mean_pct"]})
    return pd.DataFrame(rows)


def anchor_sweep(panel: dict, events, grid=(2, 4, 6, 11, 21), market_key: str = "SPY",
                 **kw) -> pd.DataFrame:
    """How far before commencement the premium is anchored — the leakage sweep."""
    kw.setdefault("boot_n", 0)
    rows = []
    for a in grid:
        tbl = build_event_table(panel, events, market_key=market_key, anchor_lag=a)
        h = headline(tbl, **kw)
        if h["n_events"] == 0:
            continue
        rows.append({"anchor_lag": a, "n": h["n_events"],
                     "runup_pct": h["legs"]["runup"]["mean_pct"],
                     "hedged_pct": h["legs"]["r_odd_hedged"]["mean_pct"],
                     "t_hac": h["legs"]["r_odd_hedged"]["t_hac"],
                     "breakeven_pct": h["breakeven_mean_pct"]})
    return pd.DataFrame(rows)


def era_cut(tbl: pd.DataFrame, split: str = "2018-01-01", **kw) -> dict:
    """Split the event list at ``split`` and re-run the headline on each half.

    The desk rule is that a sub-period *contrast* ("the run-up has grown", "the margin is
    being arbitraged away") is a claim about a **difference**, and a difference needs its
    own test — two separate point estimates are not one. So the ``"diff"`` entry carries a
    Welch *t* on each era-to-era gap, computed on the per-event vectors themselves. The
    split date is the round decade-ish boundary 2018-01-01, fixed before the numbers were
    looked at; it is not searched over, and the ``diff`` block is what stops a
    good-looking split from being read as decay it cannot support.
    """
    kw.setdefault("boot_n", 0)
    out = {}
    subs = {}
    for tag, mask in [("early", tbl["date"] < pd.Timestamp(split)),
                      ("late", tbl["date"] >= pd.Timestamp(split))]:
        sub = tbl.loc[mask].reset_index(drop=True)
        if len(sub) < 12:
            out[tag] = None
            continue
        subs[tag] = sub
        h = headline(sub, **kw)
        out[tag] = {
            "n": h["n_events"],
            "runup_pct": h["legs"]["runup"]["mean_pct"],
            "giveback_pct": h["legs"]["giveback"]["mean_pct"],
            "hedged_pct": h["legs"]["r_odd_hedged"]["mean_pct"],
            "t_hac": h["legs"]["r_odd_hedged"]["t_hac"],
            "priority_pct": h["legs"]["priority_value"]["mean_pct"],
            "breakeven_pct": h["breakeven_mean_pct"],
        }

    out["diff"] = None
    if len(subs) == 2:
        ev_e = odd_lot_returns(subs["early"], **{k: v for k, v in kw.items()
                                                 if k not in ("boot_n", "seed")})
        ev_l = odd_lot_returns(subs["late"], **{k: v for k, v in kw.items()
                                                if k not in ("boot_n", "seed")})
        diff = {}
        for col in ("runup", "giveback", "r_odd_hedged", "breakeven_premium",
                    "priority_value"):
            a = pd.to_numeric(ev_e[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(ev_l[col], errors="coerce").to_numpy(dtype=float)
            diff[col] = {"early_pct": float(np.nanmean(a) * 100.0),
                         "late_pct": float(np.nanmean(b) * 100.0),
                         "gap_pct": float((np.nanmean(b) - np.nanmean(a)) * 100.0),
                         "welch_t": welch_t(b, a)}
        out["diff"] = diff
    return out


# --------------------------------------------------------------------------- #
# Calendar-time portfolio (the deployable version, excess-of-cash)
# --------------------------------------------------------------------------- #
def calendar_time_portfolio(
    panel: dict,
    ev: pd.DataFrame,
    market_key: str = "SPY",
    cash_key: str = "BIL",
    cost_bps: float = 15.0,
    tender_fee_usd: float = 30.0,
    position_usd: float = 5000.0,
) -> pd.DataFrame:
    """Equal-weight, calendar-time version of the odd-lot trade, excess-of-cash.

    Each event contributes a position from the close of ``entry_date`` to the close of
    ``expiry_date``, earning the issuer's daily total return; on the expiry session the
    position is exchanged for the assumed clearing price, which adds a one-off
    ``p_clear / p_exp − 1``. Live positions are equal-weighted; on days with no live
    position the sleeve earns the **cash** leg (BIL). One-way cost on entry, the flat fee
    on expiry, both scaled by the position's weight.

    Returns a daily frame with ``r_port`` (the sleeve), ``r_cash``, ``r_mkt`` and
    ``n_live`` — the last being the capacity story in one column.
    """
    mkt = panel[market_key].dropna().sort_index()
    cash = panel[cash_key].dropna().sort_index() if cash_key in panel else None
    # Clip to the event span: crediting cash returns for the decades before the first
    # tender would flatter (or flatten) the sleeve's Sharpe for free.
    lo = pd.Timestamp(ev["entry_date"].min())
    hi = pd.Timestamp(ev["expiry_date"].max())
    if cash is not None:
        lo = max(lo, cash.index.min())
    idx = mkt.index[(mkt.index >= lo) & (mkt.index <= hi)]
    r_mkt = mkt.pct_change().reindex(idx)
    r_cash = (cash.reindex(mkt.index).ffill().pct_change().reindex(idx)
              if cash is not None else pd.Series(0.0, index=idx))

    ret_sum = pd.Series(0.0, index=idx)
    n_live = pd.Series(0.0, index=idx)
    fee = tender_fee_usd / position_usd if position_usd > 0 else 0.0
    c = cost_bps * 1e-4

    for _, row in ev.iterrows():
        s = panel[row["ticker"]].dropna().sort_index()
        seg = s.loc[(s.index >= row["entry_date"]) & (s.index <= row["expiry_date"])]
        if len(seg) < 2:
            continue
        r = seg.pct_change().dropna()
        r = r.reindex(r.index.intersection(idx))
        if len(r) == 0:
            continue
        # tender payoff and frictions land on the expiry session
        kick = pd.Series(0.0, index=r.index)
        kick.iloc[-1] += float(row["p_clear"]) / float(row["p_exp"]) - 1.0 - fee
        kick.iloc[0] -= c
        ret_sum.loc[r.index] += (r + kick).to_numpy()
        n_live.loc[r.index] += 1.0

    live = n_live > 0
    r_port = pd.Series(0.0, index=idx)
    r_port[live] = ret_sum[live] / n_live[live]
    r_port[~live] = r_cash[~live].fillna(0.0)

    out = pd.DataFrame({"r_port": r_port, "r_cash": r_cash.fillna(0.0),
                        "r_mkt": r_mkt.fillna(0.0), "n_live": n_live})
    return out.iloc[1:].dropna()


def portfolio_race(cal: pd.DataFrame) -> dict:
    """Excess-of-cash Sharpe race: the odd-lot sleeve vs SPY, both minus the cash leg."""
    e_port = (cal["r_port"] - cal["r_cash"]).dropna()
    e_mkt = (cal["r_mkt"] - cal["r_cash"]).dropna()
    diff = (e_port - e_mkt).dropna()
    invested = float((cal["n_live"] > 0).mean())
    return {
        "n_days": int(len(e_port)),
        "sharpe_port": annualised_sharpe(e_port),
        "sharpe_mkt": annualised_sharpe(e_mkt),
        "sharpe_adv": annualised_sharpe(e_port) - annualised_sharpe(e_mkt),
        "t_port": newey_west_t(e_port.to_numpy(), lags=10),
        "t_diff": newey_west_t(diff.to_numpy(), lags=10),
        "invested_frac": invested,
        "mean_live": float(cal.loc[cal["n_live"] > 0, "n_live"].mean())
        if invested > 0 else float("nan"),
        "vol_port": float(e_port.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "cagr_port": float((1.0 + cal["r_port"]).prod() ** (TRADING_DAYS / len(cal)) - 1.0),
    }


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict, events, premium: float = 0.13,
                     proration: float = 0.35, expiry_days: int = 21,
                     cost_bps: float = 0.0, tender_fee_usd: float = 0.0) -> dict:
    """Run the odd-lot machinery on a synthetic panel with a planted run-up / give-back.

    The unbiasedness test is run on the two **premium-free** legs: on the planted world
    the estimator must recover a positive abnormal run-up and a negative give-back (and a
    breakeven premium lifted by the run-up); on the null all three must sit at zero. The
    priority value is reported for completeness but is *not* a null statistic — it carries
    the assumed premium by construction, so on a null tape it correctly equals the premium
    on the un-filled stub rather than zero.
    """
    tbl = build_event_table(panel, events, market_key="MKT", expiry_days=expiry_days)
    if len(tbl) == 0:
        return {"n_events": 0}
    h = headline(tbl, premium=premium, proration=proration, cost_bps=cost_bps,
                 tender_fee_usd=tender_fee_usd, borrow_bps=0.0)
    return {
        "n_events": h["n_events"],
        "runup_bps": h["legs"]["runup_abn"]["mean_pct"] * 100.0,
        "runup_t": h["legs"]["runup_abn"]["t_hac"],
        "giveback_bps": h["legs"]["giveback"]["mean_pct"] * 100.0,
        "giveback_t": h["legs"]["giveback"]["t_hac"],
        "priority_bps": h["legs"]["priority_value"]["mean_pct"] * 100.0,
        "priority_t": h["legs"]["priority_value"]["t_hac"],
        "breakeven_pct": h["breakeven_mean_pct"],
    }
