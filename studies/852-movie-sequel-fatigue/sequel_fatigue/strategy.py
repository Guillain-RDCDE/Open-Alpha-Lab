"""Strategy + inference for Study 852 — Movie-Sequel Fatigue.

The claim, steelmanned: as a franchise ages, each sequel opens weaker than the last, and
the *studio's* stock should react worse to sequel N than to sequel N-1 — and a run of
tired sequels (a "down sequence") should overhang the next entry. Two testable
sub-claims, both measured on studio **opening-weekend abnormal returns**:

* **H1 — the fatigue slope.** Does a studio's opening-reaction CAR *decline with sequel
  number*? Each film reduces to one number: the studio's abnormal return (studio total
  return minus the SPY benchmark) over the opening-reaction window, anchored on the first
  session on/after the Monday after the opening weekend (the box-office number is public
  by Sunday/Monday). We regress that CAR on the sequel number across all events — raw and
  **franchise-demeaned** (a franchise fixed effect, so we ask "*within* a franchise, do
  later entries react worse?", not "are old franchises different firms"). The claim
  predicts a **negative** slope.

* **H2 — fatigue persistence.** Does a *down sequence* predict a worse next entry? Within
  each franchise we pair consecutive entries (sequel N-1 -> N) and (a) regress N's CAR on
  N-1's CAR (a pooled AR(1) on the reaction sequence), and (b) split N's CAR by whether
  N-1's reaction was negative ("fatigued") vs positive (Welch *t*). The claim predicts a
  **positive** AR(1) slope / a worse conditional mean after a down entry.

The honest prior is skeptical, and for the same reason as
[771-box-office-bomb](../../771-box-office-bomb/): a single film — even a tentpole — is a
small slice of a ~$100-200bn conglomerate (parks, streaming, cable, consumer products),
so a *scheduled, public* opening number is exactly what a semi-strong-efficient market has
already priced. And with ~40 events across ~14 franchises, this is a deliberately
**low-power** cross-event test — usually a None.

Inference primitives mirror the lab's canon (803): ``one_sample_t`` / ``welch_t`` /
``newey_west_t`` are here for cross-checks; the cross-event unit is a one-sample /
OLS-slope *t* (events are independent, non-overlapping releases), with a
label-permutation placebo (shuffle the sequel numbers) as the falsification test and a
random-pseudo-date placebo to show the per-event CARs are ordinary tracking noise. A
costed "short the fatigued sequel" timer charges the friction. One documented execution
lag throughout (anchor = first session on/after the Monday-after-opening).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

ANCHOR_LAG_DAYS = 3     # opening Friday + 3 cal days = the Monday after the opening weekend
CAR_K = 3               # opening-reaction window = [anchor-1 close .. anchor+CAR_K close]
COST_BPS = 5.0          # one-way, per leg
BORROW_BPS_YR = 50.0    # borrow on the short leg of the fatigue trade
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (lab canon)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 — the unit for independent, non-overlapping events."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 4) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 — a serial-correlation-robust
    cross-check on the per-event reaction mean."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean(); u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS y = a + b*x; returns slope, intercept, its t-stat, r, n.

    The t-stat is the classic OLS slope t (residual-variance based): b / se(b),
    se(b) = s / sqrt(Sxx), s^2 = SSR/(n-2). Events are independent, so this is the
    right precision for the cross-event slope."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or x.std() == 0:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"),
                "t": float("nan"), "r": float("nan")}
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    sxy = float(((x - xbar) * (y - ybar)).sum())
    b = sxy / sxx
    a = ybar - b * xbar
    resid = y - (a + b * x)
    ssr = float((resid ** 2).sum())
    s2 = ssr / (n - 2)
    se_b = np.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    t = b / se_b if se_b and se_b > 0 else float("nan")
    syy = float(((y - ybar) ** 2).sum())
    r = sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    return {"n": n, "slope": float(b), "intercept": float(a),
            "t": float(t), "r": float(r)}


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-event opening-reaction CAR
# --------------------------------------------------------------------------- #
def build_event_cars(prices: dict[str, pd.Series], events: pd.DataFrame | None = None,
                     car_k: int = CAR_K, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """One row per franchise entry: the studio's opening-reaction abnormal return.

    The anchor is the first trading session on/after the Monday after the opening weekend
    (``opening + ANCHOR_LAG_DAYS`` days, snapped forward) — the first session at which the
    weekend box-office is common knowledge. ``base`` is the session BEFORE the anchor (the
    opening-Friday close, before any weekend number exists). Abnormal return
    ``ar = (studio_ret) - (SPY_ret)`` is measured over:

      * ``car``  : base -> anchor + car_k  (the ~1-week opening reaction, the headline)
      * ``day0`` : base -> anchor          (the single Monday-after jump)

    A row is INCLUDED only if both the studio ticker and SPY have cached coverage across
    ``[base .. anchor + car_k]``; events outside coverage (e.g. Transformers before PARA's
    2021 continuous history) are excluded with a reason, so the funnel is auditable.
    ``short_net`` is the fatigue-trade short leg's net return (``-car`` less round-trip
    cost), used by the costed timer.
    """
    if events is None:
        events = dt.events_frame()
    spy = prices[dt.BENCHMARK]
    rt = 2.0 * cost_bps / 1e4
    rows = []
    for _, e in events.iterrows():
        franchise, seq, ticker = e["franchise"], int(e["seq"]), e["ticker"]
        row = {"franchise": franchise, "title": e.get("title", ""), "seq": seq,
               "ticker": ticker, "opening": str(pd.Timestamp(e["opening"]).date())}
        if ticker not in prices:
            row.update(included=False, reason=f"no cached prices for {ticker}")
            rows.append(row); continue
        stud = prices[ticker]
        common = stud.index.intersection(spy.index).sort_values()
        target = pd.Timestamp(e["opening"]) + pd.Timedelta(days=ANCHOR_LAG_DAYS)
        on_after = common[common >= target]
        if len(on_after) == 0:
            row.update(included=False, reason="no studio/SPY coverage at the anchor")
            rows.append(row); continue
        anchor = on_after[0]
        p = int(common.get_indexer([anchor])[0])
        if p - 1 < 0:
            row.update(included=False, reason="no pre-anchor base session")
            rows.append(row); continue
        if p + car_k >= len(common):
            row.update(included=False, reason="insufficient trailing history")
            rows.append(row); continue

        def ar(i0: int, i1: int) -> float:
            r_a = stud.loc[common[i1]] / stud.loc[common[i0]] - 1.0
            r_s = spy.loc[common[i1]] / spy.loc[common[i0]] - 1.0
            return float(r_a - r_s)

        car = ar(p - 1, p + car_k)
        day0 = ar(p - 1, p)
        row.update(included=True, reason="",
                   anchor=str(anchor.date()), car=car, day0=day0,
                   short_net=-car - rt)
        rows.append(row)
    return pd.DataFrame(rows)


def _consecutive_pairs(inc: pd.DataFrame) -> pd.DataFrame:
    """Within each franchise (ordered by seq), pair each entry's CAR with the PREVIOUS
    entry's CAR — only truly consecutive sequel numbers (N-1 -> N). Returns a frame with
    ``prev_car`` and ``car`` (the next entry's reaction)."""
    out = []
    for fr, g in inc.sort_values(["franchise", "seq"]).groupby("franchise"):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if int(g.loc[i, "seq"]) == int(g.loc[i - 1, "seq"]) + 1:
                out.append({"franchise": fr, "seq": int(g.loc[i, "seq"]),
                            "prev_car": float(g.loc[i - 1, "car"]),
                            "car": float(g.loc[i, "car"])})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# H1 — the fatigue slope (CAR vs sequel number)
# --------------------------------------------------------------------------- #
def fatigue_slope(inc: pd.DataFrame, demean: bool = False) -> dict:
    """OLS slope of opening-reaction CAR on sequel number across events.

    ``demean=True`` removes each franchise's own mean CAR and mean seq first (a franchise
    fixed effect), so the slope is the *within-franchise* fatigue tilt. The claim predicts
    a NEGATIVE slope (later sequels react worse)."""
    d = inc.copy()
    x = d["seq"].to_numpy(float)
    y = d["car"].to_numpy(float)
    if demean:
        xg = d.groupby("franchise")["seq"].transform("mean").to_numpy(float)
        yg = d.groupby("franchise")["car"].transform("mean").to_numpy(float)
        x = x - xg
        y = y - yg
    return ols_slope(x, y)


def day0_stats(inc: pd.DataFrame, col: str = "car") -> dict:
    """Mean opening-reaction CAR across events + one-sample / NW / hit-rate — is the
    average studio reaction to a sequel even different from zero?"""
    x = inc[col].to_numpy(float)
    s = one_sample_t(x)
    k = int((x[np.isfinite(x)] > 0).sum()); n = int(np.isfinite(x).sum())
    lo, hi = wilson_interval(k, n)
    s.update(t_nw=newey_west_t(x, lags=4), up_k=k, up_n=n, up_lo=lo, up_hi=hi)
    return s


# --------------------------------------------------------------------------- #
# H2 — fatigue persistence (does a down sequence predict a worse next entry?)
# --------------------------------------------------------------------------- #
def fatigue_persistence(inc: pd.DataFrame) -> dict:
    """Within-franchise AR(1) of the reaction sequence + a down/up conditional split.

    (a) pooled OLS slope of entry N's CAR on entry N-1's CAR (claim: POSITIVE — a bad
        reaction begets a bad reaction); (b) Welch t of next-CAR | prev-down vs
        next-CAR | prev-up (claim: the after-a-down mean is LOWER)."""
    pairs = _consecutive_pairs(inc)
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "ar1_slope": float("nan"), "ar1_t": float("nan"),
                "down_mean": float("nan"), "up_mean": float("nan"),
                "welch_t": float("nan"), "n_down": 0, "n_up": 0}
    sl = ols_slope(pairs["prev_car"].to_numpy(float), pairs["car"].to_numpy(float))
    down = pairs.loc[pairs["prev_car"] < 0, "car"].to_numpy(float)
    up = pairs.loc[pairs["prev_car"] >= 0, "car"].to_numpy(float)
    return {"n_pairs": int(len(pairs)), "ar1_slope": sl["slope"], "ar1_t": sl["t"],
            "ar1_r": sl["r"],
            "down_mean": float(np.nanmean(down)) if len(down) else float("nan"),
            "up_mean": float(np.nanmean(up)) if len(up) else float("nan"),
            "welch_t": welch_t(down, up), "n_down": int(len(down)), "n_up": int(len(up))}


# --------------------------------------------------------------------------- #
# Placebo 1 — permute the sequel-number labels (the H1 falsification)
# --------------------------------------------------------------------------- #
def permute_slope_pvalue(inc: pd.DataFrame, demean: bool = False,
                         n_perm: int = 5000, seed: int = 852) -> dict:
    """Shuffle the sequel-number labels across events (breaking any seq->CAR link) and
    recompute the fatigue slope ``n_perm`` times. Two-sided p = share of permuted slopes
    at least as extreme (in |.|) as observed; ``p_left`` = share <= observed (the claim's
    negative-slope tail)."""
    obs = fatigue_slope(inc, demean=demean)["slope"]
    d = inc.copy().reset_index(drop=True)
    seqs = d["seq"].to_numpy(float)
    car = d["car"].to_numpy(float)
    fr = d["franchise"].to_numpy(object)
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_perm)
    tmp = pd.DataFrame({"franchise": fr, "car": car})
    for i in range(n_perm):
        tmp["seq"] = rng.permutation(seqs)
        slopes[i] = fatigue_slope(tmp, demean=demean)["slope"]
    slopes = slopes[np.isfinite(slopes)]
    p_two = float((np.abs(slopes) >= abs(obs)).mean()) if slopes.size else float("nan")
    p_left = float((slopes <= obs).mean()) if slopes.size else float("nan")
    return {"obs_slope": float(obs), "placebo_mean": float(slopes.mean()) if slopes.size else float("nan"),
            "placebo_sd": float(slopes.std(ddof=1)) if slopes.size > 1 else float("nan"),
            "p_two": p_two, "p_left": p_left, "n_draws": int(slopes.size),
            "draws": slopes}


# --------------------------------------------------------------------------- #
# Placebo 2 — random pseudo-event dates (are the per-event CARs ordinary noise?)
# --------------------------------------------------------------------------- #
def random_date_placebo(prices: dict[str, pd.Series], inc: pd.DataFrame,
                        car_k: int = CAR_K, n_seeds: int = 20,
                        n_draws_per_seed: int = 200, base_seed: int = 852) -> dict:
    """For each INCLUDED event draw a random (non-event) same-length window on the SAME
    studio ticker vs SPY and recompute the abnormal return; average across the same n
    events; repeat. If the observed mean CAR sits in the bulk of this cloud, the studio's
    sequel reactions are indistinguishable from picking random dates."""
    obs = float(np.nanmean(inc["car"].to_numpy(float)))
    tickers = inc["ticker"].to_numpy(object)
    means = []
    # pre-build the common index per ticker
    commons = {}
    for t in set(tickers):
        commons[t] = prices[t].index.intersection(prices[dt.BENCHMARK].index).sort_values()
    spy = prices[dt.BENCHMARK]
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = []
            for t in tickers:
                common = commons[t]
                hi = len(common) - car_k - 2
                if hi <= 1:
                    continue
                q = int(rng.integers(1, hi))
                stud = prices[t]
                r_a = stud.loc[common[q + car_k]] / stud.loc[common[q - 1]] - 1.0
                r_s = spy.loc[common[q + car_k]] / spy.loc[common[q - 1]] - 1.0
                vals.append(float(r_a - r_s))
            if vals:
                means.append(np.mean(vals))
    means = np.asarray(means)
    p_two = float((np.abs(means) >= abs(obs)).mean()) if means.size else float("nan")
    return {"obs": obs, "placebo_mean": float(means.mean()) if means.size else float("nan"),
            "placebo_sd": float(means.std(ddof=1)) if means.size > 1 else float("nan"),
            "p_two": p_two, "n_draws": int(means.size)}


# --------------------------------------------------------------------------- #
# Two-era robustness cut on the fatigue slope
# --------------------------------------------------------------------------- #
def era_slopes(inc: pd.DataFrame, split: str = "2018-01-01", demean: bool = False) -> dict:
    """The fatigue slope computed on the early vs late halves of the calendar (by anchor
    date). A 'Real' stamp needs the sign to hold in BOTH eras."""
    d = inc.copy()
    d["anchor_ts"] = pd.to_datetime(d["anchor"])
    early = d[d["anchor_ts"] < pd.Timestamp(split)]
    late = d[d["anchor_ts"] >= pd.Timestamp(split)]
    return {"split": split,
            "early": fatigue_slope(early, demean=demean),
            "late": fatigue_slope(late, demean=demean),
            "n_early": int(len(early)), "n_late": int(len(late))}


# --------------------------------------------------------------------------- #
# Tradability — short the fatigued sequel (calendar-known entry), net of costs
# --------------------------------------------------------------------------- #
def fatigue_timer(inc: pd.DataFrame, cost_bps: float = COST_BPS,
                  borrow_bps_yr: float = BORROW_BPS_YR, car_k: int = CAR_K) -> dict:
    """Cost the 'fatigue' trade: at the anchor of an entry whose PREVIOUS franchise entry
    reacted negatively ("fatigued"), SHORT the studio for the ``car_k``-session reaction
    window. P&L of the short = ``-car``; charge 2 x one-way cost x NAV (round trip) plus
    borrow over the hold. Almost certainly a Mirage — few fires, tiny, noisy."""
    pairs = _consecutive_pairs(inc)
    fatigued = pairs[pairs["prev_car"] < 0]
    n = len(fatigued)
    rt = 2.0 * cost_bps / 1e4
    borrow = (borrow_bps_yr / 1e4) / 365.0 * (car_k + 1)
    if n < 2:
        return {"n": n, "gross_mean": float("nan"), "net_mean": float("nan"),
                "t_gross": float("nan"), "t_net": float("nan"),
                "cost_bps": (rt + borrow) * 1e4}
    gross = -fatigued["car"].to_numpy(float)          # short P&L
    net = gross - rt - borrow
    return {"n": int(n), "gross_mean": float(gross.mean()), "net_mean": float(net.mean()),
            "t_gross": one_sample_t(gross)["t"], "t_net": one_sample_t(net)["t"],
            "cost_bps": (rt + borrow) * 1e4}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(edge: float, seed: int, persist: float = 0.0,
                     shock_sd: float = 0.0, demean: bool = True,
                     car_k: int = CAR_K) -> dict:
    """Run the full pipeline (build reaction CARs -> fatigue slope) on a synthetic price
    world with a planted fatigue edge. A larger planted ``edge`` must drive the slope MORE
    NEGATIVE, monotonically; ``edge = 0`` must stay quiet. ``persist``/``shock_sd`` plant
    an H2 fatigue-sequence (AR(1)) reaction the persistence detector must recover."""
    close_a, close_b, ev = dt.synthetic_world(edge=edge, persist=persist, seed=seed,
                                              shock_sd=shock_sd, car_k=car_k)
    prices = {"SYN": close_a, dt.BENCHMARK: close_b}
    ev = ev.copy()
    ev["ticker"] = "SYN"
    ev["title"] = ev["franchise"] + " #" + ev["seq"].astype(str)
    cars = build_event_cars(prices, ev, car_k=car_k)
    inc = cars[cars["included"]]
    sl = fatigue_slope(inc, demean=demean)
    persist_stat = fatigue_persistence(inc)
    return {"n": int(len(inc)), "slope": sl["slope"], "t": sl["t"],
            "ar1_slope": persist_stat["ar1_slope"], "ar1_t": persist_stat["ar1_t"]}
