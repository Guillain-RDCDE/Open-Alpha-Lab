"""Event study + inference for Study 918 — Creation Halt.

The object of study is one number per pair per day: the **signed tracking spread**

    x_t = direction * ( log(F_t / F_{t-1}) - log(P_t / P_{t-1}) )

where ``F`` is the capped fund and ``P`` is an *uncapped* instrument tracking the same
thing (a twin ETF where one exists, a futures print otherwise). Because both legs are
returns on the same underlying, ``x`` is the day-on-day **change in the fund's premium**
— up to tracking error and, for the futures rulers, roll differences. It is a PROXY for
the premium: free daily NAV/indicative-value histories do not exist for these funds, and
the README says so. ``direction`` is ``+1`` for a *creation* suspension (price can only
richen) and ``−1`` for a *redemption* suspension (price can only cheapen), so every event
is signed the same way and can be pooled.

``x`` is a **long-short, self-financing** spread: one dollar long the fund, one dollar
short the twin. Its return is therefore already excess-of-cash — the cash the short leg
raises earns the cash rate and cancels — which is why the Sharpe race here is
excess-vs-excess by construction. Where the spread is *traded*, the short leg pays
borrow, both legs pay one-way cost × NAV at entry and at exit, **and the position pays a
daily rebalancing charge**, because ``exp(Σ x)`` is the return of a continuously
dollar-neutral position and holding one flat costs turnover. All three are swept.

Two mechanical contaminants are named rather than assumed away. ``x`` carries the two
legs' **expense-ratio difference** before any halt effect exists (``fee_drag_bps``); and
the ``hold='halt'`` trade exits on the resumption date, which is a **hindsight exit** —
``hold='blind'`` is the fixed-horizon rule a trader could actually have followed.

**One execution lag, stated once:** a suspension announced on day ``t`` is only acted on
at the close of ``t+1``. Every window in this module starts at ``t+1``.

The tests, in order:

1. ``event_car`` / ``event_study`` — the K-day cumulative abnormal spread after the
   announcement, standardised against that pair's own placebo distribution of all other
   K-day windows.
2. ``regime_drift`` — the mean daily spread *while* issuance is suspended versus the same
   pair outside the suspension, with an HAC *t* on each and on the difference.
3. ``resumption_fade`` — the K-day spread after issuance resumes; the premium should be
   handed back.
4. ``jackknife`` / ``resume_date_sweep`` / ``era_cut`` / ``trade_sweep`` — is the pooled
   number one event, one guessed date, one era, or one costless assumption?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk versions)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(int(lags), 0)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_mean_ci(
    x, n_boot: int = 2000, block: int = 10, seed: int = 918, alpha: float = 0.05
) -> dict:
    """Circular block-bootstrap CI for the mean of a serially dependent series."""
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
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
            "n_boot": int(n_boot), "block": int(block)}


def event_resample_ci(values, n_boot: int = 5000, seed: int = 918, alpha: float = 0.05) -> dict:
    """Bootstrap CI for the mean of a handful of per-event statistics (resample events).

    With five or six events the sampling unit is the *event*, not the day; resampling
    events with replacement is the honest interval and is deliberately wide.
    """
    v = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = v.size
    if n < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_events": int(n)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    means = v[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((means < 0).mean()), "n_events": int(n)}


# --------------------------------------------------------------------------- #
# The spread
# --------------------------------------------------------------------------- #
def pair_spread(fund: pd.Series, proxy: pd.Series, direction: int = 1) -> pd.Series:
    """Signed daily log-return spread of a capped fund against its uncapped twin.

    Both legs are reindexed onto their common trading days before differencing, so a
    24/7 print (bitcoin) is sampled on the fund's sessions and no phantom weekend
    returns enter the spread. Returns a clean, NaN-free series named ``spread``.
    """
    f = pd.Series(fund).dropna()
    p = pd.Series(proxy).dropna()
    common = f.index.intersection(p.index)
    f = f.loc[common].sort_index()
    p = p.loc[common].sort_index()
    lf = np.log(f.astype(float))
    lp = np.log(p.astype(float))
    s = (lf.diff() - lp.diff()) * float(direction)
    return s.dropna().rename("spread")


def build_spreads(prices: pd.DataFrame, events=None) -> dict[str, pd.Series]:
    """One signed spread series per event, keyed by the event key."""
    from . import data as _data

    events = _data.EVENTS if events is None else events
    out = {}
    for ev in events:
        out[ev["key"]] = pair_spread(
            prices[ev["fund"]], prices[ev["proxy"]], direction=int(ev["direction"])
        )
    return out


def build_spreads_from_frames(frames: dict[str, pd.DataFrame], events) -> dict[str, pd.Series]:
    """Synthetic counterpart of ``build_spreads`` (one two-column frame per event)."""
    out = {}
    for ev in events:
        fr = frames[ev["key"]]
        out[ev["key"]] = pair_spread(fr["fund"], fr["proxy"], direction=int(ev["direction"]))
    return out


# --------------------------------------------------------------------------- #
# Window helpers — the single execution lag lives here
# --------------------------------------------------------------------------- #
def _pos_after(index: pd.DatetimeIndex, date) -> int:
    """Index position of the first session strictly after ``date`` (the t+1 lag)."""
    return int(index.searchsorted(pd.Timestamp(date), side="right"))


def window_sum(spread: pd.Series, date, k: int) -> tuple[float, int, int]:
    """Sum of ``spread`` over the ``k`` sessions starting the day AFTER ``date``.

    Returns ``(sum, start_pos, end_pos)``; ``sum`` is NaN when fewer than ``k``
    sessions are available after the date.
    """
    idx = spread.index
    i0 = _pos_after(idx, date)
    i1 = i0 + int(k)
    if i0 >= len(idx) or i1 > len(idx):
        return float("nan"), i0, min(i1, len(idx))
    return float(spread.iloc[i0:i1].sum()), i0, i1


def placebo_sums(spread: pd.Series, k: int, exclude: list[tuple[int, int]] | None = None,
                 step: int = 1) -> np.ndarray:
    """All k-day rolling sums of ``spread`` outside the excluded position ranges.

    The excluded ranges are the event windows themselves (announcement and resumption),
    padded by ``k`` on each side so a placebo window can never overlap the event it is
    supposed to be a control for.
    """
    v = spread.to_numpy(dtype=float)
    n = v.size
    if n < k + 1:
        return np.array([])
    csum = np.concatenate([[0.0], np.cumsum(v)])
    sums = csum[k:] - csum[:-k]                      # sums[i] covers positions i..i+k-1
    keep = np.ones(sums.size, dtype=bool)
    for a, b in (exclude or []):
        lo = max(0, a - k)
        hi = min(sums.size, b + k)
        if hi > lo:
            keep[lo:hi] = False
    out = sums[keep]
    return out[::max(int(step), 1)]


# --------------------------------------------------------------------------- #
# 1. The announcement event study
# --------------------------------------------------------------------------- #
def event_car(spread: pd.Series, ev: dict, k: int = 20, placebo_step: int = 1) -> dict:
    """K-day cumulative signed spread after the suspension announcement, standardised.

    The raw CAR is compared to the same pair's own distribution of K-day sums away from
    the event (the placebo null): ``z = (CAR − placebo mean) / placebo sd`` and the
    one-sided empirical percentile. This controls for the pair's ordinary tracking
    dispersion — a 3% divergence means one thing for GBTC and quite another for VXX.

    Two cautions that the reported numbers depend on:

    * **``z`` is a scale, not a p-value.** These spreads are strongly mean-reverting and
      fat-tailed, so the k-day sums are nowhere near Gaussian. VXX's ``z = +13.84`` sits
      at empirical percentile 0.999, not at the 1e-43 a normal table would imply. The
      **percentile is the inference**; ``z`` only makes events comparable across pairs.
    * **The placebo pool is the whole sample**, including sessions after the event. That
      is standard for an event-study standardiser and it is descriptive, not tradable —
      no signal here is formed from it — but it is in-sample and is named as such.

    ``family_size`` / ``bonferroni_percentile`` below give the multiplicity bar these
    percentiles have to clear.

    **``pct`` is not as precise as its denominator suggests, so ``pct_indep`` is also
    returned.** The default placebo pool steps one session at a time, so its ~2,000
    windows overlap by 19/20 and are nowhere near 2,000 independent draws — quoting
    "percentile 0.999 of 1,978" implies a resolution of 1-in-2,000 that the data does not
    contain. ``pct_indep`` re-runs the same comparison on **non-overlapping** windows
    (``step = k``), of which VXX's 8.5-year tape holds only 99. On that honest pool VXX's
    announcement percentile is 0.990, not 0.999 — one-sided p ≈ 0.01, which does **not**
    clear the 30-look family-wise bar on its own. Report ``pct_indep``.
    """
    car, i0, i1 = window_sum(spread, ev["halt"], k)
    _, j0, j1 = window_sum(spread, ev["resume"], k)
    excl = [(i0, i1), (j0, j1)]
    pl = placebo_sums(spread, k, exclude=excl, step=placebo_step)
    pl_i = placebo_sums(spread, k, exclude=excl, step=int(k))
    if not np.isfinite(car) or pl.size < 30:
        return {"key": ev["key"], "k": int(k), "car": car, "z": float("nan"),
                "pct": float("nan"), "pct_indep": float("nan"),
                "placebo_mean": float("nan"), "placebo_sd": float("nan"),
                "n_placebo": int(pl.size), "n_placebo_indep": int(pl_i.size)}
    mu, sd = float(pl.mean()), float(pl.std(ddof=1))
    z = (car - mu) / sd if sd > 0 else float("nan")
    return {
        "key": ev["key"], "k": int(k), "car": float(car), "z": float(z),
        "pct": float((pl < car).mean()),
        "pct_indep": float((pl_i < car).mean()) if pl_i.size else float("nan"),
        "placebo_mean": mu, "placebo_sd": sd,
        "n_placebo": int(pl.size), "n_placebo_indep": int(pl_i.size),
    }


def event_study(spreads: dict[str, pd.Series], events, k: int = 20,
                announcement_only: bool = True, placebo_step: int = 1) -> pd.DataFrame:
    """Per-event announcement CARs and their placebo-standardised z-scores."""
    rows = []
    for ev in events:
        if announcement_only and not ev.get("announcement", True):
            continue
        sp = spreads.get(ev["key"])
        if sp is None or sp.empty:
            continue
        rows.append(event_car(sp, ev, k=k, placebo_step=placebo_step))
    return pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()


def family_size(n_events: int, n_horizons: int = 3, n_legs: int = 2) -> int:
    """How many placebo percentiles this design looks at before it picks a winner.

    Five announcement events × three horizons (K = 5/10/20) × two legs (announcement and
    resumption) = 30 looks. Quoting the single most extreme percentile out of 30 without
    saying so would be data snooping, so the bar is reported alongside it.
    """
    return int(n_events) * int(n_horizons) * int(n_legs)


def bonferroni_percentile(n_looks: int, alpha: float = 0.05) -> float:
    """One-sided empirical percentile a single look must clear for family-wise ``alpha``."""
    return float(1.0 - alpha / max(int(n_looks), 1))


def pooled_z(tbl: pd.DataFrame) -> dict:
    """Pool per-event z-scores: mean z, cross-event *t*, and an event-resample CI.

    With five events this *t* has four degrees of freedom. It is reported because it is
    the honest statistic for the design, not because it is powerful.
    """
    if tbl is None or len(tbl) == 0 or "z" not in tbl:
        return {"n": 0, "mean_z": float("nan"), "t": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_positive": 0}
    z = tbl["z"].dropna().to_numpy(dtype=float)
    ci = event_resample_ci(z)
    return {"n": int(z.size), "mean_z": float(z.mean()), "t": one_sample_t(z),
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "n_positive": int((z > 0).sum()),
            "mean_car": float(tbl["car"].dropna().mean())}


# --------------------------------------------------------------------------- #
# 2. The suspended-regime drift
# --------------------------------------------------------------------------- #
def fee_drag_bps(ev: dict, fees: dict | None = None) -> float:
    """Bps/day of the signed spread that is pure expense-ratio difference, not premium.

    A fund's total return is already net of its fee, so
    ``x = direction·(r_fund − r_ruler)`` contains ``direction·(fee_ruler − fee_fund)/252``
    per day before any halt effect exists. Where the ruler is an unfeed spot or futures
    print the term does not cancel — GBTC's 2.00%/yr against spot bitcoin is +0.79 bps/day
    of *mechanical* signed drift, 18% of its measured in-halt number. The fee figures are
    hardcoded prospectus values (``data.FEE_ANNUAL``) and are an ASSUMPTION, not tape.
    """
    from . import data as _data

    fees = _data.FEE_ANNUAL if fees is None else fees
    f_fund = float(fees.get(ev["fund"], 0.0))
    f_ruler = float(fees.get(ev["proxy"], 0.0))
    return float(ev["direction"]) * (f_ruler - f_fund) / TRADING_DAYS * 1e2


def regime_drift(spread: pd.Series, ev: dict, fade_buffer: int = 20,
                 fees: dict | None = None) -> dict:
    """Mean daily signed spread while issuance is suspended, versus outside it.

    The suspension window runs from the session AFTER the announcement (the execution
    lag) through the resumption date inclusive. "Outside" is every other session for the
    same pair **minus the ``fade_buffer`` sessions immediately after the resumption**.
    That exclusion matters and it cuts against the hypothesis: the post-resumption window
    is where the premium is given back, so leaving it in the control depresses the
    baseline and flatters the in-versus-out gap. On VXX it flips the control from
    −0.49 to +0.43 bps/day. Both means carry an HAC *t*; the difference carries a Welch *t*.

    ``in_bps_net_fee`` subtracts ``fee_drag_bps`` — the part of the drift that is only the
    two legs' expense-ratio difference.
    """
    idx = spread.index
    i0 = _pos_after(idx, ev["halt"])
    i1 = int(idx.searchsorted(pd.Timestamp(ev["resume"]), side="right"))
    inside = spread.iloc[i0:i1]
    outside = pd.concat([spread.iloc[:i0], spread.iloc[i1 + max(int(fade_buffer), 0):]])
    fee = fee_drag_bps(ev, fees)
    in_bps = float(inside.mean() * 1e4) if len(inside) else float("nan")
    return {
        "key": ev["key"],
        "n_in": int(len(inside)), "n_out": int(len(outside)),
        "in_bps": in_bps,
        "fee_bps": fee,
        "in_bps_net_fee": in_bps - fee,
        "out_bps": float(outside.mean() * 1e4) if len(outside) else float("nan"),
        "in_total_pct": float((np.exp(inside.sum()) - 1.0) * 100) if len(inside) else float("nan"),
        "t_in": newey_west_t(inside.to_numpy()),
        "t_out": newey_west_t(outside.to_numpy()),
        "t_diff": welch_t(inside.to_numpy(), outside.to_numpy()),
    }


def regime_table(spreads: dict[str, pd.Series], events, fade_buffer: int = 20) -> pd.DataFrame:
    rows = [regime_drift(spreads[ev["key"]], ev, fade_buffer=fade_buffer)
            for ev in events if ev["key"] in spreads]
    return pd.DataFrame(rows).set_index("key")


# --------------------------------------------------------------------------- #
# 3. The resumption fade
# --------------------------------------------------------------------------- #
def resumption_fade(spread: pd.Series, ev: dict, k: int = 20, placebo_step: int = 1) -> dict:
    """K-day signed spread after issuance resumes — the premium should be handed back.

    Same standardisation as the announcement study, so a fade is measured in units of
    the pair's own ordinary dispersion. A *negative* z is the folklore's prediction.
    ``pct_indep`` is the same percentile on non-overlapping windows — see ``event_car``
    for why the overlapping denominator flatters the result.
    """
    car, j0, j1 = window_sum(spread, ev["resume"], k)
    _, i0, i1 = window_sum(spread, ev["halt"], k)
    excl = [(i0, i1), (j0, j1)]
    pl = placebo_sums(spread, k, exclude=excl, step=placebo_step)
    pl_i = placebo_sums(spread, k, exclude=excl, step=int(k))
    if not np.isfinite(car) or pl.size < 30:
        return {"key": ev["key"], "k": int(k), "car": car, "z": float("nan"),
                "pct": float("nan"), "pct_indep": float("nan"),
                "n_placebo": int(pl.size), "n_placebo_indep": int(pl_i.size)}
    mu, sd = float(pl.mean()), float(pl.std(ddof=1))
    return {"key": ev["key"], "k": int(k), "car": float(car),
            "z": float((car - mu) / sd) if sd > 0 else float("nan"),
            "pct": float((pl < car).mean()),
            "pct_indep": float((pl_i < car).mean()) if pl_i.size else float("nan"),
            "n_placebo": int(pl.size), "n_placebo_indep": int(pl_i.size)}


def fade_study(spreads: dict[str, pd.Series], events, k: int = 20,
               placebo_step: int = 1) -> pd.DataFrame:
    rows = [resumption_fade(spreads[ev["key"]], ev, k=k, placebo_step=placebo_step)
            for ev in events if ev["key"] in spreads]
    return pd.DataFrame(rows).set_index("key") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# 4. Robustness — jackknife, date sweep, era cut
# --------------------------------------------------------------------------- #
def jackknife(tbl: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-event-out mean z and cross-event *t* — is the pool one event?"""
    if tbl is None or len(tbl) < 2:
        return pd.DataFrame()
    rows = []
    for key in tbl.index:
        sub = tbl.drop(index=key)
        z = sub["z"].dropna().to_numpy(dtype=float)
        rows.append({"dropped": key, "n": int(z.size), "mean_z": float(z.mean()),
                     "t": one_sample_t(z)})
    return pd.DataFrame(rows).set_index("dropped")


def resume_date_sweep(spreads: dict[str, pd.Series], events, shifts=(-10, -5, 0, 5, 10),
                      k: int = 20) -> pd.DataFrame:
    """Shift every APPROX resumption date by ±N business days and re-run the fade study.

    Several resumption dates are our best public reading rather than a filing date, so
    the conclusion must not hinge on them. FIRM dates are left alone.
    """
    rows = []
    for sh in shifts:
        evs = []
        for ev in events:
            e = dict(ev)
            if e.get("confidence") in ("APPROX", "SOFT"):
                e["resume"] = str((pd.Timestamp(e["resume"]) + pd.tseries.offsets.BDay(sh)).date())
            evs.append(e)
        f = fade_study(spreads, evs, k=k)
        r = regime_table(spreads, evs)
        rows.append({"shift_bdays": int(sh),
                     "mean_fade_z": float(f["z"].dropna().mean()) if len(f) else float("nan"),
                     "mean_in_bps": float(r["in_bps"].dropna().mean()) if len(r) else float("nan"),
                     "n_fade_negative": int((f["z"] < 0).sum()) if len(f) else 0})
    return pd.DataFrame(rows).set_index("shift_bdays")


def ruler_split(tbl: pd.DataFrame, events) -> dict:
    """Split the events by the quality of the uncapped ruler, not by the calendar.

    An "exact" ruler tracks the very object the capped fund holds (VIXY vs VXX,
    spot bitcoin vs GBTC), so the spread is a genuine premium proxy. A
    "curve-mismatched" ruler sits at a different point of a futures curve, so the
    spread mixes premium with roll yield. This is the cut that separates the events
    that can answer the question from the events that cannot.
    """
    kind = {ev["key"]: ev.get("ruler", "exact") for ev in events}
    out = {}
    for tag in ("exact", "curve-mismatched"):
        keys = [k for k in tbl.index if kind.get(k) == tag]
        out[tag] = pooled_z(tbl.loc[keys]) if keys else None
    return out


def spread_sharpe(spread: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a signed spread — already an excess-of-cash number.

    One dollar long the fund funded by one dollar short the twin is self-financing:
    the short proceeds earn the cash rate and the long position costs it, so the cash
    leg cancels exactly. ``excess_of_cash_check`` demonstrates that identity on the
    real tape rather than asserting it.
    """
    r = pd.Series(spread).dropna().astype(float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def excess_of_cash_check(fund: pd.Series, proxy: pd.Series, cash: pd.Series,
                         direction: int = 1) -> float:
    """Largest daily gap between the raw spread and the excess-of-cash spread.

    **This is an identity check, not an empirical finding.** ``(r_f − r_c) − (r_p − r_c)``
    cancels to ``r_f − r_p`` algebraically, so the answer is zero to floating point on any
    input and cannot come out otherwise. It is here to prove the *code path* subtracts a
    common cash leg from both arms — i.e. that nothing in this study races a raw return
    against an excess return — and for no stronger purpose. The economic claim it stands
    on is separate and is an ASSUMPTION: a one-dollar-long / one-dollar-short spread is
    self-financing only if the short rebate is the full cash rate. It is not, on a
    hard-to-borrow name, which is why the short leg is charged ``borrow_ann`` explicitly
    in ``trade_event`` and swept from 0 to 30 %/yr.
    """
    common = fund.dropna().index.intersection(proxy.dropna().index).intersection(
        cash.dropna().index)
    # Explicit shift-division rather than .pct_change(): the default fill_method has moved
    # across pandas 2.x/3.x and this study must give the same answer on both.
    f, p, c = fund.loc[common], proxy.loc[common], cash.loc[common]
    rf = f / f.shift(1) - 1.0
    rp = p / p.shift(1) - 1.0
    rc = c / c.shift(1) - 1.0
    plain = direction * (rf - rp)
    excess = direction * ((rf - rc) - (rp - rc))
    return float((plain - excess).abs().max())


def era_cut(tbl: pd.DataFrame, events, split: str = "2020-01-01") -> dict:
    """Split the events by announcement date and report each half's pooled z."""
    when = {ev["key"]: pd.Timestamp(ev["halt"]) for ev in events}
    out = {}
    for tag, mask in [("early", lambda d: d < pd.Timestamp(split)),
                      ("late", lambda d: d >= pd.Timestamp(split))]:
        keys = [k for k in tbl.index if k in when and mask(when[k])]
        sub = tbl.loc[keys] if keys else tbl.iloc[0:0]
        out[tag] = pooled_z(sub) if len(sub) else None
    return out


# --------------------------------------------------------------------------- #
# 5. Tradability — the costed, borrow-charged spread trade
# --------------------------------------------------------------------------- #
BLIND_DAYS = 60


def trade_event(spread: pd.Series, ev: dict, cost_bps: float = 10.0,
                borrow_ann: float = 3.0, hold: str = "halt",
                blind_days: int = BLIND_DAYS) -> dict:
    """The obvious trade, costed.

    ``hold='halt'``: one dollar long the capped fund, one dollar short the uncapped twin,
    entered at the close of the session AFTER the announcement and unwound at the
    resumption date. ``hold='fade'``: the reverse position for 20 sessions after
    resumption. ``hold='blind'``: the same entry, but exited after a **fixed**
    ``blind_days`` sessions.

    **The 'halt' exit is a hindsight exit** and is labelled as one wherever it is
    reported. Nobody standing at ``t+1`` knows the suspension will last 101 sessions
    (VXX) or 6 (USO), and for the APPROX events the resumption date is our own reading
    rather than a filing. ``hold='blind'`` is the no-look-ahead counterpart — a rule you
    could actually have followed — and it is the one that belongs in a tradability
    verdict.

    Costs: ``cost_bps`` one-way × NAV on BOTH legs at entry and at exit (four crossings),
    **plus a daily rebalancing charge**. The gross here is ``exp(Σ x) − 1``, which is the
    return of a position held at *constant* dollar neutrality — i.e. one that is
    rebalanced every close. That rebalancing is not free: restoring equal notionals after
    a day in which the legs diverged by ``x_t`` turns over about ``|x_t|`` of notional, so
    the trade pays ``cost_bps × Σ|x_t|``. Charging only the entry and exit crossings
    understated the cost of the long holds badly — 0.40% instead of 6.13% on the
    2,183-session GBTC regime. The short leg pays ``borrow_ann`` %/yr on notional for
    every calendar day held. The spread is self-financing, so this P&L is already
    excess-of-cash.
    """
    idx = spread.index
    if hold == "halt":
        i0 = _pos_after(idx, ev["halt"])
        i1 = int(idx.searchsorted(pd.Timestamp(ev["resume"]), side="right"))
        sign = 1.0
    elif hold == "blind":
        i0 = _pos_after(idx, ev["halt"])
        i1 = min(i0 + int(blind_days), len(idx))
        sign = 1.0
    else:
        i0 = _pos_after(idx, ev["resume"])
        i1 = min(i0 + 20, len(idx))
        sign = -1.0
    leg = spread.iloc[i0:i1]
    n = len(leg)
    if n == 0:
        return {"key": ev["key"], "n_days": 0, "gross_pct": float("nan"),
                "net_pct": float("nan"), "cost_pct": float("nan"),
                "rebal_pct": float("nan"), "borrow_pct": float("nan")}
    gross = float(np.exp(sign * leg.sum()) - 1.0)
    cal_days = float((idx[i1 - 1] - idx[i0]).days) + 1.0
    cost = 4.0 * cost_bps * 1e-4
    rebal = float(leg.abs().sum()) * cost_bps * 1e-4
    borrow = (borrow_ann / 100.0) * cal_days / 365.0
    return {"key": ev["key"], "n_days": int(n), "cal_days": cal_days,
            "gross_pct": gross * 100, "cost_pct": cost * 100, "rebal_pct": rebal * 100,
            "borrow_pct": borrow * 100,
            "net_pct": (gross - cost - rebal - borrow) * 100}


def trade_table(spreads: dict[str, pd.Series], events, cost_bps: float = 10.0,
                borrow_ann: float = 3.0, hold: str = "halt",
                blind_days: int = BLIND_DAYS) -> pd.DataFrame:
    rows = [trade_event(spreads[ev["key"]], ev, cost_bps=cost_bps, borrow_ann=borrow_ann,
                        hold=hold, blind_days=blind_days)
            for ev in events if ev["key"] in spreads]
    return pd.DataFrame(rows).set_index("key")


def trade_sweep(spreads: dict[str, pd.Series], events,
                cost_grid=(0.0, 5.0, 10.0, 25.0), borrow_grid=(0.0, 3.0, 10.0, 30.0),
                hold: str = "halt") -> pd.DataFrame:
    """Cost × borrow sweep of the mean and median per-event net P&L.

    Borrow on a capped, hard-to-borrow fund is exactly the thing that eats this trade:
    the grid runs from free to 30%/yr, which is where a squeezed ETN sat in 2020-2022.
    """
    rows = []
    for c in cost_grid:
        for b in borrow_grid:
            t = trade_table(spreads, events, cost_bps=c, borrow_ann=b, hold=hold)
            net = t["net_pct"].dropna()
            rows.append({"cost_bps": c, "borrow_ann_pct": b,
                         "mean_net_pct": float(net.mean()) if len(net) else float("nan"),
                         "median_net_pct": float(net.median()) if len(net) else float("nan"),
                         "n_positive": int((net > 0).sum()), "n": int(len(net)),
                         "t_net": one_sample_t(net.to_numpy())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Synthetic control — the machinery proof (never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(signal_strength: float = 1.0, n_events: int = 6, k: int = 20,
                     seed: int = 918, premium_per_day_bps: float = 12.0) -> dict:
    """Run the whole estimator on a synthetic panel with a known planted premium.

    At ``signal_strength=1`` the pooled announcement z and the suspended-regime drift
    must both be clearly positive and the resumption fade clearly negative. At
    ``signal_strength=0`` all three must be centred on zero. This proves the harness
    neither misses a real halt effect nor invents one — it never supports a stamp.
    """
    from . import data as _data

    frames, evs = _data.synthetic_panel(
        n_events=n_events, signal_strength=signal_strength, seed=seed,
        premium_per_day_bps=premium_per_day_bps,
    )
    spreads = build_spreads_from_frames(frames, evs)
    tbl = event_study(spreads, evs, k=k)
    reg = regime_table(spreads, evs)
    fad = fade_study(spreads, evs, k=k)
    pooled = pooled_z(tbl)
    return {
        "signal_strength": float(signal_strength),
        "pooled_z": pooled["mean_z"], "pooled_t": pooled["t"],
        "mean_in_bps": float(reg["in_bps"].mean()),
        "mean_out_bps": float(reg["out_bps"].mean()),
        "mean_t_in": float(reg["t_in"].mean()),
        "mean_fade_z": float(fad["z"].mean()),
        "n_events": int(n_events),
    }
