"""Event study + costed pair trade for Study 919 — Methodology Shock.

The claim: when an index provider changes the **rules** of an index — a special
rebalance, a float-adjustment phase, an eligibility filter, a reconstitution banding
change — the wrapper that tracks it must trade a large basket, and that forced flow
leaves a footprint a price-taker can front-run by buying the affected wrapper against
an unaffected sibling.

How we test it, in one paragraph. For every hardcoded rule change (``data.EVENTS``) we
take the affected ETF (*treated*) and a sibling ETF the same change does not touch
(*control*): QQQ vs SPY for Nasdaq-100 rule changes, SPY vs IWM for S&P eligibility and
float changes, IWM vs MDY for Russell reconstitution rules. On a clean 250-day
**estimation window** ending 21 trading days before the event we fit the market model
``r_treated = alpha + beta * r_control + e``; the **abnormal return** on an event-window
day is ``r_treated - alpha - beta * r_control``, and the **CAR** is its sum over the
window. Beta-adjusting matters: QQQ's beta to SPY is ~1.15, so a raw one-for-one
difference would mostly measure the market, not the rule change.

Execution and honesty:

- **One execution lag.** The announcement is public at the close of day 0. The tradable
  window therefore starts at day **+1**: the day-0 return, which contains the news, is
  never earned. Windows that include day 0 or earlier (``(-5, +5)``, ``(-1, +1)``) are
  reported as *descriptive* event-study statistics and are explicitly labelled
  non-tradable.
- **Costs one-way x NAV.** The tradable pair is *beta*-hedged, not naively dollar-neutral:
  1 unit long the treated wrapper against ``beta`` units short the sibling. A round trip
  therefore trades ``2 x (1 + beta)`` units of notional per unit of NAV, each charged
  ``cost_bps`` one-way. The short leg pays **borrow** for every calendar day held; the
  borrow rate is swept because it is an assumption, not a tape input. The naive 1x/1x
  version is reported as a variant, and it is a trap — with SPY's beta to IWM at 0.5-0.8
  it mostly measures the market.
- **Excess-of-cash.** Only the naive ``1x/1x`` variant is exactly dollar-neutral. The
  beta-hedged pair is long 1 unit and short ``beta`` units, so it carries a **residual net
  dollar exposure of ``1 - beta``** — between −0.23 and +0.52 of NAV across the events
  here. That residual has to be financed (or earns cash) at the bill rate, and it is
  charged explicitly (``finance_bps_ann``, an ASSUMPTION, swept) so that the reported net
  return really is an excess-of-cash return rather than one asserted to be. The
  deployed-capital race (park in BIL, overlay the pair during event windows) is therefore
  also run excess-of-cash on both sides.
- **Total return.** All closes are dividend-adjusted (``auto_adjust=True``). QQQ yields
  roughly a percent less than SPY, so a price-only comparison would plant a spurious
  negative drift in the QQQ-minus-SPY leg.
- **Survivorship: none to name.** The universe is four large, continuously listed
  wrappers picked ex ante by which index each one tracks — there is no cross-section to
  select from and no dead funds to omit. What *is* selected is the **event list**, a
  hand-assembled proxy; that is why the placebo test, the window sweep and the drop-one
  jackknife all appear below.

The honest statistical problem, stated up front: a rule-change calendar has fewer than
ten entries in three decades. The placebo randomisation test below is what makes that
tractable — it prices the pooled CAR against thousands of pseudo-events drawn from the
same tape — and ``minimum_detectable_car`` reports how big an effect this sample could
ever have resolved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def leg_date(event: dict, leg: str = "announce"):
    """The event's ``leg`` date as a Timestamp, or ``None`` when that leg does not apply.

    Mirrors ``data.leg_date`` so ``strategy`` stays importable on its own. A ``None``
    announcement date means the rule change shared an earlier press release and must not
    be double-counted as a second independent observation.
    """
    v = event.get(leg)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    ts = pd.Timestamp(v)
    return None if pd.isna(ts) else ts


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain one-sample t of mean(x) vs 0 (used across events, which are independent)."""
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


def newey_west_t(x, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * (float(u[l:] @ u[:-l]) / n)
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
    return (mid - half, mid + half)


def sharpe(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).astype(float).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def summary(returns, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an excess series)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    sd = r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year if n else float("nan")
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if n and years > 0 and wealth.iloc[-1] > 0 else float("nan")
    return {
        "n_days": int(n),
        "sharpe": float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": cagr,
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()) if n else float("nan"),
        "mean_daily_bps": float(r.mean() * 1e4) if n else float("nan"),
        "tstat": newey_west_t(r.to_numpy(), lags=5),
    }


# --------------------------------------------------------------------------- #
# Market model & abnormal returns
# --------------------------------------------------------------------------- #
def market_model(r_treated: np.ndarray, r_control: np.ndarray) -> tuple:
    """OLS ``r_treated = alpha + beta * r_control`` over the estimation window.

    Returns ``(alpha, beta, resid_sd)``. The residual standard deviation is the natural
    yardstick for "how big is this CAR?" — see ``minimum_detectable_car``.
    """
    x = np.asarray(r_control, dtype=float)
    y = np.asarray(r_treated, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 30:
        return (float("nan"), float("nan"), float("nan"))
    vx = x.var(ddof=1)
    if vx <= 0:
        return (float("nan"), float("nan"), float("nan"))
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    alpha = float(y.mean() - beta * x.mean())
    resid = y - alpha - beta * x
    return alpha, beta, float(resid.std(ddof=1))


def event_abnormal(
    treated: pd.Series,
    control: pd.Series,
    event_date,
    window: tuple = (1, 10),
    est_len: int = 250,
    est_gap: int = 21,
) -> dict:
    """Market-model abnormal returns for one event.

    ``window`` is a pair of **trading-day offsets** relative to the event day 0 (the first
    session on or after ``event_date``), inclusive. ``(1, 10)`` is the tradable window: the
    announcement is known at the close of day 0 and the position is on from day 1, so the
    day-0 return is never earned. Negative offsets are descriptive only.

    The estimation window is the ``est_len`` sessions ending ``est_gap`` sessions before
    day 0, so no event-window day contaminates alpha/beta. Returns a dict with the
    per-day abnormal returns, the CAR, the raw (unadjusted) one-for-one difference CAR,
    and the fitted alpha/beta/residual sd. ``ok=False`` if there is not enough history.
    """
    common = treated.index.intersection(control.index)
    t = treated.reindex(common).sort_index()
    c = control.reindex(common).sort_index()
    r_t = t.pct_change()
    r_c = c.pct_change()
    idx = r_t.index

    pos = int(idx.searchsorted(pd.Timestamp(event_date), side="left"))
    a, b = int(window[0]), int(window[1])
    est_hi = pos - est_gap
    est_lo = est_hi - est_len
    if est_lo < 1 or pos + b >= len(idx) or pos + a < 1:
        return {"ok": False, "event_date": pd.Timestamp(event_date)}

    alpha, beta, resid_sd = market_model(
        r_t.to_numpy()[est_lo:est_hi], r_c.to_numpy()[est_lo:est_hi]
    )
    if not np.isfinite(beta):
        return {"ok": False, "event_date": pd.Timestamp(event_date)}

    sl = slice(pos + a, pos + b + 1)
    ar = r_t.to_numpy()[sl] - alpha - beta * r_c.to_numpy()[sl]
    raw = r_t.to_numpy()[sl] - r_c.to_numpy()[sl]
    return {
        "ok": True,
        "event_date": pd.Timestamp(event_date),
        "day0": idx[pos],
        "alpha": alpha, "beta": beta, "resid_sd": resid_sd,
        "ar": ar,
        "car": float(np.nansum(ar)),
        "car_raw": float(np.nansum(raw)),
        "n_window": int(len(ar)),
        "window": (a, b),
    }


# --------------------------------------------------------------------------- #
# The pooled event study
# --------------------------------------------------------------------------- #
def run_event_study(
    prices,
    events,
    leg: str = "announce",
    window: tuple = (1, 10),
    est_len: int = 250,
    est_gap: int = 21,
    asof=None,
) -> dict:
    """Pool the market-model CARs across every usable event.

    ``prices`` is either a frame whose columns are ticker names (the real tape) or a dict
    of pair-id -> frame (the synthetic panel); each event names its ``treated`` and
    ``control`` columns. ``leg`` selects the announcement or the effective date.

    Returns a dict with the per-event table, the pooled mean CAR (in bps), the plain
    cross-event *t* (events are independent draws), the HAC *t* on the pooled daily
    abnormal returns, the hit rate with its Wilson interval, and the pooled daily AR
    series used by the bootstrap.
    """
    rows = []
    ar_pool = []
    for e in events:
        d = leg_date(e, leg)
        if d is None or (asof is not None and d > pd.Timestamp(asof)):
            continue
        frame = prices[e["pair"]] if isinstance(prices, dict) else prices
        t_col, c_col = e["treated"], e["control"]
        if t_col not in frame.columns or c_col not in frame.columns:
            continue
        res = event_abnormal(
            frame[t_col].dropna(), frame[c_col].dropna(), d,
            window=window, est_len=est_len, est_gap=est_gap,
        )
        if not res["ok"]:
            continue
        ar_pool.append(res["ar"])
        rows.append({
            "event_id": e["event_id"],
            "leg": leg,
            "date": d,
            "treated": t_col, "control": c_col,
            "beta": res["beta"],
            "car_bps": res["car"] * 1e4,
            "car_raw_bps": res["car_raw"] * 1e4,
            "resid_sd_bps": res["resid_sd"] * 1e4,
            "date_confidence": e.get("date_confidence", "exact"),
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return {"table": tbl, "n_events": 0, "mean_car_bps": float("nan"),
                "t_cross_event": float("nan"), "t_hac_daily": float("nan"),
                "hit_rate": float("nan"), "hit_ci": (float("nan"), float("nan")),
                "ar_daily": np.array([]), "window": window, "leg": leg}

    cars = tbl["car_bps"].to_numpy()
    ar_daily = np.concatenate(ar_pool) if ar_pool else np.array([])
    hits = int((cars > 0).sum())
    return {
        "table": tbl,
        "n_events": int(len(tbl)),
        "mean_car_bps": float(cars.mean()),
        "median_car_bps": float(np.median(cars)),
        "sd_car_bps": float(cars.std(ddof=1)) if len(cars) > 1 else float("nan"),
        "t_cross_event": one_sample_t(cars),
        "t_hac_daily": newey_west_t(ar_daily, lags=5),
        "hit_rate": hits / len(cars),
        "hit_ci": wilson_interval(hits, len(cars)),
        "ar_daily": ar_daily,
        "window": window,
        "leg": leg,
    }


def minimum_detectable_car(study: dict, alpha: float = 0.05) -> float:
    """The smallest pooled CAR (bps) this many events could have called significant.

    With ``n`` independent events and a cross-event CAR dispersion of ``sd``, the
    two-sided 5% threshold is ``t_crit * sd / sqrt(n)``. Printing it turns "we found
    nothing" into the far more useful "we could not have found anything smaller than X".
    """
    n = study["n_events"]
    sd = study.get("sd_car_bps", float("nan"))
    if n < 2 or not np.isfinite(sd):
        return float("nan")
    # Two-sided 5% Student-t critical value, small-n table (no scipy dependency).
    tcrit = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262}.get(n, 2.10 if n < 30 else 1.96)
    return float(tcrit * sd / np.sqrt(n))


# --------------------------------------------------------------------------- #
# A vectorised kernel for the same market-model CAR, used by the placebo test
# --------------------------------------------------------------------------- #
class PairKernel:
    """Prefix sums for one treated/control pair, so a CAR costs O(1) per position.

    The placebo test needs the *same* market-model CAR at thousands of pseudo-event
    positions. Refitting an OLS per draw is needlessly slow, so alpha, beta and the window
    sums are all recovered from four cumulative sums. The arithmetic is identical to
    ``event_abnormal`` — ``test_kernel_matches_event_abnormal`` pins them together.
    """

    def __init__(self, treated: pd.Series, control: pd.Series):
        common = treated.index.intersection(control.index).sort_values()
        rt = np.array(treated.reindex(common).pct_change().to_numpy(dtype=float), copy=True)
        rc = np.array(control.reindex(common).pct_change().to_numpy(dtype=float), copy=True)
        rt[0] = 0.0
        rc[0] = 0.0   # position 0 is never inside a legal window (see the guards below)
        self.index = common
        self.n = int(len(rt))
        self.cs_t = np.concatenate([[0.0], np.cumsum(rt)])
        self.cs_c = np.concatenate([[0.0], np.cumsum(rc)])
        self.cs_cc = np.concatenate([[0.0], np.cumsum(rc * rc)])
        self.cs_tc = np.concatenate([[0.0], np.cumsum(rt * rc)])

    def legal_positions(self, window: tuple, est_len: int = 250, est_gap: int = 21) -> np.ndarray:
        a, b = int(window[0]), int(window[1])
        lo = est_len + est_gap + 1
        hi = self.n - max(b, 0) - 1
        lo = max(lo, 1 - min(a, 0))
        return np.arange(lo, hi) if hi > lo else np.array([], dtype=int)

    def car(self, pos, window: tuple, est_len: int = 250, est_gap: int = 21) -> np.ndarray:
        """Market-model CAR (in raw return units) at each position in ``pos``."""
        p = np.asarray(pos, dtype=int)
        a, b = int(window[0]), int(window[1])
        e_hi = p - est_gap
        e_lo = e_hi - est_len
        m = float(est_len)
        sx = self.cs_c[e_hi] - self.cs_c[e_lo]
        sy = self.cs_t[e_hi] - self.cs_t[e_lo]
        sxx = self.cs_cc[e_hi] - self.cs_cc[e_lo]
        sxy = self.cs_tc[e_hi] - self.cs_tc[e_lo]
        var = (sxx - sx * sx / m) / (m - 1.0)
        cov = (sxy - sx * sy / m) / (m - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(var > 0, cov / var, np.nan)
        alpha = sy / m - beta * sx / m
        w_lo, w_hi = p + a, p + b + 1
        k = float(b - a + 1)
        sum_t = self.cs_t[w_hi] - self.cs_t[w_lo]
        sum_c = self.cs_c[w_hi] - self.cs_c[w_lo]
        return sum_t - k * alpha - beta * sum_c


# --------------------------------------------------------------------------- #
# Placebo randomisation test — the honest p-value for a tiny event list
# --------------------------------------------------------------------------- #
def placebo_test(
    prices,
    events,
    leg: str = "announce",
    window: tuple = (1, 10),
    n_draws: int = 2000,
    exclusion_days: int = 30,
    est_len: int = 250,
    est_gap: int = 21,
    seed: int = 919,
    asof=None,
) -> dict:
    """Price the observed pooled CAR against pseudo-events drawn from the same tape.

    For each draw we replace every real event date with a random session from the *same*
    treated/control pair (excluding +/-``exclusion_days`` sessions around any real event),
    then recompute the pooled mean CAR exactly as the real study does. The share of draws
    whose |pooled mean CAR| exceeds the observed one is the empirical two-sided p-value —
    no distributional assumption, and it automatically absorbs the market model's own
    small-sample quirks.
    """
    rng = np.random.default_rng(seed)
    obs = run_event_study(prices, events, leg=leg, window=window,
                          est_len=est_len, est_gap=est_gap, asof=asof)
    if obs["n_events"] == 0:
        return {"observed_bps": float("nan"), "p_two_sided": float("nan"),
                "null_mean_bps": float("nan"), "null_sd_bps": float("nan"), "n_draws": 0}

    used = obs["table"]
    # One kernel + one legal draw pool per (pair, treated, control) combination.
    kernels, pools, keys = {}, {}, []
    for _, row in used.iterrows():
        ev = next(e for e in events if e["event_id"] == row["event_id"])
        pid = ev.get("pair")
        key = (pid, row["treated"], row["control"])
        keys.append(key)
        if key in kernels:
            continue
        frame = prices[pid] if isinstance(prices, dict) else prices
        k = PairKernel(frame[row["treated"]].dropna(), frame[row["control"]].dropna())
        legal = k.legal_positions(window, est_len=est_len, est_gap=est_gap)
        # Exclude the neighbourhood of every real event that lives on this pair.
        for e in events:
            if e.get("pair") != pid:
                continue
            for lg in ("announce", "effective"):
                d = leg_date(e, lg)
                if d is None:
                    continue
                p = int(k.index.searchsorted(d, side="left"))
                legal = legal[(legal < p - exclusion_days) | (legal > p + exclusion_days)]
        kernels[key] = k
        pools[key] = legal

    # Draw all pseudo-events at once: one column per real event, one row per draw.
    cols = []
    for key in keys:
        legal = pools[key]
        if legal.size == 0:
            continue
        pos = rng.choice(legal, size=n_draws, replace=True)
        cols.append(kernels[key].car(pos, window, est_len=est_len, est_gap=est_gap) * 1e4)
    if not cols:
        return {"observed_bps": float(obs["mean_car_bps"]), "p_two_sided": float("nan"),
                "null_mean_bps": float("nan"), "null_sd_bps": float("nan"), "n_draws": 0}
    null = np.nanmean(np.vstack(cols), axis=0)
    null = null[np.isfinite(null)]
    obs_val = obs["mean_car_bps"]
    p = float((np.abs(null) >= abs(obs_val)).mean()) if null.size else float("nan")
    return {
        "observed_bps": float(obs_val),
        "p_two_sided": p,
        "null_mean_bps": float(null.mean()) if null.size else float("nan"),
        "null_sd_bps": float(null.std(ddof=1)) if null.size > 1 else float("nan"),
        "null_q025_bps": float(np.percentile(null, 2.5)) if null.size else float("nan"),
        "null_q975_bps": float(np.percentile(null, 97.5)) if null.size else float("nan"),
        "n_draws": int(null.size),
        "n_events": obs["n_events"],
        "window": window,
        "leg": leg,
    }


# --------------------------------------------------------------------------- #
# Bootstrap CIs
# --------------------------------------------------------------------------- #
def bootstrap_car_ci(cars, n_boot: int = 5000, seed: int = 919, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for the pooled mean CAR, resampling **events**.

    Events are separated by years, so the natural resampling unit is the event itself —
    there is no serial dependence between them to preserve.
    """
    x = np.asarray(cars, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return {"mean_bps": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n": int(x.size)}
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_bps": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((draws < 0).mean()), "n": int(x.size),
            "n_boot": int(n_boot)}


def block_bootstrap_ar_ci(
    ar_daily, window_len: int, n_boot: int = 5000, block: int = 5,
    seed: int = 919, alpha: float = 0.05,
) -> dict:
    """Circular block bootstrap on the pooled daily abnormal returns.

    Blocks of ``block`` consecutive event-window days preserve the short-run
    autocorrelation the event window carries; each resample is scaled up to a
    ``window_len``-day CAR so the interval is directly comparable to the headline number.
    """
    r = np.asarray(ar_daily, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 2:
        return {"car_bps": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        draws[b] = r[idx].mean() * window_len * 1e4
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"car_bps": float(r.mean() * window_len * 1e4),
            "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((draws < 0).mean()),
            "n_obs": int(n), "block": int(block), "n_boot": int(n_boot)}


# --------------------------------------------------------------------------- #
# The costed, dollar-neutral pair trade (the tradable arm)
# --------------------------------------------------------------------------- #
def pair_trade(
    prices,
    events,
    leg: str = "announce",
    hold_days: int = 10,
    cost_bps: float = 5.0,
    borrow_bps_ann: float = 50.0,
    finance_bps_ann: float = 200.0,
    hedge: str = "beta",
    est_len: int = 250,
    est_gap: int = 21,
    asof=None,
) -> dict:
    """Long the treated wrapper, short its sibling, for ``hold_days`` after the news.

    One execution lag: the announcement is public at the close of day 0, the position goes
    on at the close of day 0 for the *next* day's return — that is, the trade earns days
    +1..+``hold_days`` and never the day-0 return, which is the one containing the news.

    ``hedge`` picks the short size, and it matters far more than it looks:

    - ``"beta"`` (default) — short ``beta`` units of the sibling per 1 unit long, with
      ``beta`` fitted on the same clean pre-event estimation window the event study uses.
      This is the tradable twin of the market-model CAR: SPY's beta to IWM is ~0.5-0.8, so
      a naive one-for-one short would leave the "pure rule-change trade" carrying a large
      market exposure and measure the market, not the rule change.
    - ``"dollar"`` — the naive 1x/1x version, reported as a variant.

    Frictions, all charged one-way x NAV:

    - **cost**: ``(1 + beta) x 2 x cost_bps`` per round trip (both legs in, both legs out);
    - **borrow**: ``borrow_bps_ann`` per annum on the ``beta``-unit short notional, accrued
      over the calendar days held — an assumption, hence the sweep in ``cost_sweep``;
    - **financing**: the short proceeds fund the long only up to ``beta`` units. What is
      left, ``1 - beta`` units of NAV, is a genuine net dollar position: net long when
      ``beta < 1`` (SPY/IWM, beta 0.48-0.81), net short when ``beta > 1`` (QQQ/SPY,
      IWM/MDY). It is charged ``finance_bps_ann * (1 - beta) * days/365`` — a cost when
      net long, a credit when net short — which is exactly what turns the reported number
      into an **excess-of-cash** return. ``finance_bps_ann`` is an ASSUMPTION (a stand-in
      for the bill rate on each event date, since BIL only starts in 2007) and is swept.
      For ``hedge="dollar"`` beta is 1 and the term vanishes, as it should.

    Returns per-event gross/net returns in bps plus pooled statistics.
    """
    rows = []
    for e in events:
        d = leg_date(e, leg)
        if d is None or (asof is not None and d > pd.Timestamp(asof)):
            continue
        frame = prices[e["pair"]] if isinstance(prices, dict) else prices
        t_col, c_col = e["treated"], e["control"]
        if t_col not in frame.columns or c_col not in frame.columns:
            continue
        t = frame[t_col].dropna()
        c = frame[c_col].dropna()
        common = t.index.intersection(c.index).sort_values()
        pos = int(common.searchsorted(d, side="left"))
        if pos + hold_days >= len(common) or pos < 1:
            continue
        # Hedge ratio from the clean pre-event estimation window (never the event window).
        if hedge == "beta":
            r_t = t.reindex(common).pct_change().to_numpy()
            r_c = c.reindex(common).pct_change().to_numpy()
            e_hi, e_lo = pos - est_gap, pos - est_gap - est_len
            if e_lo < 1:
                continue
            _, beta, _ = market_model(r_t[e_lo:e_hi], r_c[e_lo:e_hi])
            if not np.isfinite(beta):
                continue
        else:
            beta = 1.0
        # Entry at the close of day 0; exit at the close of day +hold_days.
        p_t0, p_t1 = float(t.loc[common[pos]]), float(t.loc[common[pos + hold_days]])
        p_c0, p_c1 = float(c.loc[common[pos]]), float(c.loc[common[pos + hold_days]])
        gross = (p_t1 / p_t0 - 1.0) - beta * (p_c1 / p_c0 - 1.0)
        cost = 2.0 * (1.0 + abs(beta)) * cost_bps * 1e-4
        cal_days = (common[pos + hold_days] - common[pos]).days
        borrow = borrow_bps_ann * 1e-4 * abs(beta) * cal_days / 365.0
        # Residual net dollar exposure of (1 - beta) units of NAV: financed at the bill
        # rate when net long, earns it when net short. This is what makes the number an
        # excess-of-cash return instead of merely being called one.
        finance = finance_bps_ann * 1e-4 * (1.0 - beta) * cal_days / 365.0
        rows.append({
            "event_id": e["event_id"], "date": d, "beta": float(beta),
            "net_dollar_exposure": float(1.0 - beta),
            "gross_bps": gross * 1e4,
            "cost_bps_charged": cost * 1e4,
            "borrow_bps_charged": borrow * 1e4,
            "finance_bps_charged": finance * 1e4,
            "net_bps": (gross - cost - borrow - finance) * 1e4,
        })
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return {"table": tbl, "n": 0, "mean_gross_bps": float("nan"),
                "mean_net_bps": float("nan"), "t_gross": float("nan"),
                "t_net": float("nan"), "hit_rate_net": float("nan")}
    g = tbl["gross_bps"].to_numpy()
    nt = tbl["net_bps"].to_numpy()
    hits = int((nt > 0).sum())
    return {
        "table": tbl, "n": int(len(tbl)),
        "mean_gross_bps": float(g.mean()), "mean_net_bps": float(nt.mean()),
        "median_net_bps": float(np.median(nt)),
        "t_gross": one_sample_t(g), "t_net": one_sample_t(nt),
        "hit_rate_net": hits / len(nt), "hit_ci": wilson_interval(hits, len(nt)),
        "hold_days": int(hold_days), "cost_bps": cost_bps,
        "borrow_bps_ann": borrow_bps_ann, "finance_bps_ann": finance_bps_ann,
        "mean_finance_bps": float(tbl["finance_bps_charged"].mean()),
        "hedge": hedge,
    }


def cost_sweep(
    prices, events, leg: str = "announce", hold_days: int = 10,
    cost_grid=(0.0, 1.0, 5.0, 10.0, 25.0),
    borrow_grid=(0.0, 50.0, 100.0, 300.0),
    finance_bps_ann: float = 200.0,
    hedge: str = "beta",
    asof=None,
) -> pd.DataFrame:
    """Sweep the two frictions that are assumptions rather than tape: cost and borrow."""
    rows = []
    for cb in cost_grid:
        for bb in borrow_grid:
            pt = pair_trade(prices, events, leg=leg, hold_days=hold_days,
                            cost_bps=cb, borrow_bps_ann=bb,
                            finance_bps_ann=finance_bps_ann, hedge=hedge, asof=asof)
            rows.append({"cost_bps": cb, "borrow_bps_ann": bb,
                         "mean_net_bps": pt["mean_net_bps"], "t_net": pt["t_net"],
                         "hit_rate_net": pt["hit_rate_net"], "n": pt["n"]})
    return pd.DataFrame(rows)


def finance_sweep(
    prices, events, leg: str = "announce", hold_days: int = 10,
    cost_bps: float = 5.0, borrow_bps_ann: float = 50.0,
    finance_grid=(0.0, 200.0, 500.0),
    hedge: str = "beta", asof=None,
) -> pd.DataFrame:
    """Sweep the third assumption: the bill rate charged on the residual (1-beta) exposure."""
    rows = []
    for fb in finance_grid:
        pt = pair_trade(prices, events, leg=leg, hold_days=hold_days, cost_bps=cost_bps,
                        borrow_bps_ann=borrow_bps_ann, finance_bps_ann=fb,
                        hedge=hedge, asof=asof)
        rows.append({"finance_bps_ann": fb, "mean_finance_bps": pt["mean_finance_bps"],
                     "mean_net_bps": pt["mean_net_bps"], "t_net": pt["t_net"],
                     "n": pt["n"]})
    return pd.DataFrame(rows)


def deployed_capital_race(
    prices, cash: pd.Series, events, leg: str = "announce", hold_days: int = 10,
    cost_bps: float = 5.0, borrow_bps_ann: float = 50.0,
    finance_bps_ann: float = 200.0, asof=None,
) -> dict:
    """Park the NAV in cash and overlay the pair only inside event windows.

    The overlay's **excess-of-cash** return series is the pair's net daily P&L (zero on
    the idle days, which are the overwhelming majority). Racing it against a flat zero is
    the honest question: does an investor who holds T-bills and puts this trade on every
    rule change in two decades end up ahead? Reported alongside the number of live days,
    so the annualised Sharpe is read for what it is — a very sparse series.

    **Events that predate the cash series are dropped, not teleported.** BIL only starts
    in 2007, and a naive ``searchsorted`` maps every earlier event onto BIL's very first
    session — which would stack pre-2007 P&L into the first ten days of the cash era and
    invent a drawdown that never happened. ``n_dropped_pre_cash`` counts them so the
    reported event count is the number that genuinely land inside the cash window.
    """
    pt = pair_trade(prices, events, leg=leg, hold_days=hold_days,
                    cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann,
                    finance_bps_ann=finance_bps_ann, asof=asof)
    if pt["n"] == 0:
        return {"overlay": None, "n_live_events": 0, "n_dropped_pre_cash": 0}
    idx = cash.dropna().index.sort_values()
    overlay = pd.Series(0.0, index=idx, name="overlay_excess")
    live = 0
    dropped = []
    for _, row in pt["table"].iterrows():
        if len(idx) == 0 or row["date"] < idx[0]:
            dropped.append(row["event_id"])   # before the cash series even exists
            continue
        pos = int(idx.searchsorted(row["date"], side="left"))
        if pos + hold_days >= len(idx):
            dropped.append(row["event_id"])
            continue
        # Spread the round-trip P&L evenly across the holding days it was earned over.
        per_day = row["net_bps"] * 1e-4 / hold_days
        overlay.iloc[pos + 1: pos + hold_days + 1] += per_day
        live += 1
    s = summary(overlay)
    return {
        "overlay": overlay, "summary": s, "n_live_events": live,
        "n_dropped_pre_cash": len(dropped), "dropped_pre_cash": dropped,
        "n_live_days": int((overlay != 0).sum()),
        "total_excess_bps": float(overlay.sum() * 1e4),
        "pair": pt,
    }


# --------------------------------------------------------------------------- #
# Robustness — window sweep, drop-one jackknife, era cut
# --------------------------------------------------------------------------- #
DESCRIPTIVE_WINDOWS = ((-5, -1), (-1, 1), (-5, 5), (-10, 10))
TRADABLE_WINDOWS = ((1, 1), (1, 3), (1, 5), (1, 10), (1, 21))


def window_sweep(prices, events, leg: str = "announce", windows=None,
                 n_placebo: int = 2000, seed: int = 919, asof=None) -> pd.DataFrame:
    """Pooled CAR across a grid of windows, each flagged tradable or descriptive.

    A window is *tradable* only if it starts at day +1 or later; anything spanning day 0
    or before uses information that is not actionable and is labelled as such.

    Each row carries its own placebo p-value **and** a Bonferroni-adjusted one, because a
    grid of windows is a small multiple-testing exercise: with nine windows the smallest
    of nine p-values has to clear roughly 0.05/9 before it means anything.
    """
    if windows is None:
        windows = tuple(DESCRIPTIVE_WINDOWS) + tuple(TRADABLE_WINDOWS)
    rows = []
    for w in windows:
        s = run_event_study(prices, events, leg=leg, window=w, asof=asof)
        p = float("nan")
        if n_placebo:
            p = placebo_test(prices, events, leg=leg, window=w,
                             n_draws=n_placebo, seed=seed, asof=asof)["p_two_sided"]
        rows.append({
            "window": f"[{w[0]:+d},{w[1]:+d}]",
            "tradable": bool(w[0] >= 1),
            "n_events": s["n_events"],
            "mean_car_bps": s["mean_car_bps"],
            "t_cross_event": s["t_cross_event"],
            "t_hac_daily": s["t_hac_daily"],
            "hit_rate": s["hit_rate"],
            "placebo_p": p,
            "placebo_p_bonferroni": min(1.0, p * len(windows)) if np.isfinite(p) else float("nan"),
        })
    return pd.DataFrame(rows)


def jackknife(prices, events, leg: str = "announce", window: tuple = (1, 10), asof=None) -> pd.DataFrame:
    """Drop-one jackknife: is the pooled CAR carried by a single event?

    Also the study's answer to the hand-assembled event list — if one questionable date
    drives everything, the result is not about index methodology at all.
    """
    full = run_event_study(prices, events, leg=leg, window=window, asof=asof)
    rows = []
    for _, row in full["table"].iterrows():
        keep = [e for e in events if e["event_id"] != row["event_id"]]
        s = run_event_study(prices, keep, leg=leg, window=window, asof=asof)
        rows.append({
            "dropped": row["event_id"],
            "own_car_bps": row["car_bps"],
            "mean_car_bps_without": s["mean_car_bps"],
            "t_without": s["t_cross_event"],
        })
    return pd.DataFrame(rows)


def era_cut(prices, events, split: str = "2015-01-01", leg: str = "announce",
            window: tuple = (1, 10), asof=None) -> dict:
    """Split the event list by date and re-run the pooled study on each half."""
    cut = pd.Timestamp(split)
    out = {}
    for tag, sel in [("early", lambda d: d < cut), ("late", lambda d: d >= cut)]:
        sub = [e for e in events
               if leg_date(e, leg) is not None and sel(leg_date(e, leg))]
        s = run_event_study(prices, sub, leg=leg, window=window, asof=asof)
        out[tag] = None if s["n_events"] == 0 else {
            "n_events": s["n_events"],
            "mean_car_bps": s["mean_car_bps"],
            "t_cross_event": s["t_cross_event"],
            "hit_rate": s["hit_rate"],
        }
    return out


# --------------------------------------------------------------------------- #
# Synthetic control — the machinery proof (never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(
    prices, events, window: tuple = (1, 10), n_draws: int = 300, seed: int = 919,
) -> dict:
    """Run the whole detector on a synthetic tape with a known planted CAR.

    On the planted world the pooled CAR must come out large, positive and far outside the
    placebo distribution; on the null it must be indistinguishable from placebo. This
    proves the harness is unbiased — it never supports a stamp on the real tape.
    """
    s = run_event_study(prices, events, window=window)
    pl = placebo_test(prices, events, window=window, n_draws=n_draws, seed=seed)
    return {
        "n_events": s["n_events"],
        "mean_car_bps": s["mean_car_bps"],
        "t_cross_event": s["t_cross_event"],
        "hit_rate": s["hit_rate"],
        "placebo_p": pl["p_two_sided"],
        "placebo_null_sd_bps": pl["null_sd_bps"],
        "beta_mean": float(s["table"]["beta"].mean()) if s["n_events"] else float("nan"),
    }
