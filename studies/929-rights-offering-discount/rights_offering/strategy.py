"""Event study + inference for Study 929 — Rights Offering.

The claim under test. When a listed company or fund raises equity through a **rights
offering**, existing holders get the right to subscribe new shares at a price well
below the market. Retail folklore reads that discount as a *gift*: buy before the
deal, subscribe cheaply, pocket the spread. Corporate-finance theory reads it as a
*warning*: an issuer selling equity cheaply is signalling that its shares are dear (or
that it needs the cash), and the discount is a distribution mechanism, not free money —
the dilution is borne by the same shareholders it is handed to.

What this module measures, on a hand-compiled list of US rights offerings (see
``data.RIGHTS_EVENTS``, a PROXY):

1. **Abnormal returns** against SPY via a market model (alpha/beta estimated on the
   ``(-250, -31)`` window before the anchor), cumulated over three windows —
   *announcement*, *subscription period*, *post-expiry*.
2. **Is the discount compensated?** A deep-vs-shallow band split on the post-expiry
   window, plus a **label-permutation** test asking whether that split beats a random
   relabelling.
3. **Is it tradable?** A calendar-time equal-weight portfolio of in-window names, raced
   **excess-of-cash on both legs** against SPY excess-of-cash, gross and net of one-way
   costs, with a HAC t on the daily difference and a beta-adjusted alpha (see
   :func:`alpha_vs_market`). The short leg (the "dilution warning" trade) pays borrow,
   swept, and is credited its cash collateral so the costs are not one-sided. The book
   is run over the full ``(+1, +49)`` span **and** over the ex-rights-free
   ``(+29, +49)`` span, because the full span eats the mechanical dilution (below) —
   which flatters the short leg and penalises the long one.
4. **Does it survive the list's weaknesses?** A leave-one-out and a leave-one-*issuer*-out
   jackknife (the list is dominated by a few serial issuers), a placebo of random
   pseudo-anchors, an era cut, a +/- 10 calendar-day anchor jitter, and a sweep of the
   assumed rights timetable.

**One execution lag, everywhere — with one honest hole.** An announcement dated day ``t``
is known at that day's close; every tradable leg enters at the ``t+1`` close, and the
announcement-day return is an *event-study* quantity that never enters a tradable leg.
The hole: the anchors are **month-precision** (``data.RIGHTS_EVENTS``), so for roughly
half the deals the anchor sits *before* the true press release and the book is therefore
long a name whose news is not yet public. That is a genuine look-ahead channel and it is
named rather than hidden — note its direction. It can only *manufacture* an announcement
reaction the book would not have caught in real time, so it biases the study **towards**
finding an edge; the anchor jitter (+/- 10 calendar days) sweeps it, and the study finds
nothing anyway.

**Adjustment caveat.** Yahoo's ``auto_adjust`` does not neutralise the ex-rights price
drop, so the subscription window mixes real news with mechanical dilution. The
announcement and post-expiry windows are clean, and that is where the **signal** verdict
is formed. The same artefact reaches the tradable book: a ``(+1, +49)`` holding period
spans the modelled ex-rights date, so the long leg eats a price fall for which a real
subscriber was compensated by the right itself, and the short leg pockets it. Both legs
are therefore also reported over the clean ``(+29, +49)`` window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Event windows in TRADING days relative to the anchor row (index 0 = anchor session).
WINDOWS = {
    "announce": (-1, 5),      # the news window
    "run_up": (-20, -2),      # was the stock strong going in?
    "subscription": (1, 28),  # anchor+1 .. about the modelled expiry (dilution mixed in)
    "post_expiry": (29, 49),  # the 20 sessions after the modelled expiry (clean)
}
EST_WINDOW = (-250, -31)      # market-model estimation window, in trading days


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 912)
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


def newey_west_t(x, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
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


def auto_nw_t(x) -> float:
    """Newey-West t with the usual ``4 (n/100)^(2/9)`` bandwidth."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    return newey_west_t(x, lags=int(np.floor(4.0 * (len(x) / 100.0) ** (2.0 / 9.0))))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Market model & abnormal returns
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns of a close frame (so window sums are honest arithmetic).

    ``fill_method=None`` is passed explicitly: pandas 2.x pads by default (and warns),
    pandas 3.x does not. The loaded tape has no interior holes, so the two agree here —
    but the argument is pinned so the numbers cannot depend on the pandas version.
    """
    return prices.sort_index().pct_change(fill_method=None)


def market_model(r_name, r_mkt) -> tuple:
    """OLS ``r_name = alpha + beta r_mkt + e`` on the given (estimation-window) slice.

    Accepts pandas Series or plain arrays; rows with a NaN on either side are dropped.
    Returns ``(alpha, beta, n_obs)``; ``(nan, nan, n)`` when the window is too short.
    """
    y = np.asarray(r_name, dtype=float)
    x = np.asarray(r_mkt, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    if n < 40:
        return (float("nan"), float("nan"), n)
    vx = x.var(ddof=1)
    if not np.isfinite(vx) or vx <= 0:
        return (float("nan"), float("nan"), n)
    beta = float(np.cov(x, y, ddof=1)[0, 1] / vx)
    alpha = float(y.mean() - beta * x.mean())
    return (alpha, beta, n)


def _anchor_pos(index: pd.DatetimeIndex, when: pd.Timestamp):
    """Index position of the first session on or after ``when`` (None if past the tape)."""
    pos = int(index.searchsorted(pd.Timestamp(when), side="left"))
    return pos if pos < len(index) else None


def event_panel(
    prices: pd.DataFrame,
    events,
    market: str = "SPY",
    windows=None,
    est_window=None,
    rets: pd.DataFrame = None,
) -> pd.DataFrame:
    """One row per usable event: market-model betas and the cumulated abnormal returns.

    ``events`` is a sequence of ``(ticker, announce, precision, band, kind)`` tuples;
    ``rets`` lets a caller pass a pre-computed daily-return frame (the placebo loop
    re-uses one rather than differencing the price frame hundreds of times).
    Events whose estimation window or longest post-window runs off the available tape
    are dropped and counted (see :func:`coverage`).

    Abnormal return on day ``s`` is ``r_i,s - (alpha_i + beta_i r_mkt,s)`` with the
    parameters estimated on ``est_window`` *before* the anchor — nothing from inside
    any event window enters the estimate.
    """
    win = dict(WINDOWS if windows is None else windows)
    est_lo, est_hi = EST_WINDOW if est_window is None else est_window
    rets = daily_returns(prices) if rets is None else rets
    if market not in rets.columns:
        raise KeyError(f"market column {market!r} missing from prices")
    idx = rets.index
    mkt_arr = rets[market].to_numpy(dtype=float)
    cols = {c: rets[c].to_numpy(dtype=float) for c in rets.columns}

    need_lo = min(lo for lo, _ in win.values())
    need_hi = max(hi for _, hi in win.values())
    rows = []
    for tk, ann, prec, band, kind in events:
        if tk not in cols:
            continue
        pos = _anchor_pos(idx, pd.Timestamp(ann))
        if pos is None:
            continue
        if pos + min(est_lo, need_lo) < 0 or pos + need_hi >= len(idx):
            continue
        r_i = cols[tk]
        alpha, beta, n_est = market_model(r_i[pos + est_lo: pos + est_hi + 1],
                                          mkt_arr[pos + est_lo: pos + est_hi + 1])
        if not np.isfinite(beta):
            continue
        row = {
            "ticker": tk, "anchor": idx[pos], "precision": prec,
            "discount_band": band, "kind": kind,
            "alpha_est": alpha, "beta_est": beta, "n_est": n_est,
        }
        ok = True
        for name, (lo, hi) in win.items():
            seg_i = r_i[pos + lo: pos + hi + 1]
            seg_m = mkt_arr[pos + lo: pos + hi + 1]
            if seg_i.size != (hi - lo + 1) or not np.isfinite(seg_i).all() \
                    or not np.isfinite(seg_m).all():
                ok = False
                break
            row[f"car_{name}"] = float((seg_i - (alpha + beta * seg_m)).sum())
            row[f"raw_{name}"] = float(seg_i.sum())
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def coverage(prices: pd.DataFrame, events, market: str = "SPY") -> dict:
    """How many listed deals actually make it onto the tape (the honest denominator)."""
    panel = event_panel(prices, events, market=market)
    return {
        "n_listed": len(events),
        "n_usable": int(len(panel)),
        "n_tickers": int(panel["ticker"].nunique()) if len(panel) else 0,
        "dropped": int(len(events) - len(panel)),
    }


# --------------------------------------------------------------------------- #
# The headline event-study table
# --------------------------------------------------------------------------- #
def event_summary(panel: pd.DataFrame, windows=None) -> dict:
    """Mean CAR, cross-sectional t, hit rate and Wilson CI for each event window."""
    win = dict(WINDOWS if windows is None else windows)
    out = {}
    for name in win:
        col = f"car_{name}"
        if col not in panel.columns or len(panel) == 0:
            out[name] = None
            continue
        x = panel[col].to_numpy(dtype=float)
        k = int((x > 0).sum())
        lo, hi = wilson_interval(k, len(x))
        out[name] = {
            "n": int(len(x)),
            "mean_pct": float(np.mean(x) * 100.0),
            "median_pct": float(np.median(x) * 100.0),
            "t": one_sample_t(x),
            "hit_rate": float(k / len(x)),
            "hit_lo": float(lo), "hit_hi": float(hi),
            "sd_pct": float(np.std(x, ddof=1) * 100.0),
        }
    return out


def bootstrap_mean_ci(x, n_boot: int = 4000, seed: int = 929, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for a mean CAR, resampling **events** with replacement.

    Events are the sampling unit (not days): the cross-section is what is uncertain here.
    Because the list is issuer-clustered, this CI is still optimistic — the
    leave-one-issuer-out jackknife is the sterner check.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return {"mean_pct": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": int(len(x))}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[draws].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_pct": float(x.mean() * 100), "ci_low": float(lo * 100),
            "ci_high": float(hi * 100), "n": int(len(x)), "n_boot": int(n_boot)}


def block_bootstrap_mean_ci(x, n_boot: int = 4000, block: int = 21,
                            seed: int = 929, alpha: float = 0.05) -> dict:
    """Circular block bootstrap CI for the mean of a *daily* series (calendar time)."""
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


# --------------------------------------------------------------------------- #
# Jackknives — the answer to a list dominated by serial issuers
# --------------------------------------------------------------------------- #
def jackknife(panel: pd.DataFrame, window: str = "post_expiry") -> dict:
    """Leave-one-**event**-out: the range of the mean CAR and its t across deletions."""
    col = f"car_{window}"
    x = panel[col].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return {"n": n}
    means, ts = [], []
    for i in range(n):
        y = np.delete(x, i)
        means.append(y.mean() * 100)
        ts.append(one_sample_t(y))
    return {
        "n": n,
        "mean_pct": float(x.mean() * 100),
        "min_mean_pct": float(np.min(means)), "max_mean_pct": float(np.max(means)),
        "min_t": float(np.min(ts)), "max_t": float(np.max(ts)),
        "worst_event": panel["ticker"].iloc[int(np.argmin(ts))],
    }


def jackknife_issuer(panel: pd.DataFrame, window: str = "post_expiry") -> pd.DataFrame:
    """Leave-one-**issuer**-out: drop every deal by one ticker and re-run the mean/t.

    With a list this clustered (Cornerstone and Gabelli between them supply most of the
    deals), an effect that dies when one issuer leaves is an issuer story, not a
    rights-offering story.
    """
    col = f"car_{window}"
    rows = []
    for tk in sorted(panel["ticker"].unique()):
        keep = panel[panel["ticker"] != tk]
        if len(keep) < 5:
            continue
        x = keep[col].to_numpy(dtype=float)
        rows.append({"dropped": tk, "n_left": int(len(x)),
                     "mean_pct": float(x.mean() * 100), "t": one_sample_t(x)})
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Placebo — random pseudo-anchors on the same names
# --------------------------------------------------------------------------- #
def placebo(
    prices: pd.DataFrame,
    events,
    window: str = "post_expiry",
    n_draws: int = 300,
    market: str = "SPY",
    seed: int = 929,
    lo_date=None,
    hi_date=None,
) -> dict:
    """Re-run the study on random pseudo-anchor dates for the same tickers.

    Each draw keeps the ticker mix and the sample size but moves every anchor to a
    uniformly random session on that ticker's tape. The observed mean CAR is compared
    against the placebo distribution — an empirical p-value that is immune to the
    market-model and window choices.

    **Draw range matters, and is a choice.** With ``lo_date``/``hi_date`` unset the
    pseudo-anchors are drawn from the whole loaded tape, which for this study reaches
    back to 2005 and therefore includes the 2008 and 2020 crashes — regimes in which no
    deal on the list was actually announced. That inflates the placebo dispersion and
    makes the observed CAR look tamer than it is. Pass the deal era (see
    ``verify.py``) for the **era-matched** version, which is the fairer yardstick and
    the one to quote against; both are reported.

    Note this draw is *independent per event*, so it does not reproduce the calendar
    clustering of the real list — :func:`clustered_placebo` is the test that does.
    """
    rng = np.random.default_rng(seed)
    rets = daily_returns(prices)
    real = event_panel(prices, events, market=market, rets=rets)
    if len(real) < 5:
        return {"n_draws": 0}
    obs = float(real[f"car_{window}"].mean())
    tickers = list(real["ticker"])
    idx = prices.index
    lo_pos = 300 if lo_date is None else max(300, int(idx.searchsorted(pd.Timestamp(lo_date))))
    hi_pos = (len(idx) - 90 if hi_date is None
              else min(len(idx) - 90, int(idx.searchsorted(pd.Timestamp(hi_date)))))
    if hi_pos - lo_pos < 200:
        return {"n_draws": 0}
    means = np.full(n_draws, np.nan)
    for d in range(n_draws):
        fake = tuple(
            (tk, str(idx[int(rng.integers(lo_pos, hi_pos))].date()), "placebo", "moderate", "cef")
            for tk in tickers
        )
        p = event_panel(prices, fake, market=market, rets=rets)
        if len(p) >= 5:
            means[d] = float(p[f"car_{window}"].mean())
    m = means[np.isfinite(means)]
    if m.size == 0:
        return {"n_draws": 0}
    p_two = float((np.abs(m - m.mean()) >= abs(obs - m.mean())).mean())
    return {
        "observed_pct": obs * 100, "placebo_mean_pct": float(m.mean() * 100),
        "placebo_sd_pct": float(m.std(ddof=1) * 100), "n_draws": int(m.size),
        "z": float((obs - m.mean()) / m.std(ddof=1)) if m.std(ddof=1) > 0 else float("nan"),
        "p_two_sided": p_two,
        "lo_date": None if lo_date is None else str(lo_date),
        "hi_date": None if hi_date is None else str(hi_date),
    }


def clustered_placebo(
    prices: pd.DataFrame,
    events,
    window: str = "post_expiry",
    n_draws: int = 300,
    market: str = "SPY",
    seed: int = 929,
    max_shift_years: int = 8,
) -> dict:
    """Placebo that keeps the list's **calendar clustering** intact.

    :func:`placebo` moves every anchor independently, which destroys exactly the thing
    that makes a cross-sectional *t* untrustworthy here: the deals land on top of each
    other (both Cornerstone funds every September, the Gabelli funds every April), so
    their abnormal returns share a common shock. This variant instead slides the **whole
    list** by one random number of sessions, preserving every gap between deals, and
    compares the observed mean CAR against that distribution.
    """
    rng = np.random.default_rng(seed)
    rets = daily_returns(prices)
    real = event_panel(prices, events, market=market, rets=rets)
    if len(real) < 5:
        return {"n_draws": 0}
    obs = float(real[f"car_{window}"].mean())
    idx = prices.index
    base = [(str(r.ticker), int(idx.searchsorted(r.anchor))) for r in real.itertuples()]
    span = int(252 * max_shift_years)
    means = []
    for _ in range(n_draws):
        sh = int(rng.integers(-span, span))
        if abs(sh) < 60:          # a shift shorter than a window is not a placebo
            continue
        fake = tuple(
            (tk, str(idx[pos + sh].date()), "placebo", "moderate", "cef")
            for tk, pos in base if 300 <= pos + sh < len(idx) - 90
        )
        if len(fake) < max(20, len(base) // 2):
            continue
        p = event_panel(prices, fake, market=market, rets=rets)
        if len(p) >= 20:
            means.append(float(p[f"car_{window}"].mean()))
    m = np.asarray(means, dtype=float)
    if m.size < 20:
        return {"n_draws": int(m.size)}
    sd = float(m.std(ddof=1))
    return {
        "observed_pct": obs * 100, "placebo_mean_pct": float(m.mean() * 100),
        "placebo_sd_pct": sd * 100, "n_draws": int(m.size),
        "z": float((obs - m.mean()) / sd) if sd > 0 else float("nan"),
        "p_two_sided": float((np.abs(m - m.mean()) >= abs(obs - m.mean())).mean()),
    }


# --------------------------------------------------------------------------- #
# Is the discount compensated? band split + label permutation
# --------------------------------------------------------------------------- #
def discount_split(panel: pd.DataFrame, window: str = "post_expiry") -> dict:
    """Deep-band vs shallow/moderate-band mean CAR, with a Welch t on the difference.

    The band is an ASSUMPTION (see ``data.RIGHTS_EVENTS``), so this test answers a
    coarse question: do the deals we labelled *deeply* discounted behave differently
    from the rest? :func:`permutation_discount_test` says whether the answer is more
    than a coin toss.
    """
    col = f"car_{window}"
    deep = panel.loc[panel["discount_band"] == "deep", col].to_numpy(dtype=float)
    rest = panel.loc[panel["discount_band"] != "deep", col].to_numpy(dtype=float)
    return {
        "n_deep": int(len(deep)), "n_rest": int(len(rest)),
        "mean_deep_pct": float(deep.mean() * 100) if len(deep) else float("nan"),
        "mean_rest_pct": float(rest.mean() * 100) if len(rest) else float("nan"),
        "diff_pct": float((deep.mean() - rest.mean()) * 100) if len(deep) and len(rest) else float("nan"),
        "welch_t": welch_t(deep, rest),
        "t_deep": one_sample_t(deep), "t_rest": one_sample_t(rest),
    }


def permutation_discount_test(panel: pd.DataFrame, window: str = "post_expiry",
                              n_perm: int = 5000, seed: int = 929) -> dict:
    """Shuffle the discount labels: does the real deep-vs-rest gap beat a random split?"""
    col = f"car_{window}"
    x = panel[col].to_numpy(dtype=float)
    labs = (panel["discount_band"] == "deep").to_numpy()
    if labs.sum() < 3 or (~labs).sum() < 3:
        return {"n_perm": 0}
    obs = float(x[labs].mean() - x[~labs].mean())
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(labs)
        gaps[i] = x[p].mean() - x[~p].mean()
    return {
        "observed_gap_pct": obs * 100,
        "perm_sd_pct": float(gaps.std(ddof=1) * 100),
        "p_two_sided": float((np.abs(gaps) >= abs(obs)).mean()),
        "n_perm": int(n_perm),
    }


# --------------------------------------------------------------------------- #
# Tradability — a calendar-time portfolio, excess-of-cash, costed
# --------------------------------------------------------------------------- #
def calendar_time_portfolio(
    prices: pd.DataFrame,
    events,
    hold: tuple = (1, 49),
    market: str = "SPY",
    cash: str = "BIL",
    cost_bps: float = 25.0,
    borrow_bps_ann: float = 0.0,
    side: int = 1,
) -> pd.DataFrame:
    """Equal-weight daily portfolio of every name currently inside its event window.

    ``hold`` is in trading days from the anchor; ``hold=(1, 49)`` means "enter at the
    close of the session **after** the announcement, exit 49 sessions later" — the
    single execution lag, applied once. On days with no live event the book is flat and
    earns the cash leg.

    Costs: ``cost_bps`` one-way on NAV is charged on the entry and the exit day of each
    position, pro-rated by that name's weight in the book. ``side=-1`` runs the short
    ("dilution warning") version, which additionally pays ``borrow_bps_ann`` annualised
    borrow on the invested fraction — these are small, hard-to-borrow names, so the
    borrow rate is swept rather than assumed away.

    **Cash is treated symmetrically.** The long book earns the cash leg only on the days
    it is flat (its capital is in the stock). The short book is cash-collateralised, so
    it earns the cash leg on *every* day and pays borrow on top: charging it borrow while
    withholding the rebate would be a one-sided cost, and would flatter exactly the
    conclusion this study is trying to falsify.

    Returns a daily frame: ``r_port`` (net), ``r_mkt``, ``r_cash``, ``n_active``,
    ``invested`` (fraction of the book at risk).
    """
    rets = daily_returns(prices)
    idx = rets.index
    lo, hi = int(hold[0]), int(hold[1])
    names = [c for c in rets.columns if c not in (market, cash)]

    active = pd.DataFrame(0.0, index=idx, columns=names)
    for tk, ann, _prec, _band, _kind in events:
        if tk not in active.columns:
            continue
        pos = _anchor_pos(idx, pd.Timestamp(ann))
        if pos is None:
            continue
        a, b = pos + lo, min(pos + hi, len(idx) - 1)
        if a >= len(idx) or a > b:
            continue
        active.iloc[a:b + 1, active.columns.get_loc(tk)] = 1.0

    n_active = active.sum(axis=1)
    w = active.div(n_active.where(n_active > 0), axis=0).fillna(0.0)
    r_names = rets[names].fillna(0.0)
    gross = (w * r_names).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum())
    cost = turnover * (cost_bps * 1e-4)
    invested = (n_active > 0).astype(float)
    borrow = invested * (borrow_bps_ann * 1e-4) / TRADING_DAYS if side < 0 else 0.0

    r_cash = rets[cash].fillna(0.0)
    # Long: cash only on flat days. Short: cash-collateralised, so cash every day, minus
    # borrow. Both reduce to "flat day = cash day".
    cash_leg = r_cash if side < 0 else (1.0 - invested) * r_cash
    r_port = side * gross - cost - borrow + cash_leg
    out = pd.DataFrame({
        "r_port": r_port, "r_mkt": rets[market], "r_cash": r_cash,
        "n_active": n_active, "invested": invested,
    }).dropna()
    return out


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an *excess* series)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": n}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": auto_nw_t(r.to_numpy()),
    }


def alpha_vs_market(e_port, e_mkt, periods_per_year: int = TRADING_DAYS) -> dict:
    """Beta-adjusted alpha of an excess-return series against the excess market.

    Why this and not the leg's own *t*: the event book is **long-only equity** a quarter
    of the time, so its own mean excess return is mostly the equity risk premium it rents
    while it is invested. A positive own *t* is therefore not evidence of an edge. The
    honest statistic is the intercept of ``e_port = a + b e_mkt + u`` with a HAC *t*,
    which prices out the borrowed beta; it is reported alongside (never instead of) the
    Sharpe race, because a mismatched-exposure race is not a fair fight either.
    """
    y = np.asarray(e_port, dtype=float)
    x = np.asarray(e_mkt, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 60 or x.var(ddof=1) <= 0:
        return {"alpha_ann_pct": float("nan"), "beta": float("nan"), "t_alpha": float("nan")}
    beta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    a = float(y.mean() - beta * x.mean())
    return {
        "alpha_ann_pct": a * periods_per_year * 100.0,
        "beta": beta,
        # HAC t on the beta-hedged series (alpha + residual), i.e. y - beta x
        "t_alpha": auto_nw_t(y - beta * x),
        "n": int(x.size),
    }


def tradability(
    prices: pd.DataFrame,
    events,
    hold: tuple = (1, 49),
    cost_bps: float = 25.0,
    borrow_bps_ann: float = 0.0,
    side: int = 1,
    market: str = "SPY",
    cash: str = "BIL",
) -> dict:
    """The excess-of-cash race: event portfolio vs SPY, both minus the cash leg.

    Reports each arm's excess Sharpe, the Sharpe advantage, and the HAC t on the daily
    return **difference** (the Newey-West form of the Jobson-Korkie test). Because both
    arms are excess-of-cash, the cash leg cancels in the difference.
    """
    bt = calendar_time_portfolio(prices, events, hold=hold, market=market, cash=cash,
                                 cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann, side=side)
    e_port = (bt["r_port"] - bt["r_cash"]).rename("e_port")
    e_mkt = (bt["r_mkt"] - bt["r_cash"]).rename("e_mkt")
    diff = (e_port - e_mkt).dropna()
    s_p, s_m = summary(e_port), summary(e_mkt)
    al = alpha_vs_market(e_port.to_numpy(), e_mkt.to_numpy())
    return {
        "port": s_p, "mkt": s_m, "alpha": al,
        "sharpe_adv": s_p["sharpe"] - s_m["sharpe"],
        "t_diff": auto_nw_t(diff.to_numpy()),
        "t_port": auto_nw_t(e_port.to_numpy()),
        "mean_diff_bps": float(diff.mean() * 1e4),
        "invested_frac": float(bt["invested"].mean()),
        "avg_n_active": float(bt.loc[bt["n_active"] > 0, "n_active"].mean()) if (bt["n_active"] > 0).any() else 0.0,
        "e_port": e_port, "e_mkt": e_mkt, "bt": bt,
    }


def cost_sweep(prices: pd.DataFrame, events, hold: tuple = (1, 49),
               grid=(0.0, 10.0, 25.0, 50.0, 100.0), side: int = 1,
               borrow_bps_ann: float = 0.0) -> list:
    """Gross-to-net: the Sharpe advantage and HAC t at several one-way costs."""
    out = []
    for c in grid:
        t = tradability(prices, events, hold=hold, cost_bps=c, side=side,
                        borrow_bps_ann=borrow_bps_ann)
        out.append({"cost_bps": c, "sharpe_port": t["port"]["sharpe"],
                    "sharpe_adv": t["sharpe_adv"], "t_diff": t["t_diff"],
                    "mean_diff_bps": t["mean_diff_bps"]})
    return out


def borrow_sweep(prices: pd.DataFrame, events, hold: tuple = (1, 49),
                 grid=(0.0, 100.0, 300.0, 800.0), cost_bps: float = 25.0) -> list:
    """The short ("dilution warning") leg at several annualised borrow rates, in bps."""
    out = []
    for b in grid:
        t = tradability(prices, events, hold=hold, cost_bps=cost_bps, side=-1,
                        borrow_bps_ann=b)
        out.append({"borrow_bps_ann": b, "sharpe_port": t["port"]["sharpe"],
                    "sharpe_adv": t["sharpe_adv"], "t_diff": t["t_diff"],
                    "t_port": t["t_port"]})
    return out


# --------------------------------------------------------------------------- #
# Robustness to the PROXY inputs: anchor jitter, timetable, eras
# --------------------------------------------------------------------------- #
def anchor_jitter(prices: pd.DataFrame, events, window: str = "post_expiry",
                  shifts=(-10, -5, 0, 5, 10), market: str = "SPY") -> list:
    """Shift every anchor by N **calendar** days and re-run the window.

    The event list is month-precision, so a conclusion that survives only at shift 0 is
    an artefact of a date we do not actually know.
    """
    out = []
    for s in shifts:
        shifted = tuple(
            (tk, str((pd.Timestamp(ann) + pd.Timedelta(days=int(s))).date()), prec, band, kind)
            for tk, ann, prec, band, kind in events
        )
        p = event_panel(prices, shifted, market=market)
        col = f"car_{window}"
        out.append({"shift_days": int(s), "n": int(len(p)),
                    "mean_pct": float(p[col].mean() * 100) if len(p) else float("nan"),
                    "t": one_sample_t(p[col].to_numpy()) if len(p) else float("nan")})
    return out


def timetable_sweep(prices: pd.DataFrame, events, market: str = "SPY",
                    expiry_days_grid=(28, 40, 55)) -> list:
    """Re-cut the subscription / post-expiry windows under different assumed timetables.

    The modelled expiry (``data.TIMETABLE['expiry_days']``) is an ASSUMPTION; a longer
    or shorter subscription period moves the boundary between the two windows.
    """
    out = []
    for days in expiry_days_grid:
        exp_td = max(int(round(days * 252 / 365)), 5)
        win = {
            "subscription": (1, exp_td),
            "post_expiry": (exp_td + 1, exp_td + 21),
        }
        p = event_panel(prices, events, market=market, windows=win)
        row = {"expiry_days": int(days), "n": int(len(p))}
        for w in win:
            col = f"car_{w}"
            row[f"{w}_mean_pct"] = float(p[col].mean() * 100) if len(p) else float("nan")
            row[f"{w}_t"] = one_sample_t(p[col].to_numpy()) if len(p) else float("nan")
        out.append(row)
    return out


def era_cut(prices: pd.DataFrame, events, split: str = "2018-01-01",
            window: str = "post_expiry", market: str = "SPY") -> dict:
    """Split the deals by anchor date and re-run the window on each half."""
    panel = event_panel(prices, events, market=market)
    out = {}
    for tag, mask in [("early", panel["anchor"] < pd.Timestamp(split)),
                      ("late", panel["anchor"] >= pd.Timestamp(split))]:
        sub = panel[mask]
        col = f"car_{window}"
        if len(sub) < 4:
            out[tag] = None
            continue
        x = sub[col].to_numpy(dtype=float)
        out[tag] = {"n": int(len(x)), "mean_pct": float(x.mean() * 100),
                    "t": one_sample_t(x), "hit_rate": float((x > 0).mean())}
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, events, market: str = "SPY") -> dict:
    """Run the event study on a synthetic panel and report the three window means.

    On the planted world the announcement and subscription windows must come out
    clearly negative and the post-expiry window positive; on the null all three must sit
    near zero. Proves the harness is unbiased — it never supports a real-tape stamp.
    """
    panel = event_panel(prices, events, market=market)
    summ = event_summary(panel)
    out = {"n": int(len(panel))}
    for name in ("announce", "subscription", "post_expiry"):
        s = summ.get(name)
        out[f"{name}_mean_pct"] = s["mean_pct"] if s else float("nan")
        out[f"{name}_t"] = s["t"] if s else float("nan")
    if len(panel) >= 8:
        out["deep_minus_rest_pct"] = discount_split(panel)["diff_pct"]
    else:
        out["deep_minus_rest_pct"] = float("nan")
    return out
