"""Tested rules + inference for Study 603 — Treasury Auction Concession.

The claim (dealer folklore, formalised by Lou-Yan-Zhang 2013 and Fleming-Rosenberg 2008):
yields **cheapen into** big 10Y/30Y Treasury auctions and **richen after** — primary dealers
demand a price concession to warehouse the new supply. We measure it two complementary ways:

    * **Event windows.** For each auction ``A``, the pre-window yield change is
      ``close(A-5) -> close(A)`` in bps (auction results hit at 1pm ET, so the auction-day
      close already includes the stop-out) and the post-window change is
      ``close(A) -> close(A+5)``. Concession = pre > 0; richening = post < 0.

    * **Dummy regression (primary).** Daily yield changes (bps) regressed on an intercept
      plus ``D_pre`` (1 on the 5 trading days ending on an auction day) and ``D_post`` (1 on
      the 5 days after), with **Newey-West (HAC)** standard errors — the honest test when
      10Y and 30Y windows overlap in refunding weeks and daily changes autocorrelate.

Inference, the desk's shared spirit:

  * **HAC/Newey-West t** on the dummy coefficients (lag 8 ~ one window width);
  * **Welch t** for the size split (big vs small auctions, size detrended by the trailing
    median of the prior 8 same-tenor auctions — sizes grow secularly, so raw dollars would
    just split old vs new) and per-era regressions (pre-2008 / 2008-2019 / 2020+);
  * the tradability leg holds **TLT** (total-return, auto-adjusted) over the post-auction
    window: signal = the public auction calendar, entry at the **auction-day close**, so the
    first return earned is ``A -> A+1`` — exactly **one execution lag**, documented here and
    nowhere doubled. Costs are **one-way x NAV per leg** (2 legs per round trip); cash days
    earn the T-bill rate, and the race is **excess-vs-excess** (TLT minus bills vs 0).

The decisive numbers are the HAC t on ``D_pre`` / ``D_post`` on the REAL tape, and whether
the post-auction TLT round trip survives costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRE = 5      # trading days of concession build, ending ON the auction day
POST = 5     # trading days of post-auction richening
NW_LAGS = 8  # ~ one window width
TDAYS = 252.0


# --------------------------------------------------------------------------- #
# Small inference helpers
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variance). NaN if either side < 2."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_mean_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """Newey-West (HAC) t of mean(x) vs 0 — robust to serial correlation."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS betas + Newey-West HAC t-stats. X includes the intercept column."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Xe = X * e[:, None]
    S = (Xe.T @ Xe) / n
    for j in range(1, min(lags, n - 1) + 1):
        w = 1.0 - j / (lags + 1.0)
        G = (Xe[j:].T @ Xe[:-j]) / n
        S += w * (G + G.T)
    cov = n * (XtX_inv @ S @ XtX_inv)
    se = np.sqrt(np.diag(cov))
    t = np.where(se > 0, beta / se, np.nan)
    return beta, t


# --------------------------------------------------------------------------- #
# Event construction
# --------------------------------------------------------------------------- #
def snap_positions(index: pd.DatetimeIndex, dates: pd.Series,
                   pre: int = PRE, post: int = POST) -> np.ndarray:
    """Map auction dates to positions on the trading-day index (last day <= date).

    Drops auctions off the tape, too close to either edge for a full [-pre, +post] window,
    or whose snap lands more than 5 calendar days before the auction (a tape gap).
    """
    dates = pd.to_datetime(pd.Series(dates)).sort_values().to_numpy()
    pos = np.searchsorted(index.values, dates, side="right") - 1
    keep = []
    for p, d in zip(pos, dates):
        if p < pre or p >= len(index) - post:
            continue
        if (pd.Timestamp(d) - index[p]).days > 5:
            continue
        keep.append(p)
    return np.unique(np.asarray(keep, dtype=int))


def auction_events(yld: pd.Series, auction_dates: pd.Series,
                   pre: int = PRE, post: int = POST) -> pd.DataFrame:
    """Per-auction pre/post cumulative yield changes in bps.

    ``pre_bps  = (y[A] - y[A-pre]) * 100``  (concession build, ends at the auction close)
    ``post_bps = (y[A+post] - y[A]) * 100`` (post-auction move; richening if negative)
    """
    yld = yld.dropna()
    pos = snap_positions(yld.index, auction_dates, pre=pre, post=post)
    v = yld.to_numpy()
    rows = [{"date": yld.index[p],
             "pre_bps": (v[p] - v[p - pre]) * 100.0,
             "post_bps": (v[p + post] - v[p]) * 100.0} for p in pos]
    return pd.DataFrame(rows)


def baseline_windows(yld: pd.Series, auction_dates: pd.Series,
                     pre: int = PRE, post: int = POST) -> np.ndarray:
    """Unconditional ``pre``-day yield changes (bps) on days NOT inside any auction window —
    the honest 'ordinary week' baseline the event windows are compared against."""
    yld = yld.dropna()
    pos = snap_positions(yld.index, auction_dates, pre=pre, post=post)
    n = len(yld)
    near = np.zeros(n, dtype=bool)
    for p in pos:
        near[max(0, p - pre - pre):min(n, p + post + 1)] = True
    ch = (yld.to_numpy()[pre:] - yld.to_numpy()[:-pre]) * 100.0   # change ending at t
    ok = ~near[pre:]
    return ch[ok]


def dummy_regression(yld: pd.Series, auction_dates: pd.Series,
                     pre: int = PRE, post: int = POST, nw_lags: int = NW_LAGS) -> dict:
    """PRIMARY test — daily Dy (bps) on [1, D_pre, D_post] with Newey-West t.

    ``D_pre[t] = 1`` if t is one of the ``pre`` trading days ending on an auction day;
    ``D_post[t] = 1`` if t is one of the ``post`` days after one. Overlapping 10Y/30Y
    refunding-week windows are handled naturally (a day is either flagged or not), and the
    HAC kernel absorbs the serial correlation the windows induce.
    """
    yld = yld.dropna()
    dy = yld.diff().dropna() * 100.0                     # bps/day
    pos = snap_positions(yld.index, auction_dates, pre=pre, post=post)
    # dy index t corresponds to change (t-1 -> t); dy position i = yld position i+1
    n = len(dy)
    d_pre = np.zeros(n)
    d_post = np.zeros(n)
    for p in pos:
        # changes ending at yld positions p-pre+1 .. p  -> dy positions p-pre .. p-1
        d_pre[max(0, p - pre):p] = 1.0
        d_post[p:min(n, p + post)] = 1.0
    X = np.column_stack([np.ones(n), d_pre, d_post])
    beta, t = ols_hac(dy.to_numpy(), X, lags=nw_lags)
    return {"n_days": n, "n_auctions": int(len(pos)),
            "alpha_bps": float(beta[0]),
            "pre_bps_day": float(beta[1]), "t_pre": float(t[1]),
            "post_bps_day": float(beta[2]), "t_post": float(t[2]),
            "pre_days": int(d_pre.sum()), "post_days": int(d_post.sum())}


def car_profile(yld: pd.Series, auction_dates: pd.Series, k: int = 7) -> pd.Series:
    """Average cumulative yield change (bps) from day -k to +k around auctions (0 = auction
    day close), anchored at zero on day -k. The chart behind the V-shape claim."""
    yld = yld.dropna()
    pos = snap_positions(yld.index, auction_dates, pre=k, post=k)
    v = yld.to_numpy()
    paths = np.stack([(v[p - k:p + k + 1] - v[p - k]) * 100.0 for p in pos])
    return pd.Series(paths.mean(axis=0), index=range(-k, k + 1))


# --------------------------------------------------------------------------- #
# Splits — size and era
# --------------------------------------------------------------------------- #
def size_split(auctions: pd.DataFrame, yld: pd.Series, tenor: str,
               window: int = 8, pre: int = PRE, post: int = POST) -> dict:
    """Big vs small auctions — does the concession scale with size?

    Size is detrended: ``ratio = offering_amt / median(prior `window` same-tenor auctions)``
    (raw dollars grew ~20x since 1979, so an undetrended split would just compare eras).
    Big = ratio above the median ratio. Welch t on the per-auction pre-window changes.
    """
    au = auctions[auctions["tenor"] == tenor].sort_values("auction_date").copy()
    med = au["offering_amt"].rolling(window).median().shift(1)
    au["ratio"] = au["offering_amt"] / med
    au = au.dropna(subset=["ratio"])
    cut = au["ratio"].median()
    ev = auction_events(yld, au["auction_date"], pre=pre, post=post)
    au = au.merge(ev, left_on=au["auction_date"].dt.normalize().rename("d"),
                  right_on=ev["date"].dt.normalize().rename("d"), how="inner")
    big = au[au["ratio"] >= cut]
    small = au[au["ratio"] < cut]
    return {"n_big": len(big), "n_small": len(small), "cut_ratio": float(cut),
            "pre_big_bps": float(big["pre_bps"].mean()),
            "pre_small_bps": float(small["pre_bps"].mean()),
            "welch_t_pre": welch_t(big["pre_bps"].to_numpy(), small["pre_bps"].to_numpy()),
            "post_big_bps": float(big["post_bps"].mean()),
            "post_small_bps": float(small["post_bps"].mean()),
            "welch_t_post": welch_t(big["post_bps"].to_numpy(), small["post_bps"].to_numpy())}


def era_split(yld: pd.Series, auction_dates: pd.Series,
              eras: tuple = (("1979-2007", None, "2007-12-31"),
                             ("2008-2019", "2008-01-01", "2019-12-31"),
                             ("2020-2026", "2020-01-01", None))) -> list[dict]:
    """Per-era dummy regression — did the concession fade, or grow with the supply flood?"""
    out = []
    dates = pd.to_datetime(pd.Series(auction_dates))
    for name, lo, hi in eras:
        y = yld
        d = dates
        if lo is not None:
            y = y[y.index >= lo]
            d = d[d >= lo]
        if hi is not None:
            y = y[y.index <= hi]
            d = d[d <= hi]
        if len(d) < 10 or len(y) < 300:
            continue
        r = dummy_regression(y, d)
        r["era"] = name
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Tradability — the post-auction TLT round trip
# --------------------------------------------------------------------------- #
def tlt_post_auction(tlt: pd.DataFrame, auction_dates: pd.Series,
                     hold: int = POST, cost_bps: float = 2.0) -> dict:
    """Long TLT over the post-auction richening window, net of costs, excess of bills.

    Rule: the auction calendar is public weeks in advance; enter at the **auction-day
    close** (signal = the calendar date + the 1pm result, both known before the close), hold
    ``hold`` trading days, exit at that close. One execution lag exactly: position on day
    t is decided at close t-1, earning the close-to-close return of t. Overlapping windows
    (10Y Wed + 30Y Thu of a refunding week) merge into one longer holding; each entry+exit
    pays ``cost_bps`` one-way x NAV per leg. Cash days earn the 13-week bill rate; the
    reported stats are **excess of bills vs 0** (excess-vs-excess, no free ride on carry).
    """
    px = tlt["tlt"].dropna()
    ret = px.pct_change().fillna(0.0).to_numpy()
    rf = (tlt["irx"].reindex(px.index).ffill().fillna(0.0) / 100.0 / TDAYS).to_numpy()
    pos = snap_positions(px.index, auction_dates, pre=0, post=hold)
    n = len(px)
    held = np.zeros(n, dtype=bool)
    for p in pos:
        held[p + 1:min(n, p + hold + 1)] = True           # first return earned = A -> A+1
    entries = int(((~np.roll(held, 1)) & held).sum())
    exits = int((held & (~np.roll(held, -1))).sum())
    c = cost_bps / 1e4
    strat_x = np.where(held, ret - rf, 0.0)                # excess of bills
    gross_ann = float(strat_x.mean() * TDAYS)
    net_daily = strat_x.copy()
    # charge each leg once: entry cost on the first held day, exit cost on the last
    enter_days = np.where((~np.roll(held, 1)) & held)[0]
    exit_days = np.where(held & (~np.roll(held, -1)))[0]
    for i in enter_days:
        net_daily[i] -= c
    for i in exit_days:
        net_daily[i] -= c
    net_ann = float(net_daily.mean() * TDAYS)
    t_net = hac_mean_t(net_daily[held], lags=NW_LAGS)
    held_share = float(held.mean())
    per_trip_gross = float(strat_x[held].sum() / max(entries, 1) * 1e4)
    return {"n_days": n, "held_share": held_share, "n_round_trips": entries,
            "n_exits": exits, "cost_bps": cost_bps,
            "gross_ann_pct": gross_ann * 100, "net_ann_pct": net_ann * 100,
            "per_trip_gross_bps": per_trip_gross,
            "per_trip_cost_bps": 2.0 * cost_bps,
            "t_net_held": t_net,
            "start": str(px.index[0].date()), "end": str(px.index[-1].date())}


def tlt_post_alpha(tlt: pd.DataFrame, auction_dates: pd.Series,
                   hold: int = POST, nw_lags: int = NW_LAGS) -> dict:
    """Is a post-auction day an abnormally GOOD TLT day? (the carry-clean alpha test)

    TLT earns a positive unconditional term-premium carry, so a positive round-trip mean
    alone could be beta, not richening alpha. This regresses daily TLT **excess** returns
    (minus bills) on [1, D_post]: the ``D_post`` coefficient is the *extra* excess return of
    a post-auction day over an ordinary day, with a Newey-West t.
    """
    px = tlt["tlt"].dropna()
    ret = px.pct_change().fillna(0.0)
    rf = tlt["irx"].reindex(px.index).ffill().fillna(0.0) / 100.0 / TDAYS
    xs = (ret - rf).to_numpy()
    pos = snap_positions(px.index, auction_dates, pre=0, post=hold)
    n = len(px)
    d_post = np.zeros(n)
    for p in pos:
        d_post[p + 1:min(n, p + hold + 1)] = 1.0
    X = np.column_stack([np.ones(n), d_post])
    beta, t = ols_hac(xs, X, lags=nw_lags)
    return {"base_bps_day": float(beta[0] * 1e4),
            "post_extra_bps_day": float(beta[1] * 1e4), "t_post_extra": float(t[1]),
            "n_post_days": int(d_post.sum()), "n_days": n,
            "bh_excess_ann_pct": float(np.mean(xs) * TDAYS * 100)}


# --------------------------------------------------------------------------- #
# Headline summary
# --------------------------------------------------------------------------- #
def summarize_tenor(yld: pd.Series, auctions: pd.DataFrame, tenor: str) -> dict:
    """Everything the verify script prints for one tenor: events + regression + baseline."""
    dates = auctions.loc[auctions["tenor"] == tenor, "auction_date"]
    ev = auction_events(yld, dates)
    base = baseline_windows(yld, dates)
    reg = dummy_regression(yld, dates)
    return {
        "tenor": tenor, "n_auctions": len(ev),
        "pre_mean_bps": float(ev["pre_bps"].mean()),
        "post_mean_bps": float(ev["post_bps"].mean()),
        "base_mean_bps": float(base.mean()), "n_base": int(len(base)),
        "welch_pre_vs_base": welch_t(ev["pre_bps"].to_numpy(), base),
        "reg": reg,
    }
