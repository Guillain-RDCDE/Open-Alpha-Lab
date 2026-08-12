"""The event-study engine and its honest controls — Study 851 (Netflix Password
Crackdown).

The claim under test, steelmanned: Netflix's 2023 paid-sharing ("password crackdown"),
feared to spike churn, became an **upside surprise** — a "scary policy that worked".
We measure NFLX's **abnormal returns** (relative to a market benchmark) around the five
public dates of the story, and ask the only honest questions a **five-event** case
study can answer: did the average event session carry a positive abnormal return, is it
distinguishable from a random five-date calendar, and could any of it have been traded?

The machinery, one execution lag documented throughout (earnings print after the close,
so the reaction session is the next morning — encoded in ``data.EVENTS``):

* ``daily_returns`` — simple close-to-close returns.
* ``market_model_ar`` — the canonical event-study abnormal return. For each event the
  "normal" return is a **one-factor market model** ``r_asset ≈ alpha + beta·r_mkt``
  whose ``alpha``/``beta`` are OLS-estimated on a clean **estimation window** ending
  ``gap`` sessions *before* the event window (Brown & Warner 1985 / MacKinlay 1997), so
  the abnormal return over the window is genuinely out-of-sample. ``model="market"``
  (β·market), ``model="market_adjusted"`` (β≡1, a simple market-adjusted return) and
  ``model="mean"`` (constant-mean) are all available; the headline is the market model.
* ``event_car`` / ``stack_windows`` — the ``[-pre..+post]`` abnormal-return path around
  each event, stacked into an ``(n_events, window)`` matrix.
* ``car_path_stats`` — the mean cumulative abnormal-return path, anchored at 0 the
  session before the event.
* ``day0_stats`` — the event-session abnormal return itself: cross-event mean + a
  one-sample *t* (events are independent, non-overlapping calendar dates).
* ``window_car_stats`` — the cumulative abnormal return over the whole window, per event
  and cross-event (mean + one-sample *t*).
* ``placebo_distribution`` — the falsification test: the same statistic on thousands of
  random five-date pseudo-event calendars, so the observed number's percentile is the
  honest read on whether five real dates beat five random ones.
* ``buy_the_event`` — the tradable overlay: enter NFLX at the event-session close, hold
  ``hold`` sessions, one round trip of one-way costs charged twice; long-only.

Costs are one-way × NAV per leg; the overlay is long-only (no borrow, no shorting).
Everything is vectorised over the return arrays; the only Python loop is over the
handful of events (never a per-date panel loop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns."""
    return close.pct_change()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> tuple[float, float]:
    """Mean and one-sample t-stat of ``x`` (events treated as independent)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return float(np.nan if n == 0 else x.mean()), float("nan")
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(x.mean() / se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share (the event hit-rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# The market-model abnormal-return engine
# --------------------------------------------------------------------------- #
def _ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS intercept/slope of ``y`` on ``x`` (both 1-D, finite). Falls back to
    ``(mean, 0)`` if ``x`` has no variance."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if y.size < 3:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    vx = ((x - xm) ** 2).sum()
    if vx <= 0:
        return float(ym), 0.0
    beta = float(((x - xm) * (y - ym)).sum() / vx)
    alpha = float(ym - beta * xm)
    return alpha, beta


def event_car(
    r_asset: pd.Series,
    r_mkt: pd.Series,
    event_dates,
    pre: int = 1,
    post: int = 5,
    est_window: int = 120,
    gap: int = 10,
    model: str = "market",
) -> tuple[np.ndarray, list, np.ndarray]:
    """Stack the abnormal-return path ``[-pre..+post]`` around each event.

    For each event date the estimation window is the ``est_window`` sessions ending
    ``gap`` sessions before ``-pre`` (so the model never sees the event window). The
    "normal" return is:

    * ``model="market"``          → ``alpha + beta·r_mkt`` (OLS on the estimation window);
    * ``model="market_adjusted"`` → ``r_mkt`` (β ≡ 1, α ≡ 0, a market-adjusted return);
    * ``model="mean"``            → the estimation-window mean of ``r_asset`` (β ≡ 0).

    ``event_dates`` are snapped to the first session on/after each date via
    ``searchsorted``. Returns ``(matrix, kept_dates, betas)`` where ``matrix`` is
    ``(n_kept, pre+post+1)`` of abnormal returns, and ``betas`` the per-event estimated
    market beta (NaN for the non-market models).
    """
    ra = r_asset.reindex(r_asset.index.union(r_mkt.index))
    rm = r_mkt.reindex(ra.index)
    idx = ra.index
    a = ra.to_numpy(dtype=float)
    m = rm.to_numpy(dtype=float)
    win = pre + post + 1
    rows, kept, betas = [], [], []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            continue
        lo, hi = pos - pre, pos + post
        est_hi = lo - gap
        est_lo = est_hi - est_window
        if est_lo < 0 or hi >= len(idx):
            continue
        ya = a[est_lo:est_hi]
        xm = m[est_lo:est_hi]
        if model == "market":
            alpha, beta = _ols_alpha_beta(ya, xm)
            if not np.isfinite(beta):
                continue
            normal = alpha + beta * m[lo:hi + 1]
        elif model == "market_adjusted":
            beta = 1.0
            normal = m[lo:hi + 1]
        elif model == "mean":
            beta = float("nan")
            normal = np.full(win, np.nanmean(ya))
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown model {model!r}")
        ar = a[lo:hi + 1] - normal
        if not np.all(np.isfinite(ar)):
            continue
        rows.append(ar)
        kept.append(pd.Timestamp(d))
        betas.append(beta)
    if not rows:
        return np.empty((0, win)), kept, np.asarray(betas)
    return np.vstack(rows), kept, np.asarray(betas, dtype=float)


def stack_windows(r_asset, r_mkt, event_dates, pre=1, post=5, **kw) -> tuple[np.ndarray, list]:
    """Thin wrapper returning just ``(matrix, kept_dates)`` from :func:`event_car`."""
    mat, kept, _ = event_car(r_asset, r_mkt, event_dates, pre, post, **kw)
    return mat, kept


# --------------------------------------------------------------------------- #
# Headline splits
# --------------------------------------------------------------------------- #
def day0_stats(r_asset, r_mkt, event_dates, pre=1, post=5, **kw) -> dict:
    """Event-session (offset 0) abnormal return: cross-event mean + one-sample t."""
    w, kept, _ = event_car(r_asset, r_mkt, event_dates, pre, post, **kw)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan"), "kept_dates": []}
    day0 = w[:, pre]
    mean, t = one_sample_t(day0)
    hit = int((day0 > 0).sum())
    return {"n": w.shape[0], "mean": mean, "t": t, "hit": hit,
            "kept_dates": kept, "per_event": day0}


def window_car_stats(r_asset, r_mkt, event_dates, pre=1, post=5, **kw) -> dict:
    """Whole-window cumulative abnormal return [-pre..+post], per event + cross-event."""
    w, kept, _ = event_car(r_asset, r_mkt, event_dates, pre, post, **kw)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    per_event = w.sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": w.shape[0], "mean": mean, "t": t, "per_event": per_event, "kept_dates": kept}


def post_car_stats(r_asset, r_mkt, event_dates, pre=1, post=5, **kw) -> dict:
    """Post-event cumulative abnormal return [+1..+post], per event + cross-event."""
    w, kept, _ = event_car(r_asset, r_mkt, event_dates, pre, post, **kw)
    if w.shape[0] == 0:
        return {"n": 0, "mean": float("nan"), "t": float("nan")}
    per_event = w[:, pre + 1:].sum(axis=1)
    mean, t = one_sample_t(per_event)
    return {"n": w.shape[0], "mean": mean, "t": t, "per_event": per_event}


def car_path_stats(r_asset, r_mkt, event_dates, pre=1, post=5, **kw) -> pd.DataFrame:
    """Mean CAR by offset (-pre..+post), anchored so CAR(-pre)==0, own one-sample t."""
    w, kept, _ = event_car(r_asset, r_mkt, event_dates, pre, post, **kw)
    offsets = list(range(-pre, post + 1))
    if w.shape[0] == 0:
        return pd.DataFrame(columns=["offset", "mean_ar", "car", "t"]).set_index("offset")
    mean_ar = w.mean(axis=0)
    car_anchored = np.cumsum(mean_ar) - mean_ar[0]
    rows = []
    for i, k in enumerate(offsets):
        _, t = one_sample_t(w[:, i])
        rows.append({"offset": k, "mean_ar": float(mean_ar[i]),
                     "car": float(car_anchored[i]), "t": t})
    return pd.DataFrame(rows).set_index("offset")


# --------------------------------------------------------------------------- #
# Placebo — random pseudo-event calendars (the falsification test)
# --------------------------------------------------------------------------- #
def placebo_distribution(
    r_asset, r_mkt, n_events: int, pre: int = 1, post: int = 5,
    est_window: int = 120, gap: int = 10, model: str = "market",
    n_draws: int = 2000, seed: int = 851, stat: str = "day0",
) -> np.ndarray:
    """Mean statistic over ``n_draws`` sets of ``n_events`` random pseudo-event dates.

    ``stat`` is ``"day0"`` (event-session abnormal return) or ``"window_car"`` (summed
    [-pre..+post] CAR). Random dates are drawn from the interior of the tape (far enough
    from either edge to admit the estimation + event windows). A real signal must sit in
    the tail of this distribution; sitting in the bulk means five real dates do no better
    than five random ones.
    """
    ra = r_asset.reindex(r_asset.index.union(r_mkt.index))
    idx = ra.index
    n = len(idx)
    lo = est_window + gap + pre + 2
    hi = n - post - 2
    if hi <= lo or n_events <= 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.choice(np.arange(lo, hi), size=n_events, replace=False)
        dates = idx[locs]
        w, _, _ = event_car(ra, r_mkt, dates, pre, post,
                            est_window=est_window, gap=gap, model=model)
        if w.shape[0] == 0:
            out[d] = np.nan
            continue
        val = w[:, pre] if stat == "day0" else w.sum(axis=1)
        out[d] = float(np.nanmean(val))
    return out[np.isfinite(out)]


def placebo_pvalue(observed: float, placebo: np.ndarray, tail: str = "right") -> float:
    """Empirical one-sided p-value of ``observed`` within the placebo draws."""
    if placebo.size == 0 or not np.isfinite(observed):
        return float("nan")
    if tail == "left":
        return float((placebo <= observed).mean())
    return float((placebo >= observed).mean())


def block_bootstrap_ci(per_event: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                       seed: int = 851) -> tuple[float, float]:
    """Percentile CI on the cross-event mean (events resampled with replacement)."""
    x = np.asarray(per_event, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = x[rng.integers(0, n, size=n)].mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# --------------------------------------------------------------------------- #
# The tradable overlay — "buy the crackdown event" on NFLX
# --------------------------------------------------------------------------- #
def buy_the_event(close: pd.Series, event_dates, hold: int = 5,
                  cost_bps: float = 0.0) -> pd.DataFrame:
    """Long-only overlay: buy NFLX at the event-session close, hold ``hold`` sessions.

    The news is known at the close of the event session (see ``data.EVENTS`` — the
    reaction session already prices an after-close print), so the position earns
    sessions ``+1..+hold`` — one documented lag, applied once. One round trip per event:
    one-way cost charged twice (entry + exit) against NAV.
    """
    idx = close.index
    rows = []
    for d in pd.to_datetime(pd.Series(event_dates)):
        pos = idx.searchsorted(pd.Timestamp(d))
        if pos >= len(idx):
            continue
        entry, exit_ = pos, pos + hold
        if exit_ >= len(idx):
            continue
        gross = close.iat[exit_] / close.iat[entry] - 1.0
        net = gross - 2.0 * cost_bps * 1e-4
        rows.append({"entry_date": idx[entry], "hold": hold,
                     "ret_gross": float(gross), "ret_net": float(net)})
    return pd.DataFrame(rows)


def summarize_trade(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline stats for the overlay ledger: n, win-rate, mean bps, one-sample t."""
    if ledger.empty:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"), "t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mean, t = one_sample_t(r)
    return {"n": int(n), "win_rate": float((r > 0).mean()) if n else float("nan"),
            "mean_bps": mean * 1e4 if n else float("nan"), "t": t}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(asset: pd.Series, mkt: pd.Series, events,
                     pre: int = 1, post: int = 5) -> dict:
    """Run the headline day-0 market-model test on a synthetic world."""
    ra = daily_returns(asset)
    rm = daily_returns(mkt)
    return day0_stats(ra, rm, events, pre, post, model="market")
