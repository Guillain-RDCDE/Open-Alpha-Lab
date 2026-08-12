"""The single-name event-study engine and its honest controls — Study 850
(Airline Operational Meltdown).

The claim under test, steelmanned: **a very public operational meltdown — a multi-day
grounding, a mass-cancellation collapse, a viral PR disaster — is a reputational shock
that dents the *implicated* airline's own stock**, showing up as a negative abnormal
return around the event and, per the "reputation sticks" half of the folklore, a
continued negative drift over the following month (not a same-session shrug).

The machinery is a textbook market-model event study (MacKinlay 1997, *Journal of
Economic Literature*, "Event Studies in Economics and Finance"), one execution lag
documented throughout:

* ``daily_returns`` — simple close-to-close returns.
* ``market_model_params`` — for each event, OLS ``alpha, beta`` of the implicated stock
  on the market (SPY) over an **estimation window** ending ``gap`` sessions *before* the
  event (default ``[-gap-est_len .. -gap)``), so the parameters are not contaminated by
  the shock itself.
* ``event_ar_path`` — the abnormal return ``r_stock - (alpha + beta * r_mkt)`` across
  ``[-pre .. +post]`` sessions; the event date is snapped forward to the first NYSE
  session on/after the meltdown's public date (the single documented execution lag: the
  collapse is public before that session's close, so zero look-ahead).
* ``stack_event_cars`` / ``car_at`` — the cross-event CAR at several horizons: the
  event day itself (offset 0), the event week (``[0..+4]``), the event-plus-month
  (``[0..+21]``), and the pure post-event month drift (``[+1..+21]``). Each is summarised
  to one number per event, then a **one-sample t** across the (independent,
  non-overlapping) events — the planned primary.
* ``car_stats`` — mean CAR, one-sample *t*, and the down-hit rate with a Wilson interval.
* ``permutation_placebo`` — the falsification control: keep each event's **ticker** but
  read its CAR from a **random pseudo-event date** (same estimation-window machinery),
  thousands of times; a real reputational shock must sit in the left tail of that
  random-date distribution.
* ``short_the_meltdown`` — the tradable overlay: short the implicated stock at the
  event-day close, hold ``hold`` sessions, cover. Two one-way costs per round trip plus
  borrow on the short leg; the reputational-shock claim, if real and tradable, should
  make this pay.

Honest by construction: **N is tiny** (10 curated events, ~9 with price coverage) — low
power, so the default expectation is **None** unless the CAR is both large and robust.
The synthetic control only proves the estimator is unbiased; it never supports a
real-tape stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Horizon -> the AR offsets summed into that CAR.
HORIZONS: dict[str, list[int]] = {
    "day0": [0],
    "week": [0, 1, 2, 3, 4],
    "month": list(range(0, 22)),   # event day + ~one trading month
    "drift": list(range(1, 22)),   # pure post-event month drift (excludes day 0)
}


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    """Simple close-to-close daily returns."""
    return close.pct_change()


def _aligned_arrays(stock: pd.Series, mkt: pd.Series
                    ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Common-calendar aligned (index, stock_return, market_return) numpy arrays.

    Both series are inner-joined on their shared trading dates, then differenced to
    simple returns. Returned arrays share one index so positions line up.
    """
    df = pd.DataFrame({"s": stock, "m": mkt}).dropna().sort_index()
    s = df["s"].pct_change().to_numpy(dtype=float)
    m = df["m"].pct_change().to_numpy(dtype=float)
    return df.index, s, m


# --------------------------------------------------------------------------- #
# Market-model abnormal returns (vectorised over event positions)
# --------------------------------------------------------------------------- #
def _ols_alpha_beta(s_est: np.ndarray, m_est: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise OLS of ``s`` on ``m`` (each row is one estimation window).

    ``s_est``/``m_est`` are ``(K, L)`` (K windows, L obs each). Returns ``(alpha, beta)``
    each length K. Rows with zero market variance get ``beta = 0`` (fall back to a
    constant-mean model for that window).
    """
    s_est = np.atleast_2d(s_est).astype(float)
    m_est = np.atleast_2d(m_est).astype(float)
    ms = m_est.mean(axis=1, keepdims=True)
    ss = s_est.mean(axis=1, keepdims=True)
    mc = m_est - ms
    sc = s_est - ss
    var_m = np.einsum("ij,ij->i", mc, mc)
    cov = np.einsum("ij,ij->i", mc, sc)
    beta = np.where(var_m > 0, cov / np.where(var_m > 0, var_m, 1.0), 0.0)
    alpha = ss.ravel() - beta * ms.ravel()
    return alpha, beta


def car_vec(s: np.ndarray, m: np.ndarray, positions: np.ndarray, offsets: list[int],
            pre: int = 5, post: int = 21, est_len: int = 120, gap: int = 10
            ) -> np.ndarray:
    """Vectorised market-model CAR over ``offsets`` for many event ``positions``.

    For each position ``p`` (an integer row index into the aligned return arrays), the
    estimation window is ``[p-gap-est_len .. p-gap)`` (OLS ``alpha, beta``); the CAR is
    the sum over ``offsets`` of ``s[p+o] - (alpha + beta * m[p+o])``. Positions without a
    full estimation window or event window (running off either edge) return ``nan``.
    """
    positions = np.asarray(positions, dtype=int)
    n = len(positions)
    L = len(s)
    out = np.full(n, np.nan)
    lo_need = gap + est_len
    hi_need = max(offsets) if offsets else 0
    lo_pre = pre
    valid = (positions - lo_need >= 0) & (positions - lo_pre >= 0) & (positions + hi_need < L)
    if not valid.any():
        return out
    P = positions[valid]
    # estimation windows (K, est_len)
    est_cols = np.arange(-gap - est_len, -gap)
    est_idx = P[:, None] + est_cols[None, :]
    s_est = s[est_idx]
    m_est = m[est_idx]
    # drop windows carrying any nan (e.g. the very first return is nan)
    good = np.isfinite(s_est).all(axis=1) & np.isfinite(m_est).all(axis=1)
    alpha = np.full(len(P), np.nan)
    beta = np.full(len(P), np.nan)
    if good.any():
        a_g, b_g = _ols_alpha_beta(s_est[good], m_est[good])
        alpha[good] = a_g
        beta[good] = b_g
    # event-window abnormal returns at the requested offsets
    off = np.asarray(offsets, dtype=int)
    ev_idx = P[:, None] + off[None, :]
    s_ev = s[ev_idx]
    m_ev = m[ev_idx]
    ar = s_ev - (alpha[:, None] + beta[:, None] * m_ev)
    car = ar.sum(axis=1)
    car = np.where(good & np.isfinite(car), car, np.nan)
    res = np.full(n, np.nan)
    res[np.where(valid)[0]] = car
    return res


def event_ar_path(s: np.ndarray, m: np.ndarray, pos: int,
                  pre: int = 5, post: int = 21, est_len: int = 120, gap: int = 10
                  ) -> np.ndarray | None:
    """The full abnormal-return path over ``[-pre..+post]`` for one event position, or
    ``None`` if the estimation or event window runs off the tape."""
    L = len(s)
    if pos - gap - est_len < 0 or pos - pre < 0 or pos + post >= L:
        return None
    est_idx = np.arange(pos - gap - est_len, pos - gap)
    s_est, m_est = s[est_idx], m[est_idx]
    if not (np.isfinite(s_est).all() and np.isfinite(m_est).all()):
        return None
    alpha, beta = _ols_alpha_beta(s_est, m_est)
    alpha, beta = float(alpha[0]), float(beta[0])
    ev_idx = np.arange(pos - pre, pos + post + 1)
    return s[ev_idx] - (alpha + beta * m[ev_idx])


# --------------------------------------------------------------------------- #
# Stacking the real events
# --------------------------------------------------------------------------- #
def _snap_pos(index: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    """First session on/after ``date`` (the documented execution-lag snap)."""
    return int(index.searchsorted(pd.Timestamp(date)))


def stack_event_cars(events_df: pd.DataFrame, mkt: pd.Series,
                     stocks: dict[str, pd.Series], pre: int = 5, post: int = 21,
                     est_len: int = 120, gap: int = 10) -> pd.DataFrame:
    """Per-event CAR at every horizon, plus the estimated ``alpha, beta``.

    For each event whose implicated ``ticker`` is present in ``stocks``, align that
    stock against the market, snap the event date to the first session on/after it, and
    compute the market-model CAR at each horizon in :data:`HORIZONS`. Events whose
    window runs off the tape are dropped (``nan``-filtered). Returns a frame indexed by
    event date with columns ``ticker`` and one column per horizon (in return units).
    """
    rows = []
    for _, ev in events_df.iterrows():
        tkr = ev["ticker"]
        if tkr not in stocks:
            continue
        idx, s, m = _aligned_arrays(stocks[tkr], mkt)
        pos = _snap_pos(idx, ev["date"])
        if pos >= len(idx):
            continue
        row = {"date": ev["date"], "ticker": tkr, "snap": idx[pos], "pos": pos}
        ok = True
        for name, offs in HORIZONS.items():
            val = car_vec(s, m, np.array([pos]), offs, pre, post, est_len, gap)[0]
            row[name] = val
            if name == "day0" and not np.isfinite(val):
                ok = False
        # capture alpha/beta for the record
        path = event_ar_path(s, m, pos, pre, post, est_len, gap)
        row["has_path"] = path is not None
        if ok:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ticker", "snap", "pos", *HORIZONS, "has_path"])
    return pd.DataFrame(rows).set_index("date").sort_index()


def mean_ar_path(events_df: pd.DataFrame, mkt: pd.Series,
                 stocks: dict[str, pd.Series], pre: int = 5, post: int = 21,
                 est_len: int = 120, gap: int = 10) -> pd.DataFrame:
    """Mean abnormal-return path by offset ``[-pre..+post]`` across events, with each
    offset's own one-sample *t* and the anchored CAR (CAR(-pre) == 0)."""
    paths = []
    for _, ev in events_df.iterrows():
        tkr = ev["ticker"]
        if tkr not in stocks:
            continue
        idx, s, m = _aligned_arrays(stocks[tkr], mkt)
        pos = _snap_pos(idx, ev["date"])
        if pos >= len(idx):
            continue
        p = event_ar_path(s, m, pos, pre, post, est_len, gap)
        if p is not None and np.all(np.isfinite(p)):
            paths.append(p)
    offsets = list(range(-pre, post + 1))
    if not paths:
        return pd.DataFrame(columns=["offset", "mean_ar", "car", "t"]).set_index("offset")
    W = np.vstack(paths)
    mean_ar = W.mean(axis=0)
    car = np.cumsum(mean_ar) - mean_ar[0]
    rows = []
    for i, k in enumerate(offsets):
        _, t = one_sample_t(W[:, i])
        rows.append({"offset": k, "mean_ar": float(mean_ar[i]),
                     "car": float(car[i]), "t": t})
    return pd.DataFrame(rows).set_index("offset")


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
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    Included for the desk's canonical inference set; for this study the events are
    far-apart, independent calendar dates, so the HAC *t* and the plain one-sample *t*
    essentially coincide — the cross-check is intentional.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
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


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def car_stats(cars: pd.DataFrame, horizon: str = "month") -> dict:
    """Cross-event summary of one horizon's CAR: n, mean, one-sample & HAC t, down-hit
    rate with a Wilson interval."""
    x = cars[horizon].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    mean, t = one_sample_t(x)
    k_down = int((x < 0).sum())
    wlo, whi = wilson_interval(k_down, n)
    return {
        "horizon": horizon,
        "n": n,
        "mean_bps": mean * 1e4 if n else float("nan"),
        "t": t,
        "t_nw": newey_west_t(x),
        "down": k_down,
        "down_rate": (k_down / n) if n else float("nan"),
        "wilson": (wlo, whi),
    }


# --------------------------------------------------------------------------- #
# Same-ticker random-date permutation placebo (the falsification control)
# --------------------------------------------------------------------------- #
def permutation_placebo(events_df: pd.DataFrame, mkt: pd.Series,
                        stocks: dict[str, pd.Series], horizon: str = "month",
                        pre: int = 5, post: int = 21, est_len: int = 120, gap: int = 10,
                        n_draws: int = 4000, seed: int = 850) -> dict:
    """Keep each event's **ticker**, draw a **random pseudo-event date** for it, and
    recompute the mean CAR — ``n_draws`` times. Vectorised per ticker.

    Each draw pairs one random position per event; the mean across events is one
    placebo replicate, so the observed mean CAR is scored against ``n_draws`` random
    calendars of the same size **on the same tickers** (its own beta, its own vol).
    ``p`` is the left-tail share (claim predicts a *negative* CAR): fraction of random
    calendars whose mean CAR is <= observed.
    """
    offs = HORIZONS[horizon]
    rng = np.random.default_rng(seed)
    # observed per-event CARs + a random-position matrix per event, all vectorised.
    obs_list = []
    draw_cols = []  # each is an (n_draws,) vector of that event's random-date CARs
    lo_need = gap + est_len
    hi_need = max(offs) if offs else 0
    for _, ev in events_df.iterrows():
        tkr = ev["ticker"]
        if tkr not in stocks:
            continue
        idx, s, m = _aligned_arrays(stocks[tkr], mkt)
        pos = _snap_pos(idx, ev["date"])
        L = len(idx)
        obs = car_vec(s, m, np.array([pos]), offs, pre, post, est_len, gap)[0]
        if not np.isfinite(obs):
            continue
        lo, hi = lo_need + pre, L - hi_need - 1
        if hi <= lo:
            continue
        rand_pos = rng.integers(lo, hi, size=n_draws)
        rand_car = car_vec(s, m, rand_pos, offs, pre, post, est_len, gap)
        obs_list.append(obs)
        draw_cols.append(rand_car)
    if not obs_list:
        return {"n": 0, "obs_bps": float("nan"), "placebo_mean_bps": float("nan"),
                "placebo_sd_bps": float("nan"), "p_left": float("nan"),
                "n_draws": 0, "draws_bps": np.array([])}
    obs_mean = float(np.mean(obs_list))
    draws = np.nanmean(np.vstack(draw_cols), axis=0)  # (n_draws,) mean CAR per calendar
    draws = draws[np.isfinite(draws)]
    return {
        "n": len(obs_list),
        "obs_bps": obs_mean * 1e4,
        "placebo_mean_bps": float(draws.mean() * 1e4) if draws.size else float("nan"),
        "placebo_sd_bps": float(draws.std(ddof=1) * 1e4) if draws.size > 1 else float("nan"),
        "p_left": float((draws <= obs_mean).mean()) if draws.size else float("nan"),
        "n_draws": int(draws.size),
        "draws_bps": draws * 1e4,
    }


# --------------------------------------------------------------------------- #
# The tradable overlay — short the meltdown stock
# --------------------------------------------------------------------------- #
def short_the_meltdown(events_df: pd.DataFrame, stocks: dict[str, pd.Series],
                       hold: int = 21, cost_bps: float = 5.0,
                       borrow_bps_yr: float = 300.0) -> pd.DataFrame:
    """Short the implicated stock at the event-day close, hold ``hold`` sessions, cover.

    The meltdown is public at the close of the snap session (see the module docstring's
    execution lag), so the short earns sessions ``+1..+hold``. One round trip per event:
    one-way cost charged twice (entry + exit) against NAV, plus borrow on the short leg
    for the holding period. A reputational-shock claim, if real and tradable, makes the
    **short** return positive.
    """
    rows = []
    round_trip = 2.0 * cost_bps * 1e-4
    borrow = (borrow_bps_yr * 1e-4) / 252.0 * hold
    for _, ev in events_df.iterrows():
        tkr = ev["ticker"]
        if tkr not in stocks:
            continue
        s = stocks[tkr].sort_index()
        idx = s.index
        pos = int(idx.searchsorted(pd.Timestamp(ev["date"])))
        exit_ = pos + hold
        if pos >= len(idx) or exit_ >= len(idx):
            continue
        stock_move = float(s.iat[exit_] / s.iat[pos] - 1.0)
        short_gross = -stock_move
        short_net = short_gross - round_trip - borrow
        rows.append({"date": ev["date"], "ticker": tkr, "hold": hold,
                     "stock_move": stock_move, "short_gross": short_gross,
                     "short_net": short_net})
    return pd.DataFrame(rows)


def summarize_short(ledger: pd.DataFrame, col: str = "short_net") -> dict:
    """Headline stats for the short ledger: n, win-rate, mean (bps), one-sample t."""
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
def synthetic_detect(mkt: pd.Series, stocks: dict[str, pd.Series],
                     events_df: pd.DataFrame, horizon: str = "day0",
                     pre: int = 5, post: int = 21) -> dict:
    """Run the headline cross-event CAR stats on a synthetic world."""
    cars = stack_event_cars(events_df, mkt, stocks, pre=pre, post=post)
    return car_stats(cars, horizon)
