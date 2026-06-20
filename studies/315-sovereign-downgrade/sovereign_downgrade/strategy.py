"""The event-study engine and its honest controls — Study 315 (Sovereign-Downgrade).

The folklore: *"when a rating agency strips the US of its AAA, stocks tank — the downgrade
is a flashing-red macro signal, so sell the announcement and buy back the panic."* We test
it as a clean directional **abnormal-return** event study around the three US sovereign
downgrades in ``data.DOWNGRADES``:

1. Snap each announcement to the first *tradeable* session. All three were announced after
   the US cash close, so the canonical event bar ``t=0`` is the **next** session (lag=1); a
   same-session (lag=0) variant is offered as a robustness check.
2. Measure the **abnormal return** = the realized return minus the sample's mean daily
   return (a constant-mean-return market model — with three events there is no room to fit a
   beta, so we use the simplest defensible benchmark and say so). We report:
   - the **event-day** abnormal return (the announcement reaction),
   - the **cumulative abnormal return (CAR)** over the ``[0, +k]`` window (the drift after),
   - the **pre-window CAR** over ``[-k, 0)`` (was the market already falling?).
3. Race the CAR against a **synthetic control**: the same window measured around many
   random non-event dates. If a downgrade carries no special return signal, the event
   window looks exactly like a random-date window.

We also score the naive "sell the downgrade" trade: short SPY for ``hold`` sessions from the
event bar, charging costs one-way × NAV on entry and exit (a short pays borrow, modelled as
a per-session financing haircut). The cruelest fact for the bears is that the US equity tape
drifts *up*, so a blind short bleeds the drift it fights.

Inference is honest, and loud about its own weakness: with only **three** downgrade
episodes a per-event t-stat is barely meaningful. We report a HAC t-stat for completeness, a
circular block-bootstrap CI on the event-day returns, and — the test that actually carries
the verdict — a **permutation p-value** of the event mean against the synthetic-control
distribution. Three events can never clear the inference bar; we say so and stamp
accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Snap event dates onto the trading calendar
# ---------------------------------------------------------------------------
def snap_to_sessions(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    lag: int = 1,
) -> np.ndarray:
    """Integer positions in ``bars.index`` of the event bar for each announcement.

    ``lag`` shifts the event bar that many sessions after the first session on/after the
    announcement date. The canonical ``lag=1`` reflects that all three downgrades were
    announced *after* the US cash close, so the first tradeable reaction is the next
    session; ``lag=0`` (same-session) is the robustness knob. Events outside the tape are
    dropped.
    """
    idx = bars.index
    locs = []
    for e in pd.DatetimeIndex(events):
        pos = idx.searchsorted(e, side="left")  # first session >= e
        pos += lag
        if 0 <= pos < len(idx):
            locs.append(int(pos))
    return np.array(sorted(set(locs)), dtype=int)


def _log_returns(bars: pd.DataFrame) -> np.ndarray:
    """Daily log total returns of the close column (first bar = 0)."""
    close = bars["close"].to_numpy(dtype=float)
    r = np.zeros_like(close)
    r[1:] = np.log(close[1:] / close[:-1])
    return r


# ---------------------------------------------------------------------------
# Abnormal returns and cumulative abnormal returns (CARs)
# ---------------------------------------------------------------------------
def abnormal_returns(bars: pd.DataFrame) -> np.ndarray:
    """Abnormal daily return = realized log return − the sample mean daily log return.

    A constant-mean-return market model: with only three real events there is no room to fit
    an instrument-specific beta, so we benchmark against the unconditional mean. This is the
    most conservative defensible expected-return model for a single-name event study, and we
    label it as such.
    """
    r = _log_returns(bars)
    mu = r[1:].mean() if r.size > 1 else 0.0
    return r - mu


def event_day_abnormal(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    lag: int = 1,
) -> pd.DataFrame:
    """Per-event abnormal return on the event bar (the announcement-day reaction).

    Columns: ``event_date, ret, abn`` (raw log return and abnormal log return).
    """
    r = _log_returns(bars)
    abn = abnormal_returns(bars)
    locs = snap_to_sessions(bars, events, lag=lag)
    rows = []
    for e in locs:
        rows.append(
            {
                "event_date": bars.index[e],
                "ret": float(r[e]),
                "abn": float(abn[e]),
            }
        )
    return pd.DataFrame(rows)


def car(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    window: tuple[int, int] = (0, 5),
    lag: int = 1,
) -> pd.DataFrame:
    """Per-event cumulative abnormal return over a relative session window.

    ``window=(a, b)`` cumulates the abnormal returns over relative bars ``[a, b]`` inclusive
    (e.g. ``(0, 5)`` = announcement bar through five sessions later; ``(-5, -1)`` = the five
    sessions *before*). A negative CAR over ``(0, +k)`` would be the folklore's predicted
    sell-off.

    Columns: ``event_date, car``.
    """
    abn = abnormal_returns(bars)
    n = abn.size
    locs = snap_to_sessions(bars, events, lag=lag)
    a, b = window
    rows = []
    for e in locs:
        s, x = e + a, e + b
        if s < 0 or x >= n:
            continue
        rows.append(
            {
                "event_date": bars.index[e],
                "car": float(abn[s : x + 1].sum()),
            }
        )
    return pd.DataFrame(rows)


def window_cars(
    bars: pd.DataFrame,
    locs: np.ndarray,
    window: tuple[int, int] = (0, 5),
) -> np.ndarray:
    """Cumulative abnormal return over ``window`` for a set of integer positions ``locs``.

    The synthetic-control workhorse: feed it the random-date positions to build the placebo
    distribution of CARs. Out-of-range positions are dropped.
    """
    abn = abnormal_returns(bars)
    n = abn.size
    a, b = window
    out = []
    for e in locs:
        s, x = e + a, e + b
        if s >= 0 and x < n:
            out.append(float(abn[s : x + 1].sum()))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# The naive "sell the downgrade" trade (short SPY, costs + borrow)
# ---------------------------------------------------------------------------
def short_the_downgrade(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    hold: int = 5,
    lag: int = 1,
    cost_bps: float = 0.0,
    borrow_bps_per_day: float = 1.0,
) -> pd.DataFrame:
    """Score a blind short-SPY trade per event, net of one-way × NAV costs and borrow.

    The naive thesis: at the event bar, go **short** SPY for ``hold`` sessions to capture
    the predicted panic. P&L of a short over the window is ``-1 ×`` the realized log return.
    A short pays borrow each session (``borrow_bps_per_day`` of NAV) and a one-way × NAV
    transaction cost on each of the two fills (enter, exit) → ``2 × cost_bps``.

    Columns: ``event_date, pnl_gross, pnl_net``.
    """
    r = _log_returns(bars)
    n = r.size
    locs = snap_to_sessions(bars, events, lag=lag)
    rows = []
    for e in locs:
        x = e + hold
        if e < 0 or x >= n:
            continue
        # realized log return of being LONG over [e, x] is r[e+1 .. x] (close-to-close);
        # being short flips the sign. (Enter at the event bar's close, exit at x's close.)
        long_ret = float(r[e + 1 : x + 1].sum())
        pnl_gross = -long_ret
        pnl_net = (
            pnl_gross
            - borrow_bps_per_day * 1e-4 * hold
            - 2.0 * cost_bps * 1e-4
        )
        rows.append(
            {
                "event_date": bars.index[e],
                "pnl_gross": pnl_gross,
                "pnl_net": pnl_net,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Honest inference
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``x`` (no quantlab hard dependency)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray,
    block: int = 2,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 315,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (respects local dependence)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def permutation_pvalue(
    event_ret: np.ndarray,
    null_ret: np.ndarray,
    n_perm: int = 20000,
    seed: int = 315,
) -> tuple[float, float]:
    """Two-sided permutation p-value: is the event mean unusual vs the synthetic control?

    Repeatedly draws a group of the event size from the random-date (synthetic-control)
    distribution and asks how often a random draw's mean lands at least as far from the
    control centre as the observed event mean. With only three events this is the *only*
    inference that is even modestly meaningful — and it is still weak. Returns
    ``(observed_mean_minus_null_mean, p_value)``.
    """
    event_ret = np.asarray(event_ret, dtype=float)
    null_ret = np.asarray(null_ret, dtype=float)
    event_ret = event_ret[np.isfinite(event_ret)]
    null_ret = null_ret[np.isfinite(null_ret)]
    k = event_ret.size
    if k == 0 or null_ret.size == 0:
        return (float("nan"), float("nan"))
    obs = event_ret.mean() - null_ret.mean()
    pool = null_ret
    rng = np.random.default_rng(seed)
    null_means = np.array([rng.choice(pool, size=k, replace=True).mean()
                           for _ in range(n_perm)])
    centre = pool.mean()
    p = float((np.abs(null_means - centre) >= abs(event_ret.mean() - centre)).mean())
    return float(obs), p


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(x: np.ndarray) -> dict:
    """Headline statistics for a set of per-event values (abnormal returns, CARs or P&L)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return dict(n=0, hit_rate=float("nan"), mean=float("nan"),
                    median=float("nan"), std=float("nan"), tstat=float("nan"))
    return dict(
        n=int(n),
        hit_rate=float((x > 0).mean()),
        mean=float(x.mean()),
        median=float(np.median(x)),
        std=float(x.std(ddof=1)) if n > 1 else float("nan"),
        tstat=hac_tstat(x),
    )


def car_path(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 10,
    post: int = 20,
    lag: int = 1,
) -> pd.DataFrame:
    """Average cumulative-abnormal-return path around the events, aligned on t=0.

    Returns a frame indexed by relative session ``[-pre, +post]`` with the cross-event mean
    CAR (cumulated from ``-pre``, rebased to 0 at the event bar t=0) and its standard error.
    A downward-sloping path after 0 is the visual signature of the "downgrade sells off"
    thesis.
    """
    abn = abnormal_returns(bars)
    n = abn.size
    locs = snap_to_sessions(bars, events, lag=lag)
    rel = np.arange(-pre, post + 1)
    paths = []
    for e in locs:
        if e - pre < 0 or e + post >= n:
            continue
        seg = abn[e - pre : e + post + 1]
        c = np.cumsum(seg)
        c = c - c[pre]  # rebase to 0 at the event bar (index ``pre`` within the segment)
        paths.append(c)
    if not paths:
        return pd.DataFrame({"rel": rel, "mean": np.nan, "se": np.nan}).set_index("rel")
    P = np.vstack(paths)
    mean = P.mean(axis=0)
    se = P.std(axis=0, ddof=1) / np.sqrt(P.shape[0]) if P.shape[0] > 1 else np.zeros_like(mean)
    return pd.DataFrame({"rel": rel, "mean": mean, "se": se}).set_index("rel")
