"""The event-study engine and its honest controls — Study 312 (Debt-Ceiling), vol angle.

The folklore: *"debt-ceiling brinkmanship spooks markets — implied vol rises into the
X-date and collapses the moment Washington blinks, so it's a long-vol-into / short-vol-out
trade."* We test it as a clean event study around the binding deadlines in
``data.X_DATES``:

1. Snap each deadline to the first trading session on or after it (the event bar
   ``t=0``). The X-date is calendar-known in advance, so the canonical window needs no
   extra execution lag; a ``+1``-session entry lag is offered as a robustness check.
2. Measure two volatility moves on the **VIX level**:
   - the *pre-deadline ramp*: the change in VIX over the ``pre`` sessions ENDING at the
     event bar — the "long vol into the deadline" leg;
   - the *post-deadline collapse*: the change in VIX over the ``post`` sessions STARTING
     at the event bar — the "short vol on resolution" leg.
3. Race both against a **synthetic event-null**: the same windows measured around many
   random dates. If X-dates carry no special vol signal, the event windows look exactly
   like random-date windows.

We also score the canonical vol round-trip as a *trade*: go long vol ``pre`` sessions
before the deadline, flip to short vol at the deadline close, exit ``post`` sessions
later. We proxy the carry of a vol-long/short position with a VIX-futures-style term-decay
haircut and charge costs one-way × NAV — because the cruelest fact about "buy vol into the
event" is the negative roll you pay to hold it (the long-vol carry tax).

Inference is honest: a HAC (Newey-West) t-stat on the per-event moves, a circular
block-bootstrap CI, and a permutation/placebo test of the event mean vs the random-date
distribution. With only a handful of brink-going episodes in the modern VIX era, this is a
low-n event study by construction — we say so loudly and let the (wide) CI carry the verdict.
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
    lag: int = 0,
) -> np.ndarray:
    """Integer positions in ``bars.index`` of the first session on/after each event.

    ``lag`` shifts the entry bar that many sessions later (the look-ahead robustness knob;
    the canonical window uses ``lag=0`` because the X-date is known weeks in advance).
    Events outside the tape are dropped.
    """
    idx = bars.index
    locs = []
    for e in pd.DatetimeIndex(events):
        pos = idx.searchsorted(e, side="left")  # first session >= e
        pos += lag
        if 0 <= pos < len(idx):
            locs.append(int(pos))
    return np.array(sorted(set(locs)), dtype=int)


def _vix_col(bars: pd.DataFrame) -> np.ndarray:
    """The VIX level column — ``vix`` if present (synthetic), else ``close`` (real ^VIX)."""
    col = "vix" if "vix" in bars.columns else "close"
    return bars[col].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# The two vol legs: pre-deadline ramp and post-deadline collapse
# ---------------------------------------------------------------------------
def pre_ramp(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 20,
    lag: int = 0,
) -> pd.DataFrame:
    """Per-event change in the VIX level over the ``pre`` sessions ending at the event bar.

    Reports both the absolute change (vol points) and the log change (so events at
    different base vols are comparable). A positive value = vol rose *into* the deadline.

    Columns: ``event_date, vix_start, vix_end, d_level, d_log``.
    """
    vix = _vix_col(bars)
    locs = snap_to_sessions(bars, events, lag=lag)
    rows = []
    for e in locs:
        s = e - pre
        if s < 0:
            continue
        a, b = vix[s], vix[e]
        rows.append(
            {
                "event_date": bars.index[e],
                "vix_start": a,
                "vix_end": b,
                "d_level": b - a,
                "d_log": float(np.log(b / a)),
            }
        )
    return pd.DataFrame(rows)


def post_collapse(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    post: int = 20,
    lag: int = 0,
) -> pd.DataFrame:
    """Per-event change in the VIX level over the ``post`` sessions starting at the event bar.

    A *negative* value = vol fell *after* the deadline (the "short vol on resolution" leg).

    Columns: ``event_date, vix_start, vix_end, d_level, d_log``.
    """
    vix = _vix_col(bars)
    locs = snap_to_sessions(bars, events, lag=lag)
    n = len(vix)
    rows = []
    for e in locs:
        x = e + post
        if x >= n:
            continue
        a, b = vix[e], vix[x]
        rows.append(
            {
                "event_date": bars.index[e],
                "vix_start": a,
                "vix_end": b,
                "d_level": b - a,
                "d_log": float(np.log(b / a)),
            }
        )
    return pd.DataFrame(rows)


def window_log_changes(
    bars: pd.DataFrame,
    locs: np.ndarray,
    span: int,
    side: str = "post",
) -> np.ndarray:
    """Log VIX change over ``span`` sessions for a set of integer positions ``locs``.

    ``side="pre"`` measures the change ENDING at each position; ``side="post"`` measures
    the change STARTING at each position. Out-of-range positions are dropped.
    """
    vix = _vix_col(bars)
    n = len(vix)
    out = []
    for e in locs:
        if side == "pre":
            s = e - span
            if s >= 0:
                out.append(float(np.log(vix[e] / vix[s])))
        else:
            x = e + span
            if x < n:
                out.append(float(np.log(vix[x] / vix[e])))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# The canonical vol round-trip as a TRADE (with the carry tax)
# ---------------------------------------------------------------------------
def vol_round_trip(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 20,
    post: int = 20,
    lag: int = 0,
    daily_carry_bps: float = 25.0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Score the long-vol-into / short-vol-out trade per event, *net of the carry tax*.

    The naive thesis: buy vol ``pre`` sessions before the deadline (capturing the ramp),
    then at the deadline flip to short vol (capturing the collapse). We approximate the
    P&L of a long-vol position by the **log change in the VIX level** — but holding a vol
    position is not free: VIX futures roll down a (usually) upward-sloping term structure,
    so a long-vol holder bleeds ``daily_carry_bps`` of NAV per session (and a short-vol
    holder *earns* it). This negative roll is the dominant fact about every "buy
    protection into the event" trade, so we make it explicit.

    P&L per leg (in NAV terms, log-approx):
      - long-vol leg  = +d_log(pre)  − daily_carry_bps·1e-4·pre   (pays carry)
      - short-vol leg = −d_log(post) + daily_carry_bps·1e-4·post  (earns carry)
    ``cost_bps`` is a one-way × NAV transaction cost charged on each of the three fills
    (enter long, flip, exit) → 3 × cost.

    Columns: ``event_date, pnl_long, pnl_short, pnl_gross, pnl_net``.
    """
    vix = _vix_col(bars)
    locs = snap_to_sessions(bars, events, lag=lag)
    n = len(vix)
    rows = []
    for e in locs:
        s, x = e - pre, e + post
        if s < 0 or x >= n:
            continue
        d_pre = float(np.log(vix[e] / vix[s]))
        d_post = float(np.log(vix[x] / vix[e]))
        carry = daily_carry_bps * 1e-4
        pnl_long = d_pre - carry * pre        # long vol pays roll
        pnl_short = -d_post + carry * post    # short vol earns roll
        pnl_gross = pnl_long + pnl_short
        pnl_net = pnl_gross - 3.0 * cost_bps * 1e-4
        rows.append(
            {
                "event_date": bars.index[e],
                "pnl_long": pnl_long,
                "pnl_short": pnl_short,
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
    block: int = 3,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 312,
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
    seed: int = 312,
) -> tuple[float, float]:
    """Two-sided permutation p-value: is the event mean unusual vs random-date moves?

    Repeatedly re-draws a group of the event size from the random-date distribution and
    asks how often a random draw's mean lands at least as far from the random-date centre
    as the observed event mean. Returns ``(observed_mean_minus_null_mean, p_value)``.
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
    """Headline statistics for a set of per-event moves (log changes or P&L)."""
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


def vix_path(
    bars: pd.DataFrame,
    events: pd.DatetimeIndex,
    pre: int = 20,
    post: int = 40,
    lag: int = 0,
) -> pd.DataFrame:
    """Average VIX-level path around the events, aligned on the event bar (t=0).

    Returns a frame indexed by relative session ``[-pre, +post]`` with the cross-event
    *mean* VIX level rebased to 1.0 at t=0 (so different base vols are comparable) and its
    standard error. The hump shape — up into 0, down after — is the visual signature of
    the thesis.
    """
    vix = _vix_col(bars)
    locs = snap_to_sessions(bars, events, lag=lag)
    rel = np.arange(-pre, post + 1)
    paths = []
    n = len(vix)
    for e in locs:
        if e - pre < 0 or e + post >= n:
            continue
        seg = vix[e - pre : e + post + 1]
        paths.append(seg / vix[e])  # rebased to event bar
    if not paths:
        return pd.DataFrame({"rel": rel, "mean": np.nan, "se": np.nan}).set_index("rel")
    P = np.vstack(paths)
    mean = P.mean(axis=0)
    se = P.std(axis=0, ddof=1) / np.sqrt(P.shape[0]) if P.shape[0] > 1 else np.zeros_like(mean)
    return pd.DataFrame({"rel": rel, "mean": mean, "se": se}).set_index("rel")
