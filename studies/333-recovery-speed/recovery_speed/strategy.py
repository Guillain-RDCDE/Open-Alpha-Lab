"""The recovery-speed engine and its honest stats — Study 333 (Recovery-Speed).

The folklore: after a deep, market-wide drawdown, the names that climb back to their
pre-drawdown high *fastest* are the "strong" ones, and strength persists — so buy the
fast recoverers and they keep leading. This is drawdown-recovery momentum.

We make it falsifiable as a cross-sectional sort:

1. **Detect** the market drawdown trough and the recovery date (when the market first
   regains, say, a fraction of the peak).
2. At the recovery date, **score** each name by its *recovery speed* — how much of its
   own peak-to-trough loss it has clawed back, per day, since the trough.
3. **Rank** names into deciles and hold a **long top / short bottom** book over a
   forward window. The signal is known at the close of the formation bar and earns the
   *next* bar's return onward — one ``shift``, applied once (``run_event``).
4. **Stats:** the long-short forward return, its HAC/Newey-West t-stat, a circular
   block-bootstrap CI, costs charged one-way x NAV with the short leg paying borrow,
   and — because the recovery rally is itself a market move — an **excess-vs-market**
   (market-neutral) reading so we don't credit beta as alpha.

The decisive control: shuffle the recovery-speed scores across names (destroying any
link to identity) and re-rank. If the real ranking has no edge over the shuffled one,
recovery speed predicts nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Drawdown / recovery detection on the market series
# ---------------------------------------------------------------------------
def detect_drawdown(
    market: pd.Series, min_depth: float = 0.18, recover_frac: float = 0.85
) -> dict | None:
    """Locate the deepest drawdown in ``market`` and its recovery date.

    Returns a dict with integer positions ``peak``, ``trough``, ``recover`` (the first
    bar after the trough at which the market has regained ``recover_frac`` of the
    peak-to-trough loss), plus the realised ``depth``. Returns ``None`` if no drawdown
    reaches ``min_depth``.

    All positions are positional indices into ``market`` (use ``.iloc``).
    """
    px = market.to_numpy(dtype=float)
    n = len(px)
    run_max = np.maximum.accumulate(px)
    dd = px / run_max - 1.0
    trough = int(np.argmin(dd))
    depth = float(-dd[trough])
    if depth < min_depth:
        return None
    # The peak is the running-max location at the trough.
    peak = int(np.argmax(px[: trough + 1]))
    peak_px = px[peak]
    trough_px = px[trough]
    target = trough_px + recover_frac * (peak_px - trough_px)
    recover = None
    for j in range(trough + 1, n):
        if px[j] >= target:
            recover = j
            break
    if recover is None:
        return None
    return {"peak": peak, "trough": trough, "recover": recover, "depth": depth}


# ---------------------------------------------------------------------------
# Recovery-speed score per name
# ---------------------------------------------------------------------------
def recovery_speed(
    panel: pd.DataFrame, peak: int, trough: int, recover: int
) -> pd.Series:
    """Per-name recovery speed at the recovery date.

    For each name we measure how much of its OWN peak-to-trough drawdown it has clawed
    back by the recovery bar, per day since the trough:

        speed_i = (P_i[recover] - P_i[trough]) / (P_i[peak] - P_i[trough]) / days

    Names with a bigger own drawdown that bounced back fast score high. NaNs (missing
    history, no drawdown) are dropped by the caller. Uses only data up to and including
    the recovery bar — no look-ahead.
    """
    p_peak = panel.iloc[peak]
    p_trough = panel.iloc[trough]
    p_rec = panel.iloc[recover]
    own_loss = p_peak - p_trough
    clawed = p_rec - p_trough
    days = max(recover - trough, 1)
    # Only meaningful for names that actually fell (own_loss > 0).
    frac = clawed / own_loss.where(own_loss > 0)
    speed = frac / days
    return speed.replace([np.inf, -np.inf], np.nan)


def deciles(score: pd.Series, q: int = 5) -> tuple[pd.Index, pd.Index]:
    """Return (top, bottom) name indices: top ``1/q`` and bottom ``1/q`` by score."""
    s = score.dropna().sort_values()
    n = len(s)
    k = max(n // q, 1)
    bottom = s.index[:k]
    top = s.index[-k:]
    return top, bottom


# ---------------------------------------------------------------------------
# Forward event engine
# ---------------------------------------------------------------------------
def run_event(
    panel: pd.DataFrame,
    market: pd.Series,
    event: dict,
    fwd_days: int = 63,
    q: int = 5,
    cost_bps: float = 0.0,
    borrow_bps_ann: float = 0.0,
    shuffle_seed: int | None = None,
    market_neutral: bool = False,
) -> dict:
    """Run one recovery-speed long/short event and return its forward stats.

    Formation bar = ``event['recover']`` (signal known at that close). The book is held
    over the next ``fwd_days`` bars — the position is entered at the *next* bar
    (``recover + 1``), i.e. one execution lag applied once. The long leg is the top
    ``1/q`` of names by recovery speed, the short leg the bottom ``1/q`` (equal weight).

    ``cost_bps`` is a one-way cost on the NAV traded at entry+exit (both legs).
    ``borrow_bps_ann`` charges the short leg financing over the holding window.
    If ``shuffle_seed`` is set the scores are permuted across names (the null control).
    If ``market_neutral`` the per-name forward returns are measured in EXCESS of the
    market over the same window (so we don't credit the recovery-rally beta as alpha).

    Returns a dict: per-name forward returns of each leg, the long-short return (net),
    n_long/n_short, and the leg means.
    """
    rec = event["recover"]
    entry = rec + 1  # one lag: signal at rec's close, fill next bar
    exit_ = min(entry + fwd_days, len(panel) - 1)
    if entry >= len(panel):
        raise ValueError("recovery bar too close to the end of the panel")

    score = recovery_speed(panel, event["peak"], event["trough"], rec)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        vals = score.dropna()
        permuted = pd.Series(
            rng.permutation(vals.to_numpy()), index=vals.index, name=score.name
        )
        score = permuted

    top, bottom = deciles(score, q=q)

    p_entry = panel.iloc[entry]
    p_exit = panel.iloc[exit_]
    fwd = (p_exit / p_entry - 1.0)
    if market_neutral:
        mkt_fwd = market.iloc[exit_] / market.iloc[entry] - 1.0
        fwd = fwd - mkt_fwd

    long_r = fwd.loc[top].dropna()
    short_r = fwd.loc[bottom].dropna()

    days = exit_ - entry
    borrow = borrow_bps_ann * 1e-4 * days / 252.0
    cost = cost_bps * 1e-4

    long_mean = float(long_r.mean())
    short_mean = float(short_r.mean())
    # Long-short gross, then charge: entry+exit one-way cost on BOTH legs, borrow on short.
    ls_gross = long_mean - short_mean
    ls_net = ls_gross - 4.0 * cost - borrow
    return {
        "long_returns": long_r,
        "short_returns": short_r,
        "long_mean": long_mean,
        "short_mean": short_mean,
        "ls_gross": float(ls_gross),
        "ls_net": float(ls_net),
        "n_long": int(len(long_r)),
        "n_short": int(len(short_r)),
        "days_held": int(days),
        "market_neutral": bool(market_neutral),
    }


# ---------------------------------------------------------------------------
# Inference — cross-name HAC t and circular block-bootstrap CI on the spread
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-stat for the mean of ``x`` (here, the per-name LS spread)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def spread_tstat(result: dict) -> float:
    """HAC t-stat on the long-minus-short *cross-section* of forward returns.

    Stacks the long leg (sign +1) against the short leg (sign -1) demeaned to a single
    spread sample and returns its HAC t. This is the inference-bar number: is the
    long-short forward return distinguishable from zero across names?
    """
    lr = result["long_returns"].to_numpy(dtype=float)
    sr = result["short_returns"].to_numpy(dtype=float)
    if lr.size < 3 or sr.size < 3:
        return float("nan")
    # Two-sample contrast as a single centred series (Welch-style pooled spread).
    pooled = np.concatenate([lr - lr.mean() + (lr.mean() - sr.mean()),
                             sr - sr.mean()])
    # Simpler & honest: t on the difference of means with pooled variance.
    nl, ns = lr.size, sr.size
    se = np.sqrt(lr.var(ddof=1) / nl + sr.var(ddof=1) / ns)
    return float((lr.mean() - sr.mean()) / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    result: dict, n_boot: int = 2000, seed: int = 333, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap CI for the long-short spread by resampling names with replacement.

    Resamples the long and short legs (i.i.d. across names — the cross-section has no
    serial order) and returns the (alpha/2, 1-alpha/2) percentile interval of the
    long-minus-short mean.
    """
    rng = np.random.default_rng(seed)
    lr = result["long_returns"].to_numpy(dtype=float)
    sr = result["short_returns"].to_numpy(dtype=float)
    if lr.size < 2 or sr.size < 2:
        return (float("nan"), float("nan"))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        bl = rng.choice(lr, size=lr.size, replace=True)
        bs = rng.choice(sr, size=sr.size, replace=True)
        stats[b] = bl.mean() - bs.mean()
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Multi-event detection and pooling (the honest real-tape test)
# ---------------------------------------------------------------------------
def multi_events(
    market: pd.Series, min_depth: float = 0.15, recover_frac: float = 0.85,
    min_tail: int = 200,
) -> list[dict]:
    """Detect SUCCESSIVE drawdown-recovery episodes in ``market``.

    Walk forward: detect the deepest remaining drawdown, record it, then continue from
    just after its recovery bar. Each event's positions are absolute indices into
    ``market``. ``min_tail`` stops once too few bars remain. One event is never enough
    to certify a cross-sectional effect — this is how we get *independent* episodes."""
    evs: list[dict] = []
    n = len(market)
    start = 0
    while start < n - min_tail:
        sub = market.iloc[start:]
        ev = detect_drawdown(sub, min_depth=min_depth, recover_frac=recover_frac)
        if ev is None:
            break
        ev = {k: (v + start if k in ("peak", "trough", "recover") else v)
              for k, v in ev.items()}
        evs.append(ev)
        start = ev["recover"] + 1
    return evs


def pooled_spread(
    panel: pd.DataFrame,
    market: pd.Series,
    events: list[dict],
    fwd_days: int = 126,
    q: int = 5,
    market_neutral: bool = True,
    cost_bps: float = 0.0,
    borrow_bps_ann: float = 0.0,
) -> dict:
    """Pool the per-name long/short forward returns ACROSS events and infer on the pool.

    Returns the pooled long mean, short mean, long-short spread (net of costs/borrow),
    a two-sample t on the spread, a bootstrap CI, and the event count. This is the
    real-tape inference-bar number: a single event is a coincidence, the pool is evidence."""
    longs, shorts = [], []
    for ev in events:
        if ev["recover"] + 1 >= len(panel):
            continue
        r = run_event(panel, market, ev, fwd_days=fwd_days, q=q,
                      market_neutral=market_neutral)
        longs.append(r["long_returns"].to_numpy(dtype=float))
        shorts.append(r["short_returns"].to_numpy(dtype=float))
    if not longs:
        return {"n_events": 0, "spread_t": float("nan")}
    L = np.concatenate(longs)
    S = np.concatenate(shorts)
    L = L[np.isfinite(L)]
    S = S[np.isfinite(S)]
    days = fwd_days
    borrow = borrow_bps_ann * 1e-4 * days / 252.0
    cost = cost_bps * 1e-4
    ls_gross = float(L.mean() - S.mean())
    ls_net = ls_gross - 4.0 * cost - borrow
    se = np.sqrt(L.var(ddof=1) / L.size + S.var(ddof=1) / S.size)
    t = float(ls_gross / se) if se > 0 else float("nan")
    # Bootstrap CI by resampling names within each leg.
    rng = np.random.default_rng(333)
    boots = np.empty(2000)
    for b in range(2000):
        boots[b] = (rng.choice(L, L.size, replace=True).mean()
                    - rng.choice(S, S.size, replace=True).mean())
    return {
        "n_events": len(longs),
        "long_mean": float(L.mean()),
        "short_mean": float(S.mean()),
        "ls_gross": ls_gross,
        "ls_net": float(ls_net),
        "spread_t": t,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
        "n_long": int(L.size),
        "n_short": int(S.size),
        "market_neutral": bool(market_neutral),
    }


# ---------------------------------------------------------------------------
# Convenience: real vs shuffled-null edge on one event
# ---------------------------------------------------------------------------
def edge_vs_shuffle(
    panel: pd.DataFrame,
    market: pd.Series,
    event: dict,
    fwd_days: int = 63,
    q: int = 5,
    n_shuffle: int = 200,
    seed: int = 0,
    market_neutral: bool = False,
) -> dict:
    """Compare the real recovery-speed ranking against shuffled rankings.

    Returns the real long-short spread and the distribution of shuffled spreads, plus
    an empirical one-sided p-value (fraction of shuffles whose spread >= the real one).
    A real signal sits in the right tail of the shuffled null.
    """
    real = run_event(panel, market, event, fwd_days=fwd_days, q=q,
                     market_neutral=market_neutral)
    real_ls = real["ls_gross"]
    rng = np.random.default_rng(seed)
    shuffled = np.empty(n_shuffle)
    for i in range(n_shuffle):
        s = int(rng.integers(0, 2**31 - 1))
        r = run_event(panel, market, event, fwd_days=fwd_days, q=q,
                      shuffle_seed=s, market_neutral=market_neutral)
        shuffled[i] = r["ls_gross"]
    pval = float((shuffled >= real_ls).mean())
    return {
        "real_ls": float(real_ls),
        "shuffle_mean": float(shuffled.mean()),
        "shuffle_std": float(shuffled.std(ddof=1)),
        "shuffled": shuffled,
        "p_value": pval,
        "spread_t": spread_tstat(real),
    }
