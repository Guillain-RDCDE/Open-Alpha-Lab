"""The yardstick: does a mention beat a *random day* in the same universe — and does
it beat the *momentum the name already had*?

Two nulls, because a green abnormal path after a mention can fool you twice. Once
because these are volatile micro-caps that jump around anyway, so *some* of them are
always running — pick the ones that just ran and you've drawn a winner by
construction. And once because the influencer plausibly mentions names *because*
they're already moving, so "mention" and "momentum" are nearly the same event.

    conditional   = average forward abnormal return *after a mention*
    unconditional = average forward abnormal return *on a random (name, day)*  (null #1)
    excess        = conditional - unconditional

If ``excess`` ≈ 0 the mention adds nothing a coin-flip over the same universe
wouldn't. We test it with a permutation test (random baskets of the same size),
exactly as in Studies 02–03.

Then the control unique to this study, :func:`excess_vs_alternative`: pit the
mention events against **hot-streak** events (a name already in its top-decile
trailing run, from :func:`social_oracle.mentions.hot_streak_events`) on the same
forward abnormal returns, with a permutation test on the *difference* of conditional
means. A mention that beats a random day but **not** a name that was already hot is
a momentum sensor wearing an oracle's hat.

Everything is measured on **abnormal** returns (name minus market), so a rising tide
that lifted the whole feed is already netted out before we start.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def car_forward(frame: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forward cumulative abnormal return CAR(t -> t+h) for every session ``t``.

    ``NaN`` for the last ``horizon`` sessions (no full forward window). Additive CAR
    on ``r_cc - r_mkt``, matching :mod:`social_oracle.eventstudy`.
    """
    from .eventstudy import _prefix
    prefix = _prefix(frame)            # len n+1, prefix[i] = sum ab[:i]
    n = len(frame)
    out = np.full(n, np.nan)
    if horizon < n:
        # CAR(t -> t+h) = sum ab[t+1 .. t+h] = prefix[t+h+1] - prefix[t+1]
        t = np.arange(n - horizon)
        out[:n - horizon] = prefix[t + horizon + 1] - prefix[t + 1]
    return out


def _pool_and_events(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pool, event_vals)``: all valid forward abnormal CARs in the universe,
    and the subset selected by ``events``."""
    pool_parts, ev_parts = [], []
    by_ticker = {t: g["entry_pos"].to_numpy() for t, g in events.groupby("ticker")} if len(events) else {}
    for t, frame in panel.items():
        car = car_forward(frame, horizon)
        valid = car[~np.isnan(car)]
        pool_parts.append(valid)
        pos = by_ticker.get(t)
        if pos is not None and len(pos):
            ev = car[pos]
            ev_parts.append(ev[~np.isnan(ev)])
    pool = np.concatenate(pool_parts) if pool_parts else np.array([])
    ev = np.concatenate(ev_parts) if ev_parts else np.array([])
    return pool, ev


def conditional_vs_unconditional(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizons=(1, 5, 21),
    n_iter: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare mention-conditional forward abnormal returns to the random-day null.

    Default horizons are +1d / +1w / +1m. Per horizon: ``n_events, mean_cond,
    mean_uncond, excess, pct_pos_cond, pct_pos_uncond, p_greater, p_two_sided``.

    ``p_greater`` is the headline: the probability a random basket of the same size,
    drawn from every (name, day) in the universe, beats the mention basket. Small
    (< 0.05) = the mention adds something; near 0.5 = it doesn't.
    """
    rng = np.random.default_rng(seed)
    cols = ["horizon", "n_events", "mean_cond", "mean_uncond", "excess",
            "pct_pos_cond", "pct_pos_uncond", "p_greater", "p_two_sided"]
    rows = []
    for h in horizons:
        pool, ev = _pool_and_events(panel, events, h)
        k = len(ev)
        if k == 0 or k > len(pool):
            continue
        observed = ev.mean()
        uncond = pool.mean()
        draws = np.array([rng.choice(pool, size=k, replace=False).mean() for _ in range(n_iter)])
        rows.append({
            "horizon": h,
            "n_events": int(k),
            "mean_cond": float(observed),
            "mean_uncond": float(uncond),
            "excess": float(observed - uncond),
            "pct_pos_cond": float((ev > 0).mean()),
            "pct_pos_uncond": float((pool > 0).mean()),
            "p_greater": float((draws >= observed).mean()),
            "p_two_sided": float((np.abs(draws - uncond) >= abs(observed - uncond)).mean()),
        })
    if not rows:
        return pd.DataFrame(columns=cols).set_index("horizon")
    return pd.DataFrame(rows).set_index("horizon")


def excess_vs_alternative(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    alternative: pd.DataFrame,
    horizons=(1, 5, 21),
    n_iter: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """The momentum control: does ``events`` (mentions) beat ``alternative`` (hot streaks)?

    Both event sets are reduced to their conditional mean forward abnormal return at
    each horizon; we report the gap ``mean_mention - mean_alt`` and a permutation
    p-value for it. The permutation pools the two event sets and reshuffles labels,
    so the null is "mentions and hot-streaks draw forward returns from the same
    distribution" — i.e. the mention adds nothing the momentum didn't already imply.

    A mention set that clears :func:`conditional_vs_unconditional` but *not* this is
    attention chasing momentum, not foresight.

    Returns per horizon: ``n_mention, n_alt, mean_mention, mean_alt, gap,
    p_mention_gt_alt``.
    """
    rng = np.random.default_rng(seed)
    cols = ["horizon", "n_mention", "n_alt", "mean_mention", "mean_alt", "gap",
            "p_mention_gt_alt"]
    rows = []
    for h in horizons:
        _, ev = _pool_and_events(panel, events, h)
        _, alt = _pool_and_events(panel, alternative, h)
        if len(ev) == 0 or len(alt) == 0:
            continue
        m_ev, m_alt = ev.mean(), alt.mean()
        gap = m_ev - m_alt
        pooled = np.concatenate([ev, alt])
        ne = len(ev)
        draws = np.empty(n_iter)
        for i in range(n_iter):
            perm = rng.permutation(pooled)
            draws[i] = perm[:ne].mean() - perm[ne:].mean()
        rows.append({
            "horizon": h,
            "n_mention": int(ne),
            "n_alt": int(len(alt)),
            "mean_mention": float(m_ev),
            "mean_alt": float(m_alt),
            "gap": float(gap),
            "p_mention_gt_alt": float((draws >= gap).mean()),
        })
    if not rows:
        return pd.DataFrame(columns=cols).set_index("horizon")
    return pd.DataFrame(rows).set_index("horizon")
