"""The yardstick: does buying the fear gauge beat buying a *random* day — and does
it beat just buying the price drop?

Two questions, because this study has a confound Study 02 didn't: a high/spiking
VIX is nearly the same event as a deep red price day. So a green path after a VIX
event proves nothing twice over — once because the market drifts up anyway, and
once because we may simply have rediscovered the falling knife in vol coordinates.

    conditional   = average forward S&P return *after a VIX event*
    unconditional = average forward S&P return *on any random day*   (null #1)
    excess        = conditional - unconditional

If ``excess`` ≈ 0 the rule is just harvesting drift. We test it with a permutation
test (random baskets of the same size), exactly as in Study 02.

Then the question unique to Study 03 — does the gauge add anything *over the price
drop*? :func:`excess_vs_alternative` pits the VIX signal against a price-trigger
signal on the same forward returns, with a permutation test on the *difference* of
conditional means. If a VIX spike beats a random day but **not** a same-day −3%
close, the "VIX edge" is just the falling knife wearing a volatility hat.

Caveat we surface rather than hide: VIX events cluster hard (4 of Altucher's 23 are
Feb–Mar 2020), so the iid permutation here overstates significance; the block
bootstrap in :mod:`fear_gauge.robustness` stress-tests that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(market: pd.DataFrame, horizon: int) -> np.ndarray:
    """Close-to-close forward S&P return at ``horizon`` days for every bar.

    ``NaN`` for the last ``horizon`` bars (no full forward window).
    """
    close = market["Close"].to_numpy()
    fwd = np.full(len(close), np.nan)
    if horizon < len(close):
        fwd[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return fwd


def _permutation_pvalue(
    fwd: np.ndarray,
    event_mask: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """One- and two-sided p-values for the event mean vs random baskets."""
    valid = ~np.isnan(fwd)
    pool = fwd[valid]
    k = int(event_mask[valid].sum())
    if k == 0 or k > len(pool):
        return float("nan"), float("nan")

    observed = pool[event_mask[valid]].mean() if event_mask[valid].any() else np.nan
    uncond = pool.mean()

    draws = np.empty(n_iter)
    for i in range(n_iter):
        draws[i] = rng.choice(pool, size=k, replace=False).mean()

    p_greater = float((draws >= observed).mean())
    p_two = float((np.abs(draws - uncond) >= abs(observed - uncond)).mean())
    return p_greater, p_two


def conditional_vs_unconditional(
    market: pd.DataFrame,
    signal: pd.Series,
    horizons=(1, 5, 21),
    n_iter: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare event-conditional forward returns to the random-day baseline.

    Default horizons are the chart's +1d / +1w / +1m. For each horizon returns:
    ``n_events, mean_cond, mean_uncond, excess, pct_pos_cond, pct_pos_uncond,
    p_greater, p_two_sided``.

    ``p_greater`` is the headline: the probability a random basket of the same size
    beats the fear-buyer. Small (< 0.05) = the gauge adds something beyond drift;
    near 0.5 = it does not.
    """
    rng = np.random.default_rng(seed)
    sig = signal.reindex(market.index).fillna(False).to_numpy()

    cols = ["horizon", "n_events", "mean_cond", "mean_uncond", "excess",
            "pct_pos_cond", "pct_pos_uncond", "p_greater", "p_two_sided"]
    rows = []
    for h in horizons:
        fwd = forward_returns(market, h)
        valid = ~np.isnan(fwd)
        ev = sig & valid
        pool = fwd[valid]
        cond = fwd[ev]
        if cond.size == 0:
            continue
        p_greater, p_two = _permutation_pvalue(fwd, sig, n_iter, rng)
        rows.append(
            {
                "horizon": h,
                "n_events": int(ev.sum()),
                "mean_cond": float(cond.mean()),
                "mean_uncond": float(pool.mean()),
                "excess": float(cond.mean() - pool.mean()),
                "pct_pos_cond": float((cond > 0).mean()),
                "pct_pos_uncond": float((pool > 0).mean()),
                "p_greater": p_greater,
                "p_two_sided": p_two,
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols).set_index("horizon")
    return pd.DataFrame(rows).set_index("horizon")


def excess_vs_alternative(
    market: pd.DataFrame,
    signal: pd.Series,
    alternative: pd.Series,
    horizons=(1, 5, 21),
    n_iter: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """The cross-study control: does ``signal`` (VIX) beat ``alternative`` (price)?

    Both signals are reduced to their conditional mean forward return at each
    horizon; we report the gap ``mean_signal - mean_alt`` and a permutation p-value
    for it. The permutation pools the two event sets and reshuffles the labels,
    so the null is "the two triggers select forward returns from the same
    distribution" — i.e. the VIX adds nothing the price drop didn't already give.

    Use it with ``alternative`` = a same-day −3% S&P close (Study 02's T1). A VIX
    spike that clears :func:`conditional_vs_unconditional` but *not* this test is
    the falling knife re-expressed in volatility space — not a new edge.

    Returns per horizon: ``n_signal, n_alt, mean_signal, mean_alt, gap,
    p_signal_gt_alt``.
    """
    rng = np.random.default_rng(seed)
    sig = signal.reindex(market.index).fillna(False).to_numpy()
    alt = alternative.reindex(market.index).fillna(False).to_numpy()

    cols = ["horizon", "n_signal", "n_alt", "mean_signal", "mean_alt", "gap",
            "p_signal_gt_alt"]
    rows = []
    for h in horizons:
        fwd = forward_returns(market, h)
        valid = ~np.isnan(fwd)
        s = sig & valid
        a = alt & valid
        if s.sum() == 0 or a.sum() == 0:
            continue
        m_sig = fwd[s].mean()
        m_alt = fwd[a].mean()
        gap = m_sig - m_alt

        # Permute labels over the union of the two event sets.
        union = np.flatnonzero(s | a)
        ns = int(s.sum())
        pooled = fwd[union]
        draws = np.empty(n_iter)
        for i in range(n_iter):
            perm = rng.permutation(pooled)
            draws[i] = perm[:ns].mean() - perm[ns:].mean()
        p = float((draws >= gap).mean())

        rows.append({
            "horizon": h,
            "n_signal": ns,
            "n_alt": int(a.sum()),
            "mean_signal": float(m_sig),
            "mean_alt": float(m_alt),
            "gap": float(gap),
            "p_signal_gt_alt": p,
        })
    if not rows:
        return pd.DataFrame(columns=cols).set_index("horizon")
    return pd.DataFrame(rows).set_index("horizon")
