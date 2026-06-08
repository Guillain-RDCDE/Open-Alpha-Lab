"""Robustness: the ways a "follow the oracle" result can be a mirage.

1. **Clustering in time.** Mentions arrive in hype *waves* — a meme week throws off
   thirty calls that are really one bet on one theme. The iid permutation in
   :mod:`social_oracle.benchmark` overstates significance; :func:`block_bootstrap_excess`
   resamples contiguous calendar blocks so the wave structure is preserved, for an
   honest CI on the excess.

2. **Concentration in a few names.** One lucky 10-bagger can carry an entire feed's
   apparent edge. :func:`name_jackknife` drops the single most-traded name and
   recomputes — if the excess collapses, you discovered a stock, not a skill.

3. **The fade itself.** :func:`fade_curve` traces the mean abnormal CAR session by
   session, so the pop-then-fade reads off directly: up into a short-horizon peak,
   then the bleed that turns the follower's trade negative.

4. **Data-mining the knobs.** Hold period, lookback, cooldown — try enough and one
   cell shines. :func:`deflated_sharpe` discounts the best Sharpe for the number of
   configurations tried, the same Bailey–López de Prado deflation used in Study 03.

Plus :func:`split_sample`, a chronological in/out-of-sample cut: an edge that lives
only in the first half of the feed's history was overfit to it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .benchmark import car_forward, _pool_and_events
from .eventstudy import event_study


# --- Standard-normal helpers (kept dependency-free; no scipy needed) --------- #

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation (|err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0.0 else float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _event_set(events: pd.DataFrame) -> set[tuple[str, int]]:
    return {(t, int(p)) for t, p in zip(events["ticker"], events["entry_pos"])}


def block_bootstrap_excess(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizon: int = 21,
    block: int = 21,
    n_iter: int = 2000,
    seed: int = 0,
) -> dict:
    """Block-bootstrap CI for the conditional excess abnormal return.

    Builds the universe of all valid ``(name, day)`` forward abnormal returns,
    ordered by calendar date and flagged as mention-event or not, then resamples
    contiguous blocks (preserving hype-wave clustering) and recomputes the excess
    (mean_event - mean_pool) on each synthetic history. A CI that straddles zero
    means the apparent edge is within sampling noise once clustering is respected.

    Returns ``mean, ci_low, ci_high, p_excess_le_0``.
    """
    eset = _event_set(events)
    recs = []
    for t, frame in panel.items():
        car = car_forward(frame, horizon)
        dates = frame.index
        for p in range(len(frame)):
            v = car[p]
            if np.isnan(v):
                continue
            recs.append((dates[p], v, (t, p) in eset))
    if not recs:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_excess_le_0": np.nan}

    recs.sort(key=lambda r: r[0])
    fwd = np.array([r[1] for r in recs])
    sig = np.array([r[2] for r in recs], dtype=bool)
    n = len(fwd)
    if sig.sum() == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_excess_le_0": np.nan}

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    excesses = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        f, s = fwd[idx], sig[idx]
        excesses[i] = f[s].mean() - f.mean() if s.sum() else np.nan

    excesses = excesses[~np.isnan(excesses)]
    return {
        "mean": float(np.mean(excesses)),
        "ci_low": float(np.percentile(excesses, 2.5)),
        "ci_high": float(np.percentile(excesses, 97.5)),
        "p_excess_le_0": float((excesses <= 0).mean()),
    }


def fade_curve(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizons=(1, 2, 3, 5, 10, 21),
) -> pd.DataFrame:
    """Mean abnormal CAR at each forward horizon — the pop-then-fade, in one table.

    Reads the post-event leg of the event-study matrix. A row pattern of a positive,
    rising mean that peaks early and then *declines* is the signature: the follower
    who buys after the pop is buying the start of the bleed. Columns: ``mean_car,
    pct_positive, tstat, n``.
    """
    h_max = max(horizons)
    es = event_study(panel, events, horizon=h_max, pre=0)
    summ = es["summary"]
    rows = []
    for h in horizons:
        if h in summ.index:
            r = summ.loc[h]
            rows.append({"horizon": h, "mean_car": r["mean"], "pct_positive": r["pct_positive"],
                         "tstat": r["tstat"], "n": int(r["n"])})
    return pd.DataFrame(rows).set_index("horizon")


def name_jackknife(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    horizon: int = 5,
    top: int = 3,
) -> pd.DataFrame:
    """Drop the most-mentioned names one at a time; watch the excess move.

    Recomputes the mean conditional forward abnormal return with the full event set,
    then with each of the ``top`` most-frequent names removed. If pulling one name
    collapses the mean, the "edge" was that name's idiosyncratic run, not a
    repeatable response to being mentioned. Columns: ``n_events, mean_cond``.
    """
    def mean_cond(ev: pd.DataFrame) -> tuple[int, float]:
        _, vals = _pool_and_events(panel, ev, horizon)
        return len(vals), (float(vals.mean()) if len(vals) else np.nan)

    rows = []
    n, m = mean_cond(events)
    rows.append({"dropped": "(none)", "n_events": n, "mean_cond": m})
    for name in events["ticker"].value_counts().head(top).index:
        sub = events[events["ticker"] != name]
        n, m = mean_cond(sub)
        rows.append({"dropped": name, "n_events": n, "mean_cond": m})
    return pd.DataFrame(rows).set_index("dropped")


def deflated_sharpe(best_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Probabilistic Sharpe deflated for the number of configurations tried.

    A simplified Bailey–López de Prado deflated Sharpe: the probability that the best
    observed (annualised) Sharpe over ``n_trials`` configs would NOT have arisen by
    chance under a true Sharpe of zero, given ``n_obs`` return observations. Near 1 =
    unlikely a fluke; near 0.5 or below = consistent with picking the luckiest of
    many coin-flips.
    """
    if n_trials < 1 or n_obs < 2 or not np.isfinite(best_sharpe):
        return np.nan
    euler = 0.5772156649
    z = _norm_ppf(1 - 1.0 / n_trials) * (1 - euler) + _norm_ppf(1 - 1.0 / (n_trials * np.e)) * euler
    sr_daily = best_sharpe / np.sqrt(252)
    expected_max_daily = z / np.sqrt(n_obs)
    deflated = (sr_daily - expected_max_daily) * np.sqrt(n_obs)
    return float(_norm_cdf(deflated))


def split_sample(events: pd.DataFrame, frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological in-sample / out-of-sample split of the events (no shuffling).

    Pick the best knobs on the first ``frac`` of the feed's history, confirm on the
    held-out tail. An excess that evaporates out-of-sample was overfit to the window.
    """
    ev = events.sort_values("entry_date")
    cut = int(len(ev) * frac)
    return ev.iloc[:cut].copy(), ev.iloc[cut:].copy()
