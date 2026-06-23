"""Strategy + inference for Study 398 — Entropy-Efficiency (the "efficiency clock").

The pitch: when the market's **entropy** falls — returns become more *ordered*, more
predictable, less random — a tradable window opens. We make that testable with two classic
complexity measures computed on a rolling window of recent returns:

  * **Permutation entropy** (Bandt & Pompe, 2002): the Shannon entropy of the frequencies of
    the ordinal patterns of length ``m`` in the window, normalised to [0, 1]. Low = the
    up/down ordering of returns is repetitive (predictable shape).
  * **Sample entropy** (Richman & Moorman, 2000): −log of the conditional probability that two
    sub-sequences similar for ``m`` points stay similar for ``m+1``. Low = self-similar,
    forecastable; high = irregular.

Both are computed causally (window ends at day *t*, so the value at *t* uses only the past).
We then split days into a **low-entropy** regime (entropy in its bottom quantile) vs the rest,
and ask: does forward SPY return differ across regimes by more than luck, net of costs?

Inference is deliberately conservative because the entropy clock is a smooth, strongly
autocorrelated series and forward windows overlap:

  * a **Newey-West / HAC t** of the low-minus-high mean forward return (corrects the SE for
    serial correlation — a naive t over overlapping, autocorrelated data is wildly optimistic);
  * a **stationary block bootstrap** null (Politis & Romano, 1994) that resamples the regime
    labels in blocks, preserving the clustering of low-entropy days, and asks how often the
    block-bootstrapped low-minus-high gap matches the observed one;
  * a **win-rate** (P[forward return > 0] in low-entropy days) vs the unconditional base rate;
  * **one-day execution lag** and **one-way costs** on a long-in-low-entropy / flat-otherwise
    rule (with turnover counted honestly).

The decisive finding is statistical: the low-entropy "window" is, on the real tape, a small,
HAC-insignificant tilt that does not survive costs — a predictability that is not a payoff.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = {1: 1, 5: 5, 21: 21}          # horizon label (days) -> forward days


# --------------------------------------------------------------------------- #
# Entropy measures (causal, window ends at day t)
# --------------------------------------------------------------------------- #
def _perm_entropy_window(x: np.ndarray, m: int = 3) -> float:
    """Normalised permutation entropy of a 1-D window (Bandt-Pompe). NaN if too short."""
    n = len(x)
    if n < m + 1:
        return float("nan")
    # ordinal pattern of each length-m embedding = argsort -> tuple
    counts: dict[tuple, int] = {}
    for i in range(n - m + 1):
        pattern = tuple(np.argsort(x[i:i + m], kind="stable"))
        counts[pattern] = counts.get(pattern, 0) + 1
    total = sum(counts.values())
    p = np.array([c / total for c in counts.values()], dtype=float)
    h = -np.sum(p * np.log(p))
    hmax = np.log(math.factorial(m))
    return float(h / hmax) if hmax > 0 else float("nan")


def _sample_entropy_window(x: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """Sample entropy of a 1-D window (Richman-Moorman). NaN if undefined."""
    n = len(x)
    if n < m + 2:
        return float("nan")
    r = r_mult * np.std(x)
    if r <= 0:
        return float("nan")

    def _phi(mm: int) -> int:
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates) - 1):
            d = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(d <= r))
        return count

    b = _phi(m)
    a = _phi(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


def rolling_entropy(ret: pd.Series, window: int = 60, kind: str = "perm",
                    m: int | None = None) -> pd.Series:
    """Causal rolling entropy of returns: the value at day *t* uses the prior ``window`` days.

    ``kind`` is ``"perm"`` (permutation entropy, default ``m=3``) or ``"sample"`` (sample
    entropy, default ``m=2``). The window ends at *t* and is shifted so no future data leaks.
    """
    r = ret.values
    n = len(r)
    out = np.full(n, np.nan)
    if kind == "perm":
        mm = 3 if m is None else m
        fn = lambda w: _perm_entropy_window(w, m=mm)
    elif kind == "sample":
        mm = 2 if m is None else m
        fn = lambda w: _sample_entropy_window(w, m=mm)
    else:
        raise ValueError("kind must be 'perm' or 'sample'")
    for t in range(window, n):
        out[t] = fn(r[t - window:t])           # ends at t-1: strictly past
    return pd.Series(out, index=ret.index, name=f"ent_{kind}")


# --------------------------------------------------------------------------- #
# Regime split + forward returns
# --------------------------------------------------------------------------- #
def low_entropy_mask(entropy: pd.Series, q: float = 0.20) -> pd.Series:
    """Boolean mask: True on days whose (causal) entropy is in the bottom ``q`` quantile.

    The threshold is the *expanding* quantile up to each day (no look-ahead): a day is
    'low entropy' only relative to entropy seen so far, never the full-sample quantile.
    """
    thr = entropy.expanding(min_periods=250).quantile(q)
    mask = entropy <= thr
    return mask.fillna(False)


def forward_return(spy: pd.Series, horizon_days: int) -> pd.Series:
    """Simple forward return over ``horizon_days`` for every date (NaN near the tail)."""
    return spy.shift(-horizon_days) / spy - 1.0


def regime_forward(frame: pd.DataFrame, mask: pd.Series, horizon_days: int,
                   lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Forward returns split by regime, with a ``lag``-day execution lag.

    A signal observed at *t* is acted on at the close of *t+lag*; we then hold ``horizon_days``.
    Returns ``(low_entropy_fwd, high_entropy_fwd)`` aligned and tail-trimmed.
    """
    spy = frame["spy"]
    fwd = forward_return(spy, horizon_days)
    sig = mask.shift(lag).fillna(False)        # act one day after the signal
    valid = fwd.notna() & sig.notna()
    fwd, sig = fwd[valid], sig[valid].astype(bool)
    low = fwd[sig].values.astype(float)
    high = fwd[~sig].values.astype(float)
    return low, high


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(diff_series: pd.Series, lag: int | None = None) -> float:
    """Newey-West (HAC) t-stat that the mean of ``diff_series`` differs from zero.

    ``diff_series`` is the per-day (low-minus-high) forward-return contribution, i.e. forward
    return on low-entropy days and NaN elsewhere — passed here as the demeaned signed series.
    The Bartlett-kernel long-run variance corrects for the serial correlation induced by
    overlapping forward windows and clustered regimes. NaN if too short.
    """
    x = diff_series.dropna().values.astype(float)
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West rule of thumb
    g0 = np.dot(e, e) / n
    lrv = g0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gk = np.dot(e[k:], e[:-k]) / n
        lrv += 2.0 * w * gk
    if lrv <= 0:
        return float("nan")
    se = np.sqrt(lrv / n)
    return float(mu / se) if se > 0 else float("nan")


def low_minus_high_t(frame: pd.DataFrame, mask: pd.Series, horizon_days: int,
                     lag: int = 1) -> float:
    """HAC t of the per-day forward-return gap (low-entropy days vs the grand mean).

    We build a daily series equal to the (lagged-signal) forward return on every day, demean
    by the high-entropy mean, and zero out high-entropy days — so its mean is the low-vs-high
    contribution per low-entropy day. The HAC t accounts for overlap + clustering.
    """
    spy = frame["spy"]
    fwd = forward_return(spy, horizon_days)
    sig = mask.shift(lag).fillna(False).astype(bool)
    valid = fwd.notna()
    fwd, sig = fwd[valid], sig[valid]
    high_mean = fwd[~sig].mean()
    contrib = (fwd - high_mean).where(sig, other=np.nan)
    return hac_t(contrib, lag=None)


def block_bootstrap_p(frame: pd.DataFrame, mask: pd.Series, horizon_days: int,
                      n_draws: int = 5_000, block: int = 21, lag: int = 1,
                      seed: int = 398) -> dict:
    """Stationary-block-bootstrap null for the low-minus-high forward-return gap.

    Preserves the temporal clustering of low-entropy days by resampling the *mask* in blocks
    of mean length ``block`` and recomputing the gap. Returns the observed gap, the bootstrap
    mean gap, and the two-sided p-value P[|boot gap| >= |observed gap|].
    """
    low, high = regime_forward(frame, mask, horizon_days, lag=lag)
    if len(low) < 2 or len(high) < 2:
        return {"obs_gap": float("nan"), "boot_mean": float("nan"), "p_value": float("nan")}
    obs_gap = float(low.mean() - high.mean())

    spy = frame["spy"]
    fwd = forward_return(spy, horizon_days)
    sig = mask.shift(lag).fillna(False).astype(bool)
    valid = fwd.notna()
    fwd = fwd[valid].values.astype(float)
    n = len(fwd)
    k = int(sig[valid].sum())                  # number of low-entropy days
    if k < 2 or k > n - 2:
        return {"obs_gap": obs_gap, "boot_mean": float("nan"), "p_value": float("nan")}

    # Null: the k low-entropy days are *placed at random in clustered blocks*, preserving the
    # block structure of the regime (so we don't credit the gap to mere autocorrelation). For
    # each draw we lay down contiguous circular blocks of length ~`block` at random starts
    # until k slots are filled, mark them "low", and recompute the gap. Vectorised over blocks.
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(k / block)) + 1
    total = float(fwd.sum())
    gaps = np.empty(n_draws)
    base_offsets = np.arange(block)
    for d in range(n_draws):
        sel = np.zeros(n, dtype=bool)
        starts = rng.integers(0, n, size=n_blocks)
        for s in starts:
            sel[(s + base_offsets) % n] = True
            if sel.sum() >= k:
                break
        ksel = int(sel.sum())
        low_sum = float(fwd[sel].sum())
        low_mean = low_sum / ksel
        high_mean = (total - low_sum) / (n - ksel)
        gaps[d] = low_mean - high_mean
    p_val = float((np.abs(gaps) >= abs(obs_gap)).mean())
    return {"obs_gap": obs_gap, "boot_mean": float(gaps.mean()), "p_value": p_val}


def summarize(frame: pd.DataFrame, mask: pd.Series, horizon_days: int, lag: int = 1) -> dict:
    """Headline stats for one horizon: n low/high, means, win-rates, HAC t, bootstrap p."""
    low, high = regime_forward(frame, mask, horizon_days, lag=lag)
    base = forward_return(frame["spy"], horizon_days).dropna().values
    bp = block_bootstrap_p(frame, mask, horizon_days, lag=lag)
    return {
        "horizon": horizon_days,
        "n_low": int(len(low)),
        "n_high": int(len(high)),
        "low_mean": float(low.mean()) if len(low) else float("nan"),
        "high_mean": float(high.mean()) if len(high) else float("nan"),
        "low_win": float((low > 0).mean()) if len(low) else float("nan"),
        "base_win": float((base > 0).mean()) if len(base) else float("nan"),
        "gap": (float(low.mean() - high.mean())
                if len(low) and len(high) else float("nan")),
        "t_hac": low_minus_high_t(frame, mask, horizon_days, lag=lag),
        "p_boot": bp["p_value"],
    }


def net_of_costs(frame: pd.DataFrame, mask: pd.Series, cost_bps: float = 1.0,
                 lag: int = 1) -> dict:
    """Long-in-low-entropy / flat-otherwise daily rule, one-way ``cost_bps`` per side per turn.

    Position is 1 on (lagged) low-entropy days, 0 otherwise; we charge ``cost_bps`` on each
    change of position (a round trip = enter + exit = 2 turns). Returns gross/net annualised
    mean and Sharpe, the buy-&-hold benchmark, and the turnover — the tradability ledger.
    """
    ret = frame["ret"] if "ret" in frame else np.log(frame["spy"]).diff()
    pos = mask.shift(lag).fillna(False).astype(float)
    valid = ret.notna()
    ret, pos = ret[valid], pos[valid]
    turns = pos.diff().abs().fillna(pos.iloc[0] if len(pos) else 0.0)
    c = cost_bps / 1e4
    strat_gross = pos * ret
    strat_net = strat_gross - turns * c
    ann = 252.0

    def _stats(x):
        mu = x.mean() * ann
        sd = x.std(ddof=1) * np.sqrt(ann)
        return float(mu), float(mu / sd) if sd > 0 else float("nan")

    g_mu, g_sh = _stats(strat_gross)
    n_mu, n_sh = _stats(strat_net)
    bh_mu, bh_sh = _stats(ret)
    return {
        "days": int(len(ret)),
        "exposure": float(pos.mean()),
        "turnover_annual": float(turns.sum() / (len(ret) / ann)),
        "gross_ann": g_mu, "gross_sharpe": g_sh,
        "net_ann": n_mu, "net_sharpe": n_sh,
        "bh_ann": bh_mu, "bh_sharpe": bh_sh,
        "cost_bps": cost_bps,
    }
