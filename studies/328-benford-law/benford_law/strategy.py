"""The Benford engine and its honest forensic test — Study 328 (Benford-Law).

The folklore, steelmanned: leading-digit (Benford) analysis is a real forensic tool —
auditors use first-digit frequencies to flag fabricated accounting figures (Nigrini),
and a string of papers run it on macro statistics and reported financials. The viral
trading version is the leap from there: *"if a stock's prices deviate from Benford's
Law, something is wrong with it — and a deviation score should predict trouble (lower
forward returns)."*

This module supplies:

1. ``benford_pmf`` / ``leading_digits`` / ``digit_frequencies`` — the textbook law and
   the empirical first-digit histogram of any positive series.
2. ``chi_square`` and ``mad`` — the two standard goodness-of-fit statistics (Pearson
   chi-square against the Benford pmf, and Nigrini's Mean Absolute Deviation).
3. ``rolling_benford_deviation`` — a per-name, point-in-time deviation score computed on
   a trailing window of *prices* (the candidate "forensic signal").
4. ``forensic_backtest`` — the honest test of the trading claim: does a high Benford
   deviation predict *lower* forward returns? We sort names by their trailing deviation
   and measure the forward-return spread (high-deviation minus low-deviation), with a
   Newey-West HAC *t* on the spread. One execution lag, costs one-way × NAV.

A note on what Benford can and cannot see: the first-digit law is a statement about a
quantity that ranges over orders of magnitude. *Prices over decades* qualify; *daily
returns* do not (they are tightly clustered around zero, so their leading digit is
dominated by whatever the first significant figure of a ~1% move is). So we expect
prices to conform and returns to fail — and neither fact is, by itself, tradable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Benford first-digit probabilities, d = 1..9.
_DIGITS = np.arange(1, 10)
BENFORD_PMF = np.log10(1.0 + 1.0 / _DIGITS)


# ---------------------------------------------------------------------------
# The law and the empirical histogram
# ---------------------------------------------------------------------------
def benford_pmf() -> np.ndarray:
    """The Benford first-digit probabilities for digits 1..9 (sums to 1)."""
    return BENFORD_PMF.copy()


def leading_digits(values: np.ndarray | pd.Series) -> np.ndarray:
    """The leading (most-significant) decimal digit of each positive value.

    Zeros / non-finite / non-positive values are dropped (Benford is defined on
    positive magnitudes). The leading digit of ``x`` is ``floor(x / 10**floor(log10 x))``.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = np.abs(x)
    x = x[x > 0]
    if x.size == 0:
        return np.array([], dtype=int)
    exponent = np.floor(np.log10(x))
    lead = np.floor(x / np.power(10.0, exponent)).astype(int)
    # Guard against float edge cases pushing a digit to 0 or 10.
    lead = np.clip(lead, 1, 9)
    return lead


def digit_frequencies(values: np.ndarray | pd.Series) -> np.ndarray:
    """Empirical first-digit frequencies (length-9 vector, sums to 1) of ``values``.

    Returns all-NaN if there are no usable positive values.
    """
    lead = leading_digits(values)
    if lead.size == 0:
        return np.full(9, np.nan)
    counts = np.array([(lead == d).sum() for d in _DIGITS], dtype=float)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Goodness-of-fit statistics
# ---------------------------------------------------------------------------
def chi_square(values: np.ndarray | pd.Series) -> float:
    """Pearson chi-square of the first-digit histogram against the Benford pmf.

    Large values mean a poor fit to Benford. NaN if there is no usable data.
    """
    lead = leading_digits(values)
    n = lead.size
    if n == 0:
        return float("nan")
    obs = np.array([(lead == d).sum() for d in _DIGITS], dtype=float)
    exp = BENFORD_PMF * n
    return float(np.sum((obs - exp) ** 2 / exp))


def mad(values: np.ndarray | pd.Series) -> float:
    """Nigrini's Mean Absolute Deviation between observed and Benford frequencies.

    The forensic-accounting standard. Nigrini's rule-of-thumb thresholds for first-digit
    MAD: < 0.006 "close conformity", 0.006–0.012 "acceptable", 0.012–0.015 "marginal",
    > 0.015 "nonconformity". Scale-free in sample size, which is why forensics prefers
    it to chi-square. NaN if there is no usable data.
    """
    freq = digit_frequencies(values)
    if np.all(np.isnan(freq)):
        return float("nan")
    return float(np.mean(np.abs(freq - BENFORD_PMF)))


# ---------------------------------------------------------------------------
# The candidate "forensic signal" — a rolling per-name deviation score
# ---------------------------------------------------------------------------
def _lead_digit_array(vals: np.ndarray) -> np.ndarray:
    """Leading digit (1..9) per element; 0 marks a non-usable (NaN/<=0) value."""
    out = np.zeros(vals.shape, dtype=np.int8)
    ok = np.isfinite(vals) & (vals > 0)
    x = np.abs(vals[ok])
    exponent = np.floor(np.log10(x))
    lead = np.floor(x / np.power(10.0, exponent)).astype(int)
    out[ok] = np.clip(lead, 1, 9)
    return out


def _deviation_from_counts(counts: np.ndarray, statistic: str) -> float:
    """Goodness-of-fit ('mad' or 'chi2') from a length-9 digit-count vector."""
    n = counts.sum()
    if n == 0:
        return float("nan")
    if statistic == "mad":
        freq = counts / n
        return float(np.mean(np.abs(freq - BENFORD_PMF)))
    exp = BENFORD_PMF * n
    return float(np.sum((counts - exp) ** 2 / exp))


def rolling_benford_deviation(
    price: pd.Series,
    window: int = 252,
    statistic: str = "mad",
) -> pd.Series:
    """Trailing-window Benford-deviation score of a price series, stamped at each close.

    At each bar ``t`` we take the trailing ``window`` *prices* (inclusive of ``t``) and
    compute the chosen goodness-of-fit ``statistic`` ('mad' or 'chi2') against Benford.
    The value at ``t`` uses only information available at the close of ``t`` — so it can
    be used to form a position for ``t+1`` with one execution lag (applied in
    ``forensic_backtest``). Higher = further from Benford.

    Vectorised via a rolling digit histogram (one-hot leading digits summed over a
    sliding window through cumulative counts), so it is fast enough for wide panels.
    """
    vals = price.to_numpy(dtype=float)
    lead = _lead_digit_array(vals)              # 0 = unusable, else 1..9
    n = len(vals)
    # one-hot over digits 1..9, shape (n, 9)
    onehot = np.zeros((n, 9), dtype=np.int32)
    usable = lead > 0
    onehot[np.arange(n)[usable], lead[usable] - 1] = 1
    csum = np.vstack([np.zeros(9, dtype=np.int64), np.cumsum(onehot, axis=0)])
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        counts = csum[i + 1] - csum[i + 1 - window]
        out[i] = _deviation_from_counts(counts.astype(float), statistic)
    return pd.Series(out, index=price.index, name=f"benford_{statistic}")


# ---------------------------------------------------------------------------
# Honest inference helper — Newey-West HAC t on a return series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray | pd.Series) -> float:
    """Newey-West (Bartlett-kernel) HAC t-statistic on the mean of ``x``.

    Lag selection follows the standard ``floor(4 (n/100)^(2/9))`` rule. No hard
    dependency on quantlab being importable, so tests run anywhere.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    x: np.ndarray | pd.Series,
    block: int = 21,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 328,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``x`` (preserves autocorrelation)."""
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = r[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# ---------------------------------------------------------------------------
# The trading claim under test — does Benford deviation predict returns?
# ---------------------------------------------------------------------------
def forensic_backtest(
    prices: pd.DataFrame,
    window: int = 252,
    rebal: int = 21,
    quantile: float = 0.3,
    cost_bps: float = 0.0,
    statistic: str = "mad",
) -> dict:
    """Test whether a high trailing Benford deviation predicts *lower* forward returns.

    ``prices`` is a wide DataFrame (Date index × tickers) of close levels. Every
    ``rebal`` business days we rank the cross-section by each name's trailing-``window``
    Benford-deviation score, go **short** the top ``quantile`` (most "deviant", the
    forensic red-flag names) and **long** the bottom ``quantile`` (most Benford-conforming),
    equal-weight, then hold to the next rebalance. The forensic claim predicts a positive
    payoff for this long-conforming / short-deviant book.

    Honest mechanics:

    - **One execution lag.** The score at the close of the formation day decides the book
      held over the *following* period — we form on day ``t`` and earn the simple returns
      from ``t+1`` onward (a single shift, applied once).
    - **Costs one-way × NAV.** ``cost_bps`` is charged on the one-way turnover at each
      rebalance (a full long+short rebuild is ~2× NAV one-way). Shorts are part of the
      book, so the short leg pays the same spread; an explicit borrow is out of scope for
      this forensic screen (documented, not hidden).
    - **Excess-of-nothing.** The book is dollar-neutral (long minus short), so its return
      is already an excess-of-market spread; no separate cash leg to net out.

    Returns a dict with the period-return series, the HAC *t*, a block-bootstrap CI, the
    annualised mean and the number of rebalances.
    """
    px = prices.sort_index()
    rets = px.pct_change()
    cols = list(px.columns)
    dev = pd.DataFrame(
        {c: rolling_benford_deviation(px[c].dropna(), window=window, statistic=statistic)
         for c in cols}
    ).reindex(px.index)

    dates = px.index
    form_days = list(range(window, len(dates) - 1, rebal))
    period_rets = []
    period_index = []
    n_short = 0
    for k, t in enumerate(form_days):
        score = dev.iloc[t].dropna()
        if score.size < 4:
            continue
        hi_cut = score.quantile(1 - quantile)
        lo_cut = score.quantile(quantile)
        short = score[score >= hi_cut].index      # most deviant → short
        long = score[score <= lo_cut].index        # most conforming → long
        if len(short) == 0 or len(long) == 0:
            continue
        n_short += len(short)
        end = form_days[k + 1] if k + 1 < len(form_days) else len(dates) - 1
        # One lag: positions known at close of t earn t+1 .. end.
        win = rets.iloc[t + 1 : end + 1]
        long_r = win[long].mean(axis=1).fillna(0.0)
        short_r = win[short].mean(axis=1).fillna(0.0)
        gross = (long_r - short_r).sum()           # compounded≈summed over short window
        # one-way turnover ~ 2x NAV at each full rebuild (long side + short side)
        cost = 2.0 * cost_bps * 1e-4
        period_rets.append(gross - cost)
        period_index.append(dates[end])

    if not period_rets:
        return {
            "n_periods": 0, "mean_period": float("nan"), "ann_mean": float("nan"),
            "tstat": float("nan"), "ci": (float("nan"), float("nan")),
            "rets": pd.Series(dtype=float),
        }
    r = pd.Series(period_rets, index=pd.DatetimeIndex(period_index), name="long_conf_short_dev")
    periods_per_year = TRADING_DAYS_PER_YEAR / rebal
    return {
        "n_periods": int(len(r)),
        "mean_period": float(r.mean()),
        "ann_mean": float(r.mean() * periods_per_year),
        "tstat": hac_tstat(r),
        "ci": block_bootstrap_ci(r, block=6),
        "rets": r,
    }


TRADING_DAYS_PER_YEAR = 252
