"""Strategy + inference for Study 744 — Tetraphobia.

Two independent tests, two units of inference, one honest rail through each.

**A · Price clustering (the myth-check / third axis).** For each basket, count the
*trailing digit* of every daily raw close (the price expressed to its finest resolution,
``round(price * 100) mod 10`` — the last digit a trader actually chose). The universal
round-number effect over-weights digits 0 and 5 in *every* market, so that is not
tetraphobia and is excluded from the test. Tetraphobia is a specific **4-vs-8
asymmetry**: among prices ending in 4 or 8, is 8 over-represented? Under the null (no
superstition) exactly half should end in 8, so ``z = (n8 - n4) / sqrt(n4 + n8)`` is a
one-proportion z-test. A second statistic, ``z4``, asks whether digit 4 sits *below* the
mean of the eight non-round digits. The US basket is the placebo: the effect must appear
in Asia and be **absent in the US**, or it is generic microstructure, not culture.

**B · The 4/4 calendar-returns test (Signal + Tradability).** 4 April is a fixed,
publicly-known calendar date, so the "does the unlucky day underperform" question has
**no look-ahead and needs no execution lag** — the tested quantity is the return of the
4/4 session itself (snapped forward to the first session on/after 4/4 when it falls on a
weekend/holiday — the single, documented calendar convention). Each year is one
independent, non-overlapping event, so the primary statistic is a **one-sample t** of
the 4/4 return across years (per market and pooled across the three core China-sphere
ETFs), with a Wilson hit-rate and a multi-seed random-calendar placebo. 8/8 (the "lucky"
date) is the natural contrast. Tradability shorts the 4/4 session (betting on the
predicted underperformance) net of round-trip costs and one day's borrow.

Costs are one-way x NAV per leg; the short leg pays borrow. Price-clustering numbers are
price-only by construction (raw traded prices); calendar returns are total-return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

COST_BPS = 5.0     # one-way, per leg
BORROW_BPS = 1.0   # one overnight of borrow on the short leg
NONROUND = (1, 2, 3, 4, 6, 7, 8, 9)   # digits not dominated by round-number clustering


# --------------------------------------------------------------------------- #
# A · Trailing-digit clustering
# --------------------------------------------------------------------------- #
def trailing_digit_counts(closes: dict[str, pd.Series],
                          only: set[str] | None = None) -> np.ndarray:
    """Count trailing digits 0..9 across every daily close in ``closes``.

    Trailing digit = ``round(price * 100) mod 10`` — the last digit of the price at its
    finest (cent / minor-unit) resolution. ``only`` optionally restricts to a subset of
    tickers (e.g. one region). Won-priced tickers (no sub-unit digit) contribute only
    digit 0 and so add nothing to the 4-vs-8 test — a fact, not a bias.
    """
    counts = np.zeros(10, dtype=np.int64)
    for t, s in closes.items():
        if only is not None and t not in only:
            continue
        digs = np.round(s.to_numpy(dtype=float) * 100).astype(np.int64) % 10
        counts += np.bincount(digs, minlength=10)
    return counts


def tetraphobia_stats(counts: np.ndarray) -> dict:
    """Turn a digit-count vector into the tetraphobia statistics.

    * ``z8_gt_4`` — one-proportion z-test that, among prices ending 4 or 8, 8 is
      over-represented (null share 0.5). The headline.
    * ``z4`` / ``z8`` — z of digit 4 / 8 against the mean count of the eight non-round
      digits (is 4 a low outlier, 8 a high one?).
    * ``share8`` — n8 / (n4 + n8).
    * ``chi2_nonround`` — chi-square of the eight non-round digits vs uniform (df = 7),
      the overall "are the non-round digits evenly used" test.
    """
    n4, n8 = int(counts[4]), int(counts[8])
    tot48 = n4 + n8
    z8_gt_4 = (n8 - n4) / np.sqrt(tot48) if tot48 > 0 else float("nan")
    share8 = n8 / tot48 if tot48 > 0 else float("nan")
    nr = np.array([counts[d] for d in NONROUND], dtype=float)
    mean_nr = nr.mean()
    z4 = (counts[4] - mean_nr) / np.sqrt(mean_nr) if mean_nr > 0 else float("nan")
    z8 = (counts[8] - mean_nr) / np.sqrt(mean_nr) if mean_nr > 0 else float("nan")
    chi2 = float(((nr - mean_nr) ** 2 / mean_nr).sum()) if mean_nr > 0 else float("nan")
    return {"n4": n4, "n8": n8, "total": int(counts.sum()),
            "share8": float(share8), "z8_gt_4": float(z8_gt_4),
            "z4": float(z4), "z8": float(z8), "chi2_nonround": chi2}


def digit_pct(counts: np.ndarray) -> np.ndarray:
    """Digit counts as percentages (for the notebook bar chart)."""
    tot = counts.sum()
    return counts / tot * 100.0 if tot > 0 else counts.astype(float)


def region_tetraphobia(closes: dict[str, pd.Series]) -> dict[str, dict]:
    """``tetraphobia_stats`` computed separately for each Asian region tag + the US
    control — the cut that shows the effect tracks the strength of the superstition."""
    out = {}
    regions = sorted(set(dt.ASIA_CLUSTER.values()))
    for r in regions:
        tickers = {t for t, rg in dt.ASIA_CLUSTER.items() if rg == r}
        out[r] = tetraphobia_stats(trailing_digit_counts(closes, only=tickers))
    out["US"] = tetraphobia_stats(trailing_digit_counts(closes, only=set(dt.US_CONTROL)))
    return out


# --------------------------------------------------------------------------- #
# Inference primitives (shared)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 — the right unit for independent, non-overlapping
    yearly calendar events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances) — used for the 8/8-vs-4/4
    contrast. NaN if either group has < 2 finite obs."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hit_rate(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = len(x); k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# B · The 4/4 calendar-returns test
# --------------------------------------------------------------------------- #
def date_session_returns(close: pd.Series, month: int, day: int,
                         years=dt.CAL_YEARS) -> list[tuple[pd.Timestamp, float]]:
    """The close-to-close return of the first session on/after ``month/day`` each year.

    ``month/day`` is a fixed calendar date known years in advance, so there is no
    look-ahead and no execution lag: the tested return is that session's own return.
    A weekend/holiday date snaps forward to the next session (the one documented
    calendar convention). Returns [(session_date, return), ...].
    """
    r = close.pct_change()
    idx = close.index
    out = []
    for y in years:
        pos = idx.searchsorted(pd.Timestamp(f"{y}-{month:02d}-{day:02d}"))
        if pos <= 0 or pos >= len(idx):
            continue
        val = float(r.iloc[pos])
        if np.isfinite(val):
            out.append((idx[pos], val))
    return out


def market_date_stats(prices: dict[str, pd.Series], month: int, day: int,
                      tickers=None) -> dict:
    """One-sample t of the ``month/day`` session return, per market and pooled."""
    tickers = tickers or dt.CALENDAR_CORE
    per_market, pooled = {}, []
    for t in tickers:
        if t not in prices:
            continue
        rets = [v for _, v in date_session_returns(prices[t], month, day)]
        per_market[t] = {**one_sample_t(np.array(rets)), **hit_rate(np.array(rets))}
        pooled += rets
    per_market["POOLED"] = {**one_sample_t(np.array(pooled)),
                            **hit_rate(np.array(pooled))}
    return per_market


def pooled_returns(prices: dict[str, pd.Series], month: int, day: int,
                   tickers=None) -> np.ndarray:
    tickers = tickers or dt.CALENDAR_CORE
    out = []
    for t in tickers:
        if t in prices:
            out += [v for _, v in date_session_returns(prices[t], month, day)]
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Random-calendar placebo (multi-seed): is the 4/4 mean unusual vs random days?
# --------------------------------------------------------------------------- #
def placebo_pvalue(prices: dict[str, pd.Series], observed: float, n_per_market: dict,
                   tickers=None, n_seeds: int = 20, n_draws_per_seed: int = 250,
                   base_seed: int = 744, tail: str = "left") -> dict:
    """Draw, for each market, ``n`` random session returns (n = that market's 4/4 event
    count), average across markets exactly as the observed pooled mean does, repeat
    ``n_seeds x n_draws_per_seed`` times. ``tail='left'`` tests the folklore's predicted
    UNDERperformance (p = share of null means <= observed)."""
    tickers = tickers or dt.CALENDAR_CORE
    daily = {t: prices[t].pct_change().dropna().to_numpy() for t in tickers if t in prices}
    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = []
            for t, arr in daily.items():
                k = n_per_market.get(t, 0)
                if k <= 0 or len(arr) <= k:
                    continue
                locs = rng.integers(0, len(arr), size=k)
                vals.extend(arr[locs].tolist())
            if vals:
                means.append(float(np.mean(vals)))
    means = np.asarray(means)
    p = float((means <= observed).mean()) if tail == "left" else float((means >= observed).mean())
    return {"obs": observed, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)), "p_value": p,
            "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Tradability — short the 4/4 session (bet on the predicted underperformance)
# --------------------------------------------------------------------------- #
def short_the_unlucky_day(prices: dict[str, pd.Series], month: int = 4, day: int = 4,
                          tickers=None, cost_bps: float = COST_BPS,
                          borrow_bps: float = BORROW_BPS) -> dict:
    """Enter a short at the prior close, cover at the ``month/day`` session close: the
    position earns MINUS that session's return, less one round trip (2x one-way cost)
    and one day's borrow. A positive mean would mean the day really did fall; a negative
    mean means the short bled money (the day tended to rise)."""
    tickers = tickers or dt.CALENDAR_CORE
    pnl = []
    for t in tickers:
        if t not in prices:
            continue
        for _, r in date_session_returns(prices[t], month, day):
            gross = -r
            net = gross - 2.0 * cost_bps / 1e4 - borrow_bps / 1e4
            pnl.append(net)
    pnl = np.asarray(pnl, dtype=float)
    s = one_sample_t(pnl)
    return {**s, "gross_mean": float((-pooled_returns(prices, month, day, tickers)).mean())}


# --------------------------------------------------------------------------- #
# Synthetic positive controls (machinery proofs — never cited for a real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_digit_detect(bias: float, seed: int = 744) -> dict:
    """Run the trailing-digit detector on a synthetic digit stream with a planted 4->8
    bias. bias = 0 must not fire; a positive bias must light up z8_gt_4."""
    counts = np.bincount(dt.synthetic_digits(bias=bias, seed=seed), minlength=10)
    return tetraphobia_stats(counts)


def synthetic_calendar_detect(dip: float, seed: int = 744) -> dict:
    """Run the 4/4 one-sample-t detector on a synthetic tape with a planted 4/4 dip.
    dip = 0 must not fire; a negative dip must produce a significant negative t."""
    close, _ = dt.synthetic_calendar(dip=dip, seed=seed)
    rets = [v for _, v in date_session_returns(close, 4, 4)]
    return one_sample_t(np.asarray(rets))
