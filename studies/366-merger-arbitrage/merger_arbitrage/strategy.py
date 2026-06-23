"""Strategy + inference for Study 366 — Merger-Arbitrage (the deal spread).

The pitch: after an all-cash takeover is announced, the target trades a few percent
**below** the offer. Buy it, wait for the deal to close, collect the spread — a steady,
market-neutral coupon. The catch the pitch buries: when a deal **breaks**, the target snaps
back to its un-bid standalone level, a 20–40% loss in a day. Merger-arb is therefore *short
a deal-break put* — you sell insurance against deal failure, and the spread is the premium.

We test whether the spread is *more* than fair compensation for that break tail. The engine:

  * **Per-deal realized arb return.** ``(offer/entry - 1)`` if the deal closes, the
    snap-back loss if it breaks. Entry is the close **1 day after** the announcement
    (no look-ahead — you buy after the deal is public).
  * **Welch t** of the mean realized arb return against zero (is the book net positive?).
  * **A bootstrap / placebo null.** Resample the deal book with replacement many times to
    get the sampling distribution of the mean — the honest small-sample test, because a
    handful of breaks dominates the mean. The placebo p-value is P[bootstrap mean <= 0].
  * **Win-rate vs base rate.** The fraction of deals that pay (close) — a *high* win-rate
    is the whole illusion, because the few losers are enormous (negative skew).
  * **One-way costs × turnover.** Each deal is one round-trip; charge a one-way cost on
    entry and on exit (and note borrow is irrelevant — arb on cash deals is long-only here).

The decisive object is not the average spread (it's positive — that's the point) but its
**tail**: the realized mean and its t-stat once the break losses are counted honestly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Per-deal arb return (the signal)
# --------------------------------------------------------------------------- #
def arb_returns(book: pd.DataFrame) -> np.ndarray:
    """Realized per-deal arb returns from a deal frame carrying a ``ret`` column."""
    return np.asarray(book["ret"].values, dtype=float)


def annualized(book: pd.DataFrame) -> np.ndarray:
    """Per-deal annualized realized arb returns (``ann`` column)."""
    return np.asarray(book["ann"].values, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``mean(sample) - mu0`` (is the book net positive?). NaN if n<2."""
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        return float("nan")
    return float((x.mean() - mu0) / se)


def bootstrap_mean(sample: np.ndarray, n_boot: int = 20_000,
                   seed: int = 366) -> dict:
    """Bootstrap the mean of the (skewed) arb-return book.

    Resample the deal returns with replacement ``n_boot`` times; report the bootstrap mean,
    a 90% CI, and the placebo p-value P[bootstrap mean <= 0] — the honest small-sample
    answer to "is the average deal actually positive once you account for the break tail?".
    """
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_le0": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, size=n)].mean()
    return {
        "n": n,
        "mean": float(x.mean()),
        "lo": float(np.percentile(means, 5)),
        "hi": float(np.percentile(means, 95)),
        "p_le0": float((means <= 0).mean()),
    }


def win_rate(book: pd.DataFrame) -> dict:
    """Win-rate (fraction of deals that closed / paid) and the loss-side magnitude.

    The whole illusion: the win-rate is high but the loser tail is deep. Returns the
    win-rate, the mean winning return, and the mean losing return (the break snap-back)."""
    r = arb_returns(book)
    wins = r > 0
    return {
        "n": int(len(r)),
        "win_rate": float(wins.mean()) if len(r) else float("nan"),
        "mean_win": float(r[wins].mean()) if wins.any() else float("nan"),
        "mean_loss": float(r[~wins].mean()) if (~wins).any() else float("nan"),
        "n_breaks": int((~book["completed"]).sum()) if "completed" in book else int((~wins).sum()),
    }


def skewness(sample: np.ndarray) -> float:
    """Fisher-Pearson sample skewness of the arb-return book (negative = fat left tail)."""
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return float("nan")
    return float(((x - m) ** 3).mean() / s ** 3)


def summarize(book: pd.DataFrame, annualize: bool = False) -> dict:
    """Headline stats for a deal book: n, mean (per-deal or annualized), t, bootstrap,
    win-rate, break count, skew."""
    r = annualized(book) if annualize else arb_returns(book)
    bs = bootstrap_mean(r)
    wr = win_rate(book)
    return {
        "n": int(len(r)),
        "mean": float(np.nanmean(r)) if len(r) else float("nan"),
        "median": float(np.nanmedian(r)) if len(r) else float("nan"),
        "t": welch_t(r),
        "boot_lo": bs["lo"],
        "boot_hi": bs["hi"],
        "p_le0": bs["p_le0"],
        "win_rate": wr["win_rate"],
        "mean_win": wr["mean_win"],
        "mean_loss": wr["mean_loss"],
        "n_breaks": wr["n_breaks"],
        "skew": skewness(r),
    }


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(book: pd.DataFrame, cost_bps: float = 10.0) -> dict:
    """Charge a one-way cost on entry and on exit of every deal (one round-trip each).

    ``cost_bps`` is the one-way cost in basis points; each deal pays it twice (in and out),
    so the per-deal drag is ``2 * cost_bps / 1e4``. Returns gross and net mean per-deal
    realized return and the net t-stat. (Arb on announced all-cash deals is long the target
    only — no short leg, so no borrow; the binding frictions are the entry/exit spread and
    the break tail, not financing.)"""
    gross = arb_returns(book)
    c = 2.0 * cost_bps / 1e4
    net = gross - c
    return {
        "n_trades": int(len(gross)),
        "gross_mean": float(np.nanmean(gross)) if len(gross) else float("nan"),
        "net_mean": float(np.nanmean(net)) if len(net) else float("nan"),
        "net_t": welch_t(net),
        "cost_bps": cost_bps,
    }
