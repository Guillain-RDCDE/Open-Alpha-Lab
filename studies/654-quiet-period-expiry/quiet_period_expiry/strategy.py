"""Strategy + inference for Study 654 — Quiet-Period-Expiry.

The claim: **when the FINRA 25-day analyst quiet period ends, the IPO's own
underwriters initiate coverage — almost always Buy — and the stock pops** (Bradley,
Jordan & Ritter 2003). Measurements, all on the event-time abnormal-return panel built
by ``data.fetch_event_panel`` / ``data.synthetic_panel`` (abnormal = stock − SPY,
trading day *t* since the first close):

* **The headline window.** Cumulative abnormal return (CAR) over trading days
  [20..30] per IPO — one-sample *t* across IPOs (the events are cross-sectionally
  independent: 65 different companies, no shared calendar date). A same-width
  [3..13] placebo window (early post-listing, away from expiry) gives a paired
  within-IPO robustness check, and a random-window placebo (many random 11-day
  windows drawn from each IPO's own path) gives a distribution-free p-value.
* **Anatomy.** Mean abnormal return by trading day *t*, one-sample *t* per day —
  is day ~25 special, or is this a broad post-IPO drift with no bump at expiry?
* **The literal retail trade.** Buy the close of day 22, sell the close of day 27
  (a 5-session hold spanning the window), gross and net of one-way costs × 2.
* **Era contrast.** Early vs late IPOs in the basket (split at a date fixed before
  any stats were run) — has the effect faded as the "quiet period pop" folklore
  became well known and got arbitraged/anticipated?

The decisive number is the one-sample *t* on the [20..30] CAR across independent IPOs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    """One-sample t of mean(x) against zero. NaN if fewer than 2 finite obs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# CAR window — the headline
# --------------------------------------------------------------------------- #
def _car_per_ticker(panel: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    """Cumulative abnormal return per ticker over trading days [lo..hi]."""
    win = panel[(panel["t"] >= lo) & (panel["t"] <= hi)]
    return win.groupby("ticker")["abn_ret"].sum()


def car_window_stats(panel: pd.DataFrame, lo: int = dt.WINDOW_LO,
                     hi: int = dt.WINDOW_HI) -> dict:
    """One-sample t on the [lo..hi] per-IPO CAR (the headline test)."""
    car = _car_per_ticker(panel, lo, hi).to_numpy(dtype=float)
    n = len(car)
    if n < 2:
        return {"n": n, "mean_car_pct": float("nan"), "t": float("nan"),
                "hit_rate": float("nan"), "hit_lo": float("nan"), "hit_hi": float("nan")}
    k_up = int((car > 0).sum())
    lo_ci, hi_ci = wilson_interval(k_up, n)
    return {
        "n": n, "mean_car_pct": float(car.mean() * 100),
        "median_car_pct": float(np.median(car) * 100),
        "std_car_pct": float(car.std(ddof=1) * 100),
        "t": one_sample_t(car),
        "hit_up": k_up, "hit_rate": k_up / n, "hit_lo": lo_ci, "hit_hi": hi_ci,
    }


def paired_placebo_stats(panel: pd.DataFrame, lo: int = dt.WINDOW_LO, hi: int = dt.WINDOW_HI,
                         plo: int = dt.PLACEBO_LO, phi: int = dt.PLACEBO_HI) -> dict:
    """Paired within-IPO robustness: (headline-window CAR) - (same-width placebo CAR).

    Both windows come from the SAME IPO's path (same-width [plo..phi], early and away
    from expiry), so this differences out any per-IPO level effect (e.g. general
    post-IPO drift) and isolates whether [lo..hi] specifically stands out.
    """
    car_a = _car_per_ticker(panel, lo, hi)
    car_b = _car_per_ticker(panel, plo, phi)
    common = car_a.index.intersection(car_b.index)
    diff = (car_a.loc[common] - car_b.loc[common]).to_numpy(dtype=float)
    return {
        "n": len(diff), "mean_diff_pct": float(np.nanmean(diff) * 100) if len(diff) else float("nan"),
        "t": one_sample_t(diff),
    }


def random_window_placebo(panel: pd.DataFrame, lo: int = dt.WINDOW_LO, hi: int = dt.WINDOW_HI,
                          n_draws_per_seed: int = 1_000, n_seeds: int = 20,
                          base_seed: int = 654) -> dict:
    """Random-window placebo: draw a same-width random window per IPO, many times.

    For each draw, every IPO gets an independently random start offset (width held
    fixed at hi-lo+1, drawn from the range that stays inside [1, max_td]); the mean
    CAR across IPOs is recorded. p = share of draws whose mean CAR >= the observed
    [lo..hi] mean CAR (right-tail — the claim is a pop). Vectorized via a per-ticker
    cumulative-sum table (a python-level loop over ~20,000 draws x 66 tickers with a
    pandas ``reindex`` per (draw, ticker) is orders of magnitude too slow).
    """
    width = hi - lo + 1
    max_t = int(panel["t"].max())
    piv = panel.pivot(index="ticker", columns="t", values="abn_ret")
    piv = piv.reindex(columns=range(1, max_t + 1))
    arr = piv.to_numpy(dtype=float)                     # (n_tickers, max_t), NaN where missing
    arr = np.nan_to_num(arr, nan=0.0)
    n_tkr = arr.shape[0]
    # cumsum[i, k] = sum of columns 0..k-1 (0-indexed cols correspond to t = 1..max_t)
    cum = np.concatenate([np.zeros((n_tkr, 1)), np.cumsum(arr, axis=1)], axis=1)

    obs = float(_car_per_ticker(panel, lo, hi).mean())

    lo_start, hi_start = 1, max_t - width + 1           # inclusive bounds on t
    n_total = n_seeds * n_draws_per_seed
    means = np.empty(n_total, dtype=float)
    rows_idx = np.arange(n_tkr)
    k = 0
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        starts = rng.integers(lo_start, hi_start + 1, size=(n_draws_per_seed, n_tkr))
        # column positions in `cum`: start t=lo_start..hi_start -> 0-indexed col (t-1)
        end_col = starts + width - 1                    # last t (1-indexed) of the window
        window_sum = cum[rows_idx, end_col] - cum[rows_idx, starts - 1]
        means[k: k + n_draws_per_seed] = window_sum.mean(axis=1)
        k += n_draws_per_seed
    return {
        "obs": obs, "placebo_mean": float(means.mean()), "placebo_sd": float(means.std(ddof=1)),
        "p_value": float((means >= obs).mean()), "n_draws": len(means),
    }


# --------------------------------------------------------------------------- #
# Anatomy — mean abnormal return by trading day
# --------------------------------------------------------------------------- #
def per_day_stats(panel: pd.DataFrame, lo: int = 15, hi: int = 35) -> pd.DataFrame:
    """Mean abnormal return per trading day t in [lo..hi], one-sample t across IPOs."""
    rows = []
    for t in range(lo, hi + 1):
        x = panel.loc[panel["t"] == t, "abn_ret"].to_numpy(dtype=float)
        rows.append({"t": t, "n": int(np.isfinite(x).sum()),
                     "mean_pct": float(np.nanmean(x) * 100), "t_stat": one_sample_t(x)})
    return pd.DataFrame(rows).set_index("t")


# --------------------------------------------------------------------------- #
# The literal retail trade — buy day 22, sell day 27
# --------------------------------------------------------------------------- #
def timer_strategy(panel: pd.DataFrame, buy: int = dt.TIMER_BUY, sell: int = dt.TIMER_SELL,
                   cost_bps: float = 5.0) -> dict:
    """Buy the close of day ``buy``, sell the close of day ``sell``, per IPO.

    Gross return = close_sell / close_buy - 1 (the literal price move, not
    market-adjusted — this is what a retail account actually books). Net of a
    round trip = 2 x one-way ``cost_bps`` x NAV. Long-only, no borrow.
    """
    piv = panel.pivot(index="ticker", columns="t", values="close")
    if buy not in piv.columns or sell not in piv.columns:
        return {"n": 0}
    sub = piv[[buy, sell]].dropna()
    gross = (sub[sell] / sub[buy] - 1.0).to_numpy(dtype=float)
    net = gross - 2.0 * cost_bps / 1e4
    n = len(gross)
    if n < 2:
        return {"n": n}
    return {
        "n": n, "gross_pct": float(gross.mean() * 100), "net_pct": float(net.mean() * 100),
        "t_gross": one_sample_t(gross), "t_net": one_sample_t(net),
        "hit_rate": float((gross > 0).mean()), "worst_pct": float(gross.min() * 100),
        "best_pct": float(gross.max() * 100),
    }


# --------------------------------------------------------------------------- #
# Era contrast — early vs late IPOs in the basket
# --------------------------------------------------------------------------- #
def era_contrast(panel: pd.DataFrame, basket: pd.DataFrame, split: str = dt.ERA_SPLIT,
                 lo: int = dt.WINDOW_LO, hi: int = dt.WINDOW_HI) -> dict:
    """[lo..hi]-window CAR before vs since ``split`` (by IPO date), Welch t of the split."""
    car = _car_per_ticker(panel, lo, hi)
    early_tkrs = set(basket.loc[basket["ipo_date"] < split, "ticker"])
    late_tkrs = set(basket.loc[basket["ipo_date"] >= split, "ticker"])
    early = car.loc[car.index.isin(early_tkrs)].to_numpy(dtype=float)
    late = car.loc[car.index.isin(late_tkrs)].to_numpy(dtype=float)
    return {
        "n_early": len(early), "n_late": len(late),
        "early_pct": float(np.nanmean(early) * 100) if len(early) else float("nan"),
        "late_pct": float(np.nanmean(late) * 100) if len(late) else float("nan"),
        "t_early": one_sample_t(early), "t_late": one_sample_t(late),
        "t_diff": welch_t(late, early),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: pd.DataFrame, lo: int = dt.WINDOW_LO,
                     hi: int = dt.WINDOW_HI) -> dict:
    """Run the headline CAR-window test on a synthetic panel."""
    return car_window_stats(panel, lo, hi)
