"""Strategy + inference for Study 399 — Kalshi-Efficiency (event-contract calibration).

A binary event-contract priced ``p`` pays $1 on YES, $0 on NO, so a YES share bought at ``p``
has profit ``outcome - p`` (in dollars per $1 notional). Calibration asks whether realized
frequency matches price; the believer's edge is a *systematic* gap — the **favourite–longshot
bias** (buy over-priced longshots short / under-priced favourites long).

We test it with the desk's shared instruments:

  * a **reliability (calibration) curve** — realized YES frequency per price bucket vs price;
  * the **Brier-score decomposition** (Murphy 1973): reliability + resolution − uncertainty,
    so we can read the *miscalibration* component off the score directly;
  * a **longshot–favourite return spread** — the strategy P&L of going long favourites and
    short longshots, with a 1-cent illustrative **one-way fee** on every leg;
  * a **Welch t** of the spread's per-contract P&L against zero, and a **randomization null**
    (shuffle outcomes within price buckets ⇒ the spread under perfect calibration) giving an
    honest small/finite-sample p-value;
  * an **execution lag**: you trade at the *next* observed price tick, not the one your screen
    showed (no look-ahead); modelled here as the one-way fee + the realistic note that the
    cheap quote vanishes once the gap is real (cf. Study 351).

The decisive number is not the point estimate of the gap (a finite book always wiggles off the
diagonal) but whether the harvestable spread clears the t≥2 bar *net of the fee* — and on a
well-calibrated book it does not, by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default price buckets for the reliability curve (10 deciles of price).
DEFAULT_EDGES = np.linspace(0.0, 1.0, 11)

# Illustrative one-way trading fee per contract leg, in dollars (Kalshi charges a small
# per-contract fee; 1 cent is a round, transparent stand-in). Applied to every leg entered.
FEE = 0.01


# --------------------------------------------------------------------------- #
# Calibration curve
# --------------------------------------------------------------------------- #
def calibration_curve(df: pd.DataFrame, edges: np.ndarray = DEFAULT_EDGES) -> pd.DataFrame:
    """Reliability table: per price bucket, mean price, realized YES frequency, count.

    The diagonal (mean price == realized frequency) is perfect calibration. Systematic
    departures are the favourite–longshot bias.
    """
    p = df["price"].values
    y = df["outcome"].values
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "bucket": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "mean_price": float(p[m].mean()),
            "freq_yes": float(y[m].mean()),
            "n": n,
        })
    return pd.DataFrame(rows)


def brier_decomposition(df: pd.DataFrame, edges: np.ndarray = DEFAULT_EDGES) -> dict:
    """Murphy (1973) decomposition of the Brier score into reliability/resolution/uncertainty.

    Brier = mean((price - outcome)^2) = reliability - resolution + uncertainty, where
    reliability is the calibration error (0 = perfectly calibrated), resolution rewards
    sorting power, and uncertainty is the base-rate variance. A well-calibrated book has a
    near-zero *reliability* term; a favourite–longshot bias inflates it.
    """
    p = df["price"].values
    y = df["outcome"].values
    obar = y.mean()
    brier = float(np.mean((p - y) ** 2))
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
    reliability = 0.0
    resolution = 0.0
    n = len(p)
    for b in range(len(edges) - 1):
        m = idx == b
        nk = int(m.sum())
        if nk == 0:
            continue
        pbar = p[m].mean()
        ybar = y[m].mean()
        reliability += nk * (pbar - ybar) ** 2
        resolution += nk * (ybar - obar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = float(obar * (1 - obar))
    return {
        "brier": brier,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
    }


# --------------------------------------------------------------------------- #
# Longshot / favourite return spread (the harvestable strategy)
# --------------------------------------------------------------------------- #
def contract_pnl(df: pd.DataFrame, lo: float = 0.20, hi: float = 0.80,
                 fee: float = FEE) -> dict:
    """The favourite-vs-longshot trade, net of a one-way fee per leg.

    * **Long favourites** (price >= ``hi``): buy YES; per-$1 profit = ``outcome - price - fee``.
    * **Short longshots** (price <= ``lo``): sell YES (buy NO); per-$1 profit =
      ``(1 - outcome) - (1 - price) - fee`` = ``price - outcome - fee``.

    Returns the combined per-contract net P&L array (the strategy's leg-level returns), plus the
    favourite-only and longshot-only legs, so the Signal axis sees the harvestable spread *after*
    the fee and a 1-tick execution lag (the cheap quote you saw isn't the one you trade — modelled
    as the fee plus the honest caveat in the notebook).
    """
    p = df["price"].values
    y = df["outcome"].values
    fav = p >= hi
    lng = p <= lo
    fav_pnl = (y[fav] - p[fav]) - fee                 # long YES on favourites
    lng_pnl = (p[lng] - y[lng]) - fee                 # short YES on longshots
    both = np.concatenate([fav_pnl, lng_pnl])
    return {
        "pnl": both,
        "fav_pnl": fav_pnl,
        "lng_pnl": lng_pnl,
        "n_fav": int(fav.sum()),
        "n_lng": int(lng.sum()),
        "fee": fee,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample Welch/Student t of ``mean(sample) - mu0``. NaN if sample < 2."""
    sample = np.asarray(sample, dtype=float)
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu0) / se)


def randomization_pvalue(df: pd.DataFrame, lo: float = 0.20, hi: float = 0.80,
                         fee: float = FEE, n_draws: int = 20_000,
                         edges: np.ndarray = DEFAULT_EDGES, seed: int = 399) -> dict:
    """Null: outcomes are calibrated to price ⇒ shuffle YES *within* each price bucket and
    recompute the favourite/longshot spread; p = P[null spread >= observed spread].

    Drawing YES with probability equal to bucket price preserves calibration exactly, so this is
    the honest "could a well-calibrated book have produced this spread by luck?" test sized to
    the actual contract counts.
    """
    obs = contract_pnl(df, lo=lo, hi=hi, fee=fee)
    obs_mean = float(obs["pnl"].mean()) if len(obs["pnl"]) else float("nan")
    p = df["price"].values
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        y = (rng.random(len(p)) < p).astype(int)        # calibrated-by-price resample
        fav = p >= hi
        lng = p <= lo
        fav_pnl = (y[fav] - p[fav]) - fee
        lng_pnl = (p[lng] - y[lng]) - fee
        means[i] = np.concatenate([fav_pnl, lng_pnl]).mean()
    return {
        "obs_mean": obs_mean,
        "null_mean": float(means.mean()),
        "p_value": float((means >= obs_mean).mean()),
        "n": int(len(obs["pnl"])),
    }


def summarize(df: pd.DataFrame, lo: float = 0.20, hi: float = 0.80, fee: float = FEE) -> dict:
    """Headline stats: Brier reliability term, favourite/longshot spread, net per-contract P&L,
    Welch t vs zero, and the randomization p-value."""
    bd = brier_decomposition(df)
    pnl = contract_pnl(df, lo=lo, hi=hi, fee=fee)
    rp = randomization_pvalue(df, lo=lo, hi=hi, fee=fee)
    both = pnl["pnl"]
    return {
        "n": int(len(both)),
        "n_fav": pnl["n_fav"],
        "n_lng": pnl["n_lng"],
        "reliability": bd["reliability"],
        "brier": bd["brier"],
        "net_mean": float(both.mean()) if len(both) else float("nan"),
        "gross_mean": float(both.mean() + fee) if len(both) else float("nan"),
        "win_rate": float((both > 0).mean()) if len(both) else float("nan"),
        "t": welch_t(both, 0.0),
        "p_rand": rp["p_value"],
        "fee": fee,
    }
