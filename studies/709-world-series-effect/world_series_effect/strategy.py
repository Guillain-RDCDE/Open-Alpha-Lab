"""Strategy + inference for Study 709 — World-Series-Effect.

The claim, stated the way baseball-omen folklore states it: *"the League (AL vs NL) of
the World Series champion tells you something about next year's stock market — the way
the Super Bowl Indicator claims the NFC/AFC does."* We mirror that football cousin's
NFC-mnemonic and test **NL win -> bullish next year** as the primary signal, plus the
**city-mythology variant** named in the brief — **a New York franchise wins -> bullish
next year** (NY teams have won disproportionately often, so this is the natural
"hometown of Wall Street" story). Neither direction has any published mechanism; that
absence is itself part of the honest read (contrast with 637's FOMC vol-crush, which has
a one-sentence mechanism and clears the bar easily).

Measurements, run on both signal variants:

* **Mean next-year return by group** — bull-flagged seasons' next-year %-return vs
  bear-flagged, Welch *t* (single, non-overlapping annual events — the same design as
  the Super Bowl study).
* **Binomial "omen" hit rate**, tested against the honest baseline (the S&P's own
  unconditional up-rate over the sample, NOT a 50% coin — the base-rate trap that flatters
  every "predicts up" folk indicator), with a Wilson interval.
* **Permutation test** on the two-sided mean contrast, 20,000 reshuffles of the
  bull/bear label — distribution-free, handles the small n honestly.
* **Third axis ("myth-check")** — does the omen's raw hit rate beat a flat 50% coin?
  A separate, looser two-sided binomial test against p=0.5, reported purely because it's
  the question the folklore itself would ask.
* **Tradability** — a "hold the S&P only after a bull-flagged season, sit in cash
  otherwise" timing strategy vs buy-and-hold, compounded over the sample.

Execution / lag convention (the single one used throughout): the World Series is decided
by early November of season Y; a "next year" position is entered at the **December 31
close of year Y** and held through year Y+1 — the champion's league (and city) is public
information weeks before that entry, so this is a **zero-look-ahead scheduled entry**,
exactly the FOMC-calendar convention used in sibling study 637.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# --------------------------------------------------------------------------- #
# Event table
# --------------------------------------------------------------------------- #
def build_events(ws_df: pd.DataFrame, ann_ret: pd.Series) -> pd.DataFrame:
    """One row per playable World Series season with a scoreable "next year".

    Joins the hardcoded champion table to the ^GSPC calendar-year return series on
    ``target_year = ws_year + 1``; seasons whose target year isn't (yet) a COMPLETE
    year on the tape are dropped (this is what silently excludes the still-open 2025
    champion's next-year call, i.e. CY2026, until that year closes).
    """
    rows = []
    for _, r in ws_df.iterrows():
        y = int(r["ws_year"])
        ty = y + 1
        if ty in ann_ret.index:
            rows.append({
                "ws_year": y, "target_year": ty, "champion": r["champion"],
                "league": r["league"], "city": r["city"], "is_ny": bool(r["is_ny"]),
                "next_year_return": float(ann_ret.loc[ty]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
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
# The headline omen test — bull/bear split, binomial + permutation
# --------------------------------------------------------------------------- #
def omen_stats(df: pd.DataFrame, bull_mask: np.ndarray, col: str = "next_year_return",
               n_perm: int = 20_000, seed: int = 709) -> dict:
    """Mean-return contrast, Welch t, binomial hit-rate test and a permutation p.

    ``bull_mask`` flags the "bullish omen" seasons (e.g. NL win, or NY champion).
    The binomial test uses the CORRECT baseline — the sample's own unconditional
    up-rate — never a 50% coin (that mistake flatters every "predicts up" indicator,
    since the S&P is up in the large majority of years regardless). A separate,
    looser two-sided test against a flat 50% coin is reported too, purely because
    it is the question the folklore itself would ask (the "myth-check" third axis).
    """
    rets = df[col].to_numpy(dtype=float)
    up = rets > 0
    n = len(rets)
    bull_mask = np.asarray(bull_mask, dtype=bool)
    n_bull, n_bear = int(bull_mask.sum()), int((~bull_mask).sum())
    uncond_up = float(up.mean())

    hits = np.where(bull_mask, up, ~up)
    hit_rate = float(hits.mean())
    lo, hi = wilson_interval(int(hits.sum()), n)

    mean_bull = float(rets[bull_mask].mean()) if n_bull else float("nan")
    mean_bear = float(rets[~bull_mask].mean()) if n_bear else float("nan")
    contrast = mean_bull - mean_bear
    t = welch_t(rets[bull_mask], rets[~bull_mask])

    binom = scipy_stats.binomtest(
        k=int(up[bull_mask].sum()), n=n_bull, p=uncond_up, alternative="greater"
    )
    binom_p = float(binom.pvalue)

    # Two-sided permutation test on the mean contrast (distribution-free, small-n safe).
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    contrasts = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(idx)
        pb, pr = perm[:n_bull], perm[n_bull:]
        contrasts[i] = rets[pb].mean() - rets[pr].mean()
    perm_p = float((np.abs(contrasts) >= abs(contrast)).mean())

    # Myth-check: does the raw hit rate beat a flat coin (p=0.5, two-sided)?
    coin = scipy_stats.binomtest(k=int(hits.sum()), n=n, p=0.5, alternative="two-sided")

    return {
        "n": n, "n_bull": n_bull, "n_bear": n_bear,
        "uncond_up_pct": uncond_up * 100.0,
        "hit_rate_pct": hit_rate * 100.0, "hit_lo_pct": lo * 100.0, "hit_hi_pct": hi * 100.0,
        "mean_bull_pct": mean_bull, "mean_bear_pct": mean_bear, "contrast_pct": contrast,
        "welch_t": t, "binom_p": binom_p, "perm_p": perm_p,
        "coin_p": float(coin.pvalue),
    }


# --------------------------------------------------------------------------- #
# Could you trade it? — omen-timing vs buy-and-hold
# --------------------------------------------------------------------------- #
def timing_strategy(df: pd.DataFrame, bull_mask: np.ndarray,
                    col: str = "next_year_return") -> dict:
    """Hold the S&P only in bull-flagged "next years"; sit in cash otherwise.

    Compounded over the sample and annualized, vs unconditional buy-and-hold. One
    rebalance per season (at most), so transaction costs are immaterial to the
    conclusion — the strategy is dominated by the *signal's* absence and by sitting
    out roughly half the calendar, not by frictions.
    """
    r = df[col].to_numpy(dtype=float) / 100.0
    bull_mask = np.asarray(bull_mask, dtype=bool)
    n = len(r)
    strat_ret = np.where(bull_mask, r, 0.0)
    strat_cum = float(np.prod(1.0 + strat_ret))
    bah_cum = float(np.prod(1.0 + r))
    strat_ann = (strat_cum ** (1.0 / n) - 1.0) * 100.0
    bah_ann = (bah_cum ** (1.0 / n) - 1.0) * 100.0
    return {
        "n_years": n, "n_held": int(bull_mask.sum()), "n_cash": int((~bull_mask).sum()),
        "strat_ann_pct": strat_ann, "bah_ann_pct": bah_ann,
        "ann_advantage_pct": strat_ann - bah_ann,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(df: pd.DataFrame) -> dict:
    """Run the headline omen split on a synthetic world (NL-equivalent flag = bull)."""
    bull = (df["league"] == "NL").to_numpy()
    return omen_stats(df, bull, n_perm=2_000, seed=709)
