"""Strategy + inference for Study 742 — Friday-17th (*Venerdì 17*).

The claim, steelmanned: Italy's unlucky day is **Friday the 17th**, and if
superstition-driven mood moved the tape, the FTSE MIB should trade *weak* on Venerdì
17 — a systematically negative close-to-close return, worse than an ordinary Friday.

The correct inference unit is a **one-sample t across independent, non-overlapping
events** — each Venerdì 17 is one calendar date, far from the last, summarised to one
number (its close-to-close log-return). This is NOT a daily panel: there is no
overlap between events, so the event, not the day, is the unit. The battery:

* ``friday17_test`` — one-sample t of the Friday-17 return vs 0 (does the day itself
  print negative?), a **Welch** contrast vs all other Fridays (is it worse than an
  *ordinary* Friday, controlling for any generic Friday effect?), and a Wilson
  down-day hit rate (how often is Venerdì 17 actually a red day?).
* ``dom_sweep`` — the look-elsewhere kill shot: test every "middle Friday" slot the
  17th sits among (day-of-month 3, 10, 17, 24, 31) and Bonferroni-correct. The 17th
  was picked by folklore, not pre-registered; a snooper testing all five and quoting
  the most extreme inflates the false-positive rate fivefold.
* ``random_friday_placebo`` — a multi-seed random-calendar placebo: draw many random
  sets of *n* other-Friday dates from the same tape, average, and see whether the
  observed Venerdì-17 mean sits in the *tail* of that null or (as folklore usually
  turns out) squarely in its bulk.
* ``short_the_17th`` — the costed "could you trade it?" timer: short the tape into
  Venerdì 17 (calendar-known, so the position is set at the prior close — no
  look-ahead), cover at the 17th's close, pay one round trip of one-way costs plus one
  day of short borrow. Gross AND net reported; shorts pay borrow.
* ``synthetic_detect`` — the positive control: run the one-sample-t detector on a
  synthetic tape with a *planted* Friday-17 effect (and on the null world, which must
  not fire).

No look-ahead anywhere: the date label is known before the open; the return is that
session's close-to-close log-return. One documented convention, applied once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

COST_BPS = 5.0      # one-way, per leg
BORROW_BPS = 2.0    # one-day short-borrow charge on the tradability timer


# --------------------------------------------------------------------------- #
# Frame builder: close -> daily log-returns + calendar labels
# --------------------------------------------------------------------------- #
def build_frame(close: pd.Series) -> pd.DataFrame:
    """Daily log-return frame with the calendar labels the tests need.

    Columns: ``ret`` (close-to-close log-return), ``is_f17``, ``is_friday``, ``day``
    (day-of-month). The first row (no prior close) is dropped.
    """
    close = close.sort_index()
    idx = close.index
    ret = np.log(close / close.shift(1))
    df = pd.DataFrame({
        "ret": ret,
        "is_f17": dt.is_friday_17th(idx),
        "is_friday": dt.is_friday(idx),
        "day": idx.day,
    }, index=idx)
    return df.dropna(subset=["ret"])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    events (each Venerdì 17 is one observation, not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch t (unequal variances) of mean(a) - mean(b), with a two-sided p-value from
    the normal approximation (n is large enough on the 'other Fridays' side)."""
    from math import erf, sqrt
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "p": float("nan"), "diff": float("nan")}
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se <= 0:
        return {"t": float("nan"), "p": float("nan"), "diff": float(a.mean() - b.mean())}
    t = (a.mean() - b.mean()) / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return {"t": float(t), "p": float(p), "diff": float(a.mean() - b.mean())}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def down_day_rate(x: np.ndarray) -> dict:
    """Share of events with a NEGATIVE return (the folklore predicts red days),
    with a Wilson 95% interval."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    k = int((x < 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Primary test: Friday-17 vs zero, vs other Fridays, hit rate
# --------------------------------------------------------------------------- #
def friday17_test(df: pd.DataFrame) -> dict:
    """The headline: one-sample t of Venerdì-17 returns, a Welch contrast vs all other
    Fridays, and the Wilson down-day hit rate. Returns bps-scaled means."""
    f17 = df.loc[df["is_f17"], "ret"].to_numpy(dtype=float)
    other_fri = df.loc[df["is_friday"] & ~df["is_f17"], "ret"].to_numpy(dtype=float)
    all_other = df.loc[~df["is_f17"], "ret"].to_numpy(dtype=float)

    s17 = one_sample_t(f17)
    w = welch_t(f17, other_fri)
    w_all = welch_t(f17, all_other)
    hr = down_day_rate(f17)
    return {
        "n_f17": s17["n"], "mean_f17_bps": s17["mean"] * 1e4, "t_f17": s17["t"],
        "n_other_fri": int(np.isfinite(other_fri).sum()),
        "mean_other_fri_bps": float(np.nanmean(other_fri)) * 1e4,
        "contrast_fri_bps": w["diff"] * 1e4, "t_welch_fri": w["t"], "p_welch_fri": w["p"],
        "mean_all_other_bps": float(np.nanmean(all_other)) * 1e4,
        "contrast_all_bps": w_all["diff"] * 1e4, "p_welch_all": w_all["p"],
        "down_k": hr["k"], "down_n": hr["n"], "down_rate": hr["rate"],
        "down_lo": hr["lo"], "down_hi": hr["hi"],
    }


# --------------------------------------------------------------------------- #
# Look-elsewhere: the day-of-month Bonferroni sweep
# --------------------------------------------------------------------------- #
def dom_sweep(df: pd.DataFrame, doms=dt.SWEEP_DOMS) -> pd.DataFrame:
    """Test every candidate 'middle Friday' slot (17 +/- 7k) and Bonferroni-correct.

    For each day-of-month ``d`` in ``doms`` compare the Fridays landing on ``d`` to all
    other Fridays (Welch), then correct the p-values by ``k = len(doms)``. Sorted by
    raw p so the most extreme slot is on top -- the honest look-elsewhere accounting.
    """
    other_all = df.loc[df["is_friday"], "ret"]
    k = len(doms)
    rows = []
    for d in doms:
        g = df.loc[df["is_friday"] & (df["day"] == d), "ret"].to_numpy(dtype=float)
        rest = df.loc[df["is_friday"] & (df["day"] != d), "ret"].to_numpy(dtype=float)
        w = welch_t(g, rest)
        rows.append({
            "day": int(d), "n": int(np.isfinite(g).sum()),
            "mean_bps": float(np.nanmean(g)) * 1e4 if np.isfinite(g).any() else float("nan"),
            "contrast_bps": w["diff"] * 1e4, "p_raw": w["p"],
            "p_bonferroni": float(min(w["p"] * k, 1.0)) if np.isfinite(w["p"]) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Random-calendar placebo (multi-seed): is the Venerdì-17 mean in the tail?
# --------------------------------------------------------------------------- #
def random_friday_placebo(df: pd.DataFrame, n_seeds: int = 20,
                          n_draws_per_seed: int = 500, base_seed: int = 742
                          ) -> dict:
    """Draw random sets of n other-Friday dates and build the null of mean returns.

    Matched null: sample (with replacement across draws, without within a draw) the
    SAME number of dates as there are Venerdì 17s, but from the *other Fridays* of the
    same tape -- so any generic Friday effect is held constant and only the '17th'
    label is tested. ``p`` is the left-tail share (folklore predicts underperformance:
    the fraction of null means <= observed).
    """
    f17 = df.loc[df["is_f17"], "ret"].to_numpy(dtype=float)
    f17 = f17[np.isfinite(f17)]
    n = f17.size
    obs = float(f17.mean()) if n else float("nan")

    pool = df.loc[df["is_friday"] & ~df["is_f17"], "ret"].to_numpy(dtype=float)
    pool = pool[np.isfinite(pool)]
    if n == 0 or pool.size <= n:
        return {"obs_bps": obs * 1e4, "null_mean_bps": float("nan"),
                "null_sd_bps": float("nan"), "p_left": float("nan"), "n_draws": 0}

    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            idx = rng.choice(pool.size, size=n, replace=False)
            means.append(pool[idx].mean())
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4, "null_mean_bps": float(means.mean()) * 1e4,
        "null_sd_bps": float(means.std(ddof=1)) * 1e4,
        "p_left": float((means <= obs).mean()), "n_draws": int(means.size),
    }


# --------------------------------------------------------------------------- #
# Tradability: short the 17th, net of costs + borrow
# --------------------------------------------------------------------------- #
def short_the_17th(close: pd.Series, cost_bps: float = COST_BPS,
                   borrow_bps: float = BORROW_BPS) -> dict:
    """Short the tape into Venerdì 17, cover at its close; gross AND net.

    Calendar-known: you know a Venerdì 17 is coming, so the short is established at the
    PRIOR session's close (no look-ahead) and covered at the 17th's close -- the
    position captures the 17th's close-to-close return with the sign flipped. One round
    trip: one-way cost charged twice against NAV, plus one day of short borrow. A
    profitable short needs the 17th to fall by MORE than round-trip costs + borrow.
    """
    df = build_frame(close)
    f17_ret = df.loc[df["is_f17"], "ret"].to_numpy(dtype=float)
    f17_ret = f17_ret[np.isfinite(f17_ret)]
    n = f17_ret.size
    gross = -f17_ret                                   # short: profit when the day falls
    net = gross - (2.0 * cost_bps + borrow_bps) * 1e-4
    sg = one_sample_t(gross)
    sn = one_sample_t(net)
    return {
        "n": n,
        "gross_mean_bps": sg["mean"] * 1e4, "gross_t": sg["t"],
        "net_mean_bps": sn["mean"] * 1e4, "net_t": sn["t"],
        "win_rate": float((net > 0).mean()) if n else float("nan"),
        "breakeven_bps": (2.0 * cost_bps + borrow_bps),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(f17_effect: float, seed: int = 742) -> dict:
    """Run the one-sample-t detector on a synthetic tape with a planted (or null)
    Friday-17 effect. bump=0 must NOT fire; a planted bump must be recovered."""
    close, truth = dt.synthetic_daily(f17_effect=f17_effect, seed=seed)
    df = build_frame(close)
    f17 = df.loc[df["is_f17"], "ret"].to_numpy(dtype=float)
    s = one_sample_t(f17)
    s["planted_effect"] = f17_effect
    return s
