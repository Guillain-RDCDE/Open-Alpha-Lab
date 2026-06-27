"""The publication-decay engine and its honest controls — Study 536.

The pipeline has four parts:

1. **Signal construction (price-only).** From a monthly total-return panel, build the
   cross-sectional score for each classic anomaly at every month-end:
     * ``mom12_1``      trailing 12-month return skipping the most recent month,
     * ``lowvol``       negative of trailing 12-month return volatility (low vol = high score),
     * ``st_reversal``  negative of last month's return (buy losers, sell winners),
     * ``lt_reversal``  negative of the trailing 36→13 month return (De Bondt-Thaler).

2. **The long-short.** Each month, cross-sectionally rank the score, go long the top
   tercile and short the bottom tercile (dollar-neutral, equal-weight). The portfolio
   earns the NEXT month's return (one execution lag, applied once via ``shift``).

3. **The split-at-publication reckoning (the heart of the study).** Cut each anomaly's
   long-short series at its publication year and compare the mean monthly return
   BEFORE vs AFTER. The decay ratio = post / pre. McLean & Pontiff (2016) find the
   typical post/pre is ~0.5 (returns roughly halve once the anomaly is in print).

4. **Honest inference + controls.** A one-sample *t* that each leg's mean is non-zero;
   a label-shuffle placebo that randomly re-assigns the pre/post tag (destroying the
   publication boundary) to see how often a "decay" this large appears by chance;
   one-way costs x turnover (+ borrow on the short leg); and a deterministic synthetic
   positive control with a PLANTED decay the engine must recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Signal construction (price-only)
# ---------------------------------------------------------------------------
def build_signal(rets: pd.DataFrame, anomaly: str) -> pd.DataFrame:
    """Cross-sectional score for ``anomaly`` at each month-end (higher = go long).

    All signals use only information available at the close of the labelling month
    (no look-ahead); the long-short then earns the *next* month's return.
    """
    if anomaly == "mom12_1":
        # cumulative return over months t-12..t-2 (skip the most recent month)
        logret = np.log1p(rets)
        sig = logret.rolling(11).sum().shift(1)
    elif anomaly == "lowvol":
        # low total volatility = high score -> negate the trailing 12m std
        sig = -rets.rolling(12).std()
    elif anomaly == "st_reversal":
        # buy last month's losers -> negate last month's return
        sig = -rets
    elif anomaly == "lt_reversal":
        # De Bondt-Thaler: buy 3-year losers -> negate months t-36..t-13
        logret = np.log1p(rets)
        sig = -(logret.rolling(24).sum().shift(12))
    else:
        raise ValueError(f"unknown anomaly {anomaly!r}")
    return sig


def long_short_returns(
    rets: pd.DataFrame, signal: pd.DataFrame, frac: float = 1 / 3
) -> pd.Series:
    """Monthly return of a dollar-neutral top-tercile-minus-bottom-tercile portfolio.

    One execution lag: the position formed from ``signal`` at month *t* earns month
    *t+1*'s return. Names with a missing signal or return that month are dropped from
    the cross-section, so each month uses whatever names are live.
    """
    fwd = rets.shift(-1)                                   # next month's return
    out = {}
    for t in signal.index:
        s = signal.loc[t].dropna()
        r = fwd.loc[t].reindex(s.index).dropna() if t in fwd.index else pd.Series(dtype=float)
        s = s.reindex(r.index).dropna()
        n = len(s)
        if n < 6:                                          # need a real cross-section
            continue
        k = max(1, int(round(n * frac)))
        order = s.sort_values()
        short = order.index[:k]
        long = order.index[-k:]
        out[t] = float(r[long].mean() - r[short].mean())
    return pd.Series(out).sort_index()


def turnover_series(
    signal: pd.DataFrame, frac: float = 1 / 3
) -> pd.Series:
    """Fraction of the (long+short) book that changes name month-over-month.

    Used to charge realistic costs: a fully rebuilt book = turnover 1.0 per leg.
    """
    members = {}
    for t in signal.index:
        s = signal.loc[t].dropna()
        n = len(s)
        if n < 6:
            continue
        k = max(1, int(round(n * frac)))
        order = s.sort_values()
        members[t] = (set(order.index[:k]), set(order.index[-k:]))
    keys = sorted(members)
    out = {}
    prev = None
    for t in keys:
        sh, lo = members[t]
        if prev is None:
            out[t] = 1.0                                   # establish the book
        else:
            psh, plo = prev
            denom = len(sh) + len(lo)
            changed = len(sh ^ psh) / 2 + len(lo ^ plo) / 2
            out[t] = float(changed / denom) if denom else 0.0
        prev = (sh, lo)
    return pd.Series(out).sort_index()


# ---------------------------------------------------------------------------
# Honest inference
# ---------------------------------------------------------------------------
def one_sample_t(x: np.ndarray) -> float:
    """One-sample *t* that the mean of a monthly return series is non-zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")


def two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value from a Student-t statistic (scipy if available)."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy import stats

        return float(2.0 * stats.t.sf(abs(t), df))
    except Exception:
        # large-sample normal fallback
        z = abs(t)
        return float(min(1.0, 2.0 * 0.5 * np.exp(-0.717 * z - 0.416 * z * z)))


def _ann(mean_monthly: float) -> float:
    return float(mean_monthly * MONTHS_PER_YEAR)


def split_at_publication(ls: pd.Series, pub_year: int) -> dict:
    """Split a monthly long-short series at ``pub_year`` and compare pre vs post.

    Returns a dict with, for each of {pre, post}: n months, mean monthly, annualised
    mean, one-sample *t*, two-sided *p*, plus the ``decay_ratio = post_mean/pre_mean``
    and the ``decay_pct = 1 - decay_ratio`` (the McLean-Pontiff shrinkage).
    """
    years = ls.index.year if isinstance(ls.index, pd.PeriodIndex) else \
        pd.PeriodIndex(ls.index, freq="M").year
    pre = ls[years < pub_year].to_numpy(dtype=float)
    post = ls[years >= pub_year].to_numpy(dtype=float)

    def leg(x):
        x = x[np.isfinite(x)]
        t = one_sample_t(x)
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)) if x.size else float("nan"),
            "ann": _ann(float(np.mean(x))) if x.size else float("nan"),
            "t": t,
            "p": two_sided_p(t, max(x.size - 1, 1)),
        }

    pre_d, post_d = leg(pre), leg(post)
    ratio = (post_d["mean"] / pre_d["mean"]
             if pre_d["mean"] not in (0.0, float("nan")) and np.isfinite(pre_d["mean"])
             and pre_d["mean"] != 0 else float("nan"))
    return {
        "pub_year": pub_year,
        "pre": pre_d,
        "post": post_d,
        "decay_ratio": float(ratio) if np.isfinite(ratio) else float("nan"),
        "decay_pct": float(1.0 - ratio) if np.isfinite(ratio) else float("nan"),
    }


def placebo_split(
    ls: pd.Series, pub_year: int, n_boot: int = 5000, seed: int = 536
) -> dict:
    """Label-shuffle placebo: how often does a decay this large appear by chance?

    The boundary at ``pub_year`` partitions the series into ``n_pre`` and ``n_post``
    months. We repeatedly draw a RANDOM partition of the SAME sizes (destroying the
    publication boundary), recompute the decay (post_mean - pre_mean), and report the
    one-sided *p* that the real post-minus-pre drop is more negative than chance. A
    real, publication-driven decay should be rarer than the shuffled boundaries.
    """
    x = ls.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    years = (ls[np.isfinite(ls.to_numpy())].index.year
             if isinstance(ls.index, pd.PeriodIndex)
             else pd.PeriodIndex(ls.index, freq="M").year[np.isfinite(ls.to_numpy())])
    n = x.size
    n_pre = int(np.sum(np.asarray(years) < pub_year))
    if n_pre < 3 or (n - n_pre) < 3:
        return {"p": float("nan"), "n_boot": 0, "real_drop": float("nan")}
    real_drop = float(x[n_pre:].mean() - x[:n_pre].mean())     # post - pre (≤0 = decay)

    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_boot):
        perm = rng.permutation(n)
        pre = x[perm[:n_pre]]
        post = x[perm[n_pre:]]
        drop = post.mean() - pre.mean()
        if drop <= real_drop:                                  # at least as much decay
            hits += 1
    return {"p": float((hits + 1) / (n_boot + 1)), "n_boot": n_boot,
            "real_drop": real_drop}


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def apply_costs(
    ls: pd.Series, turnover: pd.Series, cost_bps: float = 10.0, borrow_bps_ann: float = 50.0
) -> pd.Series:
    """Charge one-way costs x turnover (both legs) + monthly borrow on the short leg.

    ``cost_bps`` is per leg per one-way trade; ``borrow_bps_ann`` is the annual borrow
    on the (dollar-of-)short leg, charged monthly. Returns the NET monthly series.
    """
    to = turnover.reindex(ls.index).fillna(0.0)
    # two legs, one-way each side: cost ~ 2 * turnover * cost_bps
    trade_cost = 2.0 * to * (cost_bps * 1e-4)
    borrow = (borrow_bps_ann * 1e-4) / MONTHS_PER_YEAR        # short leg is 1x NAV
    return ls - trade_cost - borrow


# ---------------------------------------------------------------------------
# Top-level runners
# ---------------------------------------------------------------------------
def run_anomaly(
    rets: pd.DataFrame,
    anomaly: str,
    pub_year: int,
    frac: float = 1 / 3,
    cost_bps: float = 10.0,
    borrow_bps_ann: float = 50.0,
    placebo_boot: int = 5000,
    seed: int = 536,
) -> dict:
    """Full teardown for one anomaly: signal → long-short → split → placebo → costs."""
    sig = build_signal(rets, anomaly)
    ls = long_short_returns(rets, sig, frac=frac)
    to = turnover_series(sig, frac=frac)
    net = apply_costs(ls, to, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)

    split_g = split_at_publication(ls, pub_year)
    split_n = split_at_publication(net, pub_year)
    plac = placebo_split(ls, pub_year, n_boot=placebo_boot, seed=seed)

    return {
        "anomaly": anomaly,
        "pub_year": pub_year,
        "n_months": int(ls.shape[0]),
        "full_mean_ann": _ann(float(ls.mean())),
        "full_t": one_sample_t(ls.to_numpy()),
        "avg_turnover": float(to.mean()),
        "gross": split_g,
        "net": split_n,
        "placebo": plac,
        "ls": ls,
        "net_series": net,
    }


def run_panel(
    rets: pd.DataFrame,
    anomalies: dict,
    frac: float = 1 / 3,
    cost_bps: float = 10.0,
    borrow_bps_ann: float = 50.0,
    placebo_boot: int = 5000,
    seed: int = 536,
) -> dict:
    """Run every anomaly and summarise the cross-anomaly decay picture."""
    per = {}
    for name, meta in anomalies.items():
        per[name] = run_anomaly(
            rets, name, meta["pub_year"], frac=frac, cost_bps=cost_bps,
            borrow_bps_ann=borrow_bps_ann, placebo_boot=placebo_boot, seed=seed,
        )
    ratios = [per[a]["gross"]["decay_ratio"] for a in per
              if np.isfinite(per[a]["gross"]["decay_ratio"])]
    return {
        "per": per,
        "median_decay_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "mean_decay_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "n_anomalies": len(per),
    }


# ---------------------------------------------------------------------------
# Synthetic positive control — proves the split engine is faithful
# ---------------------------------------------------------------------------
def synthetic_control(
    post_decay: float = 0.5, seed: int = 536, n_seeds: int = 1, **panel_kw
) -> dict:
    """Run the split engine on a panel with a PLANTED decay; recover the pre/post step.

    With ``n_seeds > 1`` the recovered decay ratio is averaged over seeds (seed-robust
    check). A faithful engine recovers ``decay_ratio ≈ 1 - post_decay``:
      - ``post_decay = 0.5`` → recovered ratio ≈ 0.5 (the canonical control),
      - ``post_decay = 1.0`` → recovered ratio ≈ 0.0 and a post leg indistinguishable
        from noise (the null: a dead post edge cannot fake significance).
    """
    from . import data as _data

    ratios, post_ts, pre_ts = [], [], []
    for s in range(n_seeds):
        rets, sig, pub_m = _data.synthetic_panel(
            post_decay=post_decay, seed=seed + s, **panel_kw
        )
        ls = long_short_returns(rets, sig)
        pub_year = rets.index[pub_m].year
        split = split_at_publication(ls, pub_year)
        if np.isfinite(split["decay_ratio"]):
            ratios.append(split["decay_ratio"])
        pre_ts.append(split["pre"]["t"])
        post_ts.append(split["post"]["t"])

    return {
        "post_decay": post_decay,
        "n_seeds": n_seeds,
        "expected_ratio": 1.0 - post_decay,
        "recovered_ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
        "pre_t_mean": float(np.nanmean(pre_ts)),
        "post_t_mean": float(np.nanmean(post_ts)),
    }
