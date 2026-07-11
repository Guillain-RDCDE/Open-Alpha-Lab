"""Strategy + inference for Study 710 — Olympic-Host-Effect.

The claim: **the HOST country's stock market rallies around its own Summer Olympics** —
national pride, an infrastructure boom, tourism, a wall of foreign attention — visible as
an abnormal return in the host's equity market versus the rest of the world in the window
around the Games.

Measurement, one number per host:

* **Abnormal return** = host-ETF total return over [-6mo .. +2mo] around the opening/
  closing ceremony, MINUS the ^GSPC price return over the identical calendar window
  (aligned to the nearest trading day). Single documented execution convention: enter at
  the close of the window-start trading day, exit at the close of the window-end trading
  day — host cities are awarded 7+ years ahead, so this is a zero-look-ahead, calendar-known
  entry, exactly like a scheduled-event study.
* **The headline** is a **one-sample t-test of the mean abnormal return across n = 6 hosts**
  (Athens 2004 has no contemporaneous ETF — see data.py). With n this small the desk's rule
  is to say so loudly: report the mean AND the median AND a nonparametric check (Wilcoxon
  signed-rank) AND a percentile bootstrap CI AND a random-window placebo, because a single
  t-stat on six numbers is exactly the kind of statistic that one fat-tailed observation can
  flip.
* **A directional myth-check**: does a *majority* of hosts actually outperform, as the
  "national pride rally" story implies they should (Wilson interval on the hit rate)? This
  catches the case where the mean is dragged around by one outlier while most individual
  events point the other way.
* **A synthetic positive control** (data.synthetic_world) is a **faithful-engine / power
  check only** — it shows the one-sample-t detector is unbiased on a null world and
  quantifies how large a *true* effect would need to be to clear t >= 2 at n = 6, given the
  real panel's own dispersion. It is never cited in support of a real-tape stamp.

The decisive number is the real-tape one-sample t across the 6 hosts; everything else here
exists to stress-test whether that single number can be trusted at this sample size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import data as dt


# --------------------------------------------------------------------------- #
# Window alignment
# --------------------------------------------------------------------------- #
def _on_or_after(series: pd.Series, date: pd.Timestamp) -> pd.Timestamp:
    idx = series.index[series.index >= date]
    if len(idx) == 0:
        raise ValueError(f"no tape on/after {date.date()}")
    return idx[0]


def _on_or_before(series: pd.Series, date: pd.Timestamp) -> pd.Timestamp:
    idx = series.index[series.index <= date]
    if len(idx) == 0:
        raise ValueError(f"no tape on/before {date.date()}")
    return idx[-1]


def host_window(host: "dt.Host") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calendar [entry_target, exit_target] for a host: [-6mo before opening,
    +2mo after closing]."""
    lo = pd.Timestamp(host.games_start) - pd.DateOffset(months=dt.WINDOW_PRE_MONTHS)
    hi = pd.Timestamp(host.games_end) + pd.DateOffset(months=dt.WINDOW_POST_MONTHS)
    return lo, hi


# --------------------------------------------------------------------------- #
# The headline panel: one abnormal return per host
# --------------------------------------------------------------------------- #
def host_abnormal_returns(real: dict[str, pd.Series],
                           hosts: tuple = dt.REAL_HOSTS) -> pd.DataFrame:
    """One row per host with a ticker: entry/exit trading dates, host & benchmark
    total/price returns over the window, the abnormal return (host − benchmark, pp), and
    the trading-day window length ``L`` (used to size the random-window placebo)."""
    bench = real[dt.BENCH_TICKER]
    rows = []
    for h in hosts:
        host_s = real[h.ticker]
        lo, hi = host_window(h)
        entry_h, exit_h = _on_or_after(host_s, lo), _on_or_before(host_s, hi)
        entry_b, exit_b = _on_or_after(bench, lo), _on_or_before(bench, hi)
        host_ret = float(host_s.loc[exit_h] / host_s.loc[entry_h] - 1.0) * 100
        bench_ret = float(bench.loc[exit_b] / bench.loc[entry_b] - 1.0) * 100
        L = int(host_s.index.get_loc(exit_h) - host_s.index.get_loc(entry_h))
        rows.append({
            "year": h.year, "city": h.city, "country": h.country, "ticker": h.ticker,
            "entry": entry_h, "exit": exit_h,
            "host_ret_pct": host_ret, "bench_ret_pct": bench_ret,
            "abn_ret_pct": host_ret - bench_ret, "L": L, "note": h.note,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t-test of mean(x) vs 0 — the study's planned primary (n hosts, each a
    single non-overlapping event)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    p = float(stats.t.sf(abs(t), df=n - 1) * 2) if np.isfinite(t) else float("nan")
    return {"n": n, "mean": mean, "median": float(np.median(x)), "sd": sd, "se": se,
            "t": t, "p": p}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def wilcoxon_test(x: np.ndarray) -> dict:
    """Wilcoxon signed-rank test of the abnormal returns against 0 — the nonparametric
    cross-check that doesn't lean on the one-sample-t's normality assumption at n = 6."""
    stat, p = stats.wilcoxon(np.asarray(x, dtype=float))
    return {"stat": float(stat), "p": float(p)}


def bootstrap_ci(x: np.ndarray, n_boot: int = 20_000, seed: int = 710,
                  alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI on the mean abnormal return (resample the n hosts with
    replacement) — honest about how wide the interval is at n = 6."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.array([rng.choice(x, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "n_boot": n_boot}


def outperform_hit_rate(df: pd.DataFrame) -> dict:
    """Share of hosts whose abnormal return is POSITIVE (host beat the benchmark) — the
    directional myth-check: does a majority actually outperform, as the 'national pride
    rally' story implies they should?"""
    k = int((df["abn_ret_pct"] > 0).sum())
    n = len(df)
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "lo": lo, "hi": hi}


def sensitivity_cut(df: pd.DataFrame, exclude_cities: tuple[str, ...]) -> dict:
    """One-sample t after dropping named confounded/outlier hosts — a sensitivity check,
    NOT a way to relabel the headline (a post-hoc cut of 1-2 of 6 points is exactly the
    kind of snooping the desk warns against; reported for transparency only)."""
    sub = df.loc[~df["city"].isin(exclude_cities), "abn_ret_pct"].values
    return {"excluded": exclude_cities, **one_sample_t(sub)}


# --------------------------------------------------------------------------- #
# Random-window placebo
# --------------------------------------------------------------------------- #
def placebo_pvalue(real: dict[str, pd.Series], df: pd.DataFrame,
                    n_seeds: int = 20, n_draws_per_seed: int = 500,
                    base_seed: int = 710) -> dict:
    """For each host, replace its actual [-6mo..+2mo] window with a RANDOM window of the
    same trading-day length ``L`` on the SAME ticker (vs the benchmark, dates aligned),
    draw a mean-abnormal-return statistic exactly like the headline, and repeat many times
    (n_seeds x n_draws_per_seed). p = share of placebo draws at least as extreme (two-
    sided) as the observed headline mean. Tests whether the observed n=6 mean is unusual
    against "any random 6 windows on these same 6 tickers", not just against a normal
    reference distribution.
    """
    bench = real[dt.BENCH_TICKER]
    obs = float(df["abn_ret_pct"].mean())
    tickers_L = list(zip(df["ticker"], df["L"]))

    def draw_one(ticker: str, L: int, rng: np.random.Generator) -> float | None:
        host_s = real[ticker]
        max_start = len(host_s.index) - L - 1
        if max_start <= 0:
            return None
        start_idx = int(rng.integers(0, max_start))
        entry_h = host_s.index[start_idx]
        exit_h = host_s.index[start_idx + L]
        entry_b = _on_or_after(bench, entry_h)
        exit_b = _on_or_before(bench, exit_h)
        hr = host_s.loc[exit_h] / host_s.loc[entry_h] - 1.0
        br = bench.loc[exit_b] / bench.loc[entry_b] - 1.0
        return float(hr - br) * 100

    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = [draw_one(t, L, rng) for t, L in tickers_L]
            if all(v is not None for v in vals):
                means.append(float(np.mean(vals)))
    means = np.asarray(means)
    p = float((np.abs(means) >= abs(obs)).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()), "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means), "draws": means}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(effect: float, seed: int, n: int = 6) -> dict:
    """Run the headline one-sample t on a synthetic (effect, seed) draw."""
    x = dt.synthetic_world(effect=effect, seed=seed, n=n)
    return one_sample_t(x)


def power_curve(effects: tuple[float, ...], n_seeds: int = 200, n: int = 6,
                 base_seed: int = 710) -> pd.DataFrame:
    """Fraction of seeds where |t| >= 2 fires, per planted effect size (pp) — how big a
    true host effect would need to be to clear the desk's bar at this n and this
    calibrated dispersion. A power/machinery diagnostic, not a market claim."""
    rows = []
    for eff in effects:
        fires = 0
        for s in range(n_seeds):
            t = synthetic_detect(eff, seed=base_seed + s, n=n)["t"]
            fires += int(abs(t) >= 2.0)
        rows.append({"effect_pct": eff, "power": fires / n_seeds})
    return pd.DataFrame(rows)
