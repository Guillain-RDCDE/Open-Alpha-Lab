"""Strategy + inference for Study 732 — Tour-de-France-Effect.

The claim: **French equities enjoy a feel-good July seasonal while the country watches
the Tour de France** -- the summer-holiday cousin of sports-sentiment folklore, recast
as a three-week *calendar window* rather than a surprise result.

Because the Tour dates are fixed and published a **year in advance**, this is a
calendar-known window, which hands the study a clean, look-ahead-free execution
convention for free -- no surprise, no information lag, no un-tradable weekend jump to
strip out (contrast study 708, where the Eurovision winner is revealed on a non-trading
Saturday night):

* **entry** = the last trading close *before* the Grand Depart. A believer knows the
  race starts tomorrow and can be positioned at this close.
* **exit**  = the first trading close on/after the final stage (the Champs-Elysees Sunday
  is a non-trading day, so this is ordinarily the Monday close). The whole race is held.

Two measurements per edition, both over that window:

* **Raw seasonal.** ``EWQ`` total return, entry->exit -- what a "buy French stocks
  during the Tour" believer actually earns. The catch: July sits inside the "Sell in May"
  summer-weakness window, so this number is fighting a known seasonal headwind.
* **Abnormal (France-specific).** ``EWQ`` minus the ``VGK`` Europe benchmark over the
  same window -- the only measurement that can separate a genuine *French* sentiment bump
  from ordinary pan-European summer beta. (VGK inception 2005 caps this to 2005->2025.)

A longer-history ``^FCHI`` (CAC 40, **price-only**, dividends NOT reinvested -- labelled
everywhere) cross-checks the raw seasonal.

Each Tour edition is a single independent, non-overlapping annual event, so the primary
statistic is a **one-sample t** of the window return across editions (not a daily panel).
A random-window placebo (drawing same-length windows from *elsewhere* in EWQ's own
history) checks whether the July-Tour window is unusual or just an ordinary three weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

COST_BPS = 5.0  # one-way, per leg


# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #
def _window_positions(index: pd.DatetimeIndex, gd: str, fs: str):
    """(entry_pos, exit_pos) on ``index``: last trading day < Grand Depart, first
    trading day >= final stage. Returns None if the window is not fully covered."""
    idx = index.sort_values()
    gd_ts, fs_ts = pd.Timestamp(gd), pd.Timestamp(fs)
    before = idx[idx < gd_ts]
    after = idx[idx >= fs_ts]
    if len(before) == 0 or len(after) == 0:
        return None
    entry = idx.get_loc(before[-1])
    exit_ = idx.get_loc(after[0])
    if exit_ <= entry:
        return None
    return entry, exit_


def _ret(series: pd.Series, gd: str, fs: str):
    """Simple return of ``series`` over the Tour window, or None if not covered."""
    pos = _window_positions(series.index, gd, fs)
    if pos is None:
        return None
    idx = series.index.sort_values()
    e, x = pos
    return float(series.loc[idx[x]] / series.loc[idx[e]] - 1.0), (x - e)


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-edition window returns
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS
                      ) -> pd.DataFrame:
    """One row per Tour edition: raw / abnormal / CAC-price window returns + costs.

    A row is INCLUDED (for the raw seasonal) if ``EWQ`` covers [entry, exit]. The
    abnormal column is populated only when ``VGK`` also covers the window (2005->);
    the CAC-price column only when ``^FCHI`` covers it. Nothing is dropped silently --
    every edition appears, with NaNs where a series doesn't reach.
    """
    ewq = prices[dt.FRANCE_ETF]
    vgk = prices[dt.EUROPE_BENCHMARK]
    cac = prices[dt.CAC_INDEX]
    common = ewq.index.intersection(vgk.index).sort_values()

    rows = []
    for year, gd, fs, note in dt.EVENTS:
        row = dict(year=year, grand_depart=gd, final_stage=fs, note=note)
        raw = _ret(ewq, gd, fs)
        if raw is None:
            row.update(included=False, reason="EWQ does not cover the window")
            rows.append(row)
            continue
        raw_ret, win = raw
        # abnormal: EWQ - VGK over the same window, on the shared trading calendar
        ar = None
        pos = _window_positions(common, gd, fs)
        if pos is not None:
            e, x = pos
            r_ewq = float(ewq.loc[common[x]] / ewq.loc[common[e]] - 1.0)
            r_vgk = float(vgk.loc[common[x]] / vgk.loc[common[e]] - 1.0)
            ar = r_ewq - r_vgk
        cac_r = _ret(cac, gd, fs)
        cac_raw = cac_r[0] if cac_r is not None else None
        row.update(
            included=True, reason="",
            win_sessions=win,
            raw_ret=raw_ret,
            raw_net=raw_ret - 2.0 * cost_bps / 1e4,
            ar=ar if ar is not None else np.nan,
            has_ar=ar is not None,
            cac_raw=cac_raw if cac_raw is not None else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    annual events (not a daily panel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else float("nan"),
                "sd": float("nan"), "t": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    return {"n": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "t": float(x.mean() / se) if se > 0 else float("nan")}


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either group < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
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
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    k = int((x > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-window placebo: is the Tour window unusual vs ordinary three weeks?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], kind: str,
                   cost_bps: float = 0.0, n_seeds: int = 20, n_draws_per_seed: int = 200,
                   base_seed: int = 732, tail: str = "right") -> dict:
    """Redraw, for each INCLUDED edition, a random same-length window from ELSEWHERE in
    the relevant series' own history; average across editions; repeat n_seeds x n_draws.

    ``kind`` = "raw" (EWQ total return, net of ``cost_bps`` per leg) or "ar" (EWQ - VGK
    abnormal). ``tail`` = "right" (claim predicts a positive bump -> p = share of null
    means >= observed) or "left". Random windows may fall in any month, so the null is
    "an ordinary three weeks for these tickers", against which the July-Tour mean is
    compared.
    """
    ewq = prices[dt.FRANCE_ETF]
    vgk = prices[dt.EUROPE_BENCHMARK]
    inc = events[events["included"]].copy()
    if kind == "ar":
        inc = inc[inc["has_ar"]]
        obs = float(inc["ar"].mean())
        idx = ewq.index.intersection(vgk.index).sort_values()
        ewq_a = ewq.reindex(idx); vgk_a = vgk.reindex(idx)
    else:
        obs = float((inc["raw_ret"] - 2.0 * cost_bps / 1e4).mean())
        idx = ewq.index.sort_values()
        ewq_a = ewq.reindex(idx)
    wins = inc["win_sessions"].astype(int).tolist()

    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = []
            for w in wins:
                hi = len(idx) - w - 1
                if hi <= 0:
                    continue
                p = int(rng.integers(0, hi))
                if kind == "ar":
                    r = (float(ewq_a.iloc[p + w] / ewq_a.iloc[p] - 1.0)
                         - float(vgk_a.iloc[p + w] / vgk_a.iloc[p] - 1.0))
                else:
                    r = float(ewq_a.iloc[p + w] / ewq_a.iloc[p] - 1.0) - 2.0 * cost_bps / 1e4
                vals.append(r)
            if vals:
                means.append(np.mean(vals))
    means = np.asarray(means)
    p = float((means >= obs).mean()) if tail == "right" else float((means <= obs).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)), "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative return by session offset from entry
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], kind: str,
             max_k: int = 16) -> pd.Series:
    """Mean cumulative return at each offset 0..max_k from entry (the pre-Depart close),
    averaged across all INCLUDED editions. ``kind`` = "raw" (EWQ) or "ar" (EWQ - VGK)."""
    ewq = prices[dt.FRANCE_ETF]
    vgk = prices[dt.EUROPE_BENCHMARK]
    if kind == "ar":
        idx = ewq.index.intersection(vgk.index).sort_values()
    else:
        idx = ewq.index.sort_values()
    inc = events[events["included"]]
    paths = []
    for _, row in inc.iterrows():
        pos = _window_positions(idx, row["grand_depart"], row["final_stage"])
        if pos is None:
            continue
        e, _x = pos
        if e + max_k >= len(idx):
            continue
        vals = []
        for k in range(0, max_k + 1):
            if kind == "ar":
                r = (float(ewq.loc[idx[e + k]] / ewq.loc[idx[e]] - 1.0)
                     - float(vgk.loc[idx[e + k]] / vgk.loc[idx[e]] - 1.0))
            else:
                r = float(ewq.loc[idx[e + k]] / ewq.loc[idx[e]] - 1.0)
            vals.append(r)
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(arr.mean(axis=0), index=range(0, max_k + 1))


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int) -> dict:
    """Run the one-sample-t abnormal detector on a synthetic paired (France, Europe)
    world with a planted per-day seasonal bump on its scheduled window calendar."""
    f, e, windows = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for entry, exit_ in windows:
        rf = f.iloc[entry + 1:exit_ + 1].sum()   # cumulative log-return over the window
        re = e.iloc[entry + 1:exit_ + 1].sum()
        ar.append(float(rf - re))
    return one_sample_t(np.asarray(ar))
