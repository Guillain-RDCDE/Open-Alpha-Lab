"""Strategy + inference for Study 731 — Wimbledon-Effect.

The claim, steelmanned: the **Wimbledon fortnight** is the UK market's "summer lull" —
a quiet, thin, low-energy tennis-and-strawberries window. Two things a believer could
mean by that, and we test both:

* **A directional seasonal.** The FTSE/UK market has a distinctive *return* signature
  during the fortnight — a reliable drift (up in the sunny-optimism version, down in the
  City-empties-and-drifts version) that you could position for. We measure it two ways:
  the **raw** EWU window return, and the **abnormal** return (EWU minus the VGK Europe
  benchmark), which strips out the Europe-wide summer drift so what's left is
  *UK-specific* to the window.
* **A volatility lull.** The literal "quiet" claim: daily moves are *smaller* during the
  fortnight than in the surrounding weeks. We measure the realized-vol ratio
  (fortnight vs a symmetric neighbourhood) per year — the study's third, myth-check axis.

**No execution lag, by construction.** The fortnight dates are published years ahead
(``data.WIMBLEDON``), so this is a *calendar-known* window: you can pre-commit to hold
EWU from the close of the last session before the fortnight to the close of the last
session inside it, with zero look-ahead and zero forecasting. Per the desk's rule,
calendar-known windows need no ``shift`` at all.

* **entry** = the last trading close *strictly before* the first Monday (the last
  "pre-Wimbledon" price).
* **exit** = the last trading close on or before the second Sunday (the men's final is a
  non-trading Sunday, so this is the Friday of week two) — a ~10-session hold.

Because each Championships year is a single independent, non-overlapping event (not a
daily panel), the primary statistic is a **one-sample t** of the window return across
years (n = contested years with EWU + VGK coverage). A random-window placebo (drawing
same-length windows at random points in the tickers' own history) checks whether the
observed mean sits inside or outside the ordinary luck cloud. Costs are one-way × NAV
per leg; the market-neutral (long EWU / short Europe) leg pays borrow on the short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as dt

COST_BPS = 5.0            # one-way, per leg
BORROW_BPS_ANNUAL = 50.0  # VGK is deep, liquid, cheap to borrow -- 0.50%/yr on the short
BASE_DAYS = 25            # sessions each side of the fortnight for the vol-lull baseline


# --------------------------------------------------------------------------- #
# Event resolution: hardcoded calendar -> per-year window returns + vol ratios
# --------------------------------------------------------------------------- #
def build_event_table(prices: dict[str, pd.Series], cost_bps: float = COST_BPS,
                      borrow_bps: float = BORROW_BPS_ANNUAL, base_days: int = BASE_DAYS,
                      ) -> pd.DataFrame:
    """One row per contested Championships year: window returns, vol ratio, capture.

    A row is INCLUDED only if EWU and VGK both cover [entry .. exit] plus the
    ``base_days`` neighbourhood on each side (so the vol baseline is well-defined).
    Excluded rows are kept (with a reason) so the funnel is auditable.
    """
    ewu = prices[dt.UK_ETF]
    vgk = prices[dt.EUROPE_BENCHMARK]
    common = ewu.index.intersection(vgk.index).sort_values()
    ewu, vgk = ewu.reindex(common), vgk.reindex(common)
    r_ewu = ewu.pct_change()
    n = len(common)

    rows = []
    for year, s, e in dt.WIMBLEDON:
        if s is None:
            rows.append(dict(year=year, included=False, reason="2020 cancelled (COVID-19)"))
            continue
        start, end = pd.Timestamp(s), pd.Timestamp(e)
        before = common[common < start]
        inwin = common[(common >= start) & (common <= end)]
        if len(before) == 0 or len(inwin) == 0:
            rows.append(dict(year=year, included=False,
                             reason="EWU/VGK predate the fortnight"))
            continue
        p_entry = common.get_loc(before[-1])
        p_exit = common.get_loc(inwin[-1])
        if p_entry - base_days < 0 or p_exit + base_days >= n:
            rows.append(dict(year=year, included=False,
                             reason="insufficient neighbourhood for vol baseline"))
            continue
        d_entry, d_exit = common[p_entry], common[p_exit]

        # window returns (calendar-known: entry close -> exit close, no look-ahead)
        raw = float(ewu.iloc[p_exit] / ewu.iloc[p_entry] - 1.0)
        raw_bench = float(vgk.iloc[p_exit] / vgk.iloc[p_entry] - 1.0)
        abn = raw - raw_bench

        # capture, net of costs (long-only EWU: 1 round trip = 2 one-way legs)
        long_net = raw - 2.0 * cost_bps / 1e4
        # market-neutral long EWU / short VGK: 2 round trips (4 legs) + borrow on short
        hold_days = p_exit - p_entry
        borrow = borrow_bps / 1e4 * (hold_days / 252.0)
        mn_net = abn - 4.0 * cost_bps / 1e4 - borrow

        # volatility lull: realized daily-vol ratio, fortnight vs symmetric neighbourhood
        win_r = r_ewu.iloc[p_entry + 1: p_exit + 1].to_numpy()
        base_r = np.concatenate([
            r_ewu.iloc[p_entry - base_days: p_entry].to_numpy(),
            r_ewu.iloc[p_exit + 1: p_exit + 1 + base_days].to_numpy(),
        ])
        win_vol = float(np.nanstd(win_r, ddof=1))
        base_vol = float(np.nanstd(base_r, ddof=1))
        log_vol_ratio = float(np.log(win_vol / base_vol)) if (win_vol > 0 and base_vol > 0) else float("nan")

        rows.append(dict(
            year=year, included=True, reason="",
            entry_date=str(d_entry.date()), exit_date=str(d_exit.date()),
            hold_days=int(hold_days),
            raw=raw, abn=abn, raw_bench=raw_bench,
            long_net=long_net, mn_net=mn_net,
            win_vol=win_vol, base_vol=base_vol, log_vol_ratio=log_vol_ratio,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> dict:
    """One-sample t of mean(x) vs 0 -- the right unit for independent, non-overlapping
    yearly window events (not a daily panel)."""
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
    """Welch t of mean(a) - mean(b) (unequal variances)."""
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


def hit_rate(x: np.ndarray, positive: bool = True) -> dict:
    """Share of events with sign matching the hypothesis, with a Wilson 95% interval."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    k = int((x > 0).sum()) if positive else int((x < 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}


# --------------------------------------------------------------------------- #
# Random-window placebo: is the observed mean inside the luck cloud?
# --------------------------------------------------------------------------- #
def placebo_pvalue(events: pd.DataFrame, prices: dict[str, pd.Series], col: str,
                   win_len: int | None = None, cost_bps: float = 0.0,
                   n_seeds: int = 20, n_draws_per_seed: int = 200, base_seed: int = 731,
                   tail: str = "two") -> dict:
    """For each INCLUDED year, redraw a random same-length window at a random point in
    the EWU/VGK common history and recompute the same column; average across the same n
    years; repeat n_seeds x n_draws_per_seed times.

    ``col`` is one of ``raw`` (EWU window return), ``abn`` (EWU - VGK), ``long_net`` or
    ``mn_net`` (net-of-cost captures). ``win_len`` defaults to each event's own hold
    length (they differ by a session or two across years); passing an int fixes it.
    ``tail``: "two" (share of |null| >= |observed|), "right" or "left".
    """
    ewu = prices[dt.UK_ETF]; vgk = prices[dt.EUROPE_BENCHMARK]
    common = ewu.index.intersection(vgk.index).sort_values()
    ewu, vgk = ewu.reindex(common), vgk.reindex(common)
    inc = events[events["included"]]
    obs = float(inc[col].mean())
    holds = inc["hold_days"].tolist()
    n_ev = len(inc)
    N = len(common)

    def window_val(p0: int, hd: int) -> float:
        p1 = p0 + hd
        raw = ewu.iloc[p1] / ewu.iloc[p0] - 1.0
        raw_b = vgk.iloc[p1] / vgk.iloc[p0] - 1.0
        abn = raw - raw_b
        if col == "raw":
            return float(raw)
        if col == "long_net":
            return float(raw - 2.0 * cost_bps / 1e4)
        if col == "mn_net":
            borrow = BORROW_BPS_ANNUAL / 1e4 * (hd / 252.0)
            return float(abn - 4.0 * cost_bps / 1e4 - borrow)
        return float(abn)  # "abn"

    means = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        for _ in range(n_draws_per_seed):
            vals = []
            for i in range(n_ev):
                hd = int(win_len) if win_len else int(holds[i])
                hi = N - hd - 1
                if hi <= 1:
                    continue
                p0 = int(rng.integers(1, hi))
                vals.append(window_val(p0, hd))
            if vals:
                means.append(float(np.mean(vals)))
    means = np.asarray(means)
    if tail == "right":
        p = float((means >= obs).mean())
    elif tail == "left":
        p = float((means <= obs).mean())
    else:
        p = float((np.abs(means) >= abs(obs)).mean())
    return {"obs": obs, "placebo_mean": float(means.mean()),
            "placebo_sd": float(means.std(ddof=1)),
            "p_value": p, "n_draws": len(means)}


# --------------------------------------------------------------------------- #
# The vol-lull third axis: is the fortnight actually quieter?
# --------------------------------------------------------------------------- #
def vol_lull_stats(events: pd.DataFrame) -> dict:
    """One-sample t of the per-year log realized-vol ratio (fortnight vs neighbourhood).

    A negative mean = the fortnight is quieter than the weeks around it (lull confirmed);
    near zero = no lull. Reports the mean ratio in plain (non-log) terms too.
    """
    inc = events[events["included"]]
    lr = inc["log_vol_ratio"].to_numpy(dtype=float)
    s = one_sample_t(lr)
    quieter = int((lr < 0).sum())
    lo, hi = wilson_interval(quieter, len(lr[np.isfinite(lr)]))
    return {"n": s["n"], "mean_log": s["mean"], "t": s["t"],
            "mean_ratio": float(np.exp(s["mean"])) if np.isfinite(s["mean"]) else float("nan"),
            "quieter_k": quieter, "quieter_rate": quieter / s["n"] if s["n"] else float("nan"),
            "quieter_lo": lo, "quieter_hi": hi}


# --------------------------------------------------------------------------- #
# Event anatomy: mean cumulative abnormal return by session offset within the window
# --------------------------------------------------------------------------- #
def car_path(events: pd.DataFrame, prices: dict[str, pd.Series], max_k: int = 10,
             col: str = "abn") -> pd.Series:
    """Mean cumulative (abnormal or raw) return at offsets 0..max_k from the entry close,
    averaged across all INCLUDED years. ``col='abn'`` for EWU-VGK, ``'raw'`` for EWU."""
    ewu = prices[dt.UK_ETF]; vgk = prices[dt.EUROPE_BENCHMARK]
    common = ewu.index.intersection(vgk.index).sort_values()
    ewu, vgk = ewu.reindex(common), vgk.reindex(common)
    inc = events[events["included"]]
    paths = []
    for _, row in inc.iterrows():
        p0 = common.get_loc(pd.Timestamp(row["entry_date"]))
        vals = []
        for k in range(0, max_k + 1):
            if p0 + k >= len(common):
                vals.append(np.nan); continue
            raw = ewu.iloc[p0 + k] / ewu.iloc[p0] - 1.0
            if col == "raw":
                vals.append(float(raw))
            else:
                raw_b = vgk.iloc[p0 + k] / vgk.iloc[p0] - 1.0
                vals.append(float(raw - raw_b))
        paths.append(vals)
    arr = np.asarray(paths)
    return pd.Series(np.nanmean(arr, axis=0), index=range(0, max_k + 1))


# --------------------------------------------------------------------------- #
# Tradability summary
# --------------------------------------------------------------------------- #
def capture_summary(events: pd.DataFrame) -> dict:
    """Headline gross/net stats for both trade constructions across years."""
    inc = events[events["included"]]
    out = {}
    for label, gross_col, net_col in (("long_only", "raw", "long_net"),
                                       ("market_neutral", "abn", "mn_net")):
        g = one_sample_t(inc[gross_col].to_numpy())
        nt = one_sample_t(inc[net_col].to_numpy())
        hr = hit_rate(inc[net_col].to_numpy())
        out[label] = {"gross_mean": g["mean"], "gross_t": g["t"],
                      "net_mean": nt["mean"], "net_t": nt["t"],
                      "win_k": hr["k"], "win_n": hr["n"], "win_rate": hr["rate"]}
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(bump: float, seed: int) -> dict:
    """Run the one-sample-t detector on a synthetic paired world with a planted
    fortnight seasonal on the asset (abnormal = asset - benchmark window return)."""
    a, b, wins = dt.synthetic_world(bump=bump, seed=seed)
    ar = []
    for w0, w1 in wins:
        ra = a.iloc[w0:w1].sum()   # cumulative asset log-return over the window
        rb = b.iloc[w0:w1].sum()
        ar.append(float(ra - rb))
    return one_sample_t(np.asarray(ar))
