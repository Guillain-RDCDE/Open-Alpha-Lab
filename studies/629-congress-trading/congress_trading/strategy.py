"""Tested rules + statistics for Study 629 — Congress Trading.

The claim (Ziobrowski et al. 2004): US senators' stock purchases earn large abnormal returns.
The only *replicable* trade is the one a member of the public can copy — buy at the first
close **after the disclosure** hits the public record (exactly ONE execution lag; the
transaction itself is already ~3 weeks old by then, and we say so).

Two inference engines:

* **Event study (color).** Per disclosed purchase, the 3/6/12-month (63/126/252 trading-day)
  buy-and-hold abnormal return vs SPY from the entry close. Pooled means carry a naive t which
  OVERSTATES precision (event windows overlap and cluster by senator/day) — quoted only as
  color, never as the verdict.

* **Calendar-time portfolio (primary — the verdict statistic).** Each day, hold every stock
  disclosed-bought in the past ``hold`` trading days, equal-weighted; the daily portfolio
  return minus SPY is a single non-overlapping daily series whose mean is tested with a
  **Newey-West (HAC) t**. This is the standard fix for overlapping event windows
  (Fama 1998; Barber-Odean calendar-time alphas) and is how the desk grades the Signal axis.

Splits (party, famous-vs-boring, size) use **Welch t** on per-event abnormal returns.
The random-dates placebo is averaged over >= 20 seeds (single-seed baselines are banned).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (63, 126, 252)          # trading days ~ 3 / 6 / 12 months


# --------------------------------------------------------------------------- #
# Core stats
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """(mean, HAC t) of a 1-d series with Newey-West (Bartlett) standard errors."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(e[:-k] @ e[k:]) / n
    se = np.sqrt(s / n)
    return float(mu), float(mu / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(mean(a)-mean(b), Welch t) for two independent samples."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = a.mean() - b.mean()
    return float(diff), float(diff / np.sqrt(va + vb))


def naive_t(x: np.ndarray) -> tuple[float, float, int]:
    """(mean, plain one-sample t, n) — quoted as color only where windows overlap."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    mu = x.mean()
    return float(mu), float(mu / (x.std(ddof=1) / np.sqrt(n))), n


# --------------------------------------------------------------------------- #
# Event study
# --------------------------------------------------------------------------- #
def event_abnormal(px: pd.DataFrame, spy: pd.Series, events: pd.DataFrame,
                   horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Per-event buy-and-hold abnormal returns vs SPY at each horizon.

    Entry = the first trading close STRICTLY AFTER the disclosure date (the one execution
    lag). ``abn_h = (P[t0+h]/P[t0] - 1) - (SPY[t0+h]/SPY[t0] - 1)`` (both total-return).
    Events whose ticker is missing from the panel, whose entry has no price, or whose
    window runs off the end of the tape are dropped (counts reported by the caller).
    """
    dates = px.index
    spy_v = spy.reindex(dates).ffill().values
    max_h = max(horizons)
    rows = []
    pos = dates.searchsorted(events["disclosure_date"].values, side="right")
    for (_, ev), t0 in zip(events.iterrows(), pos):
        tk = ev["ticker"]
        if tk not in px.columns or t0 + max_h >= len(dates):
            continue
        p = px[tk].values
        if np.isnan(p[t0]):
            continue
        rec = dict(ev)
        rec["entry_date"] = dates[t0]
        ok = True
        for h in horizons:
            if np.isnan(p[t0 + h]):
                ok = False
                break
            rec[f"abn_{h}"] = (p[t0 + h] / p[t0] - 1.0) - (spy_v[t0 + h] / spy_v[t0] - 1.0)
            rec[f"raw_{h}"] = p[t0 + h] / p[t0] - 1.0
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows)


def pooled_table(abn: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> list[dict]:
    """Pooled per-event abnormal-return means per horizon (naive t = color only)."""
    out = []
    for h in horizons:
        mu, t, n = naive_t(abn[f"abn_{h}"].values)
        out.append({"h": h, "mean_pct": mu * 100.0, "naive_t": t, "n": n})
    return out


# --------------------------------------------------------------------------- #
# Calendar-time portfolio (primary)
# --------------------------------------------------------------------------- #
def calendar_time(px: pd.DataFrame, spy: pd.Series, events: pd.DataFrame,
                  hold: int = 126, lags: int = 10,
                  cost_bps_oneway: float = 0.0) -> dict:
    """Daily equal-weight portfolio of every stock disclosed-bought in the past ``hold``
    trading days; excess return vs SPY; Newey-West t on the daily mean.

    Entry at the first close after disclosure; each event is held exactly ``hold`` days
    (one round trip). Costs, if any, are charged as one-way ``cost_bps_oneway`` x 2 legs
    spread over the holding period (a per-day drag of ``2*cost/hold`` on the invested slice).
    Returns dict with the daily series and headline stats (alpha in %/yr, HAC t).
    """
    dates = px.index
    n_d = len(dates)
    rets = px.values[1:] / px.values[:-1] - 1.0            # day i -> return dates[i+1]
    spy_ret = (spy.reindex(dates).ffill().values[1:]
               / spy.reindex(dates).ffill().values[:-1] - 1.0)
    col_of = {c: j for j, c in enumerate(px.columns)}

    active = np.zeros((n_d, len(px.columns)), dtype=np.int32)   # holding count per day
    pos = dates.searchsorted(events["disclosure_date"].values, side="right")
    n_used = 0
    for tk, t0 in zip(events["ticker"].values, pos):
        j = col_of.get(tk)
        if j is None or t0 + 1 >= n_d:
            continue
        t1 = min(t0 + hold, n_d - 1)
        # held from entry close t0 -> exit close t1: earns returns on days t0+1 .. t1
        active[t0 + 1:t1 + 1, j] += 1
        n_used += 1

    port, exc, names_held = [], [], []
    for i in range(1, n_d):
        w = active[i]
        held = w > 0
        if not held.any():
            continue
        r = rets[i - 1, held]
        ww = w[held].astype(float)
        good = ~np.isnan(r)
        if not good.any():
            continue
        pr = float(np.average(r[good], weights=ww[good]))
        port.append(pr)
        exc.append(pr - spy_ret[i - 1])
        names_held.append(int(held.sum()))

    exc = np.array(exc)
    drag_daily = 2.0 * (cost_bps_oneway / 1e4) / hold
    exc_net = exc - drag_daily
    mu, t = newey_west_t(exc_net, lags=lags)
    return {"n_days": len(exc), "n_events_used": n_used,
            "avg_names": float(np.mean(names_held)) if names_held else float("nan"),
            "mean_daily_bps": mu * 1e4, "alpha_ann_pct": mu * 252 * 100.0,
            "nw_t": t, "daily_excess": exc_net}


# --------------------------------------------------------------------------- #
# Random-dates placebo (averaged over >= 20 seeds — single-seed baselines banned)
# --------------------------------------------------------------------------- #
def placebo_random_dates(px: pd.DataFrame, spy: pd.Series, events: pd.DataFrame,
                         hold: int = 126, n_seeds: int = 20, base_seed: int = 629) -> dict:
    """Re-run the calendar-time engine with each event's disclosure date replaced by a
    uniform random date on the tape (same tickers, same event count). Averaged over
    ``n_seeds`` seeds; reports the mean and dispersion of the placebo alpha t.
    """
    lo = px.index[30]
    hi = px.index[-(max(HORIZONS) + 5)]
    span_days = (hi - lo).days
    ts, alphas = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        fake = events.copy()
        fake["disclosure_date"] = lo + pd.to_timedelta(
            rng.integers(0, span_days, size=len(events)), unit="D")
        r = calendar_time(px, spy, fake, hold=hold)
        ts.append(r["nw_t"])
        alphas.append(r["alpha_ann_pct"])
    ts, alphas = np.array(ts), np.array(alphas)
    return {"n_seeds": n_seeds, "mean_t": float(np.nanmean(ts)), "sd_t": float(np.nanstd(ts)),
            "mean_alpha_ann_pct": float(np.nanmean(alphas)),
            "sd_alpha_ann_pct": float(np.nanstd(alphas))}


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_welch(abn: pd.DataFrame, mask: pd.Series, h: int = 126) -> dict:
    """Welch t of per-event abnormal returns, group(mask) vs the rest, at horizon h."""
    a = abn.loc[mask, f"abn_{h}"].values
    b = abn.loc[~mask, f"abn_{h}"].values
    diff, t = welch_t(a, b)
    return {"n_a": len(a), "n_b": len(b),
            "mean_a_pct": float(np.nanmean(a)) * 100.0,
            "mean_b_pct": float(np.nanmean(b)) * 100.0,
            "diff_pct": diff * 100.0, "welch_t": t}
