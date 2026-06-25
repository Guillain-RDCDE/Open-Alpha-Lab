"""New-Highs-New-Lows breadth as a falsifiable mechanical rule — Study 493.

The **new-highs/new-lows (NH-NL) line** is a classic breadth indicator: count how many members
of a market basket are individually printing a fresh 52-week *high*, subtract those printing a
fresh 52-week *low*, and track the *net* over time. The folklore — taught since Charles Dow,
popularised by William O'Neil / Investor's Business Daily, and central to the Hindenburg-Omen
literature — is that **breadth leads price**: the NH-NL line tops and bottoms *before* the
index, and a surge in net new highs (a "breadth thrust") forecasts higher index returns while a
collapse warns of a top.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **52-week extremes, trailing only.** A member is "at a new high" on bar *t* if its close
   equals its trailing ``lookback``-day maximum (look-back includes *t*, no future data). The
   **net new-high fraction** ``b_t = (#high - #low) / N`` is the NH-NL breadth line.
2. **Breadth-thrust entry.** A long fires when the smoothed NH-NL line crosses **up** through a
   positive threshold (breadth expanding from neutral) — the "thrust forecasts the index" rule.
   The signal is read on the close of *t*; the trade is entered at the **next close** (one
   documented lag), then we measure the forward H-day return on the index (SPY).
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the index's drift, and (b) a **shuffled-membership placebo** that permutes which member's
   new-high series feeds each basket slot *per day* — destroying the cross-sectional breadth
   structure while keeping every member's marginal new-high rate and the daily count
   distribution. The honest "is the *breadth aggregation* load-bearing?" null.

No look-ahead: extremes use only trailing bars, breadth is read on the close of *t*, the
position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
LOOKBACK = 252            # ~52 weeks of trading days
SMOOTH = 10              # smoothing of the raw NH-NL line (a 10-day MA, the IBD-style line)
THRESH = 0.20            # breadth-thrust threshold on the smoothed net-new-high fraction


# --------------------------------------------------------------------------- #
# Breadth construction
# --------------------------------------------------------------------------- #
def _aligned_closes(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack the basket members' close columns on a common (inner-join) calendar."""
    cols = {t: b["close"] for t, b in panel.items()}
    df = pd.DataFrame(cols).dropna(how="any")
    return df


def net_new_high_line(panel: dict[str, pd.DataFrame], lookback: int = LOOKBACK,
                      smooth: int = SMOOTH) -> pd.Series:
    """The NH-NL breadth line: smoothed net fraction of members at a fresh 52-week extreme.

    For each member, "new high" on bar *t* iff close_t == trailing ``lookback``-max (incl. *t*);
    "new low" symmetrically. ``b_t = (#high - #low) / N``, smoothed with a ``smooth``-day mean.
    Trailing-only (no look-ahead). Returned over the basket's common calendar.
    """
    closes = _aligned_closes(panel)
    roll_max = closes.rolling(lookback, min_periods=lookback).max()
    roll_min = closes.rolling(lookback, min_periods=lookback).min()
    at_high = (closes >= roll_max - 1e-9)
    at_low = (closes <= roll_min + 1e-9)
    net = (at_high.sum(axis=1) - at_low.sum(axis=1)) / float(closes.shape[1])
    return net.rolling(smooth, min_periods=smooth).mean()


def breadth_thrust_entries(panel: dict[str, pd.DataFrame], index_ticker: str = "SPY",
                           lookback: int = LOOKBACK, smooth: int = SMOOTH,
                           thresh: float = THRESH) -> pd.DatetimeIndex:
    """Bars where the NH-NL line crosses **up** through ``+thresh`` — the breadth-thrust buy.

    Only the *first* bar of each up-cross is kept (the thrust, not every day breadth stays
    elevated). Entry is executed at the next close by :func:`forward_returns`.
    """
    line = net_new_high_line(panel, lookback=lookback, smooth=smooth)
    above = (line >= thresh) & line.notna()
    cross = above & ~above.shift(1, fill_value=False)
    dates = line.index[cross.to_numpy()]
    # restrict to dates that exist on the index tape
    idx = panel[index_ticker].index
    return pd.DatetimeIndex([d for d in dates if d in set(idx)])


def random_entries(index_close: pd.Series, n: int, warmup: int = LOOKBACK + SMOOTH,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = index_close.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r),
    }


def shuffled_membership_placebo(panel: dict[str, pd.DataFrame], index_ticker: str = "SPY",
                                horizon: int = 20, lookback: int = LOOKBACK,
                                smooth: int = SMOOTH, thresh: float = THRESH,
                                n_draws: int = 500, seed: int = 493) -> dict:
    """Placebo: scramble the *cross-sectional* breadth structure, keep the marginals.

    For each draw we independently permute, **per member, across time**, that member's
    new-high / new-low boolean series. Each member keeps its exact marginal rate of new highs
    and lows, but the *co-movement* that makes a genuine breadth thrust is destroyed — the days
    on which many members are simultaneously at highs become random. We rebuild the smoothed
    NH-NL line, fire the same thrust rule, and record the share of placebo runs whose mean
    forward index return **beats** the real one — the honest "is the breadth aggregation doing
    anything?" p-value, plus the observed mean.
    """
    line = net_new_high_line(panel, lookback=lookback, smooth=smooth)
    ent = breadth_thrust_entries(panel, index_ticker, lookback, smooth, thresh)
    idx_close = panel[index_ticker]["close"]
    obs = float(np.mean(forward_returns(idx_close, ent, horizon))) if len(ent) else float("nan")

    closes = _aligned_closes(panel)
    roll_max = closes.rolling(lookback, min_periods=lookback).max()
    roll_min = closes.rolling(lookback, min_periods=lookback).min()
    at_high = (closes >= roll_max - 1e-9).astype(float)
    at_low = (closes <= roll_min + 1e-9).astype(float)
    valid_mask = closes.rolling(lookback, min_periods=lookback).max().notna().all(axis=1)
    cal = closes.index
    idx_set = set(idx_close.index)

    rng = np.random.default_rng(seed)
    hi = at_high.to_numpy()
    lo = at_low.to_numpy()
    vrows = np.where(valid_mask.to_numpy())[0]
    beats, valid = 0, 0
    for _ in range(n_draws):
        hp = hi.copy(); lp = lo.copy()
        for j in range(hi.shape[1]):                 # permute each member's series in time
            perm = rng.permutation(vrows)
            hp[vrows, j] = hi[perm, j]
            lp[vrows, j] = lo[perm, j]
        net = (hp.sum(axis=1) - lp.sum(axis=1)) / float(hi.shape[1])
        ser = pd.Series(net, index=cal).rolling(smooth, min_periods=smooth).mean()
        above = (ser >= thresh) & ser.notna()
        cross = above & ~above.shift(1, fill_value=False)
        dts = pd.DatetimeIndex([d for d in cal[cross.to_numpy()] if d in idx_set])
        rr = forward_returns(idx_close, dts, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid, "n_entries": int(len(ent))}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict[str, pd.DataFrame], index_ticker: str = "SPY",
                   lookback: int = LOOKBACK, smooth: int = SMOOTH, thresh: float = THRESH,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet: breadth-thrust vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the thrust summary (gross + net), the drift-matched
    random-entry baseline, and the thrust-minus-random delta.
    """
    ent = breadth_thrust_entries(panel, index_ticker, lookback, smooth, thresh)
    idx_close = panel[index_ticker]["close"]
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(idx_close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(idx_close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            idx_close, random_entries(idx_close, max(len(ent), 50), seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
