"""The mechanical Wyckoff proxy + its honest controls — Study 446.

Wyckoff's method describes a market cycle of **accumulation → markup → distribution →
markdown**, orchestrated by a "Composite Operator" who builds positions inside sideways
**trading ranges** and then drives the trend. The two events every Wyckoff trader watches
for inside a range are:

* the **Spring** (accumulation): price briefly breaks **below** the range's support, fails
  to follow through (ideally on *low* volume = no real supply), and snaps back into the
  range. Wyckoff says this "shakes out" the weak holders and precedes the **markup** — so a
  Spring is a *buy*.
* the **Upthrust** (distribution): the mirror image — price pokes **above** resistance,
  fails (no real demand), and falls back into the range, preceding the **markdown** — a
  *sell*.

The method as a whole — phase labelling (A–E), the nine buying/selling tests, "effort vs
result" — is **irreducibly subjective**: the right phase is only clear in hindsight and two
analysts annotate the same chart differently. There is no single falsifiable rule. So we
test the **tightest mechanical version proponents accept**:

1. **Trading-range detector.** A ZigZag swing detector marks pivots; a window of recent
   pivots whose highs and lows sit inside a tight band defines a sideways **range** with a
   support and a resistance level.

2. **Spring / Upthrust events with volume confirmation.** Inside (or just after) a range,
   a Spring is a bar whose *low* pierces support but whose *close* recovers back above it;
   an Upthrust is a bar whose *high* pierces resistance but whose *close* falls back below
   it. The optional **volume filter** (``vol_confirm``) requires the test bar's volume to be
   *below* its recent average — the classic "no supply / no demand" tell.

3. **The prediction (the tradable claim).** A Spring → go **long**; an Upthrust → go
   **short**. We enter **one day after** the event bar (no look-ahead) and hold ``horizon``
   days.

Inference — the desk's shared spirit:

  * a **one-sample t** (and a HAC/Newey-West t) of the directional forward return vs 0;
  * a **label-shuffle placebo**: keep the *same* event bars, randomise the direction with a
    fair coin, many times — the honest answer to "does the Spring/Upthrust *label* add
    directional information over a coin at the same bars?";
  * a **win-rate** vs the 50% base rate;
  * **one-day execution lag** and **one-way costs × turnover** (shorts handled by sign).

No look-ahead: an event at bar t is fully observable on its own close (low/high pierce +
close recovery + that day's volume vs its trailing average); we enter at t+1's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)


# --------------------------------------------------------------------------- #
# ZigZag swing detector (the range-bracketing tool)
# --------------------------------------------------------------------------- #
def zigzag(close: np.ndarray, pct: float = 0.05) -> pd.DataFrame:
    """Percentage ZigZag pivots on a close series.

    Walk forward tracking the running extreme since the last confirmed pivot. A new pivot is
    *confirmed* the first time price reverses by at least ``pct`` from that extreme; the pivot
    is stamped at the bar of the extreme (``piv_idx``) but only *known* at the reversal bar
    (``conf_idx``).

    Returns a DataFrame of alternating pivots with columns:
      ``piv_idx`` (bar of the extreme), ``conf_idx`` (bar it was confirmed = known),
      ``price`` (extreme price), ``kind`` (+1 = swing high, -1 = swing low).
    """
    n = len(close)
    if n < 3:
        return pd.DataFrame(columns=["piv_idx", "conf_idx", "price", "kind"])

    pivots = []
    hi_idx = lo_idx = 0
    hi_px = lo_px = close[0]
    direction = 0  # 0 = undetermined, +1 = uptrend (seeking next high), -1 = downtrend

    for i in range(1, n):
        px = close[i]
        if direction >= 0 and px > hi_px:
            hi_px, hi_idx = px, i
        if direction <= 0 and px < lo_px:
            lo_px, lo_idx = px, i

        if direction == 0:
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
            elif px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i
        elif direction > 0:
            if px <= hi_px * (1.0 - pct):
                pivots.append((hi_idx, i, hi_px, +1))
                direction = -1
                lo_px, lo_idx = px, i
        else:
            if px >= lo_px * (1.0 + pct):
                pivots.append((lo_idx, i, lo_px, -1))
                direction = +1
                hi_px, hi_idx = px, i

    piv = pd.DataFrame(pivots, columns=["piv_idx", "conf_idx", "price", "kind"])
    return piv.drop_duplicates(subset=["piv_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trading ranges from the pivot sequence
# --------------------------------------------------------------------------- #
def trading_ranges(pivots: pd.DataFrame, min_pivots: int = 4,
                   max_width: float = 0.20) -> pd.DataFrame:
    """Detect sideways **trading ranges** from a ZigZag pivot sequence.

    A range is a window of ``min_pivots`` consecutive pivots whose extremes sit inside a band
    no wider than ``max_width`` (e.g. 0.20 = top within 20% of bottom). The range's
    **support** is the lowest pivot low in the window, the **resistance** the highest pivot
    high. The range is considered *established* (and thus tradable) only from the confirmation
    bar of its last pivot onward — so any Spring/Upthrust we read off it is strictly after the
    range is public.

    Returns one row per range: ``start_idx`` (first pivot's bar), ``ready_idx`` (last pivot's
    confirmation bar = when the range is known), ``support``, ``resistance``.
    """
    out = []
    n = len(pivots)
    for k in range(min_pivots - 1, n):
        seg = pivots.iloc[k - min_pivots + 1:k + 1]
        lo = float(seg["price"].min())
        hi = float(seg["price"].max())
        if lo <= 0:
            continue
        width = hi / lo - 1.0
        if width > max_width:
            continue
        out.append({
            "start_idx": int(seg.iloc[0]["piv_idx"]),
            "ready_idx": int(seg.iloc[-1]["conf_idx"]),
            "support": lo, "resistance": hi,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Spring / Upthrust events with optional volume confirmation
# --------------------------------------------------------------------------- #
def spring_upthrust_events(bars: pd.DataFrame, ranges: pd.DataFrame,
                           vol_confirm: bool = True, vol_lookback: int = 20,
                           lookahead: int = 30, pierce: float = 0.0) -> pd.DataFrame:
    """Find Spring (accumulation) and Upthrust (distribution) events inside detected ranges.

    For each established range we scan the ``lookahead`` bars after its ``ready_idx`` for the
    first qualifying event:

    * **Spring** (``dir = +1``, a buy): a bar whose **low** pierces ``support`` (low < support
      × (1 - ``pierce``)) but whose **close recovers** back above support — a false breakdown.
    * **Upthrust** (``dir = -1``, a sell): a bar whose **high** pierces ``resistance`` but
      whose **close falls back** below it — a false breakout.

    If ``vol_confirm`` is True, the event bar's volume must be **below** its trailing
    ``vol_lookback``-day average (the Wyckoff "no supply / no demand" tell). The event is fully
    observable on its own close, so the backtest enters at the *next* bar (no look-ahead).

    Returns one row per event: ``sig_idx`` (event bar = known on its close), ``dir`` (+1 / -1),
    ``kind`` ("spring"/"upthrust"), ``range_idx`` (which range it came from).
    """
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float)
    n = len(close)
    # trailing average volume (shifted so the event bar itself is excluded from its own mean)
    volavg = pd.Series(vol).rolling(vol_lookback, min_periods=5).mean().shift(1).to_numpy()

    out = []
    seen = set()  # de-dup events that several overlapping ranges would flag on the same bar
    for ri, r in ranges.iterrows():
        sup, res = r["support"], r["resistance"]
        start = int(r["ready_idx"]) + 1
        stop = min(start + lookahead, n)
        for t in range(start, stop):
            va = volavg[t]
            vol_ok = (not vol_confirm) or (np.isfinite(va) and vol[t] < va)
            # Spring: low pierces support, close recovers above support
            is_spring = (low[t] < sup * (1.0 - pierce)) and (close[t] > sup)
            # Upthrust: high pierces resistance, close falls back below resistance
            is_upthrust = (high[t] > res * (1.0 + pierce)) and (close[t] < res)
            if is_spring and vol_ok and ("spring", t) not in seen:
                out.append({"sig_idx": t, "dir": +1, "kind": "spring", "range_idx": int(ri)})
                seen.add(("spring", t))
                break
            if is_upthrust and vol_ok and ("upthrust", t) not in seen:
                out.append({"sig_idx": t, "dir": -1, "kind": "upthrust", "range_idx": int(ri)})
                seen.add(("upthrust", t))
                break
    ev = pd.DataFrame(out, columns=["sig_idx", "dir", "kind", "range_idx"])
    # one event per bar (keep the first)
    return ev.drop_duplicates(subset=["sig_idx"]).reset_index(drop=True)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of +/-1 — the placebo arm's coin (same event bars)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# --------------------------------------------------------------------------- #
# Fixed-horizon forward-return backtest
# --------------------------------------------------------------------------- #
def run_trades(bars: pd.DataFrame, entries: pd.DataFrame, horizon: int = 20,
               cost_bps: float = 1.0, directions: np.ndarray | None = None) -> pd.DataFrame:
    """Run fixed-horizon directional trades and return a per-trade ledger.

    For each event bar ``sig_idx`` (known on its own close), enter at the **next** bar's close
    (one-day execution lag) and exit ``horizon`` trading days later (or the last available
    bar). ``directions`` overrides the entry sign vector (the placebo passes a +/-1 coin
    aligned to ``entries``). ``cost_bps`` is a one-way cost; a round trip pays it twice,
    deducted from the gross return (shorts handled by the direction sign).

    Columns: ``entry_idx, dir, kind, ret_gross, ret_net``.
    """
    close = bars["close"].to_numpy(dtype=float)
    n = len(close)
    cols = ["entry_idx", "dir", "kind", "ret_gross", "ret_net"]
    if len(entries) == 0:
        return pd.DataFrame(columns=cols)
    dirs = (np.asarray(directions, dtype=int) if directions is not None
            else entries["dir"].to_numpy(dtype=int))
    kinds = (entries["kind"].to_numpy() if "kind" in entries
             else np.array(["?"] * len(entries)))
    rows = []
    for sig, d, kind in zip(entries["sig_idx"].to_numpy(dtype=int), dirs, kinds):
        e = sig + 1                       # enter one day after the event is known
        x = e + horizon
        if e >= n or x >= n:
            continue
        gross = d * (close[x] / close[e] - 1.0)
        rows.append({"entry_idx": e, "dir": int(d), "kind": kind,
                     "ret_gross": gross, "ret_net": gross - 2.0 * cost_bps * 1e-4})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def hac_t(sample: np.ndarray) -> float:
    """Newey-West (HAC) t-stat on the mean of ``sample`` (no quantlab dependency)."""
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger."""
    if len(ledger) == 0:
        return {k: float("nan") for k in
                ["win_rate", "mean_bps", "sharpe", "t", "hac_t"]} | {"n_trades": 0}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": ttest_vs_zero(r),
        "hac_t": hac_t(r),
    }


def placebo_pvalue(bars: pd.DataFrame, entries: pd.DataFrame, horizon: int = 20,
                   n_draws: int = 5000, seed: int = 446, cost_bps: float = 0.0) -> dict:
    """Label-shuffle placebo: same event bars, random +/-1 direction, many times.

    ``p`` = P[shuffled mean forward return >= observed]. The honest test for whether the
    Spring/Upthrust *label* adds directional information over a coin flip placed at the very
    same bars. Run on gross returns (cost_bps=0) so the placebo measures pure direction.
    """
    led = run_trades(bars, entries, horizon, cost_bps=cost_bps)
    if len(led) == 0:
        return {"obs": float("nan"), "p_value": float("nan"), "draws": np.array([])}
    obs = float(led["ret_gross"].mean())
    mags = (led["ret_gross"].to_numpy() * led["dir"].to_numpy())  # un-signed move
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        d = rng.choice([-1, 1], size=len(mags))
        draws[i] = float((d * mags).mean())
    p = float((draws >= obs).mean())
    return {"obs": obs, "p_value": p, "draws": draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def find_events(bars: pd.DataFrame, pct: float = 0.05, min_pivots: int = 4,
                max_width: float = 0.20, vol_confirm: bool = True,
                vol_lookback: int = 20, lookahead: int = 30,
                pierce: float = 0.0) -> pd.DataFrame:
    """Convenience: ZigZag -> trading ranges -> Spring/Upthrust events, one call."""
    piv = zigzag(bars["close"].to_numpy(dtype=float), pct=pct)
    rng = trading_ranges(piv, min_pivots=min_pivots, max_width=max_width)
    return spring_upthrust_events(bars, rng, vol_confirm=vol_confirm,
                                  vol_lookback=vol_lookback, lookahead=lookahead,
                                  pierce=pierce)


def run_experiment(bars: pd.DataFrame, pct: float = 0.05, horizon: int = 20,
                   cost_bps: float = 1.0, min_pivots: int = 4, max_width: float = 0.20,
                   vol_confirm: bool = True, vol_lookback: int = 20, lookahead: int = 30,
                   n_draws: int = 5000, seed: int = 446) -> dict:
    """Orchestrator: ranges -> Spring/Upthrust events -> backtest + inference + placebo + coin.

    Returns the headline dict for one tape/horizon: the Wyckoff long/short stats, the
    same-bars random-direction coin control, and the label-shuffle placebo p-value.
    """
    ent = find_events(bars, pct=pct, min_pivots=min_pivots, max_width=max_width,
                      vol_confirm=vol_confirm, vol_lookback=vol_lookback,
                      lookahead=lookahead)
    led = run_trades(bars, ent, horizon, cost_bps=cost_bps)
    wave = summarize(led, "ret_net")
    wave_gross = summarize(led, "ret_gross")
    coin_led = run_trades(bars, ent, horizon, cost_bps=cost_bps,
                          directions=random_directions(len(ent), seed=seed))
    coin = summarize(coin_led, "ret_gross")
    plac = placebo_pvalue(bars, ent, horizon, n_draws=n_draws, seed=seed)
    n_spring = int((ent["dir"] == 1).sum()) if len(ent) else 0
    n_upthrust = int((ent["dir"] == -1).sum()) if len(ent) else 0
    return {
        "n_entries": int(len(ent)), "n_spring": n_spring, "n_upthrust": n_upthrust,
        "wave": wave, "wave_gross": wave_gross, "coin": coin,
        "placebo_p": plac["p_value"], "placebo_obs": plac["obs"],
        "pct": pct, "horizon": horizon, "vol_confirm": vol_confirm,
    }
