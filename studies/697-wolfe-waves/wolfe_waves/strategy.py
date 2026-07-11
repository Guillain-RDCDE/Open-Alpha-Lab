"""The mechanical Wolfe-Wave detector + its honest controls — Study 697.

Bill Wolfe's "Wolfe Wave" is a five-point reversal pattern: a converging wedge (points 1-2-3-4-5,
alternating swing highs/lows) whose final leg (point 5) *overshoots* the trendline through points
1 and 3 — the "throw-over" — before snapping back. The claimed payoff is the **EPA line**
("Estimated Price at Arrival"): extend the trendline through points 1 and 4 forward, and that is
where price is supposed to travel to, on a schedule you can supposedly time off the same
geometry. Believers call it a "natural equilibrium" the market is structurally drawn back to.

Wolfe's own rules were never published as one canonical checklist — every trading-education site
states a slightly different version, which is itself evidence of how much discretion the pattern
leaves a human chartist. We encode the **one rule every source agrees on** (point 5 pierces the
1-3 trendline) plus the **channel/convergence checks** most sources add (point 3 extends the
wedge, point 4 stays inside the channel and makes a lower high/higher low), and test the one
forward-looking, falsifiable claim: **does price reach the EPA target more often, and closer to
the projected time, than a same-distance target placed on a random day?**

Pipeline:

1. **ZigZag pivots** (percentage reversal filter) — the standard swing-marking tool, identical in
   spirit to sibling 445-elliott-wave's detector, so the two studies share a vocabulary.
2. **``wolfe_candidates``** — slide a window of 5 consecutive alternating pivots and keep the ones
   whose geometry matches the wedge + throw-over rules above (both bullish, reversal UP off a
   converging low, and bearish, reversal DOWN off a converging high).
3. **The EPA target price** — the line through points 1 and 4, evaluated at the bar point 5
   becomes public knowledge (one-day execution lag from there).
4. **The EPA time target** — ``point 5's bar + (point 4's bar - point 1's bar)``, the most
   commonly cited approximation of Wolfe's "time axis" projection (sources vary; we pick one,
   fixed, ex ante, and test whether it predicts anything).
5. **Target-hit test** — walk forward (intraday high/low) and check which comes first: the EPA
   target or the invalidation stop (point 5's own extreme). Hit-rate vs a **same-distance,
   same-direction random-day placebo** on the identical tape (matched target/stop distances,
   different location) is the honest "is this the pattern, or just how far prices wander
   anyway?" null — the same idiom sibling 448-point-and-figure uses for its count target.
6. **Fixed-horizon directional return** (5/10/20/40 days), one-sample + HAC *t*, as a secondary,
   simpler robustness cut that doesn't depend on the stop/target geometry at all.

No look-ahead: a pivot at bar p is only *confirmed* once price has reversed the ZigZag threshold
from it (bar q > p); every entry uses the confirmation bar of point 5, and execution is the
**next** bar's close (one documented execution lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)


# --------------------------------------------------------------------------- #
# ZigZag swing detector (shared vocabulary with 445-elliott-wave)
# --------------------------------------------------------------------------- #
def zigzag(close: np.ndarray, pct: float = 0.04) -> pd.DataFrame:
    """Percentage ZigZag pivots on a close series.

    Walk forward tracking the running extreme since the last confirmed pivot. A new pivot is
    *confirmed* the first time price reverses by at least ``pct`` (e.g. 0.04 = 4%) from that
    extreme; the pivot is stamped at the bar of the extreme (``piv_idx``) but only *known* at
    the reversal bar (``conf_idx``) — the look-ahead-free confirmation bar.

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
# Wolfe-wave geometry detector
# --------------------------------------------------------------------------- #
def wolfe_candidates(pivots: pd.DataFrame) -> pd.DataFrame:
    """Flag 5-pivot windows matching the Wolfe-wave wedge + throw-over geometry.

    Bullish (kinds low,high,low,high,low -> reversal UP expected off point 5):
      B. point 3 extends the wedge:            p3 < p1
      A. point 5 pierces the 1-3 trendline:    p5 < line(1,3) at idx5      (the load-bearing rule)
      C. point 4 stays inside the channel:     p4 > line(1,3) at idx4  AND  p4 < p2 (lower high)

    Bearish is the exact mirror (kinds high,low,high,low,high -> reversal DOWN off point 5).

    The EPA target is the line through points 1 and 4, evaluated at the bar point 5 is
    confirmed (``sig_idx``) — the level known the instant we could act. Candidates whose target
    doesn't even land on the claimed side of point 5 (a shallow/negative 1-4 slope) are dropped:
    a "target" behind you isn't a target.
    """
    if len(pivots) < 5:
        return pd.DataFrame()
    kinds = pivots["kind"].to_numpy()
    idxs = pivots["piv_idx"].to_numpy()
    confs = pivots["conf_idx"].to_numpy()
    prices = pivots["price"].to_numpy()

    out = []
    for k in range(4, len(pivots)):
        kk = kinds[k - 4:k + 1]
        if np.array_equal(kk, [-1, 1, -1, 1, -1]):
            direction = 1
        elif np.array_equal(kk, [1, -1, 1, -1, 1]):
            direction = -1
        else:
            continue

        i1, i2, i3, i4, i5 = idxs[k - 4:k + 1]
        p1, p2, p3, p4, p5 = prices[k - 4:k + 1]
        if i3 == i1 or i4 == i1:
            continue

        def line13(x, i1=i1, i3=i3, p1=p1, p3=p3):
            return p1 + (p3 - p1) * (x - i1) / (i3 - i1)

        line13_at5, line13_at4 = line13(i5), line13(i4)

        if direction == 1:
            if not (p3 < p1 and p5 < line13_at5 and p4 > line13_at4 and p4 < p2):
                continue
        else:
            if not (p3 > p1 and p5 > line13_at5 and p4 < line13_at4 and p4 > p2):
                continue

        sig_idx = int(confs[k])
        epa_slope = (p4 - p1) / (i4 - i1)
        target_price = p1 + epa_slope * (sig_idx - i1)
        if direction == 1 and not (target_price > p5):
            continue
        if direction == -1 and not (target_price < p5):
            continue

        out.append({
            "i1": int(i1), "i2": int(i2), "i3": int(i3), "i4": int(i4), "i5": int(i5),
            "p1": float(p1), "p2": float(p2), "p3": float(p3), "p4": float(p4), "p5": float(p5),
            "dir": int(direction), "sig_idx": sig_idx,
            "target_price": float(target_price), "stop_price": float(p5),
            "time_target_idx": int(i5 + (i4 - i1)),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Target-hit test — walk forward, EPA target vs invalidation stop
# --------------------------------------------------------------------------- #
def target_hit_ledger(bars: pd.DataFrame, entries: pd.DataFrame,
                      max_horizon: int = 90) -> pd.DataFrame:
    """One row per Wolfe entry: does the EPA target get touched before the invalidation stop?

    Entry is the **next bar's close** after point 5's confirmation (one-day lag); the walk
    forward from there uses intraday High/Low so a target can be tagged on any bar, not just
    the close. A bar that touches both target and stop is scored conservatively as a loss (we
    can't know which traded first on a daily bar). No hit within ``max_horizon`` bars -> timeout
    (excluded from the hit-rate, reported separately).
    """
    if len(entries) == 0:
        return pd.DataFrame()
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    n = len(close)

    rows = []
    for _, r in entries.iterrows():
        e = int(r["sig_idx"]) + 1
        if e >= n:
            continue
        entry_price = close[e]
        target, stop, d = r["target_price"], r["stop_price"], int(r["dir"])
        if d == 1 and not (target > entry_price > stop):
            continue
        if d == -1 and not (target < entry_price < stop):
            continue
        target_dist = abs(target - entry_price) / entry_price
        stop_dist = abs(entry_price - stop) / entry_price

        outcome, bars_to = "timeout", np.nan
        for h in range(1, max_horizon + 1):
            j = e + h
            if j >= n:
                break
            hi, lo = high[j], low[j]
            if d == 1:
                hit_t, hit_s = hi >= target, lo <= stop
            else:
                hit_t, hit_s = lo <= target, hi >= stop
            if hit_s:                      # stop wins ties (conservative)
                outcome, bars_to = "loss", h
                break
            if hit_t:
                outcome, bars_to = "win", h
                break
        rows.append({
            "sig_idx": int(r["sig_idx"]), "entry_idx": e, "dir": d,
            "target_dist": target_dist, "stop_dist": stop_dist,
            "outcome": outcome, "bars_to": bars_to,
            "time_target_offset": int(r["time_target_idx"] - e),
        })
    return pd.DataFrame(rows)


def hit_rate_summary(ledger: pd.DataFrame) -> dict:
    """Hit-rate among *decided* trades (excludes timeouts), with a Wilson interval."""
    if len(ledger) == 0:
        return {"n_total": 0, "n_decided": 0, "n_timeout": 0, "n_win": 0, "n_loss": 0,
                "hit_rate": float("nan"), "wilson_lo": float("nan"), "wilson_hi": float("nan")}
    decided = ledger[ledger["outcome"] != "timeout"]
    wins = int((decided["outcome"] == "win").sum())
    losses = int((decided["outcome"] == "loss").sum())
    nd = len(decided)
    hr = wins / nd if nd else float("nan")
    lo, hi = wilson_interval(wins, nd)
    return {"n_total": len(ledger), "n_decided": nd, "n_timeout": len(ledger) - nd,
            "n_win": wins, "n_loss": losses, "hit_rate": hr, "wilson_lo": lo, "wilson_hi": hi}


def random_target_placebo(bars: pd.DataFrame, ledger: pd.DataFrame, max_horizon: int = 90,
                          n_draws: int = 500, seed: int = 697, exclude_pad: int = 10) -> dict:
    """Same-distance, same-direction random-day placebo (vectorized across draws).

    For every *decided* real trade we know (direction, target distance, stop distance). We
    replay that exact same geometry starting from ``n_draws`` random bars of the same tape
    (excluding a +/-``exclude_pad`` window around every real entry) and record the hit-rate each
    draw produces. ``p_value`` = share of placebo draws whose hit-rate is >= the observed one —
    the honest "would a random day, walked the same distance, have looked this good?" null.
    """
    decided = ledger[ledger["outcome"] != "timeout"].reset_index(drop=True)
    if len(decided) == 0:
        return {"obs_hit_rate": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan"), "p_value": float("nan"), "n_draws": 0,
                "draws": np.array([])}

    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    n = len(close)

    excl = set()
    for e in ledger["entry_idx"]:
        excl.update(range(int(e) - exclude_pad, int(e) + exclude_pad + 1))
    valid_starts = np.array([i for i in range(0, n - max_horizon - 1) if i not in excl])
    if len(valid_starts) == 0:
        return {"obs_hit_rate": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan"), "p_value": float("nan"), "n_draws": 0,
                "draws": np.array([])}

    obs_hit_rate = float((decided["outcome"] == "win").mean())
    m = len(decided)
    dirs = decided["dir"].to_numpy()
    tdist = decided["target_dist"].to_numpy()
    sdist = decided["stop_dist"].to_numpy()
    up = dirs == 1

    rng = np.random.default_rng(seed)
    draw_hit_rates = np.empty(n_draws)
    for d_i in range(n_draws):
        starts = rng.choice(valid_starts, size=m, replace=True)
        entry_price = close[starts]
        target = entry_price * (1.0 + dirs * tdist)
        stop = entry_price * (1.0 - dirs * sdist)
        win = np.zeros(m, dtype=bool)
        decided_mask = np.zeros(m, dtype=bool)
        active = np.ones(m, dtype=bool)
        for h in range(1, max_horizon + 1):
            j = starts + h
            valid = active & (j < n)
            if not valid.any():
                break
            jj = np.clip(j, 0, n - 1)
            hi, lo = high[jj], low[jj]
            hit_t = np.where(up, hi >= target, lo <= target)
            hit_s = np.where(up, lo <= stop, hi >= stop)
            newly_loss = valid & hit_s
            newly_win = valid & hit_t & ~newly_loss
            win[newly_win] = True
            decided_mask[newly_win | newly_loss] = True
            active[newly_win | newly_loss] = False
        nd = decided_mask.sum()
        draw_hit_rates[d_i] = win.sum() / nd if nd > 0 else np.nan

    draws = draw_hit_rates[~np.isnan(draw_hit_rates)]
    p = float((draws >= obs_hit_rate).mean()) if len(draws) else float("nan")
    pm = float(draws.mean()) if len(draws) else float("nan")
    psd = float(draws.std(ddof=1)) if len(draws) > 1 else float("nan")
    z = (obs_hit_rate - pm) / psd if psd and psd > 0 else float("nan")
    return {"obs_hit_rate": obs_hit_rate, "placebo_mean": pm, "placebo_sd": psd,
            "p_value": p, "z": z, "n_draws": len(draws), "draws": draws}


def pooled_target_hit(basket: dict, pct: float = 0.04, max_horizon: int = 90) -> dict:
    """Detect + walk forward on every ticker in ``basket`` ({ticker: OHLC frame}); pool the
    per-trade ledgers. Returns the pooled ledger plus per-ticker (pivots, entries) counts."""
    ledgers = []
    counts = {}
    for tk, bars in basket.items():
        piv = zigzag(bars["close"].to_numpy(float), pct=pct)
        ent = wolfe_candidates(piv)
        led = target_hit_ledger(bars, ent, max_horizon=max_horizon)
        led = led.assign(ticker=tk)
        ledgers.append(led)
        counts[tk] = {"pivots": len(piv), "entries": len(ent)}
    ledger = (pd.concat(ledgers, ignore_index=True) if ledgers
              else pd.DataFrame(columns=["sig_idx", "entry_idx", "dir", "outcome"]))
    return {"ledger": ledger, "counts": counts}


def pooled_random_target_placebo(basket: dict, per_ticker_ledgers: dict, max_horizon: int = 90,
                                 n_draws: int = 1000, seed: int = 697,
                                 exclude_pad: int = 10) -> dict:
    """The same-distance random-day placebo, pooled across the basket at the DRAW level.

    Each ticker draws its own random start bars from its own tape (so a GLD placebo trade never
    borrows a SPY bar), but win/decided counts are summed across tickers *within each draw*
    before computing that draw's pooled hit-rate — matching exactly how the observed statistic
    pools (total wins / total decided across the whole basket). ``p_value`` = share of pooled
    placebo draws whose hit-rate is >= the observed pooled hit-rate.
    """
    rng_master = np.random.default_rng(seed)
    win_draws = None
    dec_draws = None
    total_win_obs = total_dec_obs = 0

    for tk, bars in basket.items():
        led = per_ticker_ledgers.get(tk)
        if led is None or len(led) == 0:
            continue
        decided = led[led["outcome"] != "timeout"].reset_index(drop=True)
        if len(decided) == 0:
            continue
        total_win_obs += int((decided["outcome"] == "win").sum())
        total_dec_obs += len(decided)

        high = bars["high"].to_numpy(float)
        low = bars["low"].to_numpy(float)
        close = bars["close"].to_numpy(float)
        n = len(close)
        excl = set()
        for e in led["entry_idx"]:
            excl.update(range(int(e) - exclude_pad, int(e) + exclude_pad + 1))
        valid_starts = np.array([i for i in range(0, n - max_horizon - 1) if i not in excl])
        if len(valid_starts) == 0:
            continue

        m = len(decided)
        dirs = decided["dir"].to_numpy()
        tdist = decided["target_dist"].to_numpy()
        sdist = decided["stop_dist"].to_numpy()
        up = dirs == 1

        tk_seed = int(rng_master.integers(0, 1_000_000))
        rng = np.random.default_rng(tk_seed)
        win_counts = np.zeros(n_draws)
        dec_counts = np.zeros(n_draws)
        for d_i in range(n_draws):
            starts = rng.choice(valid_starts, size=m, replace=True)
            entry_price = close[starts]
            target = entry_price * (1.0 + dirs * tdist)
            stop = entry_price * (1.0 - dirs * sdist)
            win = np.zeros(m, dtype=bool)
            decided_mask = np.zeros(m, dtype=bool)
            active = np.ones(m, dtype=bool)
            for h in range(1, max_horizon + 1):
                j = starts + h
                valid = active & (j < n)
                if not valid.any():
                    break
                jj = np.clip(j, 0, n - 1)
                hi, lo = high[jj], low[jj]
                hit_t = np.where(up, hi >= target, lo <= target)
                hit_s = np.where(up, lo <= stop, hi >= stop)
                newly_loss = valid & hit_s
                newly_win = valid & hit_t & ~newly_loss
                win[newly_win] = True
                decided_mask[newly_win | newly_loss] = True
                active[newly_win | newly_loss] = False
            win_counts[d_i] = win.sum()
            dec_counts[d_i] = decided_mask.sum()

        win_draws = win_counts if win_draws is None else win_draws + win_counts
        dec_draws = dec_counts if dec_draws is None else dec_draws + dec_counts

    if win_draws is None or total_dec_obs == 0:
        return {"obs_hit_rate": float("nan"), "placebo_mean": float("nan"),
                "placebo_sd": float("nan"), "p_value": float("nan"), "z": float("nan"),
                "n_draws": 0, "n_win_obs": total_win_obs, "n_decided_obs": total_dec_obs}

    obs_hit_rate = total_win_obs / total_dec_obs
    with np.errstate(invalid="ignore", divide="ignore"):
        hr_draws = win_draws / dec_draws
    hr_draws = hr_draws[np.isfinite(hr_draws)]
    pm = float(hr_draws.mean())
    psd = float(hr_draws.std(ddof=1))
    p = float((hr_draws >= obs_hit_rate).mean())
    z = (obs_hit_rate - pm) / psd if psd > 0 else float("nan")
    return {"obs_hit_rate": obs_hit_rate, "placebo_mean": pm, "placebo_sd": psd,
            "p_value": p, "z": z, "n_draws": len(hr_draws),
            "n_win_obs": total_win_obs, "n_decided_obs": total_dec_obs}


def time_target_accuracy(ledger: pd.DataFrame) -> dict:
    """For winning trades: does the projected time target predict *when* the target is hit?

    Correlation and mean absolute error (trading days) of actual bars-to-hit vs the projected
    ``time_target_offset``. Wolfe's "time axis" is honestly murky across sources — we test our
    one fixed, ex-ante convention and report whatever it says, including if it says "nothing".
    """
    won = ledger[ledger["outcome"] == "win"]
    if len(won) < 5:
        return {"n": int(len(won)), "corr": float("nan"), "mae_days": float("nan"),
                "mean_actual": float("nan"), "mean_predicted": float("nan")}
    actual = won["bars_to"].to_numpy(float)
    predicted = np.clip(won["time_target_offset"].to_numpy(float), 1, None)
    corr = (float(np.corrcoef(actual, predicted)[0, 1])
            if np.std(predicted) > 0 and np.std(actual) > 0 else float("nan"))
    mae = float(np.mean(np.abs(actual - predicted)))
    return {"n": int(len(won)), "corr": corr, "mae_days": mae,
            "mean_actual": float(actual.mean()), "mean_predicted": float(predicted.mean())}


# --------------------------------------------------------------------------- #
# Fixed-horizon directional return — a simpler, geometry-free robustness cut
# --------------------------------------------------------------------------- #
def run_trades(bars: pd.DataFrame, entries: pd.DataFrame, horizon: int = 20,
               cost_bps: float = 1.0, directions: np.ndarray | None = None) -> pd.DataFrame:
    """Fixed-horizon directional trades: enter one bar after ``sig_idx``, exit ``horizon`` bars
    later. ``directions`` overrides the sign (the placebo passes a +/-1 coin at the same bars)."""
    close = bars["close"].to_numpy(dtype=float)
    n = len(close)
    if len(entries) == 0:
        return pd.DataFrame(columns=["entry_idx", "dir", "ret_gross", "ret_net"])
    dirs = (np.asarray(directions, dtype=int) if directions is not None
            else entries["dir"].to_numpy(dtype=int))
    rows = []
    for sig, d in zip(entries["sig_idx"].to_numpy(dtype=int), dirs):
        e = sig + 1
        x = e + horizon
        if e >= n or x >= n:
            continue
        gross = d * (close[x] / close[e] - 1.0)
        rows.append({"entry_idx": e, "dir": int(d), "ret_gross": gross,
                     "ret_net": gross - 2.0 * cost_bps * 1e-4})
    return pd.DataFrame(rows)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def ttest_vs_zero(sample: np.ndarray) -> float:
    sample = np.asarray(sample, dtype=float)
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def hac_t(sample: np.ndarray) -> float:
    """Newey-West (HAC) t-stat on the mean of ``sample``."""
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
    if len(ledger) == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "mean_bps": float("nan"),
                "t": float("nan"), "hac_t": float("nan")}
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {"n_trades": int(n),
            "win_rate": float((r > 0).mean()) if n else float("nan"),
            "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
            "t": ttest_vs_zero(r), "hac_t": hac_t(r)}
