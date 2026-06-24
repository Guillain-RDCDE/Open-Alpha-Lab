"""Strategy + honest inference for Study 448 — Point & Figure price targets.

The folklore (Cohen, *Three-Point Reversal Method*; du Plessis, *The Definitive Guide
to Point and Figure*): a Point & Figure chart throws away time and plots only price,
as columns of **Xs** (rising) and **Os** (falling). You only ever switch columns on a
**3-box reversal** (price moves 3 box sizes against the current column). The classic
signals are the **double-top buy** (a new X-column rises one box above the prior
X-column's top) and the **double-bottom sell** (the mirror). The headline promise is the
**horizontal count**: the width (in columns) of the congestion base that preceded the
breakout, times the box size, times the reversal (3), projected from the breakout — a
mechanical **price target**, and the claim is that *those targets get hit*.

We encode the tightest **objective** P&F a practitioner would accept and make it fully
falsifiable:

    * **Box size.** A fixed fraction of the instrument's median price (default 2%): a
      log-scale-ish, instrument-natural box, fixed once over the sample (no per-day fit).
    * **Columns.** Walk the daily closes; extend the current column while price keeps
      moving its way by whole boxes; switch columns only on a **3-box reversal**. A
      column is only knowable once the box-crossing close has printed (no look-ahead).
    * **Signal.** A double-top buy = a new X-column whose top exceeds the previous
      X-column's top by one box (mirror for the double-bottom sell).
    * **Target (horizontal count).** Count the columns ``w`` in the congestion phase
      (since the last opposite signal); target = breakout price ± ``w * 3 * box`` (up for
      buys, down for sells). Fixed at the signal close.
    * **Resolution.** From the bar *after* the signal, does the **high** reach the target
      (a HIT) before the **low** breaks the stop (the bottom of the signal column, a
      STOP)? We also time-cap at ``max_hold`` bars. One execution lag, resolved on
      strictly subsequent bars.

Inference, the desk's shared spirit:

  * the **hit-rate** of the count target (HIT before STOP) vs a **fair geometric
    baseline** — a target placed the **same distance** away on the **same instrument**
    from random entry days (does the *count* add anything over geometry + drift?);
  * a **one-sample / HAC t** on the per-signal P&L (target-hit minus cost), which is the
    money question — clear **t >= 2** to earn a REAL stamp;
  * a **same-distance random-target placebo** p-value (how often does a random target of
    the same distance get hit at least as often as the count target?);
  * **one-way costs** charged on every round trip (entry + exit), shorts pay borrow.

The decisive number is the hit-rate edge over the same-distance baseline and the HAC t
on the per-signal P&L: a real P&F count effect must clear **t >= 2** there. Most
fixed-geometry chart folklore does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0
DEFAULT_BOX_FRAC = 0.02      # box size as a fraction of median price
REVERSAL = 3                 # the classic 3-box reversal
DEFAULT_MAX_HOLD = 120       # time-cap on a signal's resolution (trading days)


# --------------------------------------------------------------------------- #
# The P&F column engine
# --------------------------------------------------------------------------- #
def box_size(close: pd.Series, box_frac: float = DEFAULT_BOX_FRAC) -> float:
    """Fixed P&F box size = ``box_frac`` x median close (one constant over the sample)."""
    return float(np.median(close.to_numpy(dtype=float)) * box_frac)


def build_columns(close: pd.Series, box: float, reversal: int = REVERSAL) -> list[dict]:
    """Walk daily closes into P&F columns (3-box reversal). No look-ahead.

    Returns a list of column dicts in order, each:
        ``dir``   : +1 for an X (up) column, -1 for an O (down) column
        ``top``   : highest box level reached (boxes are integer multiples of ``box``)
        ``bot``   : lowest box level reached
        ``end_i`` : integer index of the bar at which the column last extended/started
                    (the bar whose close is *knowable* — used for lag-clean resolution)

    Box levels are integer multiples of ``box``; a column extends while price makes new
    boxes in its direction and only reverses on a move of ``reversal`` boxes against it.
    """
    c = close.to_numpy(dtype=float)
    n = c.size
    if n == 0 or box <= 0:
        return []

    def lvl(p: float) -> int:
        return int(np.floor(p / box))

    cols: list[dict] = []
    cur_dir = 0
    top = bot = lvl(c[0])
    start_i = end_i = 0

    for i in range(1, n):
        b = lvl(c[i])
        if cur_dir == 0:
            if b > top:
                cur_dir, top, end_i = 1, b, i
            elif b < bot:
                cur_dir, bot, end_i = -1, b, i
            continue
        if cur_dir == 1:
            if b > top:                              # extend the X column
                top, end_i = b, i
            elif b <= top - reversal:                # 3-box reversal -> new O column
                cols.append({"dir": 1, "top": top, "bot": bot,
                             "start_i": start_i, "end_i": end_i})
                cur_dir, bot, top, start_i, end_i = -1, b, top, i, i
        else:  # cur_dir == -1
            if b < bot:                              # extend the O column
                bot, end_i = b, i
            elif b >= bot + reversal:                # 3-box reversal -> new X column
                cols.append({"dir": -1, "top": top, "bot": bot,
                             "start_i": start_i, "end_i": end_i})
                cur_dir, top, bot, start_i, end_i = 1, b, bot, i, i

    cols.append({"dir": cur_dir, "top": top, "bot": bot,
                 "start_i": start_i, "end_i": end_i})
    return [col for col in cols if col["dir"] != 0]


# --------------------------------------------------------------------------- #
# Signals + horizontal-count targets
# --------------------------------------------------------------------------- #
def signals(close: pd.Series, box: float, reversal: int = REVERSAL,
            side: str = "buy") -> pd.DataFrame:
    """Double-top buys (or double-bottom sells) with horizontal-count targets.

    A **double-top buy** is an X-column whose ``top`` exceeds the previous X-column's
    ``top`` by at least one box. The **horizontal count** is the number of columns since
    the last opposite (sell) signal — the congestion width ``w``; the target is
    ``breakout + w * reversal * box`` (mirror for sells). The breakout box level is the
    prior X-top + 1 box (where the signal triggers), priced at ``box`` units.

    Returns one row per signal: ``sig_i`` (knowable bar index = the column's ``end_i``),
    ``breakout`` (price level), ``target`` (price level), ``stop`` (the signal column's
    bottom box level, in price), ``width`` (columns counted), ``dir`` (+1 buy / -1 sell).
    No look-ahead: ``sig_i`` is the bar whose close completed the breakout box.
    """
    cols = build_columns(close, box, reversal)
    rows = []
    last_opp = 0                                     # column index of last opposite signal
    want = 1 if side == "buy" else -1
    for k in range(2, len(cols)):
        col = cols[k]
        if col["dir"] != want:
            continue
        # find the previous same-direction column (two back, since columns alternate)
        prev_same = cols[k - 2]
        if want == 1:
            triggered = col["top"] > prev_same["top"]
            breakout_lvl = prev_same["top"] + 1
        else:
            triggered = col["bot"] < prev_same["bot"]
            breakout_lvl = prev_same["bot"] - 1
        if not triggered:
            continue
        width = k - last_opp                          # congestion width in columns
        last_opp = k
        if want == 1:
            target_lvl = breakout_lvl + width * reversal
            stop_lvl = col["bot"]
        else:
            target_lvl = breakout_lvl - width * reversal
            stop_lvl = col["top"]
        rows.append({
            "sig_i": int(col["end_i"]),
            "breakout": float(breakout_lvl * box),
            "target": float(target_lvl * box),
            "stop": float(stop_lvl * box),
            "width": int(width),
            "dir": want,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Resolve a target: HIT before STOP?
# --------------------------------------------------------------------------- #
def resolve(bars: pd.DataFrame, sig: pd.DataFrame, max_hold: int = DEFAULT_MAX_HOLD,
            lag: int = 1) -> pd.DataFrame:
    """Resolve each signal: does the count TARGET get hit before the STOP?

    From the bar ``sig_i + lag`` onward (one execution lag), walk forward up to
    ``max_hold`` bars. For a buy: a **HIT** if ``high`` reaches the target; a **STOP** if
    ``low`` breaks the signal column's bottom first; else **TIMEOUT** at the last close.
    (mirror for sells). Returns a per-signal ledger with the outcome, bars held, the
    realised gross return from the breakout entry to the resolution price, and the target
    distance as a fraction of the breakout (for the same-distance baseline).
    """
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    cl = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    rows = []
    for _, s in sig.iterrows():
        i0 = int(s["sig_i"]) + lag
        if i0 >= n:
            continue
        d = int(s["dir"])
        entry = float(s["breakout"])
        target = float(s["target"])
        stop = float(s["stop"])
        last = min(i0 + max_hold - 1, n - 1)
        outcome, exit_px, held = "timeout", cl[last], last - i0 + 1
        for j in range(i0, last + 1):
            if d == 1:
                if lo[j] <= stop:
                    outcome, exit_px, held = "stop", stop, j - i0 + 1
                    break
                if hi[j] >= target:
                    outcome, exit_px, held = "hit", target, j - i0 + 1
                    break
            else:
                if hi[j] >= stop:
                    outcome, exit_px, held = "stop", stop, j - i0 + 1
                    break
                if lo[j] <= target:
                    outcome, exit_px, held = "hit", target, j - i0 + 1
                    break
            if j == last:
                outcome, exit_px, held = "timeout", cl[j], j - i0 + 1
        gross = (exit_px / entry - 1.0) * d
        dist = abs(target / entry - 1.0)
        rows.append({"sig_i": int(s["sig_i"]), "dir": d, "outcome": outcome,
                     "entry": entry, "target": target, "stop": stop,
                     "held": int(held), "gross": float(gross),
                     "target_dist": float(dist), "width": int(s["width"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Same-distance geometric baseline (the honest control)
# --------------------------------------------------------------------------- #
def baseline_hit_rate(bars: pd.DataFrame, ledger: pd.DataFrame,
                      max_hold: int = DEFAULT_MAX_HOLD, n_draws: int = 50,
                      seed: int = 448) -> dict:
    """Fair baseline: the same target/stop *distances*, placed from random entry days.

    For each real signal we keep its target distance and stop distance (as fractions),
    its direction, and its hold cap, then drop that identical bracket on ``n_draws``
    random entry bars of the same instrument. The baseline hit-rate is how often a
    same-distance target gets hit before the same-distance stop under the instrument's
    own drift + volatility — i.e. **geometry alone**. The count adds value only if its
    real hit-rate beats this.
    """
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    cl = bars["close"].to_numpy(dtype=float)
    op = bars["open"].to_numpy(dtype=float)
    n = len(bars)
    rng = np.random.default_rng(seed)
    hits = 0
    total = 0
    for _, s in ledger.iterrows():
        d = int(s["dir"])
        tdist = float(s["target_dist"])
        sdist = abs(s["stop"] / s["entry"] - 1.0)
        starts = rng.integers(0, max(n - max_hold - 1, 1), size=n_draws)
        for i0 in starts:
            entry = op[i0]
            if d == 1:
                tgt, stp = entry * (1 + tdist), entry * (1 - sdist)
            else:
                tgt, stp = entry * (1 - tdist), entry * (1 + sdist)
            last = min(i0 + max_hold - 1, n - 1)
            res = "timeout"
            for j in range(i0, last + 1):
                if d == 1:
                    if lo[j] <= stp:
                        res = "stop"; break
                    if hi[j] >= tgt:
                        res = "hit"; break
                else:
                    if hi[j] >= stp:
                        res = "stop"; break
                    if lo[j] <= tgt:
                        res = "hit"; break
            if res == "hit":
                hits += 1
            total += 1
    return {"baseline_hit_rate": (hits / total) if total else float("nan"),
            "n_baseline": total}


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t of mean(x) vs 0, auto bandwidth."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def ttest_vs_zero(x: np.ndarray) -> float:
    """Plain one-sample t of mean(x) vs 0 (i.i.d. assumption)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean() / se) if se > 0 else float("nan")


def placebo_pvalue(bars: pd.DataFrame, ledger: pd.DataFrame,
                   max_hold: int = DEFAULT_MAX_HOLD, n_draws: int = 2000,
                   seed: int = 448) -> dict:
    """Same-distance random-target placebo: is the count hit-rate beatable by chance?

    Draws ``n_draws`` independent random-entry brackets of the *same distances* and
    asks how often a batch of the same size hits at least as often as the count target's
    observed hit-rate. ``p`` = P[random hit-rate >= observed] — the honest "is it the
    count, or just geometry?" null.
    """
    obs = float((ledger["outcome"] == "hit").mean()) if len(ledger) else float("nan")
    hi = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    op = bars["open"].to_numpy(dtype=float)
    n = len(bars)
    m = len(ledger)
    rng = np.random.default_rng(seed)
    dvec = ledger["dir"].to_numpy(dtype=int)
    tvec = ledger["target_dist"].to_numpy(dtype=float)
    svec = np.abs(ledger["stop"].to_numpy(dtype=float)
                  / ledger["entry"].to_numpy(dtype=float) - 1.0)
    rates = np.empty(n_draws)
    for d in range(n_draws):
        starts = rng.integers(0, max(n - max_hold - 1, 1), size=m)
        hits = 0
        for k in range(m):
            i0 = int(starts[k])
            entry = op[i0]
            dd = dvec[k]
            if dd == 1:
                tgt, stp = entry * (1 + tvec[k]), entry * (1 - svec[k])
            else:
                tgt, stp = entry * (1 - tvec[k]), entry * (1 + svec[k])
            last = min(i0 + max_hold - 1, n - 1)
            for j in range(i0, last + 1):
                if dd == 1:
                    if lo[j] <= stp:
                        break
                    if hi[j] >= tgt:
                        hits += 1; break
                else:
                    if hi[j] >= stp:
                        break
                    if lo[j] <= tgt:
                        hits += 1; break
        rates[d] = hits / m if m else float("nan")
    p = float((rates >= obs).mean())
    return {"obs_hit_rate": obs, "placebo_mean": float(np.nanmean(rates)),
            "p_value": p, "draws": rates}


# --------------------------------------------------------------------------- #
# Costs + per-signal P&L
# --------------------------------------------------------------------------- #
def net_pnl(ledger: pd.DataFrame, cost_bps: float = 2.0,
            borrow_bps_ann: float = 50.0) -> dict:
    """Per-signal gross/net P&L. One round trip per signal; shorts pay borrow.

    ``cost_bps`` one-way x 2 (in + out) on every signal; the short leg pays
    ``borrow_bps_ann`` annualised over the bars held. Returns gross/net mean per signal,
    the net per-signal sample (for the HAC t), win-rate, and the headline t-stats.
    """
    if len(ledger) == 0:
        return {"n": 0}
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    held_yr = ledger["held"].to_numpy(dtype=float) / ANN
    borrow = np.where(ledger["dir"].to_numpy() == -1,
                      (borrow_bps_ann / 1e4) * held_yr, 0.0)
    gross = ledger["gross"].to_numpy(dtype=float)
    net = gross - round_trip - borrow
    return {
        "n": int(len(ledger)),
        "hit_rate": float((ledger["outcome"] == "hit").mean()),
        "stop_rate": float((ledger["outcome"] == "stop").mean()),
        "timeout_rate": float((ledger["outcome"] == "timeout").mean()),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "net_sample": net,
        "win_rate": float((net > 0).mean()),
        "t_net": hac_t(net),
        "t_net_iid": ttest_vs_zero(net),
        "cost_bps": cost_bps, "borrow_bps_ann": borrow_bps_ann,
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, box_frac: float = DEFAULT_BOX_FRAC,
                   reversal: int = REVERSAL, max_hold: int = DEFAULT_MAX_HOLD,
                   cost_bps: float = 2.0, n_draws: int = 2000,
                   side: str = "buy", placebo: bool = True) -> dict:
    """One-call teardown: build signals, resolve targets, baseline, placebo, costs."""
    box = box_size(bars["close"], box_frac)
    sig = signals(bars["close"], box, reversal, side=side)
    ledger = resolve(bars, sig, max_hold=max_hold)
    out = {"box": box, "n_signals": int(len(ledger)), "ledger": ledger,
           "pnl": net_pnl(ledger, cost_bps=cost_bps)}
    if len(ledger):
        out["baseline"] = baseline_hit_rate(bars, ledger, max_hold=max_hold)
        if placebo:
            out["placebo"] = placebo_pvalue(bars, ledger, max_hold=max_hold,
                                            n_draws=n_draws)
    return out
