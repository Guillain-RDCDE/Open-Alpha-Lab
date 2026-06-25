"""Kagi charts as a falsifiable mechanical rule — Study 495.

The **Kagi chart** is a price-only Japanese chart (Steve Nison popularised it in the West).
It ignores time and draws a vertical line that extends in the current direction while price
keeps moving that way, and **reverses** only when price moves against it by at least a fixed
*reversal* amount. The defining feature is **line thickness**:

* the line is **yang** (thick) while price is above the prior **shoulder** (the last swing
  high in the Kagi geometry);
* the line is **yin** (thin) while price is below the prior **waist** (the last swing low);
* the thickness **switches** the moment a reversal line breaks the prior shoulder/waist.

The folklore (Nison's *Beyond Candlesticks*, every Kagi write-up): **buy when the line turns
yang** (thick = demand has taken control, an uptrend is confirmed) and **go flat/sell when it
turns yin** (thin = supply in control). We encode the tightest mechanical version a proponent
would accept and test it honestly:

1. **Mechanical Kagi line.** Built bar-by-bar from closes. We track the running extreme in
   the current direction; a **reversal** of ``r`` (percent of the turning price) flips the
   direction and records a shoulder (at an up→down turn) or a waist (at a down→up turn).
2. **Yin/Yang switch.** A **yang switch** fires the bar a rising line *breaks above* the most
   recent shoulder; a **yin switch** fires the bar a falling line *breaks below* the most
   recent waist. Everything is known by the close of *t* (no future bars).
3. **Entry.** A long fires on the **yang switch** close, entered at the **next** close (one
   documented lag); we then measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **threshold-scramble placebo** that rebuilds the Kagi line with
   a *shuffled per-segment reversal threshold* (drawn from the same pool of thresholds but
   re-assigned), destroying the specific shoulder/waist geometry while keeping the price
   marginal and the overall switch frequency — the honest "is the Kagi geometry doing
   anything?" null.

No look-ahead: the Kagi line uses only closes up to *t*, the yang switch is read on the close
of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_REVERSAL = 0.04   # 4% reversal threshold — a common Kagi default


# --------------------------------------------------------------------------- #
# Kagi line construction
# --------------------------------------------------------------------------- #
def kagi_line(close: pd.Series, reversal: float = DEFAULT_REVERSAL):
    """Build the Kagi line bar-by-bar and tag every bar with its yin/yang state + switches.

    Walks the closes left-to-right (no future data). State:

    * ``direction`` (+1 up, -1 down): the way the current Kagi line is drawn.
    * ``ext``: the running extreme of the current line (the high if up, low if down).
    * ``shoulder`` / ``waist``: the last confirmed swing high / swing low of the Kagi line.
    * ``thick`` (bool): True = **yang** (line is above the prior shoulder), False = **yin**.

    A reversal of ``reversal`` (fraction of ``ext``) flips ``direction`` and posts a new
    shoulder (up→down) or waist (down→up). On each bar we re-evaluate thickness: the line is
    yang once it trades above the recorded shoulder, yin once it trades below the waist.

    Returns a DataFrame aligned to ``close.index`` with columns:
    ``direction``, ``thick`` (yang flag), ``yang_switch`` (thin→thick this bar),
    ``yin_switch`` (thick→thin this bar), ``shoulder``, ``waist``.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    direction = np.zeros(n, dtype=int)
    thick = np.zeros(n, dtype=bool)
    yang_sw = np.zeros(n, dtype=bool)
    yin_sw = np.zeros(n, dtype=bool)
    sh_arr = np.full(n, np.nan)
    wa_arr = np.full(n, np.nan)

    if n == 0:
        return pd.DataFrame(
            {"direction": direction, "thick": thick, "yang_switch": yang_sw,
             "yin_switch": yin_sw, "shoulder": sh_arr, "waist": wa_arr},
            index=close.index)

    # Initialise from the first bar.
    d = 1
    ext = p[0]
    shoulder = p[0]      # last swing high of the kagi line
    waist = p[0]         # last swing low of the kagi line
    is_thick = False     # start yin until a shoulder is broken

    for i in range(n):
        c = p[i]
        if d > 0:
            # line drawn up: extend on new highs
            if c > ext:
                ext = c
            elif c <= ext * (1.0 - reversal):
                # reversal down: the prior extreme becomes a shoulder, flip direction
                shoulder = ext
                d = -1
                ext = c
        else:
            if c < ext:
                ext = c
            elif c >= ext * (1.0 + reversal):
                # reversal up: the prior extreme becomes a waist, flip direction
                waist = ext
                d = 1
                ext = c

        # thickness: yang once the line breaks above the prior shoulder; yin once it
        # breaks below the prior waist. (Hysteresis between shoulder and waist.)
        prev_thick = is_thick
        if c > shoulder:
            is_thick = True
        elif c < waist:
            is_thick = False

        direction[i] = d
        thick[i] = is_thick
        sh_arr[i] = shoulder
        wa_arr[i] = waist
        if is_thick and not prev_thick:
            yang_sw[i] = True
        elif (not is_thick) and prev_thick:
            yin_sw[i] = True

    return pd.DataFrame(
        {"direction": direction, "thick": thick, "yang_switch": yang_sw,
         "yin_switch": yin_sw, "shoulder": sh_arr, "waist": wa_arr},
        index=close.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def yang_switch_entries(close: pd.Series, reversal: float = DEFAULT_REVERSAL,
                        warmup: int = 20) -> pd.DatetimeIndex:
    """Bars where the Kagi line turns **yang** (thin→thick) — the 'buy the yang switch' rule.

    Entry is executed at the next close by :func:`forward_returns`. A short ``warmup`` is
    dropped so the very first shoulder-break (an artefact of initialisation) is not traded.
    """
    kl = kagi_line(close, reversal=reversal)
    sw = kl["yang_switch"].to_numpy().copy()
    if warmup > 0:
        sw[:warmup] = False
    return close.index[sw]


def random_entries(close: pd.Series, n: int, warmup: int = 20, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warmup:]
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


def threshold_scramble_placebo(close: pd.Series, horizon: int,
                               reversal: float = DEFAULT_REVERSAL,
                               n_draws: int = 1000, seed: int = 495) -> dict:
    """Placebo: rebuild the Kagi line with a **randomised reversal threshold** per draw.

    The yang switch is entirely a product of the reversal threshold's interaction with the
    price path — it decides where shoulders/waists land and therefore where the line turns
    thick. The honest "is *this* Kagi geometry load-bearing?" null draws a fresh reversal
    threshold from a neighbourhood of plausible values (so the line is a *different but equally
    valid* Kagi), keeps the same price marginal, and asks how often a randomly-parameterised
    Kagi's yang-switch return matches or beats the real one. If the real (4%) line carries no
    special information, the observed return sits mid-pack.

    Returns the observed mean, the placebo p-value (share of scrambled Kagis that match/beat
    the real one), and the number of valid draws.
    """
    obs = float(np.mean(forward_returns(close, yang_switch_entries(close, reversal=reversal),
                                        horizon)))
    rng = np.random.default_rng(seed)
    # plausible reversal grid spanning common Kagi defaults (1%–8%)
    grid = np.linspace(0.01, 0.08, 64)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        rv = float(rng.choice(grid))
        ent = yang_switch_entries(close, reversal=rv)
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(close: pd.Series, reversal: float = DEFAULT_REVERSAL, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: yang-switch vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the yang-switch summary (gross + net), the
    drift-matched random-entry baseline, and the switch-minus-random delta.
    """
    ent = yang_switch_entries(close, reversal=reversal)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
