"""Out-of-sample test — does a fixed-period cycle actually **forecast**, and does following
it **pay**?

A spectral peak (`spectral`) tells you a cycle exists *in-sample*. That is necessary but
nowhere near sufficient: a period fitted on the whole record can hug past wiggles and
forecast nothing. The cyclist's real, falsifiable promise is a *forward* one — "VIX 40-day
low into early July, stocks rally from there" — so we test it the only honest way: **walk
forward**. At each origin we fit on the past only, project the next turn, and score the
projection against what the market actually did next. Two scores:

1. **Direction skill** (:func:`oos_direction_skill`) — does the projected cycle call the
   *sign* of the next-horizon VIX move better than a coin, and better than the same cycle
   with its phase scrambled? Pure forecast accuracy, fully causal.
2. **The trade** (:func:`phase_trade`) — the tradeable expression of "buy the rally at the
   cycle low": long the S&P for the sessions the projected cycle is *falling*, flat while it
   is *rising*. Scored on net Sharpe vs buy-and-hold and vs a random-phase null. The position
   rule reads the **one-session** projected slope, because the P&L accrues session by session:
   gating instead on the h-session-ahead change (as the direction test does, by design) lags
   the cycle by ~(h−1)/2 sessions and fails the planted-cycle positive control — a harness
   that cannot bank a cycle we *injected* has no business judging the VIX.

The random-phase null is the crux: it keeps the *period* and *amplitude* the theorist found
and scrambles only the *timing*. If the real phase does no better than an arbitrary one, the
"cycle" carries no forecast — the peak was a shape in red noise after all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cycles, spectral


def dominant_period(x, band=spectral.DEFAULT_BAND) -> float:
    """The period with the most periodogram power inside ``band`` — where a cyclist tunes."""
    periods, power = spectral.periodogram(x)
    m = (periods >= band[0]) & (periods <= band[1])
    if not m.any():
        return float("nan")
    pp, po = periods[m], power[m]
    return float(pp[int(np.argmax(po))])


# --------------------------------------------------------------------------- #
# 1 · Direction forecast skill (causal, walk-forward)
# --------------------------------------------------------------------------- #

def oos_direction_skill(
    x,
    band=spectral.DEFAULT_BAND,
    horizon: int = 20,
    min_train: int = 750,
    step: int = 10,
    width_frac: float = 0.5,
    n_null: int = 500,
    seed: int = 0,
    fixed_period: float | None = None,
) -> dict:
    """Walk-forward: does the projected cycle call the next-``horizon`` VIX direction?

    At each origin ``t`` (every ``step`` sessions, after ``min_train`` of history) we fit the
    dominant in-band period and a fixed-period cosine **on the past only**, project it
    ``horizon`` sessions forward, and take the sign of the projected change as the forecast.
    The realized sign is ``x[t+horizon] − x[t]``. Skill is the hit rate. The null repeats the
    whole walk with the fitted **phase rotated by a random offset** (period and amplitude
    kept), ``n_null`` times — so the p-value asks whether the *learned timing*, not merely
    the existence of an oscillation, beats an arbitrary phase.

    Returns ``{skill, n_calls, null_mean, null_std, p_value, horizon, median_period}``.
    """
    s = pd.Series(x, dtype=float).dropna()
    v = s.to_numpy()
    n = v.size
    rng = np.random.default_rng(seed)

    origins = list(range(min_train, n - horizon, step))
    if not origins:
        return {"skill": np.nan, "n_calls": 0, "null_mean": np.nan, "null_std": np.nan,
                "p_value": np.nan, "horizon": horizon, "median_period": np.nan}

    # Precompute, at each origin, the fitted cosine and the realized direction.
    fits, realized, periods_used = [], [], []
    for t in origins:
        train = v[:t]
        P = fixed_period if fixed_period is not None else dominant_period(train, band)
        if not np.isfinite(P):
            continue
        fit = cycles.fit_sinusoid(train, P)
        fits.append(fit)
        realized.append(np.sign(v[t + horizon - 1] - v[t - 1]))
        periods_used.append(P)
    realized = np.array(realized)

    def _skill(phase_offset: float) -> float:
        hits = 0
        for fit, real in zip(fits, realized):
            amp, ph, P = fit["amplitude"], fit["phase"] + phase_offset, fit["period"]
            w = 2.0 * np.pi / P
            last = fit["n"] - 1
            now = amp * np.cos(w * last - ph)
            fut = amp * np.cos(w * (last + horizon) - ph)
            pred = np.sign(fut - now)
            if pred == real and real != 0:
                hits += 1
        return hits / len(fits)

    skill = _skill(0.0)
    null = np.array([_skill(rng.uniform(0, 2 * np.pi)) for _ in range(n_null)])
    p_value = (int(np.sum(null >= skill)) + 1) / (n_null + 1)
    return {
        "skill": float(skill),
        "n_calls": int(len(fits)),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "p_value": float(p_value),
        "horizon": int(horizon),
        "median_period": float(np.median(periods_used)) if periods_used else np.nan,
    }


# --------------------------------------------------------------------------- #
# 2 · The trade — long the S&P when the VIX cycle is projected to fall
# --------------------------------------------------------------------------- #

@dataclass
class TradeResult:
    daily: pd.Series                 # daily strategy return (net of costs)
    stats: dict = field(default_factory=dict)
    positions: pd.Series = field(default=None)


def _fit_schedule(
    logvix: np.ndarray,
    band,
    min_train: int,
    refit: int,
    fixed_period: float | None,
) -> dict:
    """Causal cycle fit active on each day — computed **once**, reused across null draws.

    Refits the dominant period + cosine every ``refit`` sessions on the past only. The result
    (amplitude, base phase, period per day, plus the expanding in-sample σ for the amplitude
    gate) is independent of any phase rotation, so the expensive FFT/least-squares work is
    done a single time and the 200-draw null then becomes pure arithmetic. Returns arrays
    ``{amp, phase0, period, sigma}`` of length ``n`` (NaN before a fit exists).
    """
    n = logvix.size
    amp = np.full(n, np.nan)
    phase0 = np.full(n, np.nan)
    period = np.full(n, np.nan)
    # Expanding in-sample std (population) via cumulative moments — for the amplitude gate.
    c1 = np.cumsum(logvix)
    c2 = np.cumsum(logvix ** 2)
    sigma = np.full(n, np.nan)

    cur = None
    last_origin = -10**9
    for t in range(min_train, n):
        if cur is None or t - last_origin >= refit:
            P = fixed_period if fixed_period is not None else dominant_period(logvix[:t], band)
            if np.isfinite(P):
                f = cycles.fit_sinusoid(logvix[:t], P)
                cur = (f["amplitude"], f["phase"], f["period"])
            last_origin = t
        if cur is not None:
            amp[t], phase0[t], period[t] = cur
        mean = c1[t - 1] / t
        sigma[t] = np.sqrt(max(c2[t - 1] / t - mean * mean, 0.0))
    return {"amp": amp, "phase0": phase0, "period": period, "sigma": sigma}


def _positions_from_schedule(
    sched: dict, phase_offset: float = 0.0, amp_gate: float = 0.0,
) -> np.ndarray:
    """Vectorised {0,1} long/flat positions from a precomputed fit schedule.

    Long (1) when the projected cycle's **one-session** slope is negative (vol set to fall
    into the session the position is actually held), else flat (0). The one-step slope is the
    aligned read: the P&L accrues daily, so gating on a multi-session-ahead change makes the
    position lag the cycle by half the look-ahead and loses money even on a *planted* cycle
    (the positive control this module is held to). ``phase_offset`` rotates the learned phase
    (0 for the strategy, random per null draw); the amplitude gate zeroes days where the
    in-sample cycle is weak (amp < gate·σ).
    """
    amp, phase0, period, sigma = sched["amp"], sched["phase0"], sched["period"], sched["sigma"]
    valid = np.isfinite(amp) & np.isfinite(period)
    t = np.arange(amp.size, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = 2.0 * np.pi / period
        ph = phase0 + phase_offset
        now = amp * np.cos(w * t - ph)
        fut = amp * np.cos(w * (t + 1.0) - ph)
        pos = np.where(valid & ((fut - now) < 0), 1.0, 0.0)
        if amp_gate > 0.0:
            pos = np.where(valid & (amp >= amp_gate * sigma), pos, 0.0)
    return pos


def phase_trade(
    logvix: pd.Series,
    spx: pd.Series,
    band=spectral.DEFAULT_BAND,
    min_train: int = 750,
    refit: int = 5,
    width_frac: float = 0.5,
    cost_bps: float = 1.0,
    n_null: int = 200,
    seed: int = 0,
    fixed_period: float | None = None,
    amp_gate: float = 0.0,
) -> TradeResult:
    """Trade the cycle: long the S&P for the sessions the VIX cycle is projected to fall.

    The honest tradeable form of "stocks bottom on the cycle and rally": you can't buy spot
    VIX, but you *can* act on its forecast in the index. Positions are causal (refit on the
    past only) and read the fitted cycle's one-session projected slope (see
    :func:`_positions_from_schedule` for why). Net daily return = ``pos·r_spx − cost`` charged
    on every position change. Reported against **buy-and-hold** and a **random-phase null**
    Sharpe distribution — if the learned phase doesn't beat scrambled timing, the cycle adds
    nothing over being long.

    Returns a :class:`TradeResult` whose ``stats`` carry Sharpe, monthly return, exposure,
    the buy-hold Sharpe, and the null p-value.
    """
    df = pd.concat([logvix.rename("lv"), spx.rename("px")], axis=1).dropna()
    lv = df["lv"].to_numpy()
    r_spx = df["px"].pct_change().fillna(0.0).to_numpy()
    cost = cost_bps * 1e-4

    def _returns(pos: np.ndarray) -> np.ndarray:
        turn = np.abs(np.diff(pos, prepend=0.0))
        # Act on the close that triggers, earn from the next session (no look-ahead).
        held = np.concatenate([[0.0], pos[:-1]])
        return held * r_spx - turn * cost

    sched = _fit_schedule(lv, band, min_train, refit, fixed_period)
    pos = _positions_from_schedule(sched, 0.0, amp_gate)
    strat = _returns(pos)
    daily = pd.Series(strat, index=df.index, name="r_cycle")

    rng = np.random.default_rng(seed)
    null_sharpes = []
    for _ in range(n_null):
        pn = _positions_from_schedule(sched, rng.uniform(0, 2 * np.pi), amp_gate)
        null_sharpes.append(_sharpe(_returns(pn)))
    null_sharpes = np.array(null_sharpes)

    sh = _sharpe(strat)
    bh = _sharpe(r_spx[min_train:])
    active = daily.iloc[min_train:]
    # A degenerate strategy (e.g. a gate that never fires) has no return stream and hence no
    # Sharpe; its p-value is *undefined*, not tiny. Comparing NaN against the null with >= would
    # count zero exceedances and report a spuriously significant (1)/(n+1) — so the comparison
    # is made only across finite values, and NaN-vs-null is reported as NaN.
    finite_null = null_sharpes[np.isfinite(null_sharpes)]
    if np.isfinite(sh) and finite_null.size > 0:
        p_value = (int(np.sum(finite_null >= sh)) + 1) / (finite_null.size + 1)
    else:
        p_value = float("nan")
    stats = {
        "sharpe_net": float(sh),
        "monthly_net": float(active.mean() * 21),
        "exposure": float(pos[min_train:].mean()),
        "n_trades": int(np.abs(np.diff(pos)).sum()),
        "buyhold_sharpe": float(bh),
        "null_sharpe_mean": float(finite_null.mean()) if finite_null.size else float("nan"),
        "null_sharpe_p95": float(np.quantile(finite_null, 0.95)) if finite_null.size else float("nan"),
        "p_value_vs_null": float(p_value),
        "max_drawdown": float(_max_drawdown(active)),
    }
    return TradeResult(daily=daily, stats=stats, positions=pd.Series(pos, index=df.index))


def _sharpe(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def _max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())
