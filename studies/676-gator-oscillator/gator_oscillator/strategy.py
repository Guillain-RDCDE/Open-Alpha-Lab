"""Strategy + inference for Study 676 — Gator Oscillator.

Bill Williams' **Gator Oscillator** (*Trading Chaos*, 1995 / *New Trading Dimensions*,
1998) does not add new information to the [Alligator](../../421-williams-alligator/) —
it just plots the *rate of change* of the Alligator's own spread as a two-sided
histogram, so a trader can see "is the alligator's mouth opening or closing" without
eyeballing three overlapping lines:

    Jaw   = SMMA(median, 13) shifted +8 bars   (identical to the Alligator's Jaw)
    Teeth = SMMA(median,  8) shifted +5 bars   (identical to the Alligator's Teeth)
    Lips  = SMMA(median,  5) shifted +3 bars   (identical to the Alligator's Lips)

    Gator upper bar = |Jaw - Teeth|,  colored GREEN if it grew vs the prior bar, RED if
                       it shrank (plotted above the zero line)
    Gator lower bar = |Teeth - Lips|, same green/red rule (plotted below the zero line)

The folk recipe: the alligator **sleeps** while both bars are red (the mouth is closing
— lines converging, no trend) and **wakes** the moment both bars flip red -> green
together (the mouth starts opening again — a new trend is about to be "eaten"). The
believer's trade: when the gator wakes, get in — in the direction the three lines are
already fanned — before the trend runs.

This is the **paired study** to 421 (the Alligator itself), 184 (Williams Fractals) and
419/420 (the Awesome/Accelerator Oscillators, Williams' momentum cousins): all four
reuse the same Jaw/Teeth/Lips fan, so the honest question here is narrow and specific —
**does watching the histogram's color change (the Gator) add a genuine, tradable timing
edge over just knowing the fan is bullish or bearish (the Alligator)?**

Measurements:

* **Event study (Signal axis).** Pool "wake" events — the simultaneous red->green flip
  on both bars — across SPY + a 30-name liquid basket. Sign the forward return by the
  concurrent fan direction (long if Lips>Teeth>Jaw, short if Lips<Teeth<Jaw, dropped if
  the fan isn't ordered) at horizons of 1/5/10/20 trading days, entered at the next bar's
  open (one documented execution lag). HAC one-sample *t* on the signed mean, Welch *t*
  vs the unconditional base rate, and a label-shuffle placebo.
* **Trend-capture (magnitude, not direction).** Does a wake event predict a *bigger*
  move is coming, regardless of direction — |forward return| at wake events vs the
  unconditional |forward return|, Welch *t*.
* **The timer (Tradability axis).** A real position rule on SPY: enter the fan direction
  the day after a wake, hold a fixed 10-day window, one-way costs x NAV per leg + borrow
  on shorts. Raced against buy-and-hold **and** against 421's "always in the fan"
  Alligator rule — the decisive comparison for whether the Gator's timing *adds* to the
  Alligator it's built from.
* **Synthetic positive control.** A trend-persistence knob (mirroring 421) proves the
  detector fires on planted multi-week trends and stays silent on a fair-coin null.

The decisive number is the HAC *t* of the signed post-wake forward return on the REAL
tape; the honest follow-up is whether the timer beats simply riding the fan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
JAW = (13, 8)
TEETH = (8, 5)
LIPS = (5, 3)
HORIZONS = (1, 5, 10, 20)
HOLD_DAYS = 10                 # the timer's fixed holding period


# --------------------------------------------------------------------------- #
# Alligator + Gator indicators
# --------------------------------------------------------------------------- #
def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed moving average (Wilder's RMA): EWM with alpha = 1/period."""
    return series.ewm(com=period - 1, min_periods=period, adjust=False).mean()


def alligator(bars: pd.DataFrame) -> pd.DataFrame:
    """The three Alligator lines (Jaw/Teeth/Lips), forward-shifted as Williams defines.

    Median price = (high+low)/2. Each line is an SMMA, displaced forward by its shift —
    the value at bar *t* was computed from data up to *t - shift*, so it is fully known
    at the close of *t* (a conservative, look-ahead-free construction).
    """
    median = (bars["high"] + bars["low"]) / 2.0
    jaw = smma(median, JAW[0]).shift(JAW[1])
    teeth = smma(median, TEETH[0]).shift(TEETH[1])
    lips = smma(median, LIPS[0]).shift(LIPS[1])
    return pd.DataFrame({"jaw": jaw, "teeth": teeth, "lips": lips}, index=bars.index)


def gator_lines(bars: pd.DataFrame) -> pd.DataFrame:
    """The two Gator histograms: upper = |Jaw-Teeth|, lower = |Teeth-Lips|."""
    a = alligator(bars)
    upper = (a["jaw"] - a["teeth"]).abs()
    lower = (a["teeth"] - a["lips"]).abs()
    return pd.DataFrame({"upper": upper, "lower": lower}, index=bars.index)


def fan_state(bars: pd.DataFrame) -> pd.Series:
    """+1 bullish fan (Lips>Teeth>Jaw), -1 bearish fan (Lips<Teeth<Jaw), 0 otherwise."""
    a = alligator(bars)
    bull = (a["lips"] > a["teeth"]) & (a["teeth"] > a["jaw"])
    bear = (a["lips"] < a["teeth"]) & (a["teeth"] < a["jaw"])
    out = pd.Series(0, index=bars.index, dtype=int)
    out[bull] = 1
    out[bear] = -1
    out[a.isna().any(axis=1)] = 0
    return out


MIN_SLEEP = 3    # consecutive both-red bars required before a transition counts as a "wake"


def gator_wake(bars: pd.DataFrame, min_sleep: int = MIN_SLEEP) -> pd.Series:
    """True on the FIRST bar where both histograms turn green, right after a genuine
    sleep — at least ``min_sleep`` consecutive both-red (converging) bars.

    Green = today's bar taller than yesterday's (the spread is widening). A single-day
    red->green flip on both bars is common (~24% of days on SPY — any local minimum of
    two correlated noisy series) and is *not* what the folklore means by "the gator
    wakes": that phrase describes a market that has genuinely been consolidating (the
    mouth shut for a stretch) and then starts to open. Requiring a minimum sleep run is
    the honest reading of the claim, not a knob tuned for a result — ``min_sleep=3`` is
    fixed once and reused everywhere (event study, timer, synthetic control).
    """
    gl = gator_lines(bars)
    valid = (gl.notna().all(axis=1) & gl.shift(1).notna().all(axis=1)).to_numpy()
    up_green = (gl["upper"] > gl["upper"].shift(1)).to_numpy() & valid
    lo_green = (gl["lower"] > gl["lower"].shift(1)).to_numpy() & valid
    both_red = (~up_green) & (~lo_green) & valid
    both_green = up_green & lo_green & valid

    n = len(bars)
    wake = np.zeros(n, dtype=bool)
    run = 0
    for t in range(n):
        if both_green[t] and run >= min_sleep:
            wake[t] = True
        run = run + 1 if both_red[t] else 0
    return pd.Series(wake, index=bars.index, name="wake")


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag: enter next open, exit close t+horizon)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-bar forward return entering the NEXT open, exiting the close ``horizon`` later.

    Bar t's value = close[t+horizon] / open[t+1] - 1. NaN where the window overruns.
    """
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.full(n, np.nan)
    for i in range(n):
        e = i + 1
        x = e + horizon - 1
        if e < n and x < n:
            out[i] = c[x] / o[e] - 1.0
    return out


# --------------------------------------------------------------------------- #
# Event extraction across the basket
# --------------------------------------------------------------------------- #
def collect_wake_events(panel: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Pool wake events across the basket with fan direction, signed and unsigned
    forward returns. Columns: ticker, date, fan, fwd_ret, signed_ret, abs_ret."""
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 60:
            continue
        wake = gator_wake(bars).to_numpy()
        fan = fan_state(bars).to_numpy()
        fwd = forward_returns(bars, horizon)
        idx = bars.index
        for i in range(len(bars)):
            if not wake[i] or not np.isfinite(fwd[i]):
                continue
            rows.append({
                "ticker": tk, "date": idx[i], "fan": int(fan[i]), "fwd_ret": float(fwd[i]),
                "signed_ret": float(fan[i] * fwd[i]) if fan[i] != 0 else np.nan,
                "abs_ret": float(abs(fwd[i])),
            })
    return pd.DataFrame(rows)


def unconditional_base(panel: dict[str, pd.DataFrame], horizon: int) -> np.ndarray:
    """The unconditional forward-H-day return, every bar of every name (no signal)."""
    vals = []
    for bars in panel.values():
        if bars is None or len(bars) < horizon + 60:
            continue
        fwd = forward_returns(bars, horizon)
        vals.append(fwd[np.isfinite(fwd)])
    return np.concatenate(vals) if vals else np.array([])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def hac_t(sample: np.ndarray) -> float:
    """Newey-West HAC one-sample t of ``sample`` mean against 0."""
    r = np.asarray(sample, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
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


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def placebo_pvalue(panel: dict[str, pd.DataFrame], horizon: int, n_events: int,
                   n_draws: int = 5000, seed: int = 676) -> dict:
    """Label-shuffle placebo: n_events random bars, random fair-coin sign, vs observed.

    Draws ``n_events`` random bars from the unconditional pool, signs each with a fair
    coin (mimicking "a random directional call the same number of times"), and asks how
    often the random mean >= the observed signed wake-event mean.
    """
    base = unconditional_base(panel, horizon)
    if base.size == 0 or n_events <= 0:
        return {"p_value": float("nan"), "draws": np.array([])}
    ev = collect_wake_events(panel, horizon)
    signed = ev["signed_ret"].dropna().to_numpy()
    obs = float(signed.mean()) if len(signed) else float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(base, size=n_events, replace=True)
        signs = rng.choice([-1.0, 1.0], size=n_events)
        means[i] = float((signs * pick).mean())
    p = float((means >= obs).mean())
    return {"obs": obs, "p_value": p, "draws": means}


def magnitude_placebo_pvalue(panel: dict[str, pd.DataFrame], horizon: int, n_events: int,
                             n_draws: int = 5000, seed: int = 676) -> dict:
    """Label-shuffle placebo for the MAGNITUDE (trend-capture) claim: |return|, unsigned."""
    base = np.abs(unconditional_base(panel, horizon))
    if base.size == 0 or n_events <= 0:
        return {"p_value": float("nan"), "draws": np.array([])}
    ev = collect_wake_events(panel, horizon)
    obs = float(ev["abs_ret"].mean()) if len(ev) else float("nan")
    rng = np.random.default_rng(seed + 1)
    means = np.empty(n_draws)
    for i in range(n_draws):
        means[i] = float(rng.choice(base, size=n_events, replace=True).mean())
    p = float((means >= obs).mean())
    return {"obs": obs, "p_value": p, "draws": means}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(signed_mean: float, horizon: int, cost_bps: float = 5.0,
                 borrow_bps_ann: float = 50.0, short_share: float = 0.5) -> float:
    """Per-event signed return net of a one-way round trip + borrow on the short share."""
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    borrow = short_share * (borrow_bps_ann / 1e4) * (horizon / TRADING_DAYS_PER_YEAR)
    return float(signed_mean - round_trip - borrow)


# --------------------------------------------------------------------------- #
# Orchestrator — the event study, one horizon
# --------------------------------------------------------------------------- #
def summarize(panel: dict[str, pd.DataFrame], horizon: int, n_draws: int = 5000,
              placebo: bool = True, cost_bps: float = 5.0) -> dict:
    """Headline stats for one horizon: directional (signed) and magnitude (unsigned)."""
    ev = collect_wake_events(panel, horizon)
    base = unconditional_base(panel, horizon)
    n_wake = int(len(ev))
    n_dir = int(ev["fan"].ne(0).sum()) if n_wake else 0
    if n_wake == 0 or n_dir == 0:
        return {"horizon": horizon, "n_wake": n_wake, "n_dir": 0, "n_bull": 0, "n_bear": 0,
                "mean_signed": float("nan"), "base_mean": float("nan"), "win": float("nan"),
                "t_hac": float("nan"), "t_welch": float("nan"), "p_placebo": float("nan"),
                "net": float("nan"), "mean_abs": float("nan"), "base_abs": float("nan"),
                "t_welch_abs": float("nan"), "p_placebo_abs": float("nan")}
    signed = ev["signed_ret"].dropna().to_numpy()
    abs_ret = ev["abs_ret"].to_numpy()
    base_abs = np.abs(base)
    t_hac = hac_t(signed)
    t_welch = welch_t(signed, base)
    win = float((signed > 0).mean())
    mean_signed = float(signed.mean())
    p_dir = placebo_pvalue(panel, horizon, n_dir, n_draws=n_draws)["p_value"] if placebo else float("nan")
    t_welch_abs = welch_t(abs_ret, base_abs)
    mean_abs = float(abs_ret.mean())
    p_abs = magnitude_placebo_pvalue(panel, horizon, n_wake, n_draws=n_draws)["p_value"] if placebo else float("nan")
    return {
        "horizon": horizon, "n_wake": n_wake, "n_dir": n_dir,
        "n_bull": int((ev["fan"] == 1).sum()), "n_bear": int((ev["fan"] == -1).sum()),
        "mean_signed": mean_signed, "base_mean": float(base.mean()), "win": win,
        "t_hac": t_hac, "t_welch": t_welch, "p_placebo": p_dir,
        "net": net_of_costs(mean_signed, horizon, cost_bps=cost_bps),
        "mean_abs": mean_abs, "base_abs": float(base_abs.mean()),
        "t_welch_abs": t_welch_abs, "p_placebo_abs": p_abs,
    }


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   n_draws: int = 5000, cost_bps: float = 5.0) -> pd.DataFrame:
    """Run the full event study across horizons -> a tidy results frame."""
    rows = [summarize(panel, h, n_draws=n_draws, cost_bps=cost_bps) for h in horizons]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The timer — a real position rule on one instrument (Tradability axis)
# --------------------------------------------------------------------------- #
def wake_timer_signal(bars: pd.DataFrame, hold_days: int = HOLD_DAYS) -> pd.Series:
    """Enter the fan direction the day a wake fires, hold ``hold_days``, then flat.

    New wakes are ignored while a position is already open (no pyramiding). The raw
    signal is known at the close of bar *t* and shifted once (one documented lag) so it
    acts on the return of *t+1* onward, exactly ``hold_days`` sessions.
    """
    wake = gator_wake(bars).to_numpy()
    fan = fan_state(bars).to_numpy()
    n = len(bars)
    sig = np.zeros(n)
    hold_left = 0
    cur = 0
    for t in range(n):
        if hold_left <= 0 and wake[t] and fan[t] != 0:
            cur = fan[t]
            hold_left = hold_days
        if hold_left > 0:
            sig[t] = cur
            hold_left -= 1
        else:
            cur = 0
    raw = pd.Series(sig, index=bars.index, name="signal")
    return raw.shift(1).fillna(0.0).rename("signal")


def fan_signal(bars: pd.DataFrame) -> pd.Series:
    """The 421-style benchmark: long/short whenever the Alligator IS fanned, lagged once.

    This is the "always in the fan" rule the Gator's timer is measured against — the
    decisive comparison for whether the Gator's color-change timing adds anything to
    simply knowing the fan direction.
    """
    fan = fan_state(bars)
    return fan.shift(1).fillna(0.0).rename("signal")


TBILL_ANN = 0.04  # flat cash-leg proxy, 4%/yr (matches sibling 421; FRED unavailable offline)


def run_backtest(bars: pd.DataFrame, signal: pd.Series, cost_bps: float = 5.0,
                 borrow_bps_ann: float = 50.0, tbill_ann: float = TBILL_ANN) -> pd.DataFrame:
    """Apply a {-1,0,+1} position signal to daily close-to-close log returns.

    Long (signal=+1) earns the equity return; flat (0) earns a flat cash-leg proxy
    (``tbill_ann``/yr) so a timer that is out of the market most of the time is not
    unfairly penalised against a buy-and-hold benchmark that is always earning
    something — the desk's "excess-of-cash vs excess-of-cash" convention (same proxy
    as sibling study 421, FRED unavailable offline); short (-1) earns minus the equity
    return plus the cash leg, minus borrow. ``cost_bps`` (one-way) is charged on NAV
    whenever the position changes.
    """
    r_eq = np.log(bars["close"] / bars["close"].shift(1)).rename("r_equity")
    r_tb = pd.Series(tbill_ann / TRADING_DAYS_PER_YEAR, index=bars.index)
    sig = signal.reindex(bars.index).fillna(0.0)
    turnover = sig.diff().abs().fillna(sig.abs())
    cost = turnover * (cost_bps * 1e-4)
    borrow = (sig < 0).astype(float) * (borrow_bps_ann * 1e-4 / TRADING_DAYS_PER_YEAR)

    long_leg = (sig > 0).astype(float) * r_eq
    flat_leg = (sig == 0).astype(float) * r_tb
    short_leg = (sig < 0).astype(float) * (-r_eq + r_tb)
    r_strat = (long_leg + flat_leg + short_leg - cost - borrow).rename("r_strategy")
    out = pd.concat([r_eq, sig.rename("signal"), r_strat, r_eq.rename("r_bh")], axis=1)
    return out.dropna(subset=["r_equity"])


def _hac_tstat(r: np.ndarray) -> float:
    return hac_t(r)


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline annualised stats: Sharpe, vol, CAGR, max drawdown, mean bps, HAC t."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ["n_days", "cagr", "sharpe", "vol_ann", "max_drawdown", "mean_daily_bps", "tstat"]}
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))
    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")
    dd = float((cum / cum.cummax() - 1.0).min())
    return {"n_days": int(n), "cagr": cagr, "sharpe": sharpe, "vol_ann": vol_ann,
            "max_drawdown": dd, "mean_daily_bps": float(mu * 1e4), "tstat": _hac_tstat(r.to_numpy())}


def sharpe_diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC t-stat on the daily return DIFFERENCE r1 - r2 (Jobson-Korkie style)."""
    diff = (r1 - r2).dropna()
    return _hac_tstat(diff.to_numpy())


def block_permutation_pvalue(bars: pd.DataFrame, signal: pd.Series, cost_bps: float = 5.0,
                             borrow_bps_ann: float = 50.0, n_draws: int = 2000,
                             block: int = 21, seed: int = 676) -> dict:
    """Circular-block-shuffle placebo for the timer's Sharpe advantage over buy-and-hold."""
    real = run_backtest(bars, signal, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
    obs = summary(real["r_strategy"])["sharpe"] - summary(real["r_bh"])["sharpe"]

    sig = signal.reindex(bars.index).fillna(0.0).to_numpy()
    n = len(sig)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_draws)
    n_blocks = max(1, n // block)
    for i in range(n_draws):
        off = int(rng.integers(0, n_blocks)) * block
        shuffled = np.roll(sig, off)
        ssig = pd.Series(shuffled, index=bars.index, name="signal")
        bt = run_backtest(bars, ssig, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
        diffs[i] = summary(bt["r_strategy"])["sharpe"] - summary(bt["r_bh"])["sharpe"]
    p = float((diffs >= obs).mean())
    return {"obs_sharpe_diff": float(obs), "placebo_mean": float(diffs.mean()),
            "p_value": p, "draws": diffs}


def run_timer_experiment(bars: pd.DataFrame, cost_bps: float = 5.0,
                         borrow_bps_ann: float = 50.0, hold_days: int = HOLD_DAYS,
                         placebo: bool = False, n_draws: int = 2000, seed: int = 676) -> dict:
    """Race the Gator timer vs buy-and-hold vs the 421 "always in the fan" rule."""
    sig_wake = wake_timer_signal(bars, hold_days=hold_days)
    sig_fan = fan_signal(bars)

    bt_wake = run_backtest(bars, sig_wake, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)
    bt_fan = run_backtest(bars, sig_fan, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann)

    s_bh = summary(bt_wake["r_bh"])
    s_wake = summary(bt_wake["r_strategy"])
    s_fan = summary(bt_fan["r_strategy"])
    out = {
        "bh": s_bh, "wake": s_wake, "fan": s_fan,
        "t_wake_vs_bh": sharpe_diff_tstat(bt_wake["r_strategy"], bt_wake["r_bh"]),
        "t_wake_vs_fan": sharpe_diff_tstat(bt_wake["r_strategy"], bt_fan["r_strategy"]),
        "in_pos_frac": float((sig_wake.abs() > 0).mean()),
        "n_wakes": int(gator_wake(bars).sum()),
    }
    if placebo:
        pl = block_permutation_pvalue(bars, sig_wake, cost_bps=cost_bps,
                                      borrow_bps_ann=borrow_bps_ann, n_draws=n_draws, seed=seed)
        out["placebo_p"] = pl["p_value"]
        out["placebo_obs"] = pl["obs_sharpe_diff"]
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof) — reuses summarize() on a
# synthetic multi-series panel (data.synthetic_multi_panel), the same function
# the real-tape headline uses. No separate code path to keep honest.
# --------------------------------------------------------------------------- #
