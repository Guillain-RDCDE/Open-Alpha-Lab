"""QQE (Quantitative Qualitative Estimation) as a falsifiable mechanical rule — Study 471.

The **QQE** indicator, popularised on MetaTrader/TradingView (the modern restatement of
J. Welles Wilder's RSI + ATR machinery), is built in three causal steps:

1. **RSI** — Wilder's Relative Strength Index over ``rsi_len`` bars.
2. **Smoothed RSI (RSI MA)** — an EMA of the RSI over ``sf`` bars (Wilder smoothing), the
   slow line the indicator actually trades.
3. **QQE trailing band** — the *ATR of the smoothed RSI* (a Wilder-smoothed absolute change),
   multiplied by the Wilder factor ``qqe_factor`` (default **4.236**), laid out as a trailing
   band that ratchets *up* under the smoothed RSI. This is the "fast" line.

The folklore (every QQE write-up): a **long fires when the smoothed RSI crosses *above* its
QQE trailing band** — read as a *momentum ignition* that price will continue. (Symmetrically, a
cross below the upper band is a sell.) We test the long trigger.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Causal construction** — RSI, its smoothing, and the ATR-of-RSI band are all built from a
   one-sided Wilder recursion; nothing peeks forward.
2. **Band-cross long** — a long entry fires on the **first** bar where the smoothed RSI rises
   from at-or-below the trailing band to above it. Entry is at the **next** close (one
   documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **phase-scramble placebo** that rebuilds the QQE band from a
   *phase-randomised* clone of the close (Fourier surrogate: same power spectrum / marginal,
   destroyed timing), so the band-cross fires on geometry that no longer lines up with the real
   turning points — the honest "is the QQE structure load-bearing?" null.

No look-ahead: the smoothed RSI and band are causal, the cross is read on the close of *t*, the
position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# QQE construction (all causal / one-sided)
# --------------------------------------------------------------------------- #
def wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI (causal): EMA of gains/losses with alpha = 1/length."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    a = 1.0 / length
    # Wilder smoothing == EMA with alpha = 1/length; first value seeded by adjust=False EMA.
    avg_gain = gain.ewm(alpha=a, adjust=False).mean()
    avg_loss = loss.ewm(alpha=a, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.fillna(100.0)   # all-gain windows -> RSI 100
    return rsi


def qqe_bands(close: pd.Series, rsi_len: int = 14, sf: int = 5,
              qqe_factor: float = 4.236) -> pd.DataFrame:
    """The causal QQE smoothed-RSI line and its ATR-of-RSI **trailing stop** (the fast line).

    This is the canonical QQE dual-band trailing-stop construction (the one TradingView/MT4
    code implements). At each bar we maintain a *long* candidate band (``rsi_ma - delta``,
    ratcheting up below the line) and a *short* candidate band (``rsi_ma + delta``, ratcheting
    down above the line). A single **trailing stop** ``ts`` follows whichever side the smoothed
    RSI is on: while ``rsi_ma`` is above the stop, ``ts`` = the long band (under the line);
    when ``rsi_ma`` pierces below, ``ts`` flips to the short band (over the line), and so on.

    Returns a DataFrame over ``close.index`` with columns:
      * ``rsi_ma`` — the smoothed RSI (the line that trades),
      * ``ts``     — the QQE trailing stop / fast line.
    NaN during warm-up. Everything is one-sided (no future bars).
    """
    rsi = wilder_rsi(close, length=rsi_len)
    a_sf = 1.0 / sf
    rsi_ma = rsi.ewm(alpha=a_sf, adjust=False).mean()        # smoothed RSI

    # ATR of the smoothed RSI = Wilder smoothing of |Δ rsi_ma|, over the RSI-window scale,
    # then smoothed again over (2*rsi_len - 1) as in Wilder's QQE — the band half-width.
    dar_raw = rsi_ma.diff().abs()
    a_rsi = 1.0 / rsi_len
    atr_rsi = dar_raw.ewm(alpha=a_rsi, adjust=False).mean()
    atr_rsi = atr_rsi.ewm(alpha=1.0 / (2 * rsi_len - 1), adjust=False).mean()
    delta = atr_rsi * qqe_factor

    ma = rsi_ma.to_numpy(dtype=float)
    d = delta.to_numpy(dtype=float)
    n = ma.size
    ts = np.full(n, np.nan)

    long_band = 0.0
    short_band = 0.0
    trend = 1          # +1 = RSI MA above the stop (stop = long band), -1 = below (stop = short)
    prev_ma = np.nan
    prev_long = 0.0
    prev_short = 0.0
    seeded = False
    for i in range(n):
        if not np.isfinite(ma[i]) or not np.isfinite(d[i]):
            prev_ma = ma[i]
            continue
        new_long = ma[i] - d[i]
        new_short = ma[i] + d[i]
        if not seeded:
            long_band, short_band = new_long, new_short
            trend = 1
            seeded = True
        else:
            # long band ratchets up while RSI MA stays above it
            if prev_ma > prev_long and ma[i] > prev_long:
                long_band = max(prev_long, new_long)
            else:
                long_band = new_long
            # short band ratchets down while RSI MA stays below it
            if prev_ma < prev_short and ma[i] < prev_short:
                short_band = min(prev_short, new_short)
            else:
                short_band = new_short
            # trend flip: crossing the opposite band switches the active stop
            if ma[i] > prev_short:
                trend = 1
            elif ma[i] < prev_long:
                trend = -1
        ts[i] = long_band if trend == 1 else short_band
        prev_ma = ma[i]
        prev_long = long_band
        prev_short = short_band

    return pd.DataFrame({"rsi_ma": rsi_ma.to_numpy(), "ts": ts},
                        index=close.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def qqe_cross_entries(close: pd.Series, rsi_len: int = 14, sf: int = 5,
                      qqe_factor: float = 4.236, warmup: int = 60) -> pd.DatetimeIndex:
    """Bars where the smoothed RSI crosses *above* its QQE trailing band — the long trigger.

    Only the *first* bar of each cross is kept (the ignition, not every day the line stays
    above the band). Entry is executed at the next close by :func:`forward_returns`.
    """
    bands = qqe_bands(close, rsi_len=rsi_len, sf=sf, qqe_factor=qqe_factor)
    ma = bands["rsi_ma"]
    ts = bands["ts"]
    above = (ma > ts) & ts.notna()
    prev_above = above.shift(1, fill_value=False)
    cross = above & ~prev_above
    # drop the warm-up zone where the band is still settling
    if warmup > 0 and len(close) > warmup:
        cutoff = close.index[warmup]
        cross = cross & (close.index >= cutoff)
    return close.index[cross.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 60, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


def random_baseline(close: pd.Series, n: int, horizon: int, warmup: int = 60,
                    n_seeds: int = 40, cost_bps: float = 0.0) -> np.ndarray:
    """Pooled forward returns of ``n`` random entries drawn over ``n_seeds`` seeds.

    A *single* random draw is itself noisy — with only a few hundred entries the baseline mean
    can land far from the true unconditional drift purely by luck (and a lucky-low draw would
    manufacture a fake edge for the rule). Pooling many seeds estimates the **drift the rule is
    really competing against**, which is the honest comparator on an up-drifting tape.
    """
    parts = []
    for s in range(n_seeds):
        re = random_entries(close, n, warmup=warmup, seed=s)
        parts.append(forward_returns(close, re, horizon, cost_bps=cost_bps))
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


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


def _phase_scramble(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase-randomised surrogate: same power spectrum/marginal, destroyed timing."""
    n = x.size
    f = np.fft.rfft(x)
    mag = np.abs(f)
    phases = np.angle(f)
    # randomise interior phases, keep DC (and Nyquist if present) real
    rand = rng.uniform(0, 2 * np.pi, size=phases.shape)
    rand[0] = phases[0]
    if n % 2 == 0:
        rand[-1] = phases[-1]
    f2 = mag * np.exp(1j * rand)
    y = np.fft.irfft(f2, n=n)
    return y


def phase_scramble_placebo(close: pd.Series, horizon: int, rsi_len: int = 14, sf: int = 5,
                           qqe_factor: float = 4.236, n_draws: int = 500,
                           seed: int = 471) -> dict:
    """Placebo: rebuild the QQE band on a phase-randomised clone of the *returns* path.

    The Fourier surrogate keeps the close series' power spectrum and (approximately) its return
    marginal, but scrambles the timing — so the QQE band-cross fires on a series that no longer
    lines up with the real turning points. Returns the share of placebo runs whose mean
    band-cross forward return **beats** the real one — the honest "is the QQE structure adding
    anything?" p-value, plus the observed mean.
    """
    real_ent = qqe_cross_entries(close, rsi_len=rsi_len, sf=sf, qqe_factor=qqe_factor)
    obs = float(np.mean(forward_returns(close, real_ent, horizon))) if len(real_ent) else float("nan")
    if not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}

    rng = np.random.default_rng(seed)
    logp = np.log(close.to_numpy(dtype=float))
    logret = np.diff(logp)
    idx = close.index
    p0 = float(close.iloc[0])

    beats = 0
    valid = 0
    for _ in range(n_draws):
        sur_ret = _phase_scramble(logret, rng)
        sur_close = p0 * np.exp(np.concatenate([[0.0], np.cumsum(sur_ret)]))
        sc = pd.Series(sur_close, index=idx)
        ent = qqe_cross_entries(sc, rsi_len=rsi_len, sf=sf, qqe_factor=qqe_factor)
        rr = forward_returns(sc, ent, horizon)
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
def run_experiment(close: pd.Series, rsi_len: int = 14, sf: int = 5, qqe_factor: float = 4.236,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: QQE band-cross vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the band-cross summary (gross + net), the
    drift-matched random-entry baseline, and the cross-minus-random delta.
    """
    ent = qqe_cross_entries(close, rsi_len=rsi_len, sf=sf, qqe_factor=qqe_factor)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        # robust drift-matched baseline (pooled over many seeds, not one lucky draw)
        rnd = summarize(random_baseline(close, max(len(ent), 50), h, n_seeds=40))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
