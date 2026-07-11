"""The McClellan Summation Index as a falsifiable mechanical rule — Study 666.

The **Summation Index** is nothing exotic: it is the *running cumulative sum* (the
discrete integral) of the McClellan Oscillator (study
[491](../../491-mcclellan-oscillator/)) — Sherman & Marian McClellan's own extension
of their 1969 breadth oscillator into a slower, level-based indicator::

    McOsc_t  = EMA_19(net_adv)_t - EMA_39(net_adv)_t      (the oscillator, study 491)
    Summ_t   = Summ_{t-1} + McOsc_t   =   cumsum(McOsc)_t  (THIS study)

The folklore (every breadth-trading site, and the McClellans' own teaching) treats
the Summation Index as a **regime gauge**, not a single-day trigger: crossing up
through **zero** confirms a new bull phase, crossing down through zero confirms a new
bear phase, and extreme excursions (classically **+-500 / +-1000** on the *full NYSE*
scale) mark overbought/oversold turning points. We encode exactly that, three ways:

1. **Causal Summation Index.** ``Summ_t`` uses only breadth up to and including day
   *t* (a plain running sum of the causal oscillator — no centering, no look-ahead).
   A generous **warm-up is dropped** (see ``WARMUP_DAYS``) because the cumulative sum has
   no natural zero: we start it at 0 on day one of the basket's history and never
   re-base it, so the *earliest* crosses mostly reflect that arbitrary starting point,
   not real breadth history — a named quirk of any un-rebased running integral.
2. **Zero-cross events.** A bull(bear) signal fires the first day ``Summ`` crosses up
   (down) through zero — tested as a forward-return **event study** against a
   drift-matched random-entry baseline (Welch *t*), exactly as sibling study 491 tests
   the oscillator's own cross.
3. **Extreme-threshold events.** The literature's **+-500** levels are calibrated to
   *full NYSE exchange* breadth (thousands of issues); they do not transfer to a
   10-name ETF proxy without rescaling. We use a **causal, expanding/rolling z-score**
   of ``Summ`` (no look-ahead: each day's z uses only its own trailing window) and
   call a +-1sigma cross the scale-appropriate analog of the literature's +-500 level —
   stated as an explicit, honest operationalization, not a literal transcription.
4. **The long/flat regime timer.** The headline tradability test: hold the traded
   index while ``Summ_{t-1} > 0`` (one documented execution lag — the regime known at
   the close of *t-1* sets the exposure for day *t*), flat otherwise, costed on every
   regime switch. Excess over buy-and-hold (same instrument, so already
   exposure-matched) is tested with a Newey-West (HAC) one-sample *t* — daily
   regime returns are serially correlated by construction, so Welch alone would be
   wrong here.

**Dedup vs 491-mcclellan-oscillator:** that study tests the *oscillator's own*
zero-cross as a single-day trigger; this study tests its *cumulative integral* as a
multi-day regime gauge — the *"Summ_t > 0"* long/flat filter and the
extreme-threshold events have no analog in 491's single-instrument, event-only
design. The Signal axis is always **trigger/regime vs a drift-matched random or
buy-and-hold baseline**, never trigger-vs-zero, because an upward-drifting index
makes *any* long exposure look good on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
FAST, SLOW = 19, 39          # classic McClellan EMA spans (10% / 5% smoothing constants)
WARMUP_DAYS = 500             # ~2 trading years dropped as Summation-Index burn-in (named quirk)
ZSCORE_WINDOW = 252           # 1 trading year, causal rolling window for the extreme threshold


# --------------------------------------------------------------------------- #
# The oscillator and its running integral
# --------------------------------------------------------------------------- #
def mcclellan(net_adv: pd.Series, fast: int = FAST, slow: int = SLOW) -> pd.Series:
    """The McClellan oscillator = EMA_fast(net_adv) - EMA_slow(net_adv), causal."""
    a = pd.Series(np.asarray(net_adv, dtype=float), index=net_adv.index)
    ef = a.ewm(span=fast, adjust=False).mean()
    es = a.ewm(span=slow, adjust=False).mean()
    osc = ef - es
    osc.name = "mcosc"
    return osc


def summation_index(net_adv: pd.Series, fast: int = FAST, slow: int = SLOW) -> pd.Series:
    """The Summation Index = the running cumulative sum of the McClellan oscillator.

    Starts at 0 on the first day of ``net_adv`` and is never re-based — a plain,
    fully causal discrete integral of a causal oscillator, so it carries no
    look-ahead of its own.
    """
    summ = mcclellan(net_adv, fast, slow).cumsum()
    summ.name = "summ"
    return summ


def _warmup(n: int, slow: int = SLOW) -> int:
    """Bars to discard: the oscillator's own EMA warm-up plus the Summation burn-in."""
    return min(n, max(3 * slow, WARMUP_DAYS))


# --------------------------------------------------------------------------- #
# Zero-cross events (the regime-turn signal)
# --------------------------------------------------------------------------- #
def zero_cross_dates(summ: pd.Series, direction: str = "up", slow: int = SLOW,
                     apply_warmup: bool = True) -> pd.DatetimeIndex:
    """Dates ``summ`` crosses zero, after the warm-up burn-in (unless ``apply_warmup=False``,
    used only to inspect the raw/full history for documentation purposes).

    ``direction="up"``: crosses from <=0 to >0 (bull-regime confirmation).
    ``direction="down"``: crosses from >=0 to <0 (bear-regime confirmation).
    """
    s = summ.to_numpy(dtype=float)
    cross = np.zeros(s.size, dtype=bool)
    if direction == "up":
        cross[1:] = (s[1:] > 0) & (s[:-1] <= 0)
    elif direction == "down":
        cross[1:] = (s[1:] < 0) & (s[:-1] >= 0)
    else:
        raise ValueError("direction must be 'up' or 'down'")
    if apply_warmup:
        w = _warmup(s.size, slow)
        cross[:w] = False
    return summ.index[cross]


# --------------------------------------------------------------------------- #
# Extreme-threshold events (the honest analog of the literature's +-500 level)
# --------------------------------------------------------------------------- #
def rolling_zscore(summ: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Causal rolling z-score of the Summation Index (each day uses only its own
    trailing ``window`` — no look-ahead)."""
    mu = summ.rolling(window, min_periods=window).mean()
    sd = summ.rolling(window, min_periods=window).std(ddof=1)
    z = (summ - mu) / sd
    z.name = "summ_z"
    return z


def extreme_cross_dates(z: pd.Series, level: float = 1.0, direction: str = "up",
                        slow: int = SLOW) -> pd.DatetimeIndex:
    """Dates the rolling z-score crosses +-``level`` — the scale-appropriate stand-in
    for the literature's +-500 Summation Index extreme."""
    v = z.to_numpy(dtype=float)
    cross = np.zeros(v.size, dtype=bool)
    valid = ~np.isnan(v)
    if direction == "up":
        c = (v[1:] > level) & (np.nan_to_num(v[:-1], nan=level + 1) <= level)
    else:
        c = (v[1:] < -level) & (np.nan_to_num(v[:-1], nan=-level - 1) >= -level)
    cross[1:] = c & valid[1:]
    w = _warmup(v.size, slow)
    cross[:w] = False
    return z.index[cross]


# --------------------------------------------------------------------------- #
# Forward-return engine (event studies) — shared with 491's design
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost charged twice (in + out) and subtracted from each
    trade's return. Trades whose window overruns the tape are dropped.
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


def random_entries(index: pd.DatetimeIndex, n: int, slow: int = SLOW, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    w = _warmup(len(index), slow)
    valid = index[w:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero (Bartlett kernel,
    automatic lag length)."""
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


# --------------------------------------------------------------------------- #
# Event-study orchestrator (zero-cross AND extreme-threshold)
# --------------------------------------------------------------------------- #
def cross_experiment(close: pd.Series, summ: pd.Series, direction: str = "up",
                     kind: str = "zero", level: float = 1.0, cost_bps: float = 1.0,
                     random_seed: int = 7, slow: int = SLOW) -> dict:
    """Run the up-cross-or-down-cross event study, vs a drift-matched random baseline.

    ``kind`` is "zero" (the classic regime-turn cross) or "extreme" (the causal
    z-score analog of the +-500 literature level).
    """
    summ = summ.reindex(close.index).dropna()
    if kind == "zero":
        ent = zero_cross_dates(summ, direction, slow)
    elif kind == "extreme":
        z = rolling_zscore(summ)
        ent = extreme_cross_dates(z, level, direction, slow)
    else:
        raise ValueError("kind must be 'zero' or 'extreme'")

    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net_s = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close.index, max(len(ent), 50), slow=slow, seed=random_seed), h))
        a = forward_returns(close, ent, h, cost_bps=0.0)
        b = forward_returns(
            close, random_entries(close.index, max(len(ent), 50), slow=slow, seed=random_seed), h)
        res["by_h"][h] = {
            "gross": g, "net": net_s, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
            "welch_t": welch_t(a, b),
        }
    return res


def shuffled_breadth_placebo(close: pd.Series, net_adv: pd.Series, horizon: int = 20,
                             fast: int = FAST, slow: int = SLOW, direction: str = "up",
                             kind: str = "extreme", level: float = 1.0,
                             n_draws: int = 500, seed: int = 666) -> dict:
    """Placebo: time-permute the net-advances series, rebuild the Summation Index,
    re-fire the cross rule (``kind="zero"`` or ``"extreme"``).

    Shuffling ``net_adv`` in time destroys the breadth-momentum *structure* the
    cumulative sum is built to accumulate (autocorrelation, the slow multi-year
    swings) while preserving its marginal distribution exactly. If the real cross
    carries information, the observed mean should sit far in the right tail of the
    shuffled distribution. Note: with ``kind="zero"`` on the real 2005->2026 tape
    there are **zero** post-warmup crosses (see ``docs/results.md``), so this placebo
    is only meaningful with ``kind="extreme"``.
    """
    def _entries(net):
        summ = summation_index(net, fast, slow)
        if kind == "zero":
            return zero_cross_dates(summ, direction, slow)
        z = rolling_zscore(summ)
        return extreme_cross_dates(z, level, direction, slow)

    obs_r = forward_returns(close, _entries(net_adv), horizon)
    obs = float(np.mean(obs_r)) if obs_r.size else float("nan")
    if not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    vals = net_adv.to_numpy(dtype=float)
    idx = net_adv.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = pd.Series(rng.permutation(vals), index=idx, name="net_adv")
        rr = forward_returns(close, _entries(perm), horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


def shuffled_breadth_regime_placebo(close: pd.Series, net_adv: pd.Series, cost_bps: float = 5.0,
                                    fast: int = FAST, slow: int = SLOW,
                                    warmup: int = WARMUP_DAYS,
                                    n_draws: int = 500, seed: int = 666) -> dict:
    """Placebo for the headline z-score regime timer: time-permute ``net_adv``, rebuild
    the z-score regime, re-run ``regime_backtest``.

    Two-sided: the share of shuffled worlds whose ``|excess_hac_t|`` is >= the observed
    one — "is the specific EMA19-EMA39-then-cumsum-then-z-score geometry load-bearing,
    positively or negatively, for the excess we measured?"
    """
    obs_t = regime_backtest(close, regime_from_zscore(summation_index(net_adv, fast, slow)),
                            cost_bps=cost_bps, warmup=warmup)["excess_hac_t"]
    rng = np.random.default_rng(seed)
    vals = net_adv.to_numpy(dtype=float)
    idx = net_adv.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = pd.Series(rng.permutation(vals), index=idx, name="net_adv")
        regime = regime_from_zscore(summation_index(perm, fast, slow))
        rb = regime_backtest(close, regime, cost_bps=cost_bps, warmup=warmup)
        t = rb["excess_hac_t"]
        if not np.isfinite(t):
            continue
        valid += 1
        if abs(t) >= abs(obs_t):
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs_t": obs_t, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# The long/flat regime timer — the headline tradability test
# --------------------------------------------------------------------------- #
def regime_from_level(summ: pd.Series) -> pd.Series:
    """The textbook regime rule: bull while the Summation Index sits above zero."""
    r = (summ > 0).astype(float)
    r.name = "regime_level"
    return r


def regime_from_zscore(summ: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """The scale-appropriate adaptation: bull while the Summation Index sits above its
    own trailing rolling mean (z > 0) — used because, on this basket, the raw level
    never revisits zero after the first few months (see ``docs/results.md``), which
    would make the textbook rule a silent, permanent buy-and-hold."""
    z = rolling_zscore(summ, window)
    r = (z > 0).astype(float)
    r.name = "regime_zscore"
    return r


def regime_backtest(close: pd.Series, regime: pd.Series, cost_bps: float = 5.0,
                    warmup: int = WARMUP_DAYS) -> dict:
    """Long the traded index while ``regime`` is 1, flat while 0.

    **One documented execution lag**: the regime is read at the close of *t-1* and
    applied to day *t*'s return (``pos = regime.shift(1)``) — zero look-ahead. Every
    regime switch (flat->long or long->flat) is charged ``cost_bps`` one-way x NAV.
    The warm-up burn-in is dropped before any statistic is computed. Excess = timed -
    buy&hold (same instrument, so already exposure-matched); tested with a
    Newey-West (HAC) one-sample t because daily regime returns are serially
    correlated by construction.
    """
    ret = close.pct_change()
    regime = regime.reindex(close.index)
    w = min(len(close), warmup)
    pos = regime.shift(1)                    # ONE LAG: yesterday's regime sets today's exposure
    switch = pos.diff().abs().fillna(0.0)
    cost = switch * (cost_bps / 1e4)
    strat_ret = (pos * ret - cost).iloc[w:].dropna()
    bh_ret = ret.iloc[w:].dropna()
    strat_ret, bh_ret = strat_ret.align(bh_ret, join="inner")
    excess = (strat_ret - bh_ret).to_numpy(dtype=float)

    n_years = len(strat_ret) / 252.0
    n_switches = int(switch.iloc[w:].sum())

    def _ann(r: np.ndarray) -> dict:
        r = r[np.isfinite(r)]
        mu, sd = r.mean(), r.std(ddof=1)
        cagr = float(np.prod(1.0 + r) ** (252.0 / len(r)) - 1.0) if len(r) else float("nan")
        sharpe = float(mu / sd * np.sqrt(252)) if sd > 0 else float("nan")
        return {"cagr": cagr, "ann_vol": float(sd * np.sqrt(252)), "sharpe": sharpe}

    return {
        "n_days": len(strat_ret), "n_years": n_years, "n_switches": n_switches,
        "switches_per_year": n_switches / n_years if n_years > 0 else float("nan"),
        "timed": _ann(strat_ret.to_numpy()), "buy_hold": _ann(bh_ret.to_numpy()),
        "excess_mean_bps_day": float(np.nanmean(excess) * 1e4),
        "excess_hac_t": hac_t(excess),
        "frac_long": float(pos.iloc[w:].mean()),
    }


def synthetic_regime_detect(close: pd.DataFrame, net_adv: pd.Series, cost_bps: float = 5.0,
                            fast: int = FAST, slow: int = SLOW,
                            kind: str = "level") -> dict:
    """Run the headline regime backtest on a synthetic (close, net_adv) world."""
    summ = summation_index(net_adv, fast, slow)
    regime = regime_from_level(summ) if kind == "level" else regime_from_zscore(summ)
    return regime_backtest(close["close"], regime, cost_bps=cost_bps, warmup=_warmup(len(close), slow))
