"""Strategy + inference for Study 682 — Inverse-Fisher-RSI.

The claim (Ehlers, *"Using The Fisher Transform"*, TASC 2002, and Ch. 1 of *Cybernetic
Analysis for Stocks and Futures*, 2004): ordinary RSI clusters near 0/100 and lingers there,
so its overbought/oversold crossovers are mushy. Passing RSI through the **Inverse Fisher
Transform** compresses it into a crisp bounded ``[-1, +1]`` oscillator that "snaps" between
its extremes, giving cleaner, more decisive turning-point signals. Ehlers' own recipe, used
here exactly:

    v1 = 0.1 * (RSI(5) - 50)
    v2 = WMA(v1, 9)                       # 9-bar weighted moving average, the smoothing step
    IFT-RSI = (exp(2*v2) - 1) / (exp(2*v2) + 1)                # = tanh(v2)

A **bullish turn** is IFT-RSI crossing UP through -0.5 (an oversold reading snapping back);
a **bearish turn** is IFT-RSI crossing DOWN through +0.5. We test both, against:

* **The unconditional baseline** — every day is a candidate entry, identical forward-return
  formula, no conditioning.
* **A random-signal placebo** — the same number of signals per ticker, drawn from random
  eligible dates, repeated over many seeds.
* **Plain RSI(14) reversal** (cross up through 30 / down through 70 — the textbook
  overbought/oversold levels) and **plain RSI(2) reversal** (Connors-style, cross up through
  10 — see sibling [75-knee-jerk](../../75-knee-jerk/)) — the two baselines the "sharper
  turning points" claim has to beat, using the identical event-study machinery so the
  comparison is apples-to-apples.
* **A timer with costs** — turn the IFT-RSI crossover into a long-flat rule (enter on the
  bullish cross, exit on the bearish cross), one round trip = 2 x one-way cost x NAV, and
  compare its Sharpe to buy-and-hold and to a random-exposure control matched on time-in-market.

One documented execution lag throughout: a signal known at the close of day *t* earns the
return from the close of *t+1* onward (a single ``shift`` — see ``forward_return``).

The decisive numbers are the pooled Welch/HAC *t* on the REAL tape for the IFT-RSI signal vs
(a) unconditional and (b) the plain-RSI baselines, plus whether the timer clears costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (5, 10, 20)


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #
def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (Wilder-smoothed average gain/loss)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0.0, 100.0)


def wma(series: pd.Series, length: int = 9) -> pd.Series:
    """Linearly-weighted moving average (most recent bar weighted ``length``, oldest weighted 1)."""
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def ift_rsi(close: pd.Series, rsi_period: int = 5, wma_period: int = 9) -> pd.Series:
    """Ehlers' Inverse Fisher Transform of RSI — bounded in [-1, +1]. See module docstring."""
    r = rsi_wilder(close, rsi_period)
    v1 = 0.1 * (r - 50.0)
    v2 = wma(v1, wma_period)
    return (np.exp(2.0 * v2) - 1.0) / (np.exp(2.0 * v2) + 1.0)


# --------------------------------------------------------------------------- #
# Crossings and signal set
# --------------------------------------------------------------------------- #
def cross_up(s: pd.Series, level: float) -> pd.Series:
    """True on bars where ``s`` crosses UP through ``level`` (prev < level <= curr)."""
    prev = s.shift(1)
    return (prev < level) & (s >= level)


def cross_down(s: pd.Series, level: float) -> pd.Series:
    """True on bars where ``s`` crosses DOWN through ``level`` (prev > level >= curr)."""
    prev = s.shift(1)
    return (prev > level) & (s <= level)


def signal_set(close: pd.Series) -> dict[str, pd.Series]:
    """All signals tested on one instrument's close series, keyed by name.

    ``ift_bull``/``ift_bear`` are the object of study; ``rsi14_bull`` and ``rsi2_bull`` are the
    two baselines the "sharper turning points" claim has to beat, on the identical machinery.
    """
    ift = ift_rsi(close, rsi_period=5, wma_period=9)
    r14 = rsi_wilder(close, period=14)
    r2 = rsi_wilder(close, period=2)
    return {
        "ift_bull": cross_up(ift, -0.5),
        "ift_bear": cross_down(ift, 0.5),
        "rsi14_bull": cross_up(r14, 30.0),
        "rsi14_bear": cross_down(r14, 70.0),
        "rsi2_bull": cross_up(r2, 10.0),
    }


# --------------------------------------------------------------------------- #
# Forward returns — one documented execution lag: signal at close t, hold t+1 .. t+1+h
# --------------------------------------------------------------------------- #
def forward_return(close: pd.Series, h: int, lag: int = 1) -> pd.Series:
    """Log return from close(t+lag) to close(t+lag+h), indexed by t.

    ``lag=1`` is the study's single documented execution convention: a signal known at the
    close of day t is actionable at the next session's close (the desk's default one-shift
    lag for a signal with no scheduled/public-calendar exemption). Defined for every date (not
    just signal dates) so it doubles as the unconditional baseline generator.
    """
    entry = close.shift(-lag)
    exit_ = close.shift(-(lag + h))
    return np.log(exit_ / entry)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int = 5) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d.

    b is exactly the treated-minus-rest mean difference; the NW t is the serial-correlation
    -robust cross-check for a forward-return series whose windows overlap.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~np.isnan(y) & ~np.isnan(d)
    y, d = y[keep], d[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Basket-wide event + forward-return assembly
# --------------------------------------------------------------------------- #
def basket_signals(data: dict[str, pd.Series]) -> dict[str, dict[str, pd.Series]]:
    """{ticker: {signal_name: bool Series}} for every instrument in ``data``."""
    return {tkr: signal_set(close) for tkr, close in data.items()}


def basket_forward(data: dict[str, pd.Series], h: int) -> dict[str, pd.Series]:
    """{ticker: forward-return series at horizon h}."""
    return {tkr: forward_return(close, h) for tkr, close in data.items()}


def headline_stats(data: dict[str, pd.Series], signals: dict[str, dict[str, pd.Series]],
                    signal_name: str, h: int, tail: str = "right",
                    nw_lags: int | None = None) -> dict:
    """Signal-conditional forward return vs the unconditional (same-instrument) baseline.

    Pools every ticker's signal-day and non-signal-day observations, Welch t on the split, and
    a Newey-West (HAC) cross-check on the pooled dummy regression (lag = h, the overlap window).
    """
    fwd = basket_forward(data, h)
    lags = h if nw_lags is None else nw_lags

    sig_vals, base_vals, y_all, d_all, n_events = [], [], [], [], 0
    for tkr, s in fwd.items():
        s = s.dropna()
        flag = signals[tkr][signal_name].reindex(s.index).fillna(False).to_numpy()
        n_events += int(flag.sum())
        sig_vals.append(s.values[flag])
        base_vals.append(s.values[~flag])
        y_all.append(s.values)
        d_all.append(flag.astype(float))
    sig = np.concatenate(sig_vals) if sig_vals else np.array([])
    base = np.concatenate(base_vals) if base_vals else np.array([])
    y = np.concatenate(y_all) if y_all else np.array([])
    d = np.concatenate(d_all) if d_all else np.array([])

    k_up = int((sig > 0).sum())
    lo, hi = wilson_interval(k_up, len(sig))
    return {
        "signal": signal_name, "h": h, "n_sig": len(sig), "n_base": len(base),
        "sig_mean_bps": float(np.nanmean(sig)) * 1e4 if len(sig) else float("nan"),
        "base_mean_bps": float(np.nanmean(base)) * 1e4 if len(base) else float("nan"),
        "gap_bps": (float(np.nanmean(sig)) - float(np.nanmean(base))) * 1e4
                    if len(sig) and len(base) else float("nan"),
        "welch_t": welch_t(sig, base),
        "nw_t": newey_west_t(y, d, lags=lags),
        "hit_up": k_up, "hit_rate": k_up / len(sig) if len(sig) else float("nan"),
        "hit_lo": lo, "hit_hi": hi,
    }


def random_signal_placebo(data: dict[str, pd.Series], signals: dict[str, dict[str, pd.Series]],
                           signal_name: str, h: int, n_draws_per_seed: int = 200,
                           n_seeds: int = 20, base_seed: int = 682) -> dict:
    """Random-signal control: per ticker, draw as many random eligible dates as real signals.

    Repeated over ``n_seeds`` independent seeds x ``n_draws_per_seed`` draws, so no single lucky
    stream decides it. p = share of draws whose pooled mean forward return is >= the observed
    signal mean (a RIGHT-tail test for a bullish claim).
    """
    fwd = basket_forward(data, h)
    per_ticker = {}
    for tkr, s in fwd.items():
        s = s.dropna()
        flag = signals[tkr][signal_name].reindex(s.index).fillna(False)
        per_ticker[tkr] = (s.values, int(flag.sum()))

    obs = headline_stats(data, signals, signal_name, h)["sig_mean_bps"] / 1e4

    means = []
    for seed_i in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed_i)
        for _ in range(n_draws_per_seed):
            draw_vals = []
            for tkr, (vals, n_sig) in per_ticker.items():
                if n_sig == 0 or len(vals) == 0:
                    continue
                k = min(n_sig, len(vals))
                idx = rng.choice(len(vals), size=k, replace=False)
                draw_vals.append(vals[idx])
            if draw_vals:
                means.append(np.concatenate(draw_vals).mean())
    means = np.asarray(means)
    return {
        "obs": obs, "placebo_mean": float(means.mean()), "placebo_sd": float(means.std(ddof=1)),
        "p_value": float((means >= obs).mean()), "n_draws": len(means), "draws": means,
    }


# --------------------------------------------------------------------------- #
# Third axis — the timer with costs
# --------------------------------------------------------------------------- #
def build_position(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """Long-flat state machine: 1 from an enter bar until the next exit bar, else 0.

    On a bar where both fire, exit wins (conservative: an overbought reading that also
    happens to look oversold on the same print does not open a new position).
    """
    state = pd.Series(np.nan, index=enter.index)
    state[enter] = 1.0
    state[exit_] = 0.0
    return state.ffill().fillna(0.0)


def timer_with_costs(data: dict[str, pd.Series], signals: dict[str, dict[str, pd.Series]],
                      enter_name: str, exit_name: str, cost_bps: float = 5.0) -> dict:
    """Long-flat timer built from ``enter_name``/``exit_name`` crossings, pooled across the basket.

    One documented execution lag: the state decided at close t is HELD during t+1 (a single
    ``shift``). One round trip = 2 x ``cost_bps`` one-way x NAV, charged on every state change.
    Reports pooled net daily Sharpe (annualised, excess-of-cash ~ raw since idle cash accrues
    ~0% in this simplified timer) vs buy-and-hold Sharpe on the same instruments/period.
    """
    strat_rets, bh_rets, exposures = [], [], []
    for tkr, close in data.items():
        pos = build_position(signals[tkr][enter_name], signals[tkr][exit_name])
        held = pos.shift(1).fillna(0.0)               # the execution lag
        ret = np.log(close / close.shift(1))
        cost = pos.diff().abs().fillna(0.0) * (cost_bps / 1e4)
        strat = held * ret - cost.shift(1).fillna(0.0)
        strat_rets.append(strat.dropna())
        bh_rets.append(ret.dropna())
        exposures.append(float(held.mean()))
    strat_all = pd.concat(strat_rets)
    bh_all = pd.concat(bh_rets)
    return {
        "sharpe_net": _sharpe(strat_all), "sharpe_bh": _sharpe(bh_all),
        "ann_ret_net_pct": float(strat_all.mean()) * TRADING_DAYS * 100,
        "ann_ret_bh_pct": float(bh_all.mean()) * TRADING_DAYS * 100,
        "exposure": float(np.mean(exposures)), "n_obs": len(strat_all),
    }


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def random_exposure_control(data: dict[str, pd.Series], exposure: float, cost_bps: float = 5.0,
                             n_switches_hint: int = 40, n_seeds: int = 20,
                             base_seed: int = 682) -> np.ndarray:
    """Random-timing coin: an i.i.d. Bernoulli(``exposure``) long-flat mask, same avg
    time-in-market as the real timer, ``n_seeds`` independent draws. Returns net Sharpes.
    """
    sharpes = []
    for seed_i in range(n_seeds):
        rng = np.random.default_rng(base_seed + seed_i)
        strat_rets = []
        for tkr, close in data.items():
            ret = np.log(close / close.shift(1)).dropna()
            raw = rng.random(len(ret)) < exposure
            # smooth into blocks so switch frequency resembles a real timer, not daily noise
            block = n_switches_hint
            pos = pd.Series(raw, index=ret.index).rolling(block, min_periods=1).mean().round()
            held = pos.shift(1).fillna(0.0)
            cost = pos.diff().abs().fillna(0.0) * (cost_bps / 1e4)
            strat = held * ret - cost.shift(1).fillna(0.0)
            strat_rets.append(strat.dropna())
        sharpes.append(_sharpe(pd.concat(strat_rets)))
    return np.asarray(sharpes)


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(close: pd.Series, h: int = 10) -> dict:
    """Run the headline IFT-RSI bullish-cross Welch/NW split on one synthetic instrument."""
    sig = signal_set(close)
    data = {"SYN": close}
    signals = {"SYN": sig}
    s = headline_stats(data, signals, "ift_bull", h)
    return s
