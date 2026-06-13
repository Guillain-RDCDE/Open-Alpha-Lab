"""Strategy and honest controls — Study 109 (OBV-Divergence).

The folk recipe (Granville 1963): *"volume precedes price."* We test two
distinct claims:

(A) **OBV trend / cross signal**: if OBV is rising (OBV above its N-day SMA),
    prices should also rise. Trade long when OBV > SMA(OBV, N), short
    otherwise.

(B) **OBV-price divergence reversal**: when OBV and price *disagree* — OBV
    trending up while price trends down, or vice versa — a reversal of price
    is imminent. This is the stronger claim: divergence predicts the *direction*
    of the next move.

Both are tested with:
- A forward N-day return (formed after the signal bar, on the *next* day's
  open — no look-ahead).
- A random-direction control on the same signal dates, so the verdict is
  "does the signal beat a coin?" not "does it beat zero?"
- A HAC Newey-West t-stat on the per-signal returns (the inference bar).

**No look-ahead rule**: OBV and any SMA of OBV are computed on data up to and
including day *t*; the position is entered at day *t+1*'s open and the
forward return is measured from open *t+1* to close *t+N* (or close *t+1*
for a 1-day horizon).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OBV and related indicators
# ---------------------------------------------------------------------------
def obv(bars: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (Granville 1963).

    OBV accumulates signed volume: add the day's volume when close > previous
    close, subtract it when close < previous close, and carry it unchanged when
    the close is unchanged. The *level* of OBV is arbitrary (it depends on the
    starting point); what matters is the *trend* relative to the price trend.

    Returns a Series aligned to ``bars.index``, with the first value = 0.
    """
    close = bars["close"].to_numpy()
    vol = bars["volume"].to_numpy()
    n = len(close)
    out = np.empty(n, dtype=float)
    out[0] = 0.0
    for i in range(1, n):
        if close[i] > close[i - 1]:
            out[i] = out[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            out[i] = out[i - 1] - vol[i]
        else:
            out[i] = out[i - 1]
    return pd.Series(out, index=bars.index, name="obv")


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average of ``series`` with window ``n``."""
    return series.rolling(n, min_periods=n).mean()


def price_trend(close: pd.Series, n: int) -> pd.Series:
    """N-day return (log) used as a proxy for the price trend direction.

    Returns the log return from ``n`` days ago to today. Positive = up-trend,
    negative = down-trend.
    """
    return np.log(close / close.shift(n))


# ---------------------------------------------------------------------------
# Signal A — OBV trend / cross
# ---------------------------------------------------------------------------
def signal_obv_trend(
    bars: pd.DataFrame,
    obv_sma_n: int = 20,
) -> pd.Series:
    """OBV trend signal: +1 when OBV > SMA(OBV, N), -1 otherwise.

    The direction of OBV relative to its own moving average is used as a
    directional price forecast. Signal is formed at the *close* of day *t*;
    the position is taken at the *open* of day *t+1*.

    Returns a Series of {+1, -1} with the same index as ``bars`` (NaN during
    the SMA warm-up period).
    """
    o = obv(bars)
    o_sma = sma(o, obv_sma_n)
    sig = pd.Series(np.where(o > o_sma, 1.0, -1.0), index=bars.index, name="signal_trend")
    sig[o_sma.isna()] = np.nan
    return sig


# ---------------------------------------------------------------------------
# Signal B — OBV-price divergence
# ---------------------------------------------------------------------------
def signal_obv_divergence(
    bars: pd.DataFrame,
    lookback: int = 10,
    obv_sma_n: int = 5,
) -> pd.Series:
    """OBV-vs-price divergence reversal signal.

    Classic Granville divergence: when OBV is rising (positive N-day trend)
    but price is falling (negative N-day return), a bullish reversal is
    expected → signal = +1. When OBV is falling but price is rising → bearish
    divergence → signal = -1. When both agree, there is no divergence → 0.

    The signal is formed at the close of day *t* (using data up to *t*) and
    traded at the open of *t+1*.

    ``lookback`` controls the trend-detection window (N-day OBV slope vs N-day
    price return). ``obv_sma_n`` smooths OBV before computing its trend to
    reduce noise.

    Returns a Series with values in {-1, 0, +1} (0 = no divergence detected).
    """
    o = obv(bars)
    o_smooth = sma(o, obv_sma_n)

    # OBV trend: difference of smoothed OBV over the lookback window
    obv_trend = o_smooth - o_smooth.shift(lookback)
    # Price trend: log return over the lookback window
    price_ret = price_trend(bars["close"], lookback)

    # Divergence: OBV and price moving in *opposite* directions
    bullish_div = (obv_trend > 0) & (price_ret < 0)   # OBV up, price down -> expect price up
    bearish_div = (obv_trend < 0) & (price_ret > 0)   # OBV down, price up -> expect price down

    sig = pd.Series(0.0, index=bars.index, name="signal_div")
    sig[bullish_div] = 1.0   # buy the reversal
    sig[bearish_div] = -1.0  # sell the reversal

    # NaN during warm-up
    warm_up = o_smooth.isna() | o_smooth.shift(lookback).isna()
    sig[warm_up] = np.nan
    return sig


# ---------------------------------------------------------------------------
# Forward return engine — the payoff measured honestly
# ---------------------------------------------------------------------------
def forward_returns(
    bars: pd.DataFrame,
    horizon: int = 5,
    cost_bps: float = 2.0,
) -> pd.Series:
    """Per-day forward log return from open(t+1) to close(t+horizon).

    The signal is formed at the close of day *t*; the position enters at
    *open(t+1)* and exits at *close(t+horizon)*. Cost is a flat one-way
    round-trip charge on the gross return.

    Returns a Series aligned to ``bars.index`` (NaN for the last ``horizon``
    bars where the horizon is incomplete and for the last bar where next-open
    is unavailable).
    """
    open_ = bars["open"].to_numpy()
    close = bars["close"].to_numpy()
    n = len(bars)
    fwd = np.full(n, np.nan)
    # Position enters at open of t+1, exits at close of t+horizon.
    for t in range(n - horizon - 1):
        entry = open_[t + 1]
        exit_ = close[t + horizon]
        if entry > 0 and exit_ > 0:
            fwd[t] = np.log(exit_ / entry)
    return pd.Series(fwd, index=bars.index, name=f"fwd_{horizon}d")


def run_signals(
    bars: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 5,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Apply a {-1, 0, +1} signal to forward returns and build a trade ledger.

    Only the days where ``signal != 0`` generate a trade (for the trend signal,
    every non-NaN day is active; for the divergence signal, only divergence days).

    Returns a DataFrame with columns:
        ``signal_date, dir, fwd_gross, fwd_net``
    """
    fwd = forward_returns(bars, horizon=horizon, cost_bps=0.0)
    combined = pd.concat([signal.rename("dir"), fwd], axis=1).dropna()
    combined = combined[combined["dir"] != 0].copy()

    # Gross: direction * forward log return
    combined["ret_gross"] = combined["dir"] * combined[f"fwd_{horizon}d"]
    combined["ret_net"] = combined["ret_gross"] - cost_bps * 1e-4
    return combined.rename_axis("signal_date").reset_index()[
        ["signal_date", "dir", "ret_gross", "ret_net"]
    ]


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Ledger summary — the inference bar
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-signal statistics for one ledger.

    Returns signal count, win-rate, mean return (bps/signal), P&L skew, and a
    HAC Newey-West t-stat on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.

    The t-stat is the decisive number: the inference bar is |t| >= 2. Everything
    below that threshold is indistinguishable from noise, however large the
    sample and however impressive the point estimate looks.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_signals": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Newey-West HAC (Bartlett kernel, automatic lag selection).
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Pooled cross-ticker analysis helpers
# ---------------------------------------------------------------------------
def run_basket(
    bars_map: dict[str, pd.DataFrame],
    signal_fn,
    horizon: int = 5,
    cost_bps: float = 2.0,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Run ``signal_fn`` on each ticker in ``bars_map`` and pool the ledgers.

    ``signal_fn(bars) -> pd.Series`` must return a {-1, 0, +1} Series.
    When ``random_seed`` is not None, the direction is replaced by a coin flip
    (the random-direction control arm).

    Returns a pooled ledger with an additional ``ticker`` column.
    """
    frames = []
    for ticker, bars in bars_map.items():
        sig = signal_fn(bars)
        led = run_signals(bars, sig, horizon=horizon, cost_bps=cost_bps)
        if random_seed is not None and len(led):
            dirs = random_directions(len(led), seed=random_seed)
            led["ret_gross"] = dirs * led["ret_gross"].abs()  # same magnitudes, random sign
            led["ret_net"] = led["ret_gross"] - cost_bps * 1e-4
            led["dir"] = dirs
        led["ticker"] = ticker
        frames.append(led)
    if not frames:
        return pd.DataFrame(columns=["signal_date", "dir", "ret_gross", "ret_net", "ticker"])
    return pd.concat(frames, ignore_index=True)
