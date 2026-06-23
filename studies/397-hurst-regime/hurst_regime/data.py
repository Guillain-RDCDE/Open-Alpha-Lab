"""Data layer for Study 397 — Hurst-Regime (rolling Hurst on real markets + synthetic control).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a handful of liquid markets (SPY, QQQ, GLD, TLT,
  EFA) via yfinance (no key), cached under ``_cache/prices.csv`` (a wide CSV indexed by date,
  one column per ticker). The believers' claim is that the *Hurst exponent* of a price series
  tells you whether it is in a **trending** regime (H > 0.5, momentum pays) or a
  **mean-reverting** regime (H < 0.5, fade pays) — so a Hurst-gated switch between the two
  styles should beat either style alone.

* **Synthetic.** A deterministic, fixed-seed generator that produces fractional Gaussian /
  AR(1) price paths with a **known** Hurst exponent and a **planted edge knob** (``edge``).
  With ``edge = 0`` the regime label carries *no* forecasting power and the switch must NOT
  manufacture significance; with a large planted edge the Hurst gate must light up. It is the
  positive control: the estimator must recover the planted H, and the inference must recover
  the planted edge.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "prices.csv")

# A small, transparent set of liquid markets with long history. SPY is the headline tape;
# the others let us ask whether any Hurst-regime edge is market-specific or universal.
MARKETS = ["SPY", "QQQ", "GLD", "TLT", "EFA"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "1995-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the market basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/prices.csv``. Never imported by the offline
    notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    raw = yf.download(MARKETS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def load_real(ticker: str = "SPY", path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cached prices -> a one-column frame for ``ticker`` (the headline tape).

    Returns a frame indexed by date with a single ``price`` column (adjusted close) for the
    requested market, dropping leading NaNs.
    """
    df = load_prices(path)
    s = df[ticker].dropna()
    return pd.DataFrame({"price": s})


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def _fgn(n: int, H: float, rng: np.random.Generator) -> np.ndarray:
    """Fractional Gaussian noise with Hurst exponent ``H`` (Davies–Harte / circulant method).

    Returns ``n`` standardised increments whose long-range dependence corresponds to ``H``.
    H = 0.5 is white noise (random walk prices); H > 0.5 is persistent (trending increments);
    H < 0.5 is anti-persistent (mean-reverting increments).
    """
    # autocovariance of fGn at lag k
    k = np.arange(0, n)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
               + np.abs(k + 1) ** (2 * H))
    g[0] = 1.0
    # circulant embedding
    r = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(r).real
    lam[lam < 0] = 0.0                       # guard tiny negative eigenvalues
    m = len(r)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    x = np.fft.fft(np.sqrt(lam / m) * w).real
    return x[:n]


def synthetic_prices(n_days: int = 6_000, H: float = 0.5, edge: float = 0.0,
                     seed: int = 397, sig_daily: float = 0.010,
                     mu_daily: float = 0.0002) -> pd.DataFrame:
    """Deterministic single-H price path with a **known** Hurst exponent ``H``.

    The increments are fractional Gaussian noise with Hurst ``H`` (so the estimator must
    recover ``H``). Used to validate that :func:`hurst_rs` is faithful. The ``edge`` knob (if
    set) plants a small momentum/reversal lean keyed to the series' own H — but the headline
    positive control for the *gate* is :func:`synthetic_regimes` (a path whose regime
    alternates, where switching styles is the whole job).

    Returns a frame indexed by a decorative business-day range with a single ``price`` column.
    """
    rng = np.random.default_rng(seed)
    z = _fgn(n_days, H, rng)
    z = (z - z.mean()) / (z.std() + 1e-12)
    ret = mu_daily + sig_daily * z
    if edge != 0.0:
        sign = np.sign(ret)
        lean = sign if H > 0.5 else -sign
        boost = np.zeros(n_days)
        boost[1:] = edge * sig_daily * lean[:-1]
        ret = ret + boost
    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("1990-01-02", periods=n_days)
    return pd.DataFrame({"price": price}, index=idx)


def synthetic_regimes(n_days: int = 6_000, edge: float = 0.0, seed: int = 397,
                      block: int = 504, sig_daily: float = 0.010,
                      mu_daily: float = 0.0002) -> pd.DataFrame:
    """Deterministic path whose **regime alternates** — the positive control for the *gate*.

    The series is built in contiguous blocks of ``block`` trading days that alternate between a
    nominal **trending** block and a nominal **mean-reverting** block. The ``edge`` knob is the
    *amount of true Hurst separation* between the two block types: the trending block uses
    Hurst ``0.5 + 0.22*edge`` and the reverting block ``0.5 - 0.22*edge``.

    * ``edge = 0`` ⇒ **both block types are H = 0.5 (random walk)**. The blocks are
      statistically identical; the regime label is a pure decoration with no forecastable
      payoff. A Hurst gate has nothing real to key off and must NOT beat the best static style
      reliably — the honest null.
    * ``edge = 1`` ⇒ a strong, clean H separation (≈0.72 vs ≈0.28). Now the right style truly
      flips block to block and a gate that reads H correctly must light up.

    Returns ``frame`` with a ``price`` column **and** a boolean ``trend_regime`` column (the
    ground-truth label — handy for diagnostics; never used by the live strategy).
    """
    rng = np.random.default_rng(seed)
    ret = np.empty(n_days)
    is_trend = np.zeros(n_days, dtype=bool)
    H_trend = 0.5 + 0.22 * edge
    H_revert = 0.5 - 0.22 * edge
    pos = 0
    flip = True
    while pos < n_days:
        end = min(pos + block, n_days)
        m = end - pos
        H = H_trend if flip else H_revert
        z = _fgn(m, H, rng)
        z = (z - z.mean()) / (z.std() + 1e-12)
        ret[pos:end] = mu_daily + sig_daily * z
        is_trend[pos:end] = flip
        pos = end
        flip = not flip
    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("1990-01-02", periods=n_days)
    return pd.DataFrame({"price": price, "trend_regime": is_trend}, index=idx)


# --------------------------------------------------------------------------- #
# Hurst estimator (shared by real + synthetic; lives here so the control can self-check)
# --------------------------------------------------------------------------- #
def hurst_rs(x: np.ndarray, min_chunk: int = 8) -> float:
    """Classical rescaled-range (R/S) estimate of the Hurst exponent of a 1-D series ``x``.

    ``x`` is a series of *increments* (e.g. log-returns). Splits the window into chunks of
    several sizes, computes the mean rescaled range R/S per size, and regresses
    ``log(R/S)`` on ``log(size)``; the slope is the Hurst exponent. Returns NaN if the window
    is too short to form at least two valid chunk sizes.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_chunk:
        return float("nan")
    sizes = []
    s = min_chunk
    while s <= n // 2:
        sizes.append(s)
        s = int(s * 1.6) + 1
    if len(sizes) < 2:
        return float("nan")
    logn, logrs = [], []
    for size in sizes:
        n_chunks = n // size
        rs_vals = []
        for c in range(n_chunks):
            seg = x[c * size:(c + 1) * size]
            m = seg.mean()
            y = np.cumsum(seg - m)
            r = y.max() - y.min()
            sd = seg.std(ddof=0)
            if sd > 0 and r > 0:
                rs_vals.append(r / sd)
        if rs_vals:
            logn.append(np.log(size))
            logrs.append(np.log(np.mean(rs_vals)))
    if len(logn) < 2:
        return float("nan")
    slope = np.polyfit(logn, logrs, 1)[0]
    return float(slope)


def rolling_hurst(returns: pd.Series, window: int = 252, min_chunk: int = 8,
                  stride: int = 5) -> pd.Series:
    """Rolling R/S Hurst exponent of a return series over a trailing ``window``.

    Returns a Series aligned to ``returns`` with NaN until the first full window. Pure
    trailing — the value at date *t* uses only returns up to and including *t* (no look-ahead).

    The R/S estimate is recomputed every ``stride`` trading days and forward-filled in between
    (the Hurst exponent is slowly varying over a year-long window, so a 5-day stride is
    indistinguishable from daily recomputation but ~5× faster). ``stride = 1`` gives the exact
    daily estimate. The forward-fill only ever carries a *past* value forward, so it introduces
    no look-ahead.
    """
    vals = returns.values
    n = len(vals)
    out = np.full(n, np.nan)
    last = None
    for i in range(window, n + 1):
        if (i - window) % stride == 0 or i == n:
            last = hurst_rs(vals[i - window:i], min_chunk=min_chunk)
        out[i - 1] = last
    return pd.Series(out, index=returns.index)
