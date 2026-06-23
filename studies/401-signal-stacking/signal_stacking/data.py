"""Data layer for Study 401 (Signal-Stacking).

The viral claim is that you can take **K weak signals** — each barely better than a
coin flip, none significant on its own — *normalise them into one composite z-score*,
and the combination "produces something retail has never seen." This study builds the
machine that tests that claim, so the data layer supplies a panel with the *same shape*
(a matrix of standardised daily signals plus the next-day return they try to time) and
**two knobs that decide whether the magic can possibly happen**:

- ``signal_ic`` — the per-signal *information coefficient* (the correlation between a
  signal known at the close of day *t* and the return it earns over *t→t+1*). This is the
  honest measure of "how weak". ``signal_ic = 0`` is the **null**: every signal is pure
  noise, individually *and* in any combination — stacking coin-flips can only manufacture
  luck. ``signal_ic > 0`` (a realistic 0.03–0.06, i.e. a ~52% hit-rate) is the
  **positive control**: each signal carries a faint *real* edge the stacker should be able
  to compound.

- ``signal_corr`` — the pairwise correlation *between* signals. This is the knob the viral
  thread never mentions and the one that decides everything. Decorrelated weak edges add
  up like √K (the legitimate kernel of the claim); redundant ones (high ``signal_corr``)
  collapse to a ceiling — fifty copies of the same signal are worth barely more than one.

The generative model is deliberately analytic so every information coefficient and
cross-correlation is *known in advance* (and asserted in the tests):

    z_t            ~ N(0, 1)                                   # the latent return driver
    fwd_ret_t      = drift + ret_vol * z_t                     # the t→t+1 return to time
    c_t            ~ N(0, 1)                                   # a shared noise factor
    signal_k,t     = ic * z_t + sqrt(1 - ic^2) *
                     ( sqrt(rho) * c_t + sqrt(1 - rho) * u_k,t )      # u_k,t ~ N(0,1) iid

so that, by construction,

    corr(signal_k, fwd_ret)        = ic                        # each signal's true IC
    corr(signal_j, signal_k)       = ic^2 + (1 - ic^2) * rho   # the cross-correlation
    var(signal_k)                  = 1                          # already standardised

Each ``signal_k,t`` is known at the **close of day t** and earns ``fwd_ret_t`` (the
*next-day* return) — one execution lag, encoded once here, never shifted again in
``strategy.py``. There is no look-ahead baked into the panel.

``load_real`` builds the *same shape* from a real daily price series (default SPY
total-return): a basket of textbook weak timing signals (momentum, MA-gaps, oscillators,
a vol screen, a few pure decoys) standardised the same way, aligned to the next-day
return. The real tape is *not* a clean null — a faint short-horizon structure may be
buried in it — which is exactly why the honest arbiter (a permutation test on the
composite, plus an in/out-of-sample split) matters: it tells you whether stacking *added*
anything beyond the best single signal and beyond luck.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The engineered weak signals on the real tape — the columns the stacker combines.
# Every one is built from information available at the close of t (lagged where needed).
FEATURE_NAMES = [
    "mom_5",          # 5-day momentum
    "mom_20",         # 20-day momentum
    "mom_60",         # 60-day momentum
    "ma_gap_10",      # price vs 10-day moving average
    "ma_gap_50",      # price vs 50-day moving average
    "ma_gap_200",     # price vs 200-day moving average
    "zscore_20",      # rolling 20-day z-score of return
    "rsi_proxy_14",   # bounded oscillator proxy
    "vol_screen_20",  # (negated) 20-day realised vol — a low-vol tilt
    "noise_a",        # pure decoy
    "noise_b",        # pure decoy
    "noise_c",        # pure decoy
]


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic, offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_days: int = 2520,
    n_signals: int = 10,
    signal_ic: float = 0.0,
    signal_corr: float = 0.0,
    drift: float = 0.0003,
    ret_vol: float = 0.01,
    seed: int = 401,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible daily panel of standardised signals + the next-day return they time.

    Returns ``(signals, fwd_ret, truth)`` where ``signals`` is ``n_days x n_signals`` with
    every column a unit-variance score *known at the close of t*, and ``fwd_ret[t]`` is the
    *t→t+1* return a position formed at the close of t would earn (one lag, encoded here).

    ``signal_ic`` is each signal's true information coefficient (``corr(signal, fwd_ret)``);
    ``signal_corr`` is the latent cross-signal correlation. See the module docstring for the
    exact generative model and the closed-form moments the tests assert.
    """
    if not -1.0 < signal_ic < 1.0:
        raise ValueError("signal_ic must be in (-1, 1)")
    if not 0.0 <= signal_corr < 1.0:
        raise ValueError("signal_corr must be in [0, 1)")

    rng = np.random.default_rng(seed)

    z = rng.normal(0.0, 1.0, n_days)                       # latent return driver
    fwd_ret = drift + ret_vol * z                          # the t→t+1 return to time

    c = rng.normal(0.0, 1.0, n_days)                       # shared noise factor
    ic = float(signal_ic)
    rho = float(signal_corr)
    a = np.sqrt(max(0.0, 1.0 - ic * ic))
    common = np.sqrt(rho) * c
    cols = {}
    for k in range(n_signals):
        u = rng.normal(0.0, 1.0, n_days)
        e = common + np.sqrt(1.0 - rho) * u                # unit-var noise, pairwise corr=rho
        cols[f"sig_{k:02d}"] = ic * z + a * e
    signals = pd.DataFrame(cols)

    # Decorative daily date labels via a SAFE range (well under the ns-Timestamp span).
    idx = pd.date_range("2005-01-03", periods=n_days, freq="B")
    signals.index = pd.DatetimeIndex(idx, name="date")
    fwd = pd.Series(fwd_ret, index=signals.index, name="fwd_ret")

    truth = {
        "n_days": n_days,
        "n_signals": n_signals,
        "signal_ic": ic,
        "signal_corr": rho,
        "drift": drift,
        "ret_vol": ret_vol,
        "seed": seed,
        # Closed-form moments (what a faithful estimator must recover):
        "expected_signal_ic": ic,
        "expected_pair_corr": ic * ic + (1.0 - ic * ic) * rho,
    }
    return signals, fwd, truth


# ---------------------------------------------------------------------------
# Real tape — daily prices via the shared cache, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str, ticker: str = "SPY") -> str:
    return os.path.join(cache_dir, f"stacking_{ticker}.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1995-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Real daily tape (default SPY total-return) → the SAME ``(signals, fwd_ret)`` shape.

    Cache-first: reads the parquet under ``_cache/`` and never touches the network unless
    ``fetch=True``. On a cache miss with ``fetch=False`` it raises ``FileNotFoundError``
    (the offline contract), so the reproducible core and the tests stay deterministic.

    A basket of textbook *weak* timing signals (momentum, MA-gaps, an oscillator, a low-vol
    screen, three pure decoys) is built only from information at the close of t and
    standardised. ``fwd_ret[t]`` is the t→t+1 return — one lag, encoded here.

    The real tape is *not* a clean null. There is no survivorship issue (a single liquid
    ETF), so no opt-in guard is needed; the honesty axis here is whether *stacking* adds
    anything over the best single signal and over luck — judged by the permutation test and
    the in/out-of-sample split in ``strategy.py``.
    """
    cpath = _cache_path(cache_dir, ticker)
    if not fetch:
        if not os.path.exists(cpath):
            raise FileNotFoundError(
                f"No cached tape at {cpath}. "
                f"Call load_real(fetch=True) once to populate (needs network)."
            )
        px = pd.read_parquet(cpath)["close"]
    else:
        px = _fetch_prices(ticker, start)
        os.makedirs(cache_dir, exist_ok=True)
        px.to_frame("close").to_parquet(cpath)

    px = px.dropna()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    ret = px.pct_change().fillna(0.0)
    signals = _signals_from_price(px, ret)
    # Forward (next-day) return aligned to the close-of-t signal: fwd_ret[t] = ret[t+1].
    fwd = ret.shift(-1).rename("fwd_ret")
    valid = signals.dropna().index.intersection(fwd.dropna().index)
    signals = signals.loc[valid]
    fwd = fwd.loc[valid]
    return signals, fwd


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column on its own full-sample mean/std (time-series standardisation)."""
    mu = df.mean()
    sd = df.std(ddof=0).replace(0.0, 1.0)
    return (df - mu) / sd


def _signals_from_price(px: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """A basket of textbook weak timing signals, standardised — the real-tape stack."""
    raw = pd.DataFrame(index=px.index)
    raw["mom_5"] = ret.rolling(5).sum()
    raw["mom_20"] = ret.rolling(20).sum()
    raw["mom_60"] = ret.rolling(60).sum()
    raw["ma_gap_10"] = px / px.rolling(10).mean() - 1.0
    raw["ma_gap_50"] = px / px.rolling(50).mean() - 1.0
    raw["ma_gap_200"] = px / px.rolling(200).mean() - 1.0
    raw["zscore_20"] = (ret - ret.rolling(20).mean()) / (ret.rolling(20).std(ddof=0) + 1e-12)
    up = ret.clip(lower=0).rolling(14).mean()
    dn = (-ret.clip(upper=0)).rolling(14).mean()
    raw["rsi_proxy_14"] = (up - dn) / (up + dn + 1e-12)
    raw["vol_screen_20"] = -ret.rolling(20).std(ddof=0)        # low-vol tilt (negated vol)
    # Deterministic decoys seeded off a fixed seed (reproducible across reloads).
    rng = np.random.default_rng(401)
    raw["noise_a"] = rng.normal(0.0, 1.0, len(px))
    raw["noise_b"] = rng.normal(0.0, 1.0, len(px))
    raw["noise_c"] = rng.normal(0.0, 1.0, len(px))
    raw = raw[FEATURE_NAMES].dropna()
    return _standardise(raw)


def _fetch_prices(ticker: str, start: str) -> pd.Series:
    """Daily total-return closes via the shared loader, falling back to yfinance."""
    try:
        from quantlab import data as qld

        bars = qld.load(ticker, start=start, mode="total_return")
        s = bars["close"] if "close" in bars else bars.iloc[:, 0]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        return s.dropna()
    except Exception:
        pass

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"no price data returned for {ticker}")
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.dropna()


def fingerprint(signals: pd.DataFrame, fwd: pd.Series) -> str:
    """A short content fingerprint of the panel, for the as-of stamp."""
    a = np.ascontiguousarray(signals.to_numpy().ravel())
    b = np.ascontiguousarray(fwd.to_numpy().ravel())
    h = hashlib.sha1()
    h.update(a.tobytes())
    h.update(b.tobytes())
    return h.hexdigest()[:12]
