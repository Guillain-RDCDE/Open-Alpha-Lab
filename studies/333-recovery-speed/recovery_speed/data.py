"""Data layer for Study 333 (Recovery-Speed).

The claim is *cross-sectional*: after a market-wide drawdown, rank names by how fast
they recover and bet on the fast recoverers. We need a **panel** of names, so both
tapes return the same shape — a (dates x tickers) close-price frame plus a market
series.

Two tapes, one shape:

- ``synthetic_panel`` — a *deterministic, offline* generator. Every name is a market
  factor plus an idiosyncratic random walk, and a tunable knob ``recovery_edge``
  controls the only thing a recovery-speed rule could possibly harvest: whether a
  name's *post-drawdown recovery speed* carries information about its *forward*
  return. ``recovery_edge=0`` is the **null** — recovery speed is pure noise, so any
  rule that "works" on it is data-mining. ``recovery_edge>0`` plants a genuine
  recovery-momentum effect the engine must be able to bank (the **positive control**).
  A market-wide drawdown is planted at a known location so the event study has teeth.

- ``load_real`` — a cache-first real panel via per-ticker parquet (the desk's shared
  ``_cache`` convention, same as ``quantlab.data`` / Study 301). Cache-only by default
  so the test-suite and reproducible core never touch the network. The hardcoded
  universe is a small set of long-lived large-caps; because it is a *current,
  surviving* universe projected backwards, any cross-sectional result on it is biased
  upward — survivorship is named on the Signal axis (see ``SURVIVORSHIP_NOTE``).

No look-ahead is baked in here — the one-bar execution lag lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# A small, long-lived large-cap universe for the real tape. It is a CURRENT, surviving
# set projected backwards — delisted losers are absent, so any cross-sectional result
# on it is an UPPER BOUND. The caveat travels with every published number.
REAL_UNIVERSE = [
    "AAPL", "MSFT", "JNJ", "PG", "KO", "JPM", "XOM", "WMT", "GE", "IBM",
    "INTC", "CSCO", "CAT", "MCD", "DIS", "HD", "BA", "MMM", "PFE", "MRK",
]
MARKET_PROXY = "SPY"

SURVIVORSHIP_NOTE = (
    "The real universe is a CURRENT set of survivors projected backwards; delisted "
    "names (the ones that drew down and never recovered) are absent. A recovery-speed "
    "cross-section on survivors is biased UPWARD — the names that recovered slowly and "
    "then died were removed before the test saw them. Treat any real-tape magnitude as "
    "an upper bound; the synthetic null/control is what isolates the mechanism."
)


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_names: int = 60,
    n_days: int = 1500,
    recovery_edge: float = 0.0,
    market_vol: float = 0.011,
    idio_vol: float = 0.014,
    beta_lo: float = 0.6,
    beta_hi: float = 1.4,
    drawdown_at: float = 0.45,
    drawdown_depth: float = 0.30,
    drawdown_len: int = 40,
    start: str = "2014-01-02",
    seed: int = 333,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A reproducible (dates x names) close-price panel with a planted market drawdown.

    Each name *i* has return ``r_it = beta_i * m_t + a_i + e_it`` where ``m_t`` is the
    market factor (with one engineered drawdown-and-recovery episode), ``beta_i`` its
    market loading, ``a_i`` a tiny idiosyncratic drift, and ``e_it`` idiosyncratic noise.

    The **recovery_edge** knob couples a name's recovery speed to its forward drift:
    a per-name latent ``q_i ~ N(0,1)`` is added BOTH to how fast the name climbs out of
    the drawdown (faster for high ``q_i``) AND to its post-recovery drift
    (``recovery_edge * q_i`` per day). So:

    - ``recovery_edge = 0`` → recovery speed is uncorrelated with forward returns: the
      **null**. A recovery-speed rule must look like a coin here.
    - ``recovery_edge > 0`` → fast recoverers genuinely keep leading: the **positive
      control** the engine must bank.

    Returns ``(panel, market, truth)``:
      * ``panel``  — DataFrame, index=dates, columns=tickers (synthetic ``N0001``...),
        values = close prices (start 100).
      * ``market`` — Series of the market close (a tradable proxy).
      * ``truth``  — the planted parameters, incl. ``q`` (per-name latent) and the
        drawdown window, so tests can check the mechanism directly.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Market factor. Keep ambient market noise modest so the *planted* episode is the
    # deepest drawdown — the detector then lands on it deterministically, and the V is
    # clean enough that recovery-speed scores reflect the planted per-name dynamics
    # rather than a random pre-episode dip.
    # Low ambient market noise keeps the running-max peak pinned just before the planted
    # episode, so the detector lands on OUR drawdown (a clean V), and the recovery bar it
    # finds sits inside the per-name recovery leg where recovery speed is informative.
    m = rng.normal(0.00010, market_vol * 0.35, n_days)
    d0 = int(drawdown_at * n_days)
    fall = drawdown_len // 2          # sharp decline
    rise = drawdown_len               # V-shaped recovery, fully retracing the fall
    m[d0:d0 + fall] += np.log(1.0 - drawdown_depth) / fall
    # Fully retrace the planted fall over the rise window, so a recover_frac~0.5 detector
    # fires partway up the recovery leg, right where per-name recovery speed is informative.
    m[d0 + fall:d0 + fall + rise] += -np.log(1.0 - drawdown_depth) / rise

    betas = rng.uniform(beta_lo, beta_hi, n_names)
    drifts = rng.normal(0.0, 0.00015, n_names)
    q = rng.standard_normal(n_names)  # per-name recovery-speed latent

    rec_start = d0 + fall             # trough
    rec_end = d0 + fall + rise        # market back at the peak
    post_start = rec_end

    closes = np.empty((n_days, n_names))
    for i in range(n_names):
        e = rng.normal(0.0, idio_vol, n_days)
        r = betas[i] * m + drifts[i] + e
        # In the FIRST HALF of the recovery leg, high-q names climb back FASTER and low-q
        # names lag — this is what makes recovery SPEED differ across names. The fast
        # movers then give some back in the second half, so the timing tilt is mean-zero
        # over the full leg (a SPEED difference, not a level difference). At
        # recovery_edge=0 it therefore leaves NO persistent forward edge — the null.
        half = rise // 2
        ramp = np.concatenate([
            np.full(half, 1.0),                          # fast climb (q>0) early
            np.full(rise - half, -half / (rise - half)), # symmetric give-back later
        ])
        r[rec_start:rec_end] += 0.0016 * q[i] * ramp
        # The plantable edge: high-q (fast-recovering) names keep leading AFTER the
        # recovery leg ends (only if recovery_edge>0; at 0 this vanishes and recovery
        # speed predicts nothing forward).
        r[post_start:] += recovery_edge * q[i]
        closes[:, i] = 100.0 * np.exp(np.cumsum(r))

    tickers = [f"N{j:04d}" for j in range(1, n_names + 1)]
    panel = pd.DataFrame(
        closes, index=pd.DatetimeIndex(sessions, name="date"), columns=tickers
    )
    market = pd.Series(
        100.0 * np.exp(np.cumsum(m)), index=panel.index, name=MARKET_PROXY
    )
    truth = {
        "recovery_edge": recovery_edge,
        "n_names": n_names,
        "n_days": n_days,
        "seed": seed,
        "q": q,
        "tickers": tickers,
        "drawdown_idx": d0,
        "recovery_window": (rec_start, rec_end),
        "post_idx": post_start,
        "drawdown_depth": drawdown_depth,
    }
    return panel, market, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first per-ticker parquet, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def _load_one(ticker: str, fetch: bool, cache_dir: str, start: str) -> pd.Series:
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached tape for {ticker} at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only on an explicit network opt-in

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    s = bars["close"].copy()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.name = ticker
    return s


def load_real(
    universe: list[str] | None = None,
    market: str = MARKET_PROXY,
    start: str = "2004-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_survivorship_bias: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Real (dates x names) total-return-adjusted close panel + market, cache-first.

    Network is touched only on explicit ``fetch=True``. Returns ``(panel, market,
    meta)`` aligned on the common date index. Auto-adjusted closes from yfinance are
    total-return proxies (splits + dividends), labelled as such.

    SURVIVORSHIP: this is a current-survivor universe projected backwards. Because the
    cross-section is the heart of the Signal claim, the loader **refuses to run** unless
    you pass ``allow_survivorship_bias=True`` — an explicit, documented opt-in. The
    caveat (``SURVIVORSHIP_NOTE``) must travel with any published number.
    """
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_real builds a cross-section on a CURRENT-survivor universe, which "
            "biases any recovery-speed result UPWARD. Pass allow_survivorship_bias=True "
            "to opt in, and carry data.SURVIVORSHIP_NOTE wherever you publish the number."
        )
    uni = list(universe) if universe is not None else list(REAL_UNIVERSE)
    cols = {}
    for t in uni:
        try:
            cols[t] = _load_one(t, fetch, cache_dir, start)
        except FileNotFoundError:
            if not fetch:
                raise
    if not cols:
        raise FileNotFoundError("No cached names available for the real panel.")
    panel = pd.DataFrame(cols).sort_index()
    mkt = _load_one(market, fetch, cache_dir, start)
    common = panel.dropna(how="all").index.intersection(mkt.index)
    panel = panel.loc[common].dropna(axis=1, how="any")
    mkt = mkt.loc[common]
    meta = {
        "universe": list(panel.columns),
        "market": market,
        "adjustment": "auto_adjust (total-return proxy)",
        "survivorship_note": SURVIVORSHIP_NOTE,
        "n_names": panel.shape[1],
        "n_days": panel.shape[0],
    }
    return panel, mkt, meta


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of a panel's price matrix, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
