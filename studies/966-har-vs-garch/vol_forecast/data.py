"""Data layer for Study 966 — Forecasting Tomorrow's Vol.

Close-only: every model here reads the same single input, the daily total-return
close, so no model can win by having been fed a better tape. (The competing question — whether
a *richer bar* beats a better model — is study 965's, and the two are deliberately kept
apart.)

Two tapes, one shape:

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``), cached as parquet in the **shared** desk cache
  ``studies/_cache``. ``fetch`` is the only thing here that touches the network;
  ``load_prices`` reads the cache offline and never imports yfinance, so the test-suite
  and CI stay green on a fresh checkout where ``_cache/`` is git-ignored and absent.
- ``synthetic_panel`` — a deterministic, offline generator with a ``signal_strength``
  knob: at ``1.0`` the effect this study hunts is **planted** at a known size, at ``0.0``
  the panel is the matching null. Every test runs on it; nothing in the suite needs a
  network or a cache.

The sample is pinned at ``AS_OF`` so a rerun months later cannot quietly grow the window.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache — most of these tapes are already sitting there.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Six tapes, deliberately including two that are not equities: a volatility model that
# only works on the S&P is a folklore, not a model.
TICKERS = ("SPY", "QQQ", "IWM", "EEM", "TLT", "GLD")

# Study-wide as-of: the sample is sliced here so reruns cannot creep.
AS_OF = "2026-06-30"
START = "1993-01-01"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return closes, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(tickers=TICKERS, start: str = START, end: str | None = None,
          cache_dir: str = DEFAULT_CACHE, retries: int = 4) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- **and** dividend-adjusted: a price-only tape would bias
    every comparison here toward whichever leg happens to yield less.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                asof: str = AS_OF) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close panel.

    Returns a date-indexed frame with one column per ticker, sliced to ``asof``. A
    ticker that lists late is simply NaN before its inception — never back-filled.
    Raises ``FileNotFoundError`` if any ticker is missing: the offline core and the
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call vol_forecast.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]




# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_vol_path(
    n_years: int = 20,
    vol_ann: float = 0.17,
    persistence: float = 0.97,        # AR(1) coefficient of log-variance
    vol_of_vol: float = 0.35,
    signal_strength: float = 1.0,     # 0 = constant vol (the null), 1 = full clustering
    df_t: float = 6.0,                # Student-t degrees of freedom for the shock
    start: str = "2004-01-02",
    seed: int = 966,
) -> tuple[pd.Series, pd.Series, dict]:
    """Daily returns with a *known* conditional volatility path — the only fair scoreboard.

    Log-variance follows an AR(1) with coefficient ``persistence`` scaled by
    ``signal_strength`` (at 0 the path is flat: constant volatility, the null under which no
    model may beat a rolling average). Shocks are Student-t so the tape is fat-tailed the way
    a real one is, which is what makes a squared-return proxy noisy and QLIKE necessary.

    Returns ``(returns, sigma, truth)``: the daily returns, the conditional sigma that
    generated them, and the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n)
    sd_d = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)

    phi = float(persistence) * float(signal_strength)
    shock = vol_of_vol * float(signal_strength) * np.sqrt(max(1.0 - phi ** 2, 1e-9))
    lv = np.zeros(n)
    e = rng.normal(0.0, 1.0, n)
    for t in range(1, n):
        lv[t] = phi * lv[t - 1] + shock * e[t]
    sigma = sd_d * np.exp(lv - 0.5 * np.var(lv))

    z = rng.standard_t(df_t, n) / np.sqrt(df_t / (df_t - 2.0))
    r = pd.Series(sigma * z, index=pd.DatetimeIndex(dates, name="date"), name="ret")
    truth = {"n_days": n, "n_years": n_years, "seed": seed, "vol_ann": vol_ann,
             "persistence": phi, "signal_strength": float(signal_strength), "df_t": df_t}
    return r, pd.Series(sigma, index=r.index, name="sigma"), truth


def synthetic_panel(n_assets: int = 3, n_years: int = 20, signal_strength: float = 1.0,
                    seed: int = 966, cash_rate_ann: float = 0.02):
    """Panel wrapper (prices, cash, truth) so the shared data-layer tests apply unchanged."""
    cols, sig = {}, {}
    for i in range(n_assets):
        r, s, truth = synthetic_vol_path(n_years=n_years, signal_strength=signal_strength,
                                         seed=seed + i)
        cols[f"A{i}"] = 100.0 * (1.0 + r).cumprod()
        sig[f"A{i}"] = s
    prices = pd.DataFrame(cols)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(np.cumprod(np.full(len(prices), cash_daily)), index=prices.index,
                     name="cash")
    truth = {"n_assets": n_assets, "n_years": n_years, "n_days": len(prices), "seed": seed,
             "signal_strength": float(signal_strength),
             "alpha_vol_eff": 0.0 if signal_strength == 0 else 0.05 * signal_strength,
             "cash_rate_ann": cash_rate_ann, "sigma": pd.DataFrame(sig)}
    return prices, cash, truth


def synthetic_daily(n_years: int = 20, signal_strength: float = 1.0, seed: int = 966):
    """Single-name convenience wrapper, matching the desk's shared shape."""
    prices, cash, truth = synthetic_panel(n_assets=1, n_years=n_years,
                                          signal_strength=signal_strength, seed=seed)
    return pd.DataFrame({"asset": prices.iloc[:, 0], "cash": cash}), truth

