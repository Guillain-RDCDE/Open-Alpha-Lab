"""Data layer for Study 263 (Insider-Buying).

The claim: aggregate corporate-insider buying (SEC Form 4 open-market purchases),
typically summarised as a *buy/sell ratio* (number or dollar value of insider buys
divided by sells), is a contrarian sentiment gauge -- it spikes near market bottoms
when insiders "back up the truck" and collapses near tops. Lakonishok & Lee (2001)
and Seyhun (1998) document modest aggregate predictive power at the 6-12 month
horizon. This study tests the aggregate market-timing version on the S&P 500.

Three components, all offline for the core:

- ``INSIDER_BS_RATIO`` -- a hardcoded, curated MONTHLY aggregate insider buy/sell
  ratio (2003-01 .. 2025-12). **This is a curated proxy, NOT a live Form-4 feed.**
  The level and the well-known regime spikes are anchored to documented episodes
  (insiders bought heavily at the Mar-2009 bottom, Mar-2020 COVID crash, and the
  Oct-2022 lows; they sold into the 2007 and 2021 tops). Between anchors the series
  is filled deterministically. Because the magnitudes are reconstructed rather than
  pulled from a clean public source, **the signal axis is capped at WEAK** -- a
  curated proxy cannot earn a REAL stamp no matter how good the t-stat looks. The
  honesty banner is named on the Signal axis and in docs/results.md.

- ``synthetic_panel`` -- a deterministic, offline generator that plants a known
  amount of contrarian predictive power (``beta`` knob). ``beta = 0`` is the null;
  tests confirm the machinery is truthful before looking at the curated tape.

- ``fetch_sp500`` -- real S&P 500 (^GSPC) monthly closes via yfinance, cache-only by
  default (fetch=False); network only on explicit fetch=True (lazy import). Cache at
  ``_cache/gspc_monthly.parquet``. Price-only (not total return) -- labeled as such.

No look-ahead: the buy/sell ratio observed at month-end t (insiders' Form-4 filings
are public with a 2-business-day lag, so end-of-month t data is genuinely available
by the start of month t+1) predicts the return EARNED in month t+1. Positions are
formed at the close of month t and held through month t+1 (one-month execution lag).

Survivorship: the S&P 500 is an index of survivors by construction; the timing
overlay does not pick stocks, so survivorship is mild here, but it is named.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
GSPC_CACHE = os.path.join(DEFAULT_CACHE, "gspc_monthly.parquet")

SEED = 263

# ---------------------------------------------------------------------------
# Curated monthly aggregate insider buy/sell ratio -- the offline core
# ---------------------------------------------------------------------------
# A buy/sell ratio of 1.0 means insiders bought as many shares/dollars as they
# sold; values near 0.10-0.20 are normal (insiders sell far more than they buy,
# because options/grants make selling routine). Spikes toward 0.5-1.0 mark the
# rare "everybody's buying" episodes that the contrarian story cares about.
#
# Anchors (documented episodes; magnitudes are STYLISED, not exact):
#   2003-03  ~0.45  post-2002 bottom, insiders buying
#   2007-07  ~0.10  pre-crisis top, heavy selling
#   2008-11  ~0.70  Lehman aftermath buying spike
#   2009-03  ~0.85  the generational buying spike at the bottom
#   2011-08  ~0.55  US-downgrade selloff buying
#   2015-08  ~0.40  China-devaluation dip buying
#   2018-12  ~0.50  Q4-2018 selloff buying
#   2020-03  ~0.90  COVID crash -- the largest buying spike on record
#   2021-11  ~0.08  meme/SPAC top, insiders dumping
#   2022-10  ~0.55  bear-market low buying
#   2024-07  ~0.15  AI-driven highs, light buying
#
# Sources for the *shape* (not the exact numbers):
#   - Lakonishok, J. & Lee, I. (2001). "Are Insider Trades Informative?" RFS 14(1).
#   - Seyhun, H. N. (1998). "Investment Intelligence from Insider Trading." MIT Press.
#   - Vickers Weekly Insider; Washington Service / InsiderInsights commentary archives.
#
# As-of 2026-06-17. This is a CURATED PROXY -- see module docstring.

_ANCHORS: dict[str, float] = {
    "2003-01": 0.42, "2003-03": 0.45, "2003-12": 0.30,
    "2004-06": 0.22, "2005-06": 0.18, "2006-06": 0.15,
    "2007-07": 0.10, "2007-12": 0.14,
    "2008-09": 0.40, "2008-11": 0.70,
    "2009-03": 0.85, "2009-12": 0.30,
    "2010-06": 0.28, "2011-08": 0.55, "2011-12": 0.30,
    "2012-06": 0.25, "2013-06": 0.18, "2014-06": 0.16,
    "2015-08": 0.40, "2015-12": 0.28,
    "2016-02": 0.45, "2016-12": 0.22,
    "2017-06": 0.16, "2018-02": 0.30, "2018-12": 0.50,
    "2019-06": 0.22, "2019-12": 0.18,
    "2020-03": 0.90, "2020-09": 0.30,
    "2021-06": 0.14, "2021-11": 0.08,
    "2022-06": 0.40, "2022-10": 0.55, "2022-12": 0.35,
    "2023-06": 0.20, "2023-10": 0.32,
    "2024-07": 0.15, "2024-12": 0.18,
    "2025-04": 0.38, "2025-12": 0.20,
}


def _build_curated_ratio() -> pd.Series:
    """Construct the monthly curated buy/sell ratio by deterministic interpolation.

    Linear interpolation between documented anchor points, plus a small
    deterministic (seeded) jitter so the series is not piecewise-linear-flat.
    Clipped to [0.05, 1.0]. Fully reproducible.
    """
    idx = pd.date_range("2003-01-31", "2025-12-31", freq="ME")
    anchor_ts = pd.to_datetime([k + "-01" for k in _ANCHORS]) + pd.offsets.MonthEnd(0)
    anchor_vals = np.array(list(_ANCHORS.values()), dtype=float)
    anchor_series = pd.Series(anchor_vals, index=anchor_ts).sort_index()

    # Reindex anchors onto the full monthly grid and interpolate by time.
    base = anchor_series.reindex(anchor_series.index.union(idx)).interpolate(
        method="time"
    ).reindex(idx)
    base = base.ffill().bfill()

    # Deterministic jitter (seeded), small relative to level.
    rng = np.random.default_rng(SEED)
    jitter = rng.normal(0.0, 0.025, size=len(idx))
    ratio = (base.to_numpy() + jitter).clip(0.05, 1.0)

    out = pd.Series(ratio, index=idx, name="bs_ratio")
    out.index.name = "date"
    return out


INSIDER_BS_RATIO: pd.Series = _build_curated_ratio()


def insider_ratio() -> pd.Series:
    """Return a copy of the curated monthly aggregate insider buy/sell ratio.

    Index = month-end dates (2003-01 .. 2025-12); value = buy/sell ratio in [0.05, 1.0].
    **Curated proxy, not a live Form-4 feed** -- see module docstring. The signal axis
    is capped at WEAK because the magnitudes are reconstructed.
    """
    return INSIDER_BS_RATIO.copy()


# ---------------------------------------------------------------------------
# Synthetic positive control -- deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 276,
    beta: float = 0.04,
    base_ret: float = 0.006,
    ret_vol: float = 0.043,
    ratio_mean: float = 0.28,
    ratio_vol: float = 0.18,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """Monthly (ratio, forward-return) panel with a planted contrarian effect.

    The buy/sell ratio is an AR(1)-ish demeaned signal; the next month's return is::

        base_ret + beta * (ratio_t - ratio_mean) + noise

    so a HIGH ratio (insiders buying) predicts a HIGHER next-month return when
    ``beta > 0``. ``beta = 0`` is the null (ratio carries no information).

    Returns ``(df, truth)`` where ``df`` has columns ``bs_ratio`` (the signal at
    month t) and ``fwd_ret`` (the return earned in month t+1), and ``truth`` records
    the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2003-01-31", periods=n_months, freq="ME")

    # AR(1) ratio centred on ratio_mean, clipped to a plausible band.
    ratio = np.empty(n_months)
    ratio[0] = ratio_mean
    rho = 0.85
    for t in range(1, n_months):
        ratio[t] = ratio_mean + rho * (ratio[t - 1] - ratio_mean) + rng.normal(0.0, ratio_vol * (1 - rho ** 2) ** 0.5)
    ratio = np.clip(ratio, 0.05, 1.0)

    noise = rng.normal(0.0, ret_vol, n_months)
    fwd = base_ret + beta * (ratio - ratio_mean) + noise

    df = pd.DataFrame({"bs_ratio": ratio, "fwd_ret": fwd}, index=idx)
    df.index.name = "date"
    truth = {
        "n_months": n_months,
        "beta": beta,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "ratio_mean": ratio_mean,
        "ratio_vol": ratio_vol,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- S&P 500 (^GSPC) monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_sp500(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """S&P 500 (^GSPC) monthly close; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (lazy yfinance import),
    then the result is cached as parquet. Returns a DataFrame with a ``close``
    column indexed by month-end dates. **Price-only** (not total return); the
    ~1.8%/yr dividend yield is therefore excluded from the index level -- labeled
    on every benchmark race.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC monthly close at {cache_path}. "
                "Call fetch_sp500(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start="2002-12-01",
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    close = close.dropna()
    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"close": close.astype(float)})
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached ^GSPC monthly: {out.shape[0]} months -> {cache_path}")
    return out


def build_real_panel(gspc: pd.DataFrame) -> pd.DataFrame:
    """Join the curated buy/sell ratio with realised S&P 500 forward returns.

    Returns a DataFrame indexed by month-end with columns:
      ``bs_ratio`` (signal at month t),
      ``ret``      (return earned IN month t, price-only),
      ``fwd_ret``  (return earned IN month t+1 -- what the signal at t predicts).

    No look-ahead: ``fwd_ret`` at row t is the t->t+1 return; the signal at row t is
    the ratio observed at the end of month t.
    """
    ratio = insider_ratio()
    close = gspc["close"].copy()
    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    ret = close.pct_change()

    df = pd.DataFrame({"bs_ratio": ratio})
    df["ret"] = ret.reindex(df.index)
    df["fwd_ret"] = df["ret"].shift(-1)
    df = df.dropna(subset=["bs_ratio", "ret"])
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """A short content fingerprint of a Series/DataFrame, for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.select_dtypes("number").to_numpy(dtype=float).ravel()
    else:
        arr = pd.Series(obj).to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
