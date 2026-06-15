"""Data layer for Study 168 (Advance-Decline).

Two tapes, one shape:

- ``synthetic_daily`` — deterministic, offline. Generates a cross-section of N stocks and the
  implied Advance-Decline (A/D) line from their daily returns, with a tunable ``divergence``
  knob that plants a gap between the index price path and the cumulative breadth line at
  controllable moments. ``divergence = 0`` is the null: index and breadth co-move perfectly
  (up to idiosyncratic noise). This is the study's null in a bottle.

- ``fetch_spy`` / ``fetch_panel`` — real daily OHLCV for SPY and a panel of S&P 500
  constituents from Yahoo Finance (via ``quantlab.universe``), cache-first by default.

SURVIVORSHIP NOTE: the constituent universe is pulled from ``quantlab.universe.sp500_symbols``
(current S&P 500 membership). This means the breadth panel is conditioned on having survived
to today — the classic survivorship bias. As the METHODOLOGY requires, this is named on the
Signal axis wherever the result is published. The caller must opt in explicitly via
``allow_survivorship_bias=True`` before the real panel is returned.

No look-ahead: the A/D line is computed from daily closes; divergence signals are formed on
day t; positions entered at day t+1's open (see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
STUDY_CACHE = DEFAULT_CACHE  # per-study scratch

# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------

def synthetic_daily(
    n_stocks: int = 200,
    n_days: int = 756,          # ~3 years of trading days
    vol_ann: float = 0.20,      # annualised vol per stock
    corr: float = 0.35,         # common-factor correlation across stocks
    divergence: float = 0.0,    # ≠ 0 → plant breadth−index divergence episodes
    n_div_episodes: int = 4,    # how many divergence windows to plant
    div_len: int = 20,          # length of each divergence window (trading days)
    seed: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate a synthetic daily panel + implied A/D line and index.

    Model: daily stock returns are drawn from a one-factor model

        r_{i,t} = sqrt(corr) * f_t + sqrt(1-corr) * eps_{i,t}

    where f_t and eps_{i,t} are i.i.d. N(0, σ_daily). The 'index' is the
    equal-weighted average return. When ``divergence != 0``, we plant
    ``n_div_episodes`` windows of length ``div_len`` days where a large share of
    stocks *underperform* relative to the factor, so the cumulative A/D count
    (advancers minus decliners) trends down while the factor-driven index
    holds up — creating an artificial bearish divergence.

    Returns
    -------
    panel : pd.DataFrame
        Shape (n_days, n_stocks) of daily returns.
    spy : pd.DataFrame
        Columns: ``open``, ``high``, ``low``, ``close``, ``ad_line``, ``index_ret``.
        Indexed by business dates starting 2020-01-02.
    truth : dict
        Planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_days, name="date")
    sig_d = vol_ann / np.sqrt(252.0)

    # Factor + idiosyncratic
    factor = rng.normal(0.0, sig_d, n_days)
    idio   = rng.normal(0.0, sig_d * np.sqrt(1.0 - corr), (n_days, n_stocks))
    returns = np.sqrt(corr) * factor[:, None] + idio        # (n_days, n_stocks)

    # Plant divergence episodes: in these windows, a fraction of stocks are given
    # a negative drift while the common factor stays positive.
    div_windows: list[tuple[int, int]] = []
    if divergence != 0.0:
        # Space episodes evenly, avoiding the last 30 days (need room for forward returns)
        spacing = max(1, (n_days - 60) // (n_div_episodes + 1))
        for ep in range(n_div_episodes):
            start_day = spacing * (ep + 1)
            end_day   = min(start_day + div_len, n_days - 30)
            div_windows.append((start_day, end_day))
            # During this window the common factor (index) gets a small positive push
            # but many stocks diverge downward
            n_drag = int(n_stocks * 0.60)   # 60% of stocks drag breadth down
            drag_idx = rng.choice(n_stocks, size=n_drag, replace=False)
            for t in range(start_day, end_day):
                returns[t, drag_idx] -= divergence * sig_d

    # Build index (equal-weight) and A/D line
    index_ret  = returns.mean(axis=1)                          # daily EW return
    advance    = (returns > 0).sum(axis=1).astype(float)
    decline    = (returns < 0).sum(axis=1).astype(float)
    ad_net     = advance - decline                             # daily net advancers
    ad_line    = ad_net.cumsum()                               # cumulative A/D line

    # Build a OHLCV-like SPY proxy from the index returns
    index_level = 300.0 * np.exp(np.cumsum(index_ret))
    open_  = np.empty_like(index_level)
    open_[0] = 300.0
    open_[1:] = index_level[:-1]
    wick = np.abs(rng.normal(0.0, sig_d * 0.5, n_days)) * index_level
    spy = pd.DataFrame({
        "open":      open_,
        "high":      np.maximum(open_, index_level) + wick,
        "low":       np.minimum(open_, index_level) - wick,
        "close":     index_level,
        "ad_line":   ad_line,
        "index_ret": index_ret,
    }, index=dates)

    panel = pd.DataFrame(returns, index=dates,
                         columns=[f"S{i:03d}" for i in range(n_stocks)])

    truth = {
        "n_stocks":        n_stocks,
        "n_days":          n_days,
        "vol_ann":         vol_ann,
        "corr":            corr,
        "divergence":      divergence,
        "n_div_episodes":  n_div_episodes,
        "div_len":         div_len,
        "div_windows":     div_windows,
        "seed":            seed,
    }
    return panel, spy, truth


# ---------------------------------------------------------------------------
# Real data — SPY daily OHLCV, cache-first
# ---------------------------------------------------------------------------

def _spy_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "ad_spy_daily.parquet")


def fetch_spy(
    start: str = "2005-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily SPY OHLCV, cache-first; columns: open, high, low, close, volume.

    Network is touched only on ``fetch=True``; the cached parquet is returned
    otherwise. Call once with ``fetch=True`` to populate the local cache.
    """
    path = _spy_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached SPY daily tape at {path}. "
                "Call fetch_spy(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network
        raw = yf.download("SPY", start=start, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no data for SPY")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Real data — S&P 500 constituent panel for A/D line, cache-first + GUARDED
# ---------------------------------------------------------------------------

_PANEL_CACHE_NAME = "ad_panel_daily.parquet"


def _panel_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _PANEL_CACHE_NAME)


def fetch_panel(
    start: str = "2005-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    allow_survivorship_bias: bool = False,
) -> pd.DataFrame:
    """Daily close-returns for S&P 500 constituents, cache-first — GUARDED.

    The panel is built from *current* S&P 500 membership (``quantlab.universe.sp500_symbols``),
    projected backwards. This is the standard survivorship-biased universe: delisted losers
    are absent. The bias direction here is somewhat ambiguous (we're measuring breadth, not
    individual stock returns), but it still means fallen names — which by definition declined —
    are excluded from our A/D counts, likely overstating breadth. The caller must pass
    ``allow_survivorship_bias=True`` to receive the data, and must document the bias.

    Returns a (date × ticker) DataFrame of daily log-returns.
    """
    import sys
    sys.path.insert(0, REPO_ROOT)
    from quantlab.hf_data import SurvivorshipBiasError

    if not allow_survivorship_bias:
        raise SurvivorshipBiasError(
            "Refusing to build/load the A/D constituent panel: it uses current S&P 500 "
            "membership projected backwards — delisted names (losers) are absent, which "
            "likely overstates breadth. Pass allow_survivorship_bias=True to override "
            "and document the bias alongside your results."
        )
    warnings.warn(
        "Survivorship bias acknowledged: A/D panel is current S&P 500 membership "
        "projected backwards — fallen names absent, breadth likely overstated.",
        stacklevel=2,
    )

    path = _panel_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached A/D constituent panel at {path}. "
                "Call fetch_panel(fetch=True, allow_survivorship_bias=True) once."
            )
        return pd.read_parquet(path)

    from quantlab.universe import sp500_symbols
    import yfinance as yf

    syms = sp500_symbols(allow_survivorship_bias=True)
    # Download in one batched call — cheaper than per-symbol downloads.
    raw = yf.download(
        syms, start=start, auto_adjust=True, progress=False,
        group_by="ticker", threads=True,
    )
    # Extract Close prices — handle both yfinance MultiIndex layouts
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()
        if "Close" in lvl0:
            # (field, ticker) layout
            px = raw["Close"]
        elif "Close" in lvl1:
            # (ticker, field) layout
            px = raw.xs("Close", axis=1, level=1, drop_level=True)
        else:
            # Try lowercase
            close_cols = [c for c in lvl0 if str(c).lower() == "close"]
            if close_cols:
                px = raw[close_cols[0]]
            else:
                raise KeyError(f"Cannot find Close in MultiIndex columns: {lvl0[:5]}")
    else:
        if "Close" in raw.columns:
            px = raw[["Close"]].rename(columns={"Close": raw.columns[0]})
        else:
            px = raw

    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.index.name = "date"
    # Daily log returns
    lr = np.log(px / px.shift(1)).dropna(how="all")
    # Keep only columns with ≥750 observations (~3 years) from our start date
    lr = lr.loc[start:]
    lr = lr.loc[:, lr.notna().sum() >= 750]

    os.makedirs(cache_dir, exist_ok=True)
    lr.to_parquet(path)
    return lr


# ---------------------------------------------------------------------------
# A/D line builder — shared by both synthetic and real paths
# ---------------------------------------------------------------------------

def build_ad_line(returns: pd.DataFrame) -> pd.Series:
    """Cumulative Advance-Decline line from a daily returns panel.

    Counts the number of constituents with positive returns on each day
    (advancers) minus those with negative returns (decliners), then cumulates.
    Returns a Series indexed by date, named ``'ad_line'``.
    """
    adv = (returns > 0).sum(axis=1).astype(float)
    dec = (returns < 0).sum(axis=1).astype(float)
    ad_net = adv - dec
    return ad_net.cumsum().rename("ad_line")


def fingerprint(df: pd.DataFrame, col: str = "close") -> str:
    """Short content fingerprint of a price series, for the as-of stamp."""
    arr = df[col].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
