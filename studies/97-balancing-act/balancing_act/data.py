"""Data layer for Study 97 (Balancing-Act).

Two kinds of tape, one shape (a daily frame of total-return *price* columns, one per
asset):

- ``synthetic_two_asset`` — a *deterministic, offline* generator of two assets with a
  tunable correlation and (roughly) matched Sharpes. A ``corr`` knob plants the only
  thing a fixed-weight blend can possibly harvest: diversification. ``corr < 0`` makes
  the two assets negatively correlated, so a 60/40-style blend genuinely lifts the
  risk-adjusted return (lower vol for similar return → higher Sharpe). ``corr > 0`` makes
  them move together, so blending buys *no* diversification benefit. This is the study's
  positive control (negative corr) and its null (positive corr) in a bottle.
- ``load_real`` — the real daily **total-return** tapes via the shared cross-asset panel
  (``_cache/cross_asset_etfs.parquet``, built with ``yfinance auto_adjust=True``) with a
  ``quantlab.data`` fallback per ticker. SPY (stocks), IEF (7-10y Treasuries), TLT (20y+
  Treasuries) and SHY (1-3y Treasuries, the cash/excess-of-cash proxy). The Treasury ETFs
  begin **2002-07-30**, which bounds the joint window honestly.

No look-ahead is baked in here — the annual-rebalance bookkeeping lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
PANEL = os.path.join(DEFAULT_CACHE, "cross_asset_etfs.parquet")

# The Treasury ETFs bound the joint window: IEF and TLT both list 2002-07-30.
TREASURY_INCEPTION = "2002-07-30"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (correlation control)
# ---------------------------------------------------------------------------
def synthetic_two_asset(
    n_days: int = 6000,
    corr: float = -0.3,
    mu_a: float = 0.08,
    mu_b: float = 0.05,
    vol_a: float = 0.16,
    vol_b: float = 0.07,
    start: str = "2002-07-30",
    seed: int = 97,
) -> tuple[pd.DataFrame, dict]:
    """Two total-return price series ('STK', 'BND') with a planted correlation.

    The pair is built so the two legs have *similar* Sharpes but different volatilities,
    mirroring stocks-vs-bonds. The ``corr`` knob is the only lever that matters:

    - ``corr < 0`` → the legs hedge each other, so a fixed-weight blend cuts portfolio
      volatility more than it cuts return → the blend's Sharpe **beats** the better leg.
      This is the positive control: diversification is real, the harness should bank it.
    - ``corr > 0`` → the legs move together, so blending earns no diversification; the
      blend's Sharpe sits **between** the two legs, never above the best. This is the null.

    Returns ``(frame, truth)`` where ``frame`` has columns ``['STK', 'BND']`` (price
    levels starting at 100) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    corr = float(np.clip(corr, -0.99, 0.99))
    cov = np.array([[1.0, corr], [corr, 1.0]])
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_days, 2)) @ chol.T
    da = mu_a / 252.0 + (vol_a / np.sqrt(252.0)) * z[:, 0]
    db = mu_b / 252.0 + (vol_b / np.sqrt(252.0)) * z[:, 1]
    stk = 100.0 * np.exp(np.cumsum(da))
    bnd = 100.0 * np.exp(np.cumsum(db))
    frame = pd.DataFrame({"STK": stk, "BND": bnd}, index=idx)
    frame.index.name = "Date"
    realised = np.corrcoef(da, db)[0, 1]
    truth = {
        "corr": corr,
        "realised_corr": float(realised),
        "n_days": n_days,
        "seed": seed,
        "mu_a": mu_a, "mu_b": mu_b, "vol_a": vol_a, "vol_b": vol_b,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def load_real(
    tickers: tuple[str, ...] = ("SPY", "IEF", "TLT", "SHY"),
    start: str = TREASURY_INCEPTION,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for ``tickers`` (one column each).

    Primary source is the shared cross-asset panel (``_cache/cross_asset_etfs.parquet``,
    yfinance ``auto_adjust=True`` ⇒ dividends + splits folded in ⇒ total return), which is
    the fair series for a buy-and-hold comparison. On a cache miss for any ticker it falls
    back to :func:`quantlab.data.fetch` with ``mode="total_return"``. The frame is the
    inner join across tickers (dropna), so it begins at the latest inception — the Treasury
    ETFs (IEF/TLT, 2002-07-30) bound the joint window. Columns are upper-case tickers.
    """
    cols: dict[str, pd.Series] = {}
    panel = None
    if use_cache and os.path.exists(PANEL):
        panel = pd.read_parquet(PANEL)
        panel.index = pd.DatetimeIndex(panel.index).tz_localize(None)

    for tk in tickers:
        if panel is not None and tk in panel.columns:
            cols[tk] = panel[tk].rename(tk)
        else:
            import sys

            sys.path.insert(0, REPO_ROOT)
            from quantlab import data as qdata

            raw = qdata.fetch(tk, start=start, end=end, mode="total_return",
                              use_cache=use_cache)
            cols[tk] = raw["Close"].rename(tk)

    frame = pd.concat(cols.values(), axis=1, join="inner").dropna()
    frame = frame.loc[start:]
    if end is not None:
        frame = frame.loc[:end]
    frame.index.name = "Date"
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a price frame, for the as-of stamp.

    Hashes the column names and the raw float bytes of every column, so a different
    window, a different asset, or drifted data all change the fingerprint loudly.
    """
    h = hashlib.sha1()
    for c in frame.columns:
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
