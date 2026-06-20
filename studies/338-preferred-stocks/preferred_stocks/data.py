"""Data layer for Study 338 (Preferred-Stocks).

Two kinds of tape, one shape (a daily frame of total-return *price* columns, one per
asset — PFF, SPY, IEF):

- ``synthetic_three_asset`` — a *deterministic, offline* generator with one knob,
  ``pref_beta``, that controls how equity-like the "preferred" leg is:

    PFF_return_t = pref_beta * STK_t + (1 - pref_beta) * BND_t + coupon + idio

  * ``pref_beta`` near **0** → the preferred leg loads on bonds: it cushions in the
    stock crash (the *null* the marketing wants to be true — "bond-like").
  * ``pref_beta`` near **1** → the preferred leg loads on stocks: it crashes *with*
    equities (the *positive control* — the thing this study expects to find on the
    real tape).

  The crash is planted explicitly (a multi-day equity drawdown window) so the harness
  can measure who the preferred leg follows *into the crash*.

- ``load_real`` — the real daily **total-return** tapes, cache-first. SPY (stocks) and
  IEF (7-10y Treasuries) live in the shared cross-asset cache; PFF (iShares Preferred &
  Income Securities ETF) is fetched via :func:`quantlab.data.fetch` with a study-local
  cache. PFF lists **2007-03-30**, which bounds the joint window honestly — and it means
  PFF's first full test was the 2008 crisis.

No look-ahead is baked in here — the crash bookkeeping lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")
LOCAL_CACHE = os.path.join(STUDY_ROOT, "_cache")

# PFF (iShares Preferred & Income Securities ETF) lists 2007-03-30 — it bounds the
# joint window, and means preferreds' first stress test on this tape was the GFC.
PFF_INCEPTION = "2007-03-30"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (equity-vs-bond loading control)
# ---------------------------------------------------------------------------
def synthetic_three_asset(
    n_days: int = 4500,
    pref_beta: float = 0.85,
    mu_stk: float = 0.09,
    mu_bnd: float = 0.03,
    vol_stk: float = 0.18,
    vol_bnd: float = 0.06,
    coupon: float = 0.055,
    idio_vol: float = 0.05,
    crash_start: int = 1500,
    crash_len: int = 60,
    crash_stk: float = -0.45,
    crash_bnd: float = +0.10,
    start: str = "2007-03-30",
    seed: int = 338,
) -> tuple[pd.DataFrame, dict]:
    """Three total-return price series ('PFF', 'SPY', 'BND') with a planted crash.

    The preferred leg is a convex combination of the equity and bond legs plus a fat
    coupon and idiosyncratic noise::

        PFF_ret = pref_beta * STK_ret + (1 - pref_beta) * BND_ret + coupon/252 + idio

    A multi-day equity crash is planted at ``[crash_start, crash_start+crash_len)``:
    over that window the equity leg drifts to ``crash_stk`` and the bond leg to
    ``crash_bnd`` (flight to quality). What the preferred leg *does* in that window is
    governed entirely by ``pref_beta`` — that is the whole experiment:

    - ``pref_beta`` high (→1): PFF follows stocks down — the **positive control** (the
      "equity-in-disguise" finding this study expects on the real tape).
    - ``pref_beta`` low (→0): PFF follows bonds up — the **null** ("bond-like safety",
      the marketing claim being tested).

    Returns ``(frame, truth)`` where ``frame`` has columns ``['PFF', 'SPY', 'BND']``
    (price levels starting at 100) and ``truth`` records the planted parameters,
    including the crash window dates.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    pref_beta = float(np.clip(pref_beta, 0.0, 1.0))

    # Baseline daily returns for the two reference legs.
    stk = mu_stk / 252.0 + (vol_stk / np.sqrt(252.0)) * rng.standard_normal(n_days)
    bnd = mu_bnd / 252.0 + (vol_bnd / np.sqrt(252.0)) * rng.standard_normal(n_days)

    # Plant the crash: spread the target drawdown evenly across the window.
    # Clamp into range so short synthetic tapes (tests) never index out of bounds.
    lo = min(max(crash_start, 0), max(n_days - 1, 0))
    hi = min(lo + crash_len, n_days)
    if hi > lo:
        stk[lo:hi] = np.log1p(crash_stk) / (hi - lo)
        bnd[lo:hi] = np.log1p(crash_bnd) / (hi - lo)

    idio = (idio_vol / np.sqrt(252.0)) * rng.standard_normal(n_days)
    pff = pref_beta * stk + (1.0 - pref_beta) * bnd + coupon / 252.0 + idio

    def _to_px(r):
        return 100.0 * np.exp(np.cumsum(r))

    frame = pd.DataFrame(
        {"PFF": _to_px(pff), "SPY": _to_px(stk), "BND": _to_px(bnd)}, index=idx
    )
    frame.index.name = "Date"

    beta_stk = float(np.corrcoef(pff, stk)[0, 1])
    beta_bnd = float(np.corrcoef(pff, bnd)[0, 1])
    truth = {
        "pref_beta": pref_beta,
        "corr_pff_stk": beta_stk,
        "corr_pff_bnd": beta_bnd,
        "n_days": n_days,
        "seed": seed,
        "coupon": coupon,
        "crash_peak": idx[max(lo - 1, 0)],
        "crash_trough": idx[hi - 1] if hi > lo else idx[-1],
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def _load_one(ticker: str, start: str, end: str | None, use_cache: bool) -> pd.Series:
    """One total-return Close series, cache-first (shared panel → local parquet → fetch)."""
    # 1) shared per-ticker total-return parquet (SPY/IEF live here)
    shared = os.path.join(SHARED_CACHE, f"{ticker}_total_return.parquet")
    if use_cache and os.path.exists(shared):
        df = pd.read_parquet(shared)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df["Close"].rename(ticker)
    # 2) study-local parquet (where we stash PFF)
    local = os.path.join(LOCAL_CACHE, f"{ticker}_total_return.parquet")
    if use_cache and os.path.exists(local):
        df = pd.read_parquet(local)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df["Close"].rename(ticker)
    # 3) live fetch via quantlab (network) — cache it locally for next time
    import sys

    sys.path.insert(0, REPO_ROOT)
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start="1993-01-01", end=None,
                      mode="total_return", use_cache=use_cache)
    if use_cache:
        os.makedirs(LOCAL_CACHE, exist_ok=True)
        raw.to_parquet(local)
    return raw["Close"].rename(ticker)


def load_real(
    tickers: tuple[str, ...] = ("PFF", "SPY", "IEF"),
    start: str = PFF_INCEPTION,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for ``tickers`` (one column each).

    Total-return adjusted (``auto_adjust=True`` ⇒ dividends + splits folded in), the fair
    series for income instruments like preferreds whose return is mostly the coupon. The
    frame is the inner join across tickers (dropna), so it begins at the latest inception —
    **PFF (2007-03-30)** bounds the joint window. Columns are upper-case tickers.

    Cache order per ticker: shared ``_cache/<tk>_total_return.parquet`` (SPY, IEF) →
    study-local ``_cache/`` (PFF) → live :func:`quantlab.data.fetch` on a miss.
    """
    cols = {tk: _load_one(tk, start, end, use_cache) for tk in tickers}
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
