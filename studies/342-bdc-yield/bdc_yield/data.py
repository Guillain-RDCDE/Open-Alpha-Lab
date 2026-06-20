"""Data layer for Study 342 (BDC-Yield).

Two kinds of tape, one shape (a daily frame of total-return *price* columns, one per
asset — BIZD, SPY, IEF):

- ``synthetic_three_asset`` — a *deterministic, offline* generator with one knob,
  ``bdc_beta``, that controls how equity-like the BDC leg is:

    BIZD_return_t = bdc_beta * STK_t + (1 - bdc_beta) * BND_t + distribution + idio

  * ``bdc_beta`` near **0** → the BDC leg loads on bonds: it cushions in the credit
    crash (the *null* the ~10%-yield marketing wants to be true — "high income, low risk").
  * ``bdc_beta`` near **1** → the BDC leg loads on stocks: it crashes *with* equities,
    leveraged (the *positive control* — levered private-credit equity, what this study
    expects to find on the real tape).

  A multi-day equity/credit crash is planted explicitly so the harness can measure who
  the BDC leg follows *into the crash*, and a fat distribution that is partly funded by
  the leg's own NAV (the "return of capital" the high headline yield often hides).

- ``load_real`` — the real daily **total-return** tapes, cache-first. SPY (stocks) and
  IEF (7-10y Treasuries) live in the shared cross-asset cache; BIZD (VanEck BDC Income
  ETF) is fetched via :func:`quantlab.data.fetch` with a study-local cache. BIZD lists
  **2013-02-12**, which bounds the joint window honestly — so its first real stress test
  on this tape was the 2020 COVID crash and the 2022 credit-spread widening.

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

# BIZD (VanEck BDC Income ETF) lists 2013-02-12 — it bounds the joint window, and means
# BDCs' first real stress test on this tape was the 2020 COVID crash, not the GFC.
BIZD_INCEPTION = "2013-02-12"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (equity-vs-bond loading control)
# ---------------------------------------------------------------------------
def synthetic_three_asset(
    n_days: int = 3300,
    bdc_beta: float = 1.15,
    mu_stk: float = 0.09,
    mu_bnd: float = 0.03,
    vol_stk: float = 0.18,
    vol_bnd: float = 0.06,
    distribution: float = 0.10,
    return_of_capital: float = 0.04,
    idio_vol: float = 0.06,
    crash_start: int = 1200,
    crash_len: int = 45,
    crash_stk: float = -0.40,
    crash_bnd: float = +0.08,
    start: str = "2013-02-12",
    seed: int = 342,
) -> tuple[pd.DataFrame, dict]:
    """Three total-return price series ('BIZD', 'SPY', 'BND') with a planted credit crash.

    The BDC leg is a (possibly >1, i.e. *levered*) loading on the equity and bond legs
    plus a fat distribution net of the part that is really a return of the leg's own
    capital, plus idiosyncratic noise::

        BIZD_ret = bdc_beta * STK_ret + (1 - bdc_beta) * BND_ret
                   + (distribution - return_of_capital)/252 + idio

    ``bdc_beta`` may exceed 1 — BDCs run balance-sheet leverage, so they can fall *more*
    than the market they track. A multi-day equity/credit crash is planted at
    ``[crash_start, crash_start+crash_len)``: over that window the equity leg drifts to
    ``crash_stk`` and the bond leg to ``crash_bnd`` (flight to quality). What the BDC leg
    *does* in that window is governed entirely by ``bdc_beta`` — that is the experiment:

    - ``bdc_beta`` high (≥1): BIZD follows stocks down, amplified — the **positive
      control** (the "levered private-credit equity" finding this study expects).
    - ``bdc_beta`` low (→0): BIZD follows bonds up — the **null** ("high income, low
      risk", the marketing claim being tested).

    The ``return_of_capital`` knob models the open secret of the high headline yield: a
    chunk of the ~10% "distribution" is funded by paying out the leg's own NAV, so the
    *total* return is far below the quoted yield. Returns ``(frame, truth)`` where
    ``frame`` has columns ``['BIZD', 'SPY', 'BND']`` (price levels starting at 100) and
    ``truth`` records the planted parameters, including the crash window dates.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    bdc_beta = float(max(bdc_beta, 0.0))

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
    net_carry = (distribution - return_of_capital) / 252.0
    bizd = bdc_beta * stk + (1.0 - bdc_beta) * bnd + net_carry + idio

    def _to_px(r):
        return 100.0 * np.exp(np.cumsum(r))

    frame = pd.DataFrame(
        {"BIZD": _to_px(bizd), "SPY": _to_px(stk), "BND": _to_px(bnd)}, index=idx
    )
    frame.index.name = "Date"

    corr_stk = float(np.corrcoef(bizd, stk)[0, 1])
    corr_bnd = float(np.corrcoef(bizd, bnd)[0, 1])
    truth = {
        "bdc_beta": bdc_beta,
        "corr_bizd_stk": corr_stk,
        "corr_bizd_bnd": corr_bnd,
        "n_days": n_days,
        "seed": seed,
        "distribution": distribution,
        "return_of_capital": return_of_capital,
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
    # 2) study-local parquet (where we stash BIZD)
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
    tickers: tuple[str, ...] = ("BIZD", "SPY", "IEF"),
    start: str = BIZD_INCEPTION,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for ``tickers`` (one column each).

    Total-return adjusted (``auto_adjust=True`` ⇒ distributions + splits folded in) — the
    only fair series for a ~10%-yield instrument like BIZD, whose price *drifts down* as it
    pays out, so a price-only series would massively understate the gross return and a
    distribution-yield headline would massively overstate the *net* one. The frame is the
    inner join across tickers (dropna), so it begins at the latest inception — **BIZD
    (2013-02-12)** bounds the joint window. Columns are upper-case tickers.

    Cache order per ticker: shared ``_cache/<tk>_total_return.parquet`` (SPY, IEF) →
    study-local ``_cache/`` (BIZD) → live :func:`quantlab.data.fetch` on a miss.
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
