"""Data layer for Study 340 (Bank-Loans).

Two kinds of tape, one shape (a daily frame of total-return *price* columns, one per
asset — BKLN, TLT, IEF, SPY):

- ``synthetic_four_asset`` — a *deterministic, offline* generator with two knobs that
  control how a "bank-loan" leg is built out of a **rate factor** and an **equity/credit
  factor**::

      LOAN_ret = -dur * RATE_ret + credit_beta * STK_ret + coupon/252 + idio

  where ``RATE_ret`` is the daily change in a synthetic rate level (driving the duration
  legs) and ``STK_ret`` is the equity factor.

  * ``dur`` near **0** → the loan leg has *no duration*: it barely moves when rates jump
    (the **positive control** — the "rate protection" the brochure sells, which IS real).
  * ``credit_beta`` high → the loan leg loads on equities and crashes *with* stocks (the
    catch the brochure hides — credit risk wearing a coupon).

  A rate-spike window (rates jump → long bonds fall) AND an equity-crash window (stocks
  fall → credit spreads blow out) are planted, so the harness can verify both: the loan
  leg survives the rate spike (low ``dur``) yet falls in the equity crash (high
  ``credit_beta``).

- ``load_real`` — the real daily **total-return** tapes, cache-first. TLT/IEF/SPY live in
  the shared cross-asset cache; BKLN (Invesco Senior Loan ETF) is fetched via
  :func:`quantlab.data.fetch` with a study-local cache. BKLN lists **2011-03-03**, which
  bounds the joint window honestly.

No look-ahead is baked in here — the regime/crash bookkeeping lives in ``strategy.py``.
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

# BKLN (Invesco Senior Loan ETF) lists 2011-03-03 — it bounds the joint window, and means
# the loan leg's first big tests on this tape were the 2018 rate-shock and the 2020 crash.
BKLN_INCEPTION = "2011-03-03"


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core (duration-vs-credit loading control)
# ---------------------------------------------------------------------------
def synthetic_four_asset(
    n_days: int = 3700,
    dur: float = 0.10,
    credit_beta: float = 0.55,
    mu_stk: float = 0.09,
    vol_stk: float = 0.17,
    vol_rate: float = 0.006,      # daily vol of the rate level (in rate "points")
    tlt_dur: float = 17.0,        # long-bond duration (years)
    ief_dur: float = 7.5,         # intermediate-bond duration (years)
    bnd_carry: float = 0.025,     # treasury carry/yield drift
    coupon: float = 0.05,         # loan coupon
    idio_vol: float = 0.04,
    rate_spike_start: int = 1200,
    rate_spike_len: int = 80,
    rate_spike_bp: float = 0.020,   # +200bp jump in the rate level over the window
    crash_start: int = 2200,
    crash_len: int = 40,
    crash_stk: float = -0.34,
    crash_rate_bp: float = -0.010,  # flight-to-quality: rates fall in the crash
    start: str = "2011-03-03",
    seed: int = 340,
) -> tuple[pd.DataFrame, dict]:
    """Four total-return price series ('BKLN', 'TLT', 'IEF', 'SPY') with two planted shocks.

    The loan leg is built from a rate factor and an equity/credit factor::

        LOAN_ret = -dur * d(rate) + credit_beta * STK_ret + coupon/252 + idio

    Treasury legs are duration × the same rate factor (plus carry); equity is the STK
    factor. Two windows are planted:

    - a **rate spike** at ``[rate_spike_start, +rate_spike_len)`` (rates jump
      ``rate_spike_bp``) — long bonds (TLT) fall hard; the loan leg should barely move when
      ``dur`` is near 0 (the positive control for "rate protection").
    - an **equity crash** at ``[crash_start, +crash_len)`` (stocks fall ``crash_stk``,
      rates fall ``crash_rate_bp`` in a flight to quality) — Treasuries rally; the loan leg
      should fall when ``credit_beta`` is high (the catch the brochure hides).

    Returns ``(frame, truth)`` where ``frame`` has columns ``['BKLN','TLT','IEF','SPY']``
    (price levels starting at 100) and ``truth`` records the planted parameters and the two
    window endpoints.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    dur = float(max(dur, 0.0))
    credit_beta = float(np.clip(credit_beta, 0.0, 2.0))

    # Two orthogonal factors: a rate-level change and an equity return.
    drate = (vol_rate / np.sqrt(252.0)) * rng.standard_normal(n_days)  # daily Δrate
    stk = mu_stk / 252.0 + (vol_stk / np.sqrt(252.0)) * rng.standard_normal(n_days)

    def _clamp(s, L):
        lo = min(max(s, 0), max(n_days - 1, 0))
        hi = min(lo + L, n_days)
        return lo, hi

    # Plant the rate spike (rates rise) — spread evenly across the window.
    rlo, rhi = _clamp(rate_spike_start, rate_spike_len)
    if rhi > rlo:
        drate[rlo:rhi] += rate_spike_bp / (rhi - rlo)

    # Plant the equity crash (stocks down, rates down -> flight to quality).
    clo, chi = _clamp(crash_start, crash_len)
    if chi > clo:
        stk[clo:chi] = np.log1p(crash_stk) / (chi - clo)
        drate[clo:chi] += crash_rate_bp / (chi - clo)

    # Bond legs: price return = -duration * Δrate + carry drift.
    tlt = -tlt_dur * drate + bnd_carry / 252.0
    ief = -ief_dur * drate + bnd_carry / 252.0

    idio = (idio_vol / np.sqrt(252.0)) * rng.standard_normal(n_days)
    loan = -dur * drate + credit_beta * stk + coupon / 252.0 + idio

    def _to_px(r):
        return 100.0 * np.exp(np.cumsum(r))

    frame = pd.DataFrame(
        {"BKLN": _to_px(loan), "TLT": _to_px(tlt), "IEF": _to_px(ief), "SPY": _to_px(stk)},
        index=idx,
    )
    frame.index.name = "Date"

    corr_loan_stk = float(np.corrcoef(loan, stk)[0, 1])
    corr_loan_rate = float(np.corrcoef(loan, drate)[0, 1])
    corr_tlt_rate = float(np.corrcoef(tlt, drate)[0, 1])
    truth = {
        "dur": dur,
        "credit_beta": credit_beta,
        "corr_loan_stk": corr_loan_stk,
        "corr_loan_rate": corr_loan_rate,
        "corr_tlt_rate": corr_tlt_rate,
        "n_days": n_days,
        "seed": seed,
        "coupon": coupon,
        "rate_spike_peak": idx[max(rlo - 1, 0)],
        "rate_spike_end": idx[rhi - 1] if rhi > rlo else idx[-1],
        "crash_peak": idx[max(clo - 1, 0)],
        "crash_trough": idx[chi - 1] if chi > clo else idx[-1],
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return prices, cache-first
# ---------------------------------------------------------------------------
def _load_one(ticker: str, start: str, end: str | None, use_cache: bool) -> pd.Series:
    """One total-return Close series, cache-first (shared panel -> local parquet -> fetch)."""
    # 1) shared per-ticker total-return parquet (TLT/IEF/SPY live here)
    shared = os.path.join(SHARED_CACHE, f"{ticker}_total_return.parquet")
    if use_cache and os.path.exists(shared):
        df = pd.read_parquet(shared)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df["Close"].rename(ticker)
    # 2) study-local parquet (where we stash BKLN)
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
    tickers: tuple[str, ...] = ("BKLN", "TLT", "IEF", "SPY"),
    start: str = BKLN_INCEPTION,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily **total-return** price frame for ``tickers`` (one column each).

    Total-return adjusted (``auto_adjust=True`` => dividends + splits folded in), the fair
    series for income instruments whose return is mostly the coupon. The frame is the inner
    join across tickers (dropna), so it begins at the latest inception — **BKLN
    (2011-03-03)** bounds the joint window. Columns are upper-case tickers.

    Cache order per ticker: shared ``_cache/<tk>_total_return.parquet`` (TLT/IEF/SPY) ->
    study-local ``_cache/`` (BKLN) -> live :func:`quantlab.data.fetch` on a miss.
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

    Hashes the column names and the raw float bytes of every column, so a different window,
    a different asset, or drifted data all change the fingerprint loudly.
    """
    h = hashlib.sha1()
    for c in frame.columns:
        h.update(str(c).encode())
        h.update(np.ascontiguousarray(frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
