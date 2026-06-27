"""Data layer for Study 531 (Enterprise-Multiple).

Two tapes, one shape -- a monthly panel of cross-sectional EV/EBITDA signal values
and forward monthly returns for a fixed survivor basket of large-cap names:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``em_premium``
  knob controls how strongly a low enterprise multiple predicts next-month returns;
  ``em_premium=0`` is the null (the sort is a coin), so tests / the positive control can
  assert the edge appears only when the premium is planted. Built on a ``pd.period_range``
  monthly index to dodge the ns-Timestamp overflow trap on CI pandas.

- ``fetch_panel`` -- the real tape. For each ticker we pull annual fundamentals from
  yfinance (``Ticker.financials`` for EBITDA, ``Ticker.balance_sheet`` for total debt,
  cash, and shares outstanding) plus daily adjusted prices. Enterprise value is
  ``market cap + total debt - cash`` (market cap = shares x price), and the signal is the
  *enterprise multiple* EV/EBITDA. The multiple is computed per fiscal year, then held
  constant (forward-filled) through the following 12 months with a reporting lag, so the
  monthly cross-section is always formed from already-public data.

**Survivorship bias** is explicit: the basket is names *still trading in 2026*. The natural
shorts -- expensive names that later imploded and were delisted -- are absent, so any
long-cheap/short-expensive premium is an upper bound. Named on the SIGNAL axis; opt-in
guard via ``allow_survivorship_bias`` (default True so the study runs).

No look-ahead: the fiscal-year-Y multiple is treated as public only after a reporting lag
(``report_lag_months=4``, i.e. fundamentals filed by ~April of Y+1), and the position is
entered the *next* trading day (one execution lag), so the signal is always stale before
it trades.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# A fixed ~40-name large-cap survivor basket (still trading in 2026), spread across
# sectors so the cross-section is not a single-industry bet. Same flavour as the
# baskets in studies 238 / 330 / 363.
BASKET = [
    # Tech / comms
    "AAPL", "MSFT", "GOOGL", "ORCL", "CSCO", "IBM", "INTC", "QCOM", "TXN", "ADBE",
    # Consumer staples / discretionary
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "HD", "NKE", "SBUX", "TGT",
    # Health care
    "JNJ", "PFE", "MRK", "ABT", "UNH", "AMGN", "MDT",
    # Financials
    "JPM", "BAC", "WFC", "AXP", "USB",
    # Industrials / energy / materials
    "XOM", "CVX", "CAT", "HON", "UPS", "MMM", "LMT", "DE",
]

# yfinance row labels we rely on (stable across recent yfinance versions).
_EBITDA_KEY = "EBITDA"
_TOTAL_DEBT_KEY = "Total Debt"
_CASH_KEY = "Cash And Cash Equivalents"
_CASH_KEY_ALT = "Cash Cash Equivalents And Short Term Investments"
_SHARES_KEY = "Ordinary Shares Number"
_SHARES_KEY_ALT = "Share Issued"


@dataclass(frozen=True)
class WorldTruth:
    """The planted premium parameters for a synthetic panel."""

    em_premium: float  # extra monthly return per cross-sectional (negated) EM z-score

    @property
    def has_edge(self) -> bool:
        return self.em_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic tape (deterministic, offline) -- the positive control / power check
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 40,
    n_months: int = 180,
    em_premium: float = 0.004,
    base_ret: float = 0.007,
    ret_vol: float = 0.055,
    seed: int = 531,
) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible firm x month panel with a known enterprise-multiple effect.

    Each month every firm draws an enterprise multiple (EV/EBITDA) and the *cheap*
    firms (low EM) earn a premium next month:

        ret_{t+1} = base_ret + em_premium * z_cheap + noise

    where ``z_cheap`` is the cross-sectional z-score of the *negated* multiple (so low
    EM -> high z_cheap -> higher expected return, the value direction). Setting
    ``em_premium=0`` is the null -- the sort becomes a coin flip.

    Returns ``(em_panel, fwd_ret, truth)`` -- both DataFrames indexed by a monthly
    ``PeriodIndex`` (built with ``pd.period_range`` to avoid ns overflow), columns =
    firm identifiers. ``em_panel`` is the signal *known at the start* of each month;
    ``fwd_ret`` is that month's realised return (already aligned -- row t is the return
    earned by the position formed from row-t's signal).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2011-01", periods=n_months, freq="M")
    firms = [f"F{j:02d}" for j in range(n_firms)]

    # A persistent firm-level multiple level (slow-moving) plus monthly jitter, so the
    # sort has the realistic property that a firm tends to stay cheap or expensive.
    level = rng.uniform(5.0, 18.0, size=n_firms)  # plausible EV/EBITDA levels
    em = np.empty((n_months, n_firms))
    for tt in range(n_months):
        em[tt] = level + rng.standard_normal(n_firms) * 1.2
    em = np.clip(em, 2.0, 40.0)

    # Cross-sectional z-score of the negated multiple -> "cheapness".
    neg = -em
    mu = neg.mean(axis=1, keepdims=True)
    sd = neg.std(axis=1, keepdims=True) + 1e-9
    z_cheap = (neg - mu) / sd

    noise = ret_vol * rng.standard_normal((n_months, n_firms))
    fwd = base_ret + em_premium * z_cheap + noise

    em_panel = pd.DataFrame(em, index=idx, columns=firms)
    fwd_panel = pd.DataFrame(fwd, index=idx, columns=firms)
    return em_panel, fwd_panel, WorldTruth(em_premium)


# ---------------------------------------------------------------------------
# Real tape (yfinance fundamentals + prices), cached to THIS study's _cache/
# ---------------------------------------------------------------------------
def fetch_panel(
    tickers: list[str] | None = None,
    fetch: bool = False,
    cache_dir: str = STUDY_CACHE,
    report_lag_months: int = 4,
    allow_survivorship_bias: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly EV/EBITDA signal panel and aligned forward monthly returns.

    Returns ``(em_panel, fwd_ret)`` -- both month (``PeriodIndex``) x ticker DataFrames.
    Row t of ``em_panel`` is the enterprise multiple *already public* at the start of
    month t; row t of ``fwd_ret`` is the simple return realised over month t (so a sort
    on ``em_panel.loc[t]`` is paid by ``fwd_ret.loc[t]`` -- no look-ahead).

    **Cache-first:** reads ``cache_dir/em_panel.parquet`` and ``cache_dir/fwd_ret.parquet``
    if present. With ``fetch=True`` it (re)downloads from yfinance (guarded with retries)
    and rewrites the cache + a manifest. With ``fetch=False`` and no cache it returns empty
    frames (callers must handle gracefully).

    **Survivorship caveat:** the basket is current/2026 survivors -- magnitudes are upper
    bounds. ``allow_survivorship_bias=False`` raises, to make the opt-in explicit.
    """
    if not allow_survivorship_bias:
        raise RuntimeError(
            "Study 531 uses a survivor basket (names still trading in 2026). "
            "Pass allow_survivorship_bias=True to acknowledge the upper-bound caveat."
        )
    tickers = list(tickers) if tickers is not None else list(BASKET)
    em_p = os.path.join(cache_dir, "em_panel.parquet")
    fwd_p = os.path.join(cache_dir, "fwd_ret.parquet")

    if not fetch and os.path.exists(em_p) and os.path.exists(fwd_p):
        em = pd.read_parquet(em_p)
        fwd = pd.read_parquet(fwd_p)
        em.index = pd.PeriodIndex(em.index, freq="M")
        fwd.index = pd.PeriodIndex(fwd.index, freq="M")
        return em, fwd

    if not fetch:
        return pd.DataFrame(), pd.DataFrame()

    # ---- live download -------------------------------------------------------
    fy_multiple = {}  # ticker -> Series indexed by fiscal-year-end Timestamp
    prices = {}       # ticker -> monthly close Series
    for tk in tickers:
        mult = _fetch_fy_multiple(tk)
        px = _fetch_monthly_close(tk)
        if mult is None or px is None or mult.empty or px.empty:
            continue
        fy_multiple[tk] = mult
        prices[tk] = px

    if not fy_multiple:
        return pd.DataFrame(), pd.DataFrame()

    # Common monthly grid.
    all_idx = pd.period_range(
        min(p.index.min() for p in prices.values()),
        max(p.index.max() for p in prices.values()),
        freq="M",
    )
    em_cols, fwd_cols = {}, {}
    for tk in fy_multiple:
        mult = fy_multiple[tk]      # fiscal-year-end -> EV/EBITDA
        px = prices[tk]             # monthly PeriodIndex -> close

        # Forward-fill the fiscal-year multiple into each subsequent month, but only
        # after a reporting lag (signal becomes public report_lag_months after FY end).
        sig = pd.Series(index=all_idx, dtype=float)
        for fy_end, val in mult.items():
            if pd.isna(val):
                continue
            start = (pd.Period(fy_end, freq="M") + report_lag_months)
            stop = start + 12  # held until the next fiscal year's data is public
            mask = (all_idx >= start) & (all_idx < stop)
            sig.loc[mask] = val
        em_cols[tk] = sig

        # Monthly simple returns aligned so row t = return over month t.
        ret = px.reindex(all_idx).pct_change().shift(-1)
        fwd_cols[tk] = ret

    em = pd.DataFrame(em_cols).reindex(all_idx)
    fwd = pd.DataFrame(fwd_cols).reindex(all_idx)

    # Keep only months with at least a handful of names on both sides.
    keep = (em.notna().sum(axis=1) >= 10) & (fwd.notna().sum(axis=1) >= 10)
    em, fwd = em.loc[keep], fwd.loc[keep]

    # Drop any trailing partial month (no full forward return).
    valid = fwd.notna().any(axis=1)
    em, fwd = em.loc[valid], fwd.loc[valid]

    os.makedirs(cache_dir, exist_ok=True)
    em.to_parquet(em_p)
    fwd.to_parquet(fwd_p)
    _write_manifest(cache_dir, em, fwd, tickers)
    return em, fwd


def _fetch_fy_multiple(ticker: str, retries: int = 3) -> pd.Series | None:
    """Per-fiscal-year EV/EBITDA for one ticker from yfinance fundamentals.

    EV = market-cap (shares x fiscal-year-end price) + total debt - cash.
    Returns a Series indexed by fiscal-year-end Timestamp, or None on failure.
    """
    import yfinance as yf

    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            fin = t.financials
            bs = t.balance_sheet
            if fin is None or bs is None or fin.empty or bs.empty:
                return None
            if _EBITDA_KEY not in fin.index or _TOTAL_DEBT_KEY not in bs.index:
                return None
            ebitda = fin.loc[_EBITDA_KEY]
            debt = bs.loc[_TOTAL_DEBT_KEY]
            cash = (
                bs.loc[_CASH_KEY] if _CASH_KEY in bs.index
                else bs.loc[_CASH_KEY_ALT] if _CASH_KEY_ALT in bs.index
                else pd.Series(0.0, index=bs.columns)
            )
            shares = (
                bs.loc[_SHARES_KEY] if _SHARES_KEY in bs.index
                else bs.loc[_SHARES_KEY_ALT] if _SHARES_KEY_ALT in bs.index
                else None
            )
            if shares is None:
                return None

            # Fiscal-year-end closing prices for the market-cap term.
            hist = t.history(period="max", interval="1mo", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            close = hist["Close"]
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)

            out = {}
            for col in ebitda.index:
                fy_end = pd.Timestamp(col).tz_localize(None)
                eb = ebitda.get(col, np.nan)
                if pd.isna(eb) or eb <= 0:
                    continue
                d = debt.get(col, 0.0)
                c = cash.get(col, 0.0)
                sh = shares.get(col, np.nan)
                if pd.isna(sh) or sh <= 0:
                    continue
                # closing price in the month of the fiscal-year end
                px_window = close[close.index <= fy_end]
                if px_window.empty:
                    continue
                price = float(px_window.iloc[-1])
                mktcap = sh * price
                ev = mktcap + (0.0 if pd.isna(d) else float(d)) - (0.0 if pd.isna(c) else float(c))
                if ev <= 0:
                    continue
                out[fy_end] = ev / float(eb)
            if not out:
                return None
            s = pd.Series(out).sort_index()
            # Drop implausible multiples (data errors).
            return s.where((s > 0) & (s < 100))
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _fetch_monthly_close(ticker: str, retries: int = 3) -> pd.Series | None:
    """Monthly adjusted close for one ticker, indexed by monthly PeriodIndex."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max", interval="1mo", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            close = hist["Close"].copy()
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            close = close.resample("ME").last()
            close.index = close.index.to_period("M")
            return close.dropna()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _write_manifest(cache_dir: str, em: pd.DataFrame, fwd: pd.DataFrame, tickers: list[str]) -> None:
    manifest = {
        "study": "531-enterprise-multiple",
        "tickers_requested": tickers,
        "tickers_used": list(em.columns),
        "n_months": int(em.shape[0]),
        "month_min": str(em.index.min()),
        "month_max": str(em.index.max()),
        "fp_em": fingerprint(em),
        "fp_fwd": fingerprint(fwd),
    }
    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint (SHA-1 of flattened values), for the as-of stamp."""
    arr = np.ascontiguousarray(df.fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
