"""Data layer for Study 907 — Senior Loans vs High-Yield.

The claim under test, steelmanned: *"Senior secured bank loans (BKLN, SRLN) sit **above**
high-yield bonds (HYG, JNK) in the capital stack — first lien, better recovery, and a
floating coupon — at a **similar yield**. So a loan sleeve gives you the same fat carry as
junk bonds but with less risk: a **seniority premium** you collect for free."* We test
whether long-loans (or loans-vs-HY, both excess-of-cash) actually earns a **real
risk-adjusted edge**, through the credit-stress episodes where seniority is supposed to
matter most (2015-16 energy default wave, the 2020 liquidity shock), net of costs and the
loans' own liquidity gaps.

Two kinds of tape, one shape (a daily frame of total-return *price* columns):

* **Real tape** — daily **total-return** closes (``auto_adjust=True`` folds coupons +
  splits in, the fair series for income instruments) for the two flagship **loan** ETFs
  (**BKLN** Invesco Senior Loan, **SRLN** SPDR Blackstone Senior Loan), the two flagship
  **high-yield** ETFs (**HYG** iShares, **JNK** SPDR), an intermediate Treasury (**IEF**,
  a duration reference) and **BIL** (1-3m T-bill, the **cash / risk-free** leg every Sharpe
  is measured *excess of*). Pulled once via yfinance and cached under this study's OWN
  ``_cache/`` as a CSV; every later call reads the cache OFFLINE. **BKLN lists 2011-03-03**
  (it bounds the flagship-pair window); **SRLN lists 2013-04-04** (it joins the composite
  loan sleeve mid-sample — a short-history caveat named on the Signal axis).

* **Synthetic world — the positive control.** :func:`synthetic_pair` builds a deterministic,
  seeded pair of *loans* and *HY* return streams driven by a shared **credit factor**, with
  the loan leg carrying **lower volatility** (seniority + floating rate) and a tunable knob
  ``sharpe_edge`` — the *risk-adjusted* advantage planted into the loan leg. At
  ``sharpe_edge = 0`` the two legs are engineered to the **same excess-Sharpe** (the null:
  lower vol is exactly offset by lower carry, so there is nothing to find); at
  ``sharpe_edge > 0`` the loan leg's excess-Sharpe genuinely exceeds HY's and the detector
  must recover it. A stress window (HY falls harder than loans) is planted so the
  stress-table machinery can be exercised. The synthetic control ONLY proves the machinery
  is unbiased — it never supports the real-tape stamp.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network) runs once to build
the cache; ``load_prices()`` reads it directly and is what the notebooks' offline cells use.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE_DIR = os.path.join(STUDY_ROOT, "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "loans_hy_prices.csv")

# The six tapes. LOANS = {BKLN, SRLN}; HY = {HYG, JNK}; IEF = duration reference; BIL = cash.
TICKERS = ["BKLN", "SRLN", "HYG", "JNK", "IEF", "BIL"]
LOAN_LEGS = ("BKLN", "SRLN")
HY_LEGS = ("HYG", "JNK")

# BKLN (Invesco Senior Loan ETF) lists 2011-03-03 — it bounds the flagship-pair window.
# SRLN (SPDR Blackstone Senior Loan) lists 2013-04-04 — it joins the composite mid-sample.
BKLN_INCEPTION = "2011-03-03"
AS_OF = "2026-06-30"  # last complete calendar month at build; the partial current month is dropped


# ---------------------------------------------------------------------------
# Real tape — daily total-return closes, cache-first
# ---------------------------------------------------------------------------
def fetch(start: str = "2005-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download the six total-return close tapes and cache them (network, run once).

    ``auto_adjust=True`` => dividend + split adjusted (total return), the fair series for
    coupon-driven instruments. Retries up to ``retries`` times on a transient failure.
    """
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError("Could not fetch the loans/HY tapes (network).")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    """True iff the real-tape cache is present (absent on CI — the suite is synthetic-only)."""
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Wide daily **total-return** close frame (one column per ticker), cache-first.

    Sliced to the as-of month-end (the partial current month is dropped). Columns are the
    six upper-case tickers; rows are the union of trading days (NaN before a late lister's
    inception — BKLN 2011-03-03, SRLN 2013-04-04).
    """
    if not os.path.exists(path):
        return fetch(path=path).loc[:asof]
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df[[c for c in TICKERS if c in df.columns]]
    return df[df.index <= pd.Timestamp(asof)]


# ---------------------------------------------------------------------------
# Synthetic world — deterministic offline control (planted risk-adjusted edge)
# ---------------------------------------------------------------------------
def synthetic_pair(
    n_days: int = 3800,
    sharpe_edge: float = 0.0,
    base_sharpe: float = 0.40,      # per-year excess Sharpe both legs share at edge=0
    vol_loans: float = 0.055,       # loan leg annual vol (lower — seniority + floating rate)
    vol_hy: float = 0.082,          # HY leg annual vol (higher)
    rho: float = 0.80,              # loans/HY correlation via the shared credit factor
    rf_annual: float = 0.02,        # cash (BIL) annual yield
    stress_start: int = 2200,
    stress_len: int = 30,
    stress_hy: float = -0.10,       # HY total loss over the planted stress window
    stress_loans: float = -0.06,    # loans lose less (seniority) over the same window
    start: str = BKLN_INCEPTION,
    seed: int = 907,
) -> tuple[pd.DataFrame, dict]:
    """Deterministic loans/HY/cash price frame with a PLANTED risk-adjusted edge.

    Both legs are driven by a shared credit factor ``f`` (correlation ``rho``); the loan leg
    is given ``vol_loans`` < ``vol_hy``. At ``sharpe_edge = 0`` each leg's per-day excess
    mean is set to ``base_sharpe`` × its own daily vol, so **both legs have the same
    excess-Sharpe** (the null — lower vol is exactly paid for by lower carry, nothing to
    find). ``sharpe_edge`` then shifts the loan leg's daily excess mean by
    ``sharpe_edge/sqrt(252) × vol_loans``, so the loan leg's annual excess-Sharpe exceeds
    HY's by ≈ ``sharpe_edge``. A stress window ``[stress_start, +stress_len)`` plants HY
    falling ``stress_hy`` and loans falling the smaller ``stress_loans`` (seniority in a
    selloff), for the stress-table machinery.

    Returns ``(frame, truth)`` where ``frame`` has columns ``['LOANS','HY','CASH']`` (price
    levels from 100) and ``truth`` records the planted knobs.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=int(n_days))  # daily, n<10000 -> OOB-safe
    n = len(idx)
    sd_l = float(vol_loans) / np.sqrt(252.0)
    sd_h = float(vol_hy) / np.sqrt(252.0)
    rf_d = float(rf_annual) / 252.0
    s0 = float(base_sharpe) / np.sqrt(252.0)          # per-day common Sharpe at edge=0
    edge_d = float(sharpe_edge) / np.sqrt(252.0)       # per-day extra Sharpe for loans

    mean_l = rf_d + (s0 + edge_d) * sd_l               # excess mean loans (+edge)
    mean_h = rf_d + s0 * sd_h                          # excess mean HY

    f = rng.standard_normal(n)                         # shared credit factor
    e_l = rng.standard_normal(n)
    e_h = rng.standard_normal(n)
    a = np.sqrt(max(1.0 - rho * rho, 0.0))
    loans = mean_l + sd_l * (rho * f + a * e_l)
    hy = mean_h + sd_h * (rho * f + a * e_h)

    lo = min(max(int(stress_start), 0), max(n - 1, 0))
    hi = min(lo + int(stress_len), n)
    if hi > lo:
        loans[lo:hi] += np.log1p(stress_loans) / (hi - lo)
        hy[lo:hi] += np.log1p(stress_hy) / (hi - lo)

    cash = np.full(n, rf_d)
    frame = pd.DataFrame(
        {
            "LOANS": 100.0 * np.exp(np.cumsum(loans)),
            "HY": 100.0 * np.exp(np.cumsum(hy)),
            "CASH": 100.0 * np.exp(np.cumsum(cash)),
        },
        index=idx,
    )
    frame.index.name = "Date"
    truth = {
        "sharpe_edge": float(sharpe_edge),
        "base_sharpe": float(base_sharpe),
        "vol_loans": float(vol_loans),
        "vol_hy": float(vol_hy),
        "rho": float(rho),
        "n_days": n,
        "seed": int(seed),
        "stress_peak": idx[max(lo - 1, 0)],
        "stress_trough": idx[hi - 1] if hi > lo else idx[-1],
        "stress_hy": float(stress_hy),
        "stress_loans": float(stress_loans),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Fingerprint (content stamp for the as-of line)
# ---------------------------------------------------------------------------
def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame (columns + rounded values + dates).

    NaN is canonicalised so late-lister gaps hash stably; any change to a value, a date, or
    the row count changes the digest loudly.
    """
    h = hashlib.sha1()
    for ts in frame.index:
        h.update(str(pd.Timestamp(ts).date()).encode())
    for c in sorted(map(str, frame.columns)):
        h.update(b"\x1e")
        h.update(c.encode())
        for v in np.round(frame[c].to_numpy(dtype=float), 6):
            h.update(b"nan" if not np.isfinite(v) else np.float64(v).tobytes())
    return h.hexdigest()[:12]
