"""Data layer for Study 521 (Cash-Based Operating Profitability).

Two tapes, one schema -- a (fiscal-year x ticker) signal frame plus aligned next-year
returns:

- ``synthetic_panel`` -- a *deterministic, offline* generator. A tunable ``cop_premium``
  knob controls how strongly cash-based operating profitability predicts next-year
  returns; ``cop_premium = 0`` is the null hypothesis (no predictive power). Tests and
  the synthetic positive control never touch the network.

- ``fetch_panel`` -- the real tape: yfinance annual fundamentals (income statement,
  cash-flow statement, balance sheet) plus daily adjusted prices for a fixed ~40-name
  large-cap survivor basket, cached to **this study's own** ``_cache/`` (gitignored).
  Returns ``(cop, gpa, fwd_ret)`` or empty frames if the cache is absent and the
  network fails (e.g. on CI).

THE SIGNAL (Ball, Gerakos, Linnainmaa & Nikolaev 2016, *Accruals, cash flows, and
operating profitability in the cross section of stock returns*, JFE):

    cash-based OP = (accounting operating profit  -  accruals)  /  total assets

We strip the accrual component out of operating profitability using the working-capital
changes that yfinance reports directly on the cash-flow statement:

    cash_OP_t = ( OperatingIncome_t + D&A_t + ChangeInWorkingCapital_t ) / TotalAssets_t

where ``ChangeInWorkingCapital`` already nets receivables, inventory, payables and
accrued-expense changes (the BGLN accrual block). For the head-to-head contrast we also
compute the accrual-laden gross-profitability sibling (Study 122):

    gpa_t = GrossProfit_t / TotalAssets_t

No look-ahead: the signal for fiscal year *y* is built from year-*y* filings (reported
after the fiscal year ends). We conservatively lag by one full year -- the signal on
year-*y* data predicts the *next* 12-month forward return, entered one trading day after
the fiscal-year-end date is public (the standard Novy-Marx / Fama-French annual lag).

The basket is **survivorship-biased** (names still trading in 2026): positive results
are **upper-bound** estimates. We name this on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Fixed large-cap survivor basket (~40 names, sector-spread, all still trading 2026).
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "ORCL", "CRM", "ADBE",
    "JPM", "BAC", "WFC", "GS", "MA", "V", "AXP",
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "HD", "NKE",
    "XOM", "CVX", "CAT", "GE", "HON", "UPS", "LIN",
]
_seen: set[str] = set()
UNIVERSE = [t for t in UNIVERSE if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    cop_premium: float  # slope: how much z(cash-OP) lifts next-year return

    @property
    def has_premium(self) -> bool:
        return self.cop_premium != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_firms: int = 200,
    n_years: int = 16,
    cop_premium: float = 0.06,
    base_ret: float = 0.10,
    ret_vol: float = 0.30,
    accrual_noise: float = 0.4,
    seed: int = 521,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WorldTruth]:
    """A reproducible (year x ticker) cash-OP panel with a known effect.

    Each firm-year draws an accounting operating-profitability ratio from a right-skewed
    log-normal. We split it into a *cash* component and an *accrual* component; only the
    cash component carries predictive power for next-year returns:

        gross/accrual-laden OP  = cash_OP + accrual
        next-year return        = base_ret + cop_premium * z(cash_OP) + noise

    ``cop_premium = 0`` is the null. The accrual component is pure noise vs returns -- so a
    *faithful* engine should find a stronger spread sorting on cash-OP than on the
    accrual-laden ratio, exactly the BGLN (2016) result.

    Returns ``(cop, gpa, fwd_ret, truth)`` -- parallel (year x ticker) frames.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2009, 2009 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]

    cash_op = rng.lognormal(mean=-1.2, sigma=0.5, size=(n_years, n_firms))
    accrual = accrual_noise * rng.standard_normal((n_years, n_firms)) * cash_op.std()
    accrual_laden = cash_op + accrual  # the "gross/operating profitability" sibling

    zc = (cash_op - cash_op.mean(axis=1, keepdims=True)) / (
        cash_op.std(axis=1, keepdims=True) + 1e-9
    )
    fwd = base_ret + cop_premium * zc + ret_vol * rng.standard_normal((n_years, n_firms))

    idx = pd.Index(years, name="year")
    cop = pd.DataFrame(cash_op, index=idx, columns=firms)
    gpa = pd.DataFrame(accrual_laden, index=idx, columns=firms)
    fwd_ret = pd.DataFrame(fwd, index=idx, columns=firms)
    return cop, gpa, fwd_ret, WorldTruth(cop_premium)


# ---------------------------------------------------------------------------
# Real panel -- yfinance, study-local cache
# ---------------------------------------------------------------------------
def _row(df: pd.DataFrame, *names: str) -> pd.Series | None:
    """First matching row of a yfinance statement frame (by exact label), or None."""
    if df is None or df.empty:
        return None
    for nm in names:
        if nm in df.index:
            return df.loc[nm]
    return None


def _fetch_one(ticker: str, retries: int = 3):
    """Pull annual statements + daily prices for one ticker, with retries."""
    import yfinance as yf

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            fin = tk.financials
            cf = tk.cashflow
            bs = tk.balance_sheet
            px = tk.history(period="max", auto_adjust=True)["Close"]
            if fin is None or fin.empty or bs is None or bs.empty:
                raise ValueError("empty statements")
            return fin, cf, bs, px
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"{ticker}: {last_exc}")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cash-OP signal, gross-profitability sibling, and aligned next-year returns.

    Cache-first: reads ``<cache_dir>/cop.parquet``, ``gpa.parquet`` and
    ``fwd.parquet`` if present. On a cache miss and ``fetch=True``, pulls yfinance
    annual fundamentals + daily prices for :data:`UNIVERSE`, builds the panels, and
    writes the parquets. Returns empty frames if both cache and network fail.

    Construction per fiscal year *y* for each ticker:

      * ``cash_OP_y`` = (OperatingIncome_y + D&A_y + ChangeInWorkingCapital_y) / Assets_y
      * ``gpa_y``     = GrossProfit_y / Assets_y
      * ``fwd_y``     = the 252-trading-day forward total return entered one trading day
        after the fiscal-year-end date (the one-year reporting lag).

    Survivorship caveat (named): the basket is names still trading in 2026.
    """
    cop_p = os.path.join(cache_dir, "cop.parquet")
    gpa_p = os.path.join(cache_dir, "gpa.parquet")
    fwd_p = os.path.join(cache_dir, "fwd.parquet")
    if all(os.path.exists(p) for p in (cop_p, gpa_p, fwd_p)):
        return (
            pd.read_parquet(cop_p),
            pd.read_parquet(gpa_p),
            pd.read_parquet(fwd_p),
        )
    if not fetch:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cop_rows: dict[tuple[int, str], float] = {}
    gpa_rows: dict[tuple[int, str], float] = {}
    fwd_rows: dict[tuple[int, str], float] = {}

    for tk in UNIVERSE:
        try:
            fin, cf, bs, px = _fetch_one(tk)
        except Exception:  # noqa: BLE001
            continue
        if px is None or px.empty:
            continue
        px = px.copy()
        px.index = pd.to_datetime(px.index).tz_localize(None)

        op_inc = _row(fin, "Operating Income", "Total Operating Income As Reported")
        gp = _row(fin, "Gross Profit")
        assets = _row(bs, "Total Assets")
        dna = _row(cf, "Depreciation Amortization Depletion", "Depreciation And Amortization")
        dwc = _row(cf, "Change In Working Capital")

        for col in fin.columns:
            fy = pd.Timestamp(col)
            year = int(fy.year)
            a = assets.get(col) if assets is not None else None
            if a is None or not np.isfinite(a) or a == 0:
                continue

            # Cash-based operating profitability: strip the accrual block.
            oi = op_inc.get(col) if op_inc is not None else None
            d = dna.get(col) if dna is not None else 0.0
            w = dwc.get(col) if dwc is not None else 0.0
            d = 0.0 if (d is None or not np.isfinite(d)) else float(d)
            w = 0.0 if (w is None or not np.isfinite(w)) else float(w)
            if oi is not None and np.isfinite(oi):
                cop_rows[(year, tk)] = float((oi + d + w) / a)

            # Accrual-laden gross profitability (Study 122 sibling).
            g = gp.get(col) if gp is not None else None
            if g is not None and np.isfinite(g):
                gpa_rows[(year, tk)] = float(g / a)

            # Forward return: enter 1 trading day after the fiscal-year-end is public.
            entry = px.index[px.index > fy]
            if len(entry) == 0:
                continue
            e0 = entry[0]
            future = px.loc[e0:]
            if len(future) < 252:
                continue
            p0 = float(future.iloc[0])
            p1 = float(future.iloc[min(251, len(future) - 1)])
            if p0 > 0:
                fwd_rows[(year, tk)] = p1 / p0 - 1.0

    if not cop_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cop = _pivot(cop_rows)
    gpa = _pivot(gpa_rows)
    fwd = _pivot(fwd_rows).reindex(index=cop.index, columns=cop.columns)

    os.makedirs(cache_dir, exist_ok=True)
    cop.to_parquet(cop_p)
    gpa.to_parquet(gpa_p)
    fwd.to_parquet(fwd_p)
    return cop, gpa, fwd


def _pivot(rows: dict[tuple[int, str], float]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    s = pd.Series(rows)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["year", "ticker"])
    df = s.unstack("ticker").sort_index()
    df.index.name = "year"
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a panel (for the as-of stamp in docs/results.md)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
