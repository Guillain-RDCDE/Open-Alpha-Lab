"""Data layer for Study 623 — IPO Long-Run Underperformance.

Two real tracks plus a synthetic control, all offline-friendly once cached:

* **Track A — Ritter's published cohort table (hardcoded).** Table 19 of Jay Ritter's
  "Initial Public Offerings: Updated Long-run Statistics" (April 7, 2026 edition; Table 19
  itself updated February 16, 2026), the updated Table I of Ritter & Welch (2002, JF):
  for every IPO cohort year 1980-2024, the number of IPOs, the average first-day return,
  the average one-year return, and the average **3-year buy-and-hold return** raw,
  **market-adjusted** (vs CRSP value-weighted) and **style-adjusted** (vs size + book-to-market
  matched seasoned firms). Fetched once from
  https://site.warrington.ufl.edu/ritter/files/IPOs-long-run-returns-on-IPOs.pdf
  (linked from https://site.warrington.ufl.edu/ritter/ipo-data/) and hardcoded below with the
  source comment, per desk convention for published event/cohort tables. A copy of the PDF is
  kept in ``_cache/`` for provenance. NOTE: the 2023 and 2024 cohorts are truncated windows
  (returns run only through Dec 31, 2025, i.e. < 3 full years).

* **Track B — the live tape (yfinance, cached).** Daily adjusted (total-return) closes for
  the Renaissance IPO ETF (ticker ``IPO``, inception 2013-10-16 — a rules-based basket of
  recent IPOs, held ~2-3 years then dropped: exactly Ritter's "IPO aftermarket" exposure,
  investable), against ``SPY`` (market), ``IWM`` (small caps), ``IWO`` (Russell 2000 Growth —
  the small-growth *style* benchmark for the third axis) and ``^IRX`` (13-week T-bill rate,
  the risk-free leg). Cached wide under ``_cache/ipo_tape.csv``; everything downstream is
  cache-first and offline.

* **Synthetic.** A deterministic monthly world with a market leg and an "IPO cohort
  portfolio" leg carrying a TUNABLE planted alpha drift (``edge``; negative = the planted
  Ritter drag) plus a null (``edge=0``). The machinery proof: the HAC-alpha detector must
  recover a planted drag and must NOT fire on the null. Spans ~13 years — far below the
  250-year pandas ns-Timestamp ceiling.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
TAPE_CACHE = os.path.join(CACHE_DIR, "ipo_tape.csv")

TICKERS = ["IPO", "SPY", "IWM", "IWO", "^IRX"]

# The stamped sample end: June 2026 is the last complete calendar month as of the run date.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# Track A — Ritter Table 19 (hardcoded published cohort table)
# --------------------------------------------------------------------------- #
# Source: Jay R. Ritter, "Initial Public Offerings: Updated Long-run Statistics",
# Table 19 (updated February 16, 2026) — the updated Table I of Ritter & Welch (2002),
# "A Review of IPO Activity, Pricing, and Allocations", Journal of Finance 57(4).
# https://site.warrington.ufl.edu/ritter/files/IPOs-long-run-returns-on-IPOs.pdf
# Columns: cohort year, number of IPOs, EW average first-day return (%), EW average
# one-year return (%), EW average 3-year buy-and-hold return (%) from the first closing
# market price: raw, market-adjusted (CRSP VW), style-adjusted (size & B/M matched firm).
# Universe screens (Ritter's): offer price >= $5, no units/ADRs/REITs/CEFs/banks/S&Ls/
# natural-resource LPs/small best-efforts; CRSP-listed within 6 months.
# 2023 and 2024 cohorts are TRUNCATED (returns through Dec 31, 2025 only).
RITTER_TABLE19 = [
    # (year,   n, first_day, one_year, bhr3_raw, bhr3_mkt_adj, bhr3_style_adj)
    (1980,   71, 14.3,   28.7,  89.8,  37.0,  18.5),
    (1981,  192,  5.9,  -10.5,  12.3, -27.0,  11.0),
    (1982,   77, 11.0,  101.8,  37.5, -31.5, -12.0),
    (1983,  451,  9.9,  -19.2,  15.9, -37.7,  -4.4),
    (1984,  171,  3.7,   20.0,  50.2, -28.5,  29.0),
    (1985,  186,  6.4,   23.6,   5.6, -41.3, -12.3),
    (1986,  393,  6.2,    9.5,  16.9, -22.6,  -1.3),
    (1987,  285,  5.6,  -21.5,  -2.6, -19.1, -11.2),
    (1988,  105,  5.5,   28.7,  58.0,   9.7,  38.7),
    (1989,  116,  8.0,   -5.5,  48.1,  13.2,   7.2),
    (1990,  110, 10.8,    4.0,   9.7, -35.9, -38.4),
    (1991,  286, 11.9,   10.5,  31.2,  -1.8,   5.8),
    (1992,  412, 10.3,   20.5,  37.4,  -0.2,  11.1),
    (1993,  510, 12.7,    3.0,  44.1,  -8.7,  -9.5),
    (1994,  402,  9.6,   27.8,  78.0,  -5.7,  -0.9),
    (1995,  462, 21.4,   26.5,  28.6, -58.0, -24.7),
    (1996,  677, 17.2,    7.1,  25.2, -56.8,   7.0),
    (1997,  474, 14.0,    8.0,  58.3,  -2.0,  22.0),
    (1998,  283, 21.8,   18.4,  22.9,   5.1,  -4.9),
    (1999,  476, 71.2,   22.1, -47.6, -32.5, -60.6),
    (2000,  380, 56.4,  -52.9, -60.1, -30.9, -56.9),
    (2001,   80, 14.0,  -14.3,  18.0,  14.6, -27.8),
    (2002,   66,  9.1,    3.1,  68.6,  39.0,  -0.4),
    (2003,   63, 11.7,   25.7,  34.0,  -7.7, -11.2),
    (2004,  173, 12.3,   17.8,  51.4,   6.9,  -7.0),
    (2005,  159, 10.3,   19.0,  14.6,   3.1,  -2.5),
    (2006,  157, 12.1,   21.4, -28.8, -11.1,  -4.5),
    (2007,  159, 14.0,  -28.4, -16.5,  -0.4,   0.5),
    (2008,   21,  5.7,  -34.4,  11.4,   8.1,   5.1),
    (2009,   41,  9.8,   11.5,  37.0,  -5.1, -18.3),
    (2010,   91,  9.4,   15.7,  36.4,  -9.6, -18.5),
    (2011,   81, 13.9,  -12.2,  38.6,  -8.7, -11.6),
    (2012,   93, 17.7,   35.7,  81.9,  31.8,  33.4),
    (2013,  158, 20.9,   12.8,  12.1, -14.4, -16.1),
    (2014,  206, 15.5,   20.1,  17.1,  -9.7, -12.5),
    (2015,  118, 19.2,  -23.8,  24.5,  -9.9, -27.3),
    (2016,   75, 14.5,   23.3,  70.5,  29.5,  27.0),
    (2017,  106, 12.9,   32.4,  52.8,  22.6,  35.6),
    (2018,  134, 18.6,   -6.8,  79.1,  23.4,  55.8),
    (2019,  113, 23.5,   33.0,  12.5, -25.1,  -8.8),
    (2020,  165, 41.6,    9.9, -48.1, -78.6, -75.7),
    (2021,  311, 32.1,  -49.1, -49.1, -68.6, -42.3),
    (2022,   38, 48.9,  -27.7, -29.5, -72.2, -69.9),
    (2023,   54, 11.9,  -11.6, -30.9, -88.3, -84.7),   # truncated: through Dec 31, 2025
    (2024,   72, 15.4,  -17.4,  -4.7, -33.2, -12.3),   # truncated: through Dec 31, 2025
]

# Ritter's own bottom line (same table, sample-size-weighted aggregate row):
# 9,253 IPOs 1980-2024, avg first-day +18.9%, avg 3-yr BHR +19.1% raw,
# -20.5% market-adjusted, -8.9% style-adjusted.
RITTER_AGGREGATE = {
    "n_ipos": 9253, "first_day": 18.9, "one_year": 5.6,
    "bhr3_raw": 19.1, "bhr3_mkt_adj": -20.5, "bhr3_style_adj": -8.9,
}


def ritter_cohorts(drop_truncated: bool = False) -> pd.DataFrame:
    """Ritter Table 19 as a DataFrame indexed by cohort year.

    ``drop_truncated=True`` removes the 2023-2024 cohorts whose 3-year windows are
    incomplete (returns only run through Dec 31, 2025).
    """
    df = pd.DataFrame(
        RITTER_TABLE19,
        columns=["year", "n_ipos", "first_day", "one_year",
                 "bhr3_raw", "bhr3_mkt_adj", "bhr3_style_adj"],
    ).set_index("year")
    if drop_truncated:
        df = df.loc[:2022]
    return df


# --------------------------------------------------------------------------- #
# Track B — the live tape (Renaissance IPO ETF vs benchmarks)
# --------------------------------------------------------------------------- #
def fetch_tape(start: str = "2013-01-01", end: str | None = None,
               path: str = TAPE_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the ETF tape once and cache it (network-only; never in the offline path)."""
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
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = TAPE_CACHE) -> bool:
    return os.path.exists(path)


def load_tape(path: str = TAPE_CACHE) -> pd.DataFrame:
    """Cached wide daily frame: adjusted closes for IPO/SPY/IWM/IWO + ^IRX (percent)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def monthly_panel(tape: pd.DataFrame, as_of: str = AS_OF) -> pd.DataFrame:
    """Monthly simple total returns for IPO/SPY/IWM/IWO plus the monthly risk-free rate.

    Sliced to the pinned ``as_of`` (last complete month) BEFORE resampling, so a stamped
    run never contains a partial month. Returns start at the first full month of the IPO
    ETF's life (Nov 2013). ``rf`` is the 13-week T-bill discount yield (^IRX, %/yr)
    averaged over the month and de-annualised to a monthly simple rate.
    """
    t = tape[tape.index <= pd.Timestamp(as_of)].copy()
    px = t[["IPO", "SPY", "IWM", "IWO"]]
    mret = px.resample("ME").last().pct_change()
    rf_ann = t["^IRX"].ffill().resample("ME").mean() / 100.0
    rf_m = (1.0 + rf_ann) ** (1.0 / 12.0) - 1.0
    out = mret.copy()
    out["rf"] = rf_m
    out = out.dropna(subset=["IPO", "SPY", "IWM", "IWO"])
    # first full calendar month of the IPO ETF (inception 2013-10-16 -> start Nov 2013)
    out = out[out.index >= "2013-11-01"]
    return out


# --------------------------------------------------------------------------- #
# Synthetic control — a world where we KNOW the truth
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 152, edge: float = 0.0, beta: float = 1.25,
                    seed: int = 623, mkt_mu: float = 0.008, mkt_sig: float = 0.042,
                    idio_sig: float = 0.030) -> pd.DataFrame:
    """Deterministic monthly world: a market leg and an 'IPO portfolio' leg.

    The IPO leg is ``rf + beta * (mkt - rf) + edge + idiosyncratic noise`` — i.e. a
    levered-beta basket with a TUNABLE planted monthly alpha ``edge`` (negative = the
    planted Ritter drag; 0 = the null world where any measured alpha is pure noise).
    Same shape as :func:`monthly_panel` (columns IPO/SPY/IWM/IWO/rf; the small-cap columns
    are decorative correlated legs so the downstream code paths run unchanged).

    Fixed seed -> byte-identical output. Index is a ~13-year monthly period_range
    (decorative; far below the 250-year ns-Timestamp ceiling).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2013-11", periods=n_months, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    rf = np.full(n_months, 0.0015)
    mkt = rng.normal(mkt_mu, mkt_sig, n_months)
    ipo = rf + beta * (mkt - rf) + edge + rng.normal(0.0, idio_sig, n_months)
    iwm = rf + 1.1 * (mkt - rf) + rng.normal(0.0, 0.020, n_months)
    iwo = rf + 1.2 * (mkt - rf) + rng.normal(0.0, 0.022, n_months)
    return pd.DataFrame({"IPO": ipo, "SPY": mkt, "IWM": iwm, "IWO": iwo, "rf": rf},
                        index=idx)
