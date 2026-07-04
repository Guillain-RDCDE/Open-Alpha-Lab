"""Data layer for Study 624 — Buffer ETFs: the Cost of Comfort.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for the flagship laddered buffer ETF (**BUFR**) and the four
  quarterly **Innovator U.S. Equity Power Buffer** single-series funds (PJAN / PAPR / PJUL /
  POCT, 15% buffer, annual reset), plus **SPY** (the reference asset every one of these funds
  writes its options on) and **BIL** (1-3 month T-bills, the cash leg of the "dumb mix").
  Two adjustment modes, cached separately, because the study needs both:

  - ``bec_adj.csv`` — **auto-adjusted (total-return)** closes for every ticker: fund and mix
    performance is always measured in total return.
  - ``bec_raw.csv`` — **unadjusted** closes for SPY: the defined-outcome terms (cap / buffer)
    are written on SPY's **price** return over the outcome period, dividends excluded — the
    delivery check must therefore use the raw price series.

* **Synthetic.** A deterministic, fixed-seed generator with a TUNABLE planted effect and a
  null (see :func:`synthetic_world` and :func:`synthetic_outcomes`): (a) a monthly world where
  the "buffer fund" either *is* exactly the beta-matched SPY/BIL mix (null — the gap detector
  must NOT fire) or carries a planted annual **structuring drag** (the detector must recover
  it); (b) an outcome-period world with a *known* planted cap and buffer that the delivery
  checker must read back. Date spans stay well under 250 years.

Pure numpy + pandas + stdlib on the offline path. ``fetch_panel`` (network) is used once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
ADJ_CACHE = os.path.join(CACHE_DIR, "bec_adj.csv")
RAW_CACHE = os.path.join(CACHE_DIR, "bec_raw.csv")

# Headline as-of: the tape is sliced here so reruns cannot drift. June 2026 is the last
# complete calendar month at publication (2026-07-03).
AS_OF = "2026-06-30"

# ---------------------------------------------------------------------------
# The funds and their STATED terms.
# Sources (fetched 2026-07-03, hardcoded per desk convention):
#   * FT Vest / First Trust BUFR fund page (ftportfolios.com, "FT Vest Laddered Buffer ETF"):
#     inception 2020-08-10, net expense ratio 0.95%/yr (fund-of-funds: 12 monthly FT Vest
#     Buffer ETFs, each ~10% buffer on SP500 PRICE return, annual reset, laddered monthly).
#   * Innovator ETFs "U.S. Equity Power Buffer ETF" series pages (innovatoretfs.com):
#     PJUL inception 2018-08-07, POCT 2018-10-01, PJAN 2019-01-02, PAPR 2019-04-01;
#     each: 15% buffer against SPY PRICE-return losses over a 12-month outcome period
#     (resetting the 1st trading day of the name month), upside capped at a level reset
#     annually (caps have ranged ~8-16% gross across vintages), expense ratio 0.79%/yr.
#   * Both issuers state terms are on the reference asset's PRICE return, before fees.
# ---------------------------------------------------------------------------
FUNDS = {
    "BUFR": dict(name="FT Vest Laddered Buffer ETF", er_pct=0.95, buffer_pct=10.0,
                 reset_month=None, laddered=True, inception="2020-08-10"),
    "PJAN": dict(name="Innovator U.S. Equity Power Buffer - January", er_pct=0.79,
                 buffer_pct=15.0, reset_month=1, laddered=False, inception="2019-01-02"),
    "PAPR": dict(name="Innovator U.S. Equity Power Buffer - April", er_pct=0.79,
                 buffer_pct=15.0, reset_month=4, laddered=False, inception="2019-04-01"),
    "PJUL": dict(name="Innovator U.S. Equity Power Buffer - July", er_pct=0.79,
                 buffer_pct=15.0, reset_month=7, laddered=False, inception="2018-08-07"),
    "POCT": dict(name="Innovator U.S. Equity Power Buffer - October", er_pct=0.79,
                 buffer_pct=15.0, reset_month=10, laddered=False, inception="2018-10-01"),
}
BENCH = ["SPY", "BIL"]
TICKERS = list(FUNDS) + BENCH

# SPY's trailing 12-month dividend yield over the sample, ~1.4%/yr (yfinance dividend
# history 2019-2026; used only in the descriptive cost decomposition).
SPY_DIV_YIELD_PCT = 1.4


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2018-08-01", end: str | None = None,
                adj_path: str = ADJ_CACHE, raw_path: str = RAW_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted + raw closes for all tickers and cache them (network, run once)."""
    import yfinance as yf

    def _dl(auto_adjust: bool) -> pd.DataFrame:
        for _ in range(retries):
            try:
                px = yf.download(TICKERS, start=start, end=end, auto_adjust=auto_adjust,
                                 progress=False)["Close"]
                if px is not None and len(px) > 0:
                    return px.dropna(how="all")
            except Exception:
                time.sleep(2.0)
        raise RuntimeError("yfinance download failed")

    adj = _dl(True)
    raw = _dl(False)
    os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    adj.to_csv(adj_path)
    raw.to_csv(raw_path)
    return adj, raw


def have_real(adj_path: str = ADJ_CACHE, raw_path: str = RAW_CACHE) -> bool:
    return os.path.exists(adj_path) and os.path.exists(raw_path)


def load_real(asof: str | None = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (adjusted total-return closes, raw price closes), sliced to the as-of."""
    adj = pd.read_csv(ADJ_CACHE, index_col=0, parse_dates=True).sort_index()
    raw = pd.read_csv(RAW_CACHE, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        adj = adj[adj.index <= pd.Timestamp(asof)]
        raw = raw[raw.index <= pd.Timestamp(asof)]
    return adj, raw


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple returns (month-end to month-end); drops a trailing partial month."""
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < last_bucket:            # tape stops before that month truly ended
        ret = ret.iloc[:-1]
    return ret


# --------------------------------------------------------------------------- #
# Synthetic worlds (machinery proofs — never market evidence)
# --------------------------------------------------------------------------- #
def synthetic_world(drag_pct: float = 0.0, n_months: int = 240, w: float = 0.55,
                    seed: int = 62400) -> pd.DataFrame:
    """Monthly world for the GAP DETECTOR: index, cash, and a 'buffer fund'.

    The fund is the ``w``/(1-w) index/cash mix plus small idiosyncratic tracking noise
    (the option wrapper's NAV wiggle) minus a planted annual structuring drag of
    ``drag_pct`` (the tunable effect). With ``drag_pct = 0`` the fund is an unbiased copy
    of the mix — the mix-minus-fund HAC t must NOT fire; with a planted drag the detector
    must recover it. Deterministic (fixed seed); span = 20 years (well under the 250-year
    CI trap). Returns a DataFrame of monthly simple returns: columns IDX, CASH, FUND.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2000-01", periods=n_months, freq="M")
    idx = rng.normal(0.008, 0.045, size=n_months)
    cash = np.full(n_months, 0.0015)
    noise = rng.normal(0.0, 0.003, size=n_months)
    fund = w * idx + (1.0 - w) * cash + noise - (drag_pct / 100.0) / 12.0
    return pd.DataFrame({"IDX": idx, "CASH": cash, "FUND": fund},
                        index=pidx.to_timestamp(how="end").normalize())


def synthetic_outcomes(cap_pct: float = 10.0, buffer_pct: float = 15.0,
                       fee_pct: float = 0.79, n_years: int = 40,
                       seed: int = 6240) -> pd.DataFrame:
    """Outcome-period world for the DELIVERY CHECKER: known planted cap + buffer.

    Each row is one 12-month outcome period: the reference PRICE return is drawn, and the
    fund period return is the textbook defined-outcome payoff — min(ref, cap) above zero,
    0 inside the buffer, ref + buffer beyond it — minus the fee, plus small execution noise.
    The checker must read back the planted buffer (floor honored) and an implied cap close
    to ``cap_pct``. Deterministic; 40 annual periods.
    """
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.07, 0.16, size=n_years)
    cap, buf, fee = cap_pct / 100.0, buffer_pct / 100.0, fee_pct / 100.0
    payoff = np.where(ref >= 0, np.minimum(ref, cap),
                      np.where(ref >= -buf, 0.0, ref + buf))
    fund = payoff - fee + rng.normal(0.0, 0.002, size=n_years)
    per = pd.period_range("1980", periods=n_years, freq="Y")
    return pd.DataFrame({"ref_ret": ref, "fund_ret": fund},
                        index=per.to_timestamp(how="end").normalize())
