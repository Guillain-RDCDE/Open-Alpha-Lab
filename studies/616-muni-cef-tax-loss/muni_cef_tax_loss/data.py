"""Data layer for Study 616 — Muni-CEF-Tax-Loss ("Muni CEF Tax-Loss Season").

Two sources, both offline-friendly once cached:

* **Real tape.** Daily adjusted (total-return) closes for a fixed panel of **12 seasoned
  municipal-bond closed-end funds** across five sponsors (Nuveen, BlackRock, Eaton Vance,
  Invesco, Western Asset) plus two benchmarks: **MUB** (iShares National Muni Bond ETF — the
  headline benchmark, an open-end vehicle that trades at NAV, from 2007-09) and **VWLTX**
  (Vanguard Long-Term Tax-Exempt mutual fund — the pre-MUB robustness benchmark, NAV-priced,
  full 2000→ history). Everything cached wide under ``_cache/mct_prices.csv``.

  Because a muni CEF's NAV is a levered muni-bond portfolio while MUB/VWLTX **price at NAV**,
  the fund-minus-benchmark **excess total return is a transparent proxy for discount motion**:
  when holders dump the CEF for a tax loss its *price* falls relative to the muni market
  (discount widens → negative excess), and the January snap-back is the reverse. Same proxy
  logic as study 367 (closed-end-fund-discount), which tests the *level* edge; this study
  tests the *seasonal flow*.

* **Synthetic.** A deterministic, fixed-seed generator producing a monthly fund-minus-benchmark
  excess-return panel with **plantable knobs**: a December ``dump`` (mean excess shifted down in
  December) and a January ``snap`` (shifted up in January). With both knobs at 0 the seasonal
  machinery must NOT manufacture significance; with large planted knobs it must light up. The
  monthly index uses ``period_range`` (spans far below the 250-year ns-Timestamp limit).

Pure numpy + pandas + stdlib for the offline path. ``fetch_panel`` (network) is used only once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "mct_prices.csv")

# Fixed panel: 12 seasoned national muni CEFs still trading in 2026, spread across sponsors.
# This is a *survivors* panel (merged/liquidated muni CEFs cannot appear) — named on the
# Signal axis. NEA/NVG/NZF/NUV/NXP = Nuveen; BTT/MYI/MQY/MHD = BlackRock (BTT is a 2030
# target-term fund, listed 2012); EIM = Eaton Vance; VMO = Invesco; MMU = Western Asset.
FUNDS = ["NEA", "NVG", "NZF", "NUV", "NXP", "BTT", "MYI", "MQY", "MHD", "EIM", "VMO", "MMU"]
BENCH = "MUB"          # headline benchmark (NAV-priced muni ETF, from 2007-09)
ALT_BENCH = "VWLTX"    # robustness benchmark (NAV-priced muni mutual fund, full history)

# Frozen as-of: the sample is sliced here so published numbers never creep. June 2026 is the
# last complete calendar month at the study's as-of date (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2000-01-01", end: str | None = None,
                path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the fund + benchmark adjusted closes and cache them (network, run once)."""
    import yfinance as yf

    tickers = FUNDS + [BENCH, ALT_BENCH]
    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = PRICES_CACHE, asof: str | None = AS_OF) -> pd.DataFrame:
    """Cached wide adjusted-close frame, sliced to the frozen as-of."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if asof is not None:
        px = px[px.index <= pd.Timestamp(asof)]
    return px


# --------------------------------------------------------------------------- #
# Monthly total returns + fund-minus-benchmark excess (the discount-motion proxy)
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple total returns (month-end to month-end).

    Drops the final calendar month if the tape stops before that month genuinely ended, so a
    stamped run never contains a partial month.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret.iloc[1:]


def monthly_excess(prices: pd.DataFrame, bench: str = BENCH,
                   funds: list[str] | None = None) -> pd.DataFrame:
    """Monthly fund total return minus benchmark total return, per fund.

    Both legs are total-return (auto-adjusted closes), so the monthly distributions the CEFs
    pay do not pollute the seasonal comparison. Because the benchmark prices at NAV, this
    excess is the study's transparent **discount-motion proxy**: negative = the CEF's price
    lagged the muni market that month (discount widening), positive = snap-back.
    """
    funds = funds or [f for f in FUNDS if f in prices.columns]
    mret = monthly_returns(prices)
    b = mret[bench]
    out = {}
    for f in funds:
        if f not in mret.columns:
            continue
        e = mret[f] - b
        out[f] = e
    return pd.DataFrame(out).dropna(how="all")


# --------------------------------------------------------------------------- #
# Synthetic control — plantable December dump / January snap-back
# --------------------------------------------------------------------------- #
def synthetic_excess(n_funds: int = 12, n_years: int = 19, dump: float = 0.0,
                     snap: float = 0.0, seed: int = 616,
                     sig_monthly: float = 0.020) -> pd.DataFrame:
    """Deterministic monthly fund-minus-benchmark excess panel with PLANTED seasonal knobs.

    Every fund's monthly excess is i.i.d. noise around zero, plus ``-dump`` in every December
    and ``+snap`` in every January. With ``dump = snap = 0`` the panel is seasonally dead: the
    winter snap-back statistic must NOT reach significance however the noise falls. With large
    planted knobs it must light up. Fund-level noise carries a common (cross-fund) component so
    the panel mimics the real tape's one-effective-observation-per-winter correlation.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2008-01", periods=n_years * 12, freq="M")
    idx = pidx.to_timestamp(how="end").normalize()
    months = np.array([p.month for p in pidx])
    common = rng.normal(0.0, sig_monthly * 0.7, size=len(pidx))     # shared muni-CEF factor
    cols = {}
    for j in range(n_funds):
        own = rng.normal(0.0, sig_monthly * 0.7, size=len(pidx))
        e = common + own
        e[months == 12] -= dump
        e[months == 1] += snap
        cols[f"F{j:02d}"] = e
    return pd.DataFrame(cols, index=idx)
