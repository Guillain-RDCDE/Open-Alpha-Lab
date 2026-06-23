"""Data layer for Study 367 — Closed-End Fund Discount (CEF price + NAV proxy).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a basket of long-listed US equity
  closed-end funds (CEFs) plus the benchmark ETFs that stand in for each fund's
  mandate (yfinance, no key), cached under ``_cache/cef_prices.csv`` (a wide CSV
  indexed by date, one column per ticker). True per-fund daily **NAV** is not on
  yfinance, so we build a **transparent NAV proxy**: each CEF is mapped to a published
  benchmark (``BENCH_MAP``); the proxy *discount* is the fund's price measured relative
  to that benchmark, demeaned per fund (a fund whose price has fallen relative to its
  mandate is "at a discount" in proxy terms). This is an explicit, labelled proxy of the
  same object the literature uses (price / NAV − 1).

* **Synthetic.** A deterministic, fixed-seed generator that builds a panel of funds whose
  log-discount is a *mean-reverting* (AR(1)) process around zero, with a controllable
  **planted edge**: when ``edge`` > 0 the next-month fund return carries a component
  proportional to (−discount), i.e. wide-discount funds really do out-earn. With
  ``edge = 0`` the discount carries no return information and the test must NOT manufacture
  significance; with a large planted edge the long-wide minus long-narrow spread must light
  up. The offline core runs with no network.

Pure numpy + pandas + stdlib for the offline path. ``fetch_cefs`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "cef_prices.csv")

# A transparent, fixed basket of long-listed US equity closed-end funds mapped to the
# published benchmark ETF that stands in for each fund's mandate. This mapping is the
# *proxy NAV*: a utilities CEF tracks XLU, an S&P-500 CEF tracks SPY, an energy CEF tracks
# XLE, etc. (Survivorship is acknowledged on the Signal axis: a fixed surviving-CEF basket
# is discounts-of-the-survivors, a mild bias we name in writing.)
BENCH_MAP = {
    "ADX": "SPY", "GAM": "SPY", "CET": "SPY", "USA": "SPY", "GGT": "SPY",
    "SOR": "SPY", "CSQ": "SPY", "EOS": "SPY", "ETV": "SPY", "CGO": "SPY",
    "GAB": "SPY", "RVT": "IWM", "FUND": "IWM", "PEO": "XLE", "HQH": "XLV",
    "BME": "XLV", "BST": "QQQ", "UTF": "XLU",
}
BENCHMARKS = sorted(set(BENCH_MAP.values()))
CEFS = sorted(BENCH_MAP)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_cefs(start: str = "1995-01-01", end: str | None = None,
               path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the CEF basket + benchmark ETFs via yfinance and cache a wide CSV.

    Network-only; used once to build ``_cache/cef_prices.csv``. Never imported by the
    offline notebook cells. Keeps columns with a long, mostly-complete history.
    """
    import yfinance as yf

    tickers = sorted(set(CEFS) | set(BENCHMARKS))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.30]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def proxy_discount(prices: pd.DataFrame, bench_map: dict | None = None) -> pd.DataFrame:
    """Build the per-fund **proxy discount** panel from prices.

    For each CEF with a benchmark in ``bench_map`` present in ``prices``::

        disc_t = log(price_t) − log(bench_t) − mean_t( log(price) − log(bench) )

    i.e. the fund's price *relative to its benchmark*, demeaned over the fund's life so the
    series is centred on its own average (the absolute price/NAV anchor is unobservable; the
    *time-variation* of the discount — what mean-reverts — is what we measure). A negative
    ``disc_t`` means the fund is cheap *relative to its mandate* right now: a proxy discount.

    Returns a frame indexed by date with one column per usable CEF (NaN before the fund or
    its benchmark exists). This is a PROXY for the true price/NAV discount, named as such.
    """
    bench_map = bench_map or BENCH_MAP
    out = {}
    for cef, bench in bench_map.items():
        if cef not in prices.columns or bench not in prices.columns:
            continue
        sub = prices[[cef, bench]].dropna()
        if len(sub) < 252:
            continue
        d = np.log(sub[cef]) - np.log(sub[bench])
        out[cef] = (d - d.mean()).reindex(prices.index)
    return pd.DataFrame(out, index=prices.index)


def cef_prices(prices: pd.DataFrame, bench_map: dict | None = None) -> pd.DataFrame:
    """The CEF price columns that have a usable benchmark (for forward returns)."""
    bench_map = bench_map or BENCH_MAP
    cols = [c for c in bench_map if c in prices.columns
            and bench_map[c] in prices.columns]
    return prices[cols].copy()


def load_real(path: str = DEFAULT_CACHE) -> dict:
    """Convenience: cached prices -> {'disc': proxy-discount panel, 'px': CEF prices}."""
    px = load_prices(path)
    return {"disc": proxy_discount(px), "px": cef_prices(px)}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_funds: int = 18, n_months: int = 360, edge: float = 0.0,
                    seed: int = 367, rho: float = 0.92, disc_sig: float = 0.06,
                    ret_sig: float = 0.05) -> dict:
    """Deterministic CEF panel with a mean-reverting log-discount and a PLANTED edge.

    For each fund the log-discount ``d`` follows an AR(1)::

        d_{t+1} = rho * d_t + N(0, disc_sig)

    so the discount mean-reverts toward zero (the literature's stylised fact). The monthly
    fund return is a market component plus idiosyncratic noise plus an *edge* term::

        ret_{t+1} = mkt_{t+1} + N(0, ret_sig) + edge * (−d_t)

    When ``edge`` = 0 the discount carries **no** forward-return information: a long-wide /
    short-narrow sort must NOT find significance however the noise falls (the whole
    null-discipline test). When ``edge`` > 0 wide-discount funds (large −d) really do
    out-earn next month, and the sort must light up in proportion. Returns a dict with the
    monthly ``disc`` panel and the monthly ``ret`` panel (both funds × months), plus a
    decorative ``period`` index (``pd.period_range`` — no OutOfBounds for long spans).
    """
    rng = np.random.default_rng(seed)
    d = np.zeros((n_months, n_funds))
    d[0] = rng.normal(0, disc_sig / np.sqrt(1 - rho ** 2), size=n_funds)
    eps = rng.normal(0, disc_sig, size=(n_months, n_funds))
    for t in range(1, n_months):
        d[t] = rho * d[t - 1] + eps[t]

    mkt = rng.normal(0.006, 0.04, size=n_months)              # common market factor
    idio = rng.normal(0, ret_sig, size=(n_months, n_funds))
    ret = mkt[:, None] + idio
    # planted edge: wide discount (−d) predicts next-month return
    ret[1:] += edge * (-d[:-1])

    idx = pd.period_range("1996-01", periods=n_months, freq="M")
    cols = [f"F{i:02d}" for i in range(n_funds)]
    disc = pd.DataFrame(d, index=idx, columns=cols)
    rets = pd.DataFrame(ret, index=idx, columns=cols)
    return {"disc": disc, "ret": rets}
