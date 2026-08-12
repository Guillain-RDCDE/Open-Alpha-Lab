"""Data layer for Study 908 — Optimized-Roll Commodities.

The claim under test: a **broad commodity index that rolls naively into the front
month** bleeds a "contango tax" — every month it sells the cheap expiring contract and
buys the more expensive next one up a rising futures curve, a mechanical negative roll
yield. An **optimized-roll** index instead picks the cheapest-to-hold contract along the
curve (and, for the most aggressive version, screens the whole commodity basket for the
ones in backwardation), dodging much of that drag. The structural bet is that the
optimized wrapper delivers a **higher excess-of-cash return / Sharpe** than a front-month
index over the full sample, net of costs — the roll-yield edge, packaged.

We test it on live, liquid US ETFs (yfinance, ``auto_adjust=True`` total-return closes):

  * **USCI** — United States Commodity Index Fund, tracking the *SummerHaven Dynamic
    Commodity Index*: from a 27-commodity universe it holds the **14** with the strongest
    backwardation / price momentum and rolls each into an **optimized** (cheapest-carry)
    contract. The most aggressive "second-generation" optimized-roll wrapper. Inception
    **2010-08-10** — so the common sample starts there.
  * **DBC** — Invesco DB Commodity Index Tracking Fund. A broad 14-commodity index with
    DB's "Optimum Yield" roll rule (picks, within the next 13 months, the contract that
    maximizes implied roll yield). *Semi-optimized* — the study's primary benchmark, and
    an honest middle case: it is NOT a pure front-month roller.
  * **GSG** — iShares S&P GSCI Commodity-Indexed Trust. Production-weighted, ~62% energy,
    and rolls **naively into the front month** — the cleanest *front-month* comparator.
  * **DJP** — iPath Bloomberg Commodity Index Total Return ETN. Broad, front-month roll
    (Bloomberg roll schedule) — a second front-month comparator, less energy-heavy.
  * **PDBC** — Invesco Optimum Yield Diversified Commodity Strategy (the active, K-1-free
    cousin of DBC, same Optimum Yield roll). Inception 2014-11 — a corroborating
    optimized roller on the shorter window.
  * **BIL** — SPDR Bloomberg 1-3 Month T-Bill ETF: the **cash leg**. Every return series
    is taken *excess of BIL* so the race is excess-vs-excess (a commodity index carries a
    big collateral-yield component — at 2023-26 bill rates ~5%/yr — that is NOT a roll
    edge and must be netted out of both sides).

Cache-first: ``fetch`` (network, yfinance) runs once and writes ``_cache/roll_prices.csv``;
everything else is offline. ``AS_OF = 2026-06-30`` (the partial current month is dropped).

Synthetic world: a deterministic monthly generator with a TUNABLE planted roll edge
(knob ``roll_edge_annual``) — a shared commodity spot factor, a front-month index that
eats a contango drag, and an optimized index that recovers ``roll_edge_annual`` of it.
``roll_edge_annual = 0`` is the null (optimized and front-month differ only by noise).
The index is a ``PeriodIndex`` (never ``.to_timestamp`` with a large ``periods`` — the
ns-Timestamp OOB trap).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "roll_prices.csv")

# Total-return closes for every leg, one wide frame.
TICKERS = ["USCI", "DBC", "PDBC", "DJP", "GSG", "BIL"]

CASH = "BIL"                    # the cash leg every return is taken excess of
OPTIMIZED = ["USCI", "PDBC"]    # optimized-roll wrappers (PDBC on the shorter window)
FRONT = ["GSG", "DJP"]          # naive front-month rollers
SEMI = ["DBC"]                  # DB "Optimum Yield" — semi-optimized, the primary benchmark

# The headline race: optimized USCI vs the benchmarks (semi-optimized DBC + front GSG/DJP).
BENCHMARKS = ["DBC", "GSG", "DJP"]

# Expense ratios (fund fact sheets, 2026) — already embedded in the total-return NAV, but
# they are the honest tradability differentiator, so we carry them.
EXPENSE_RATIOS = {"USCI": 1.03, "DBC": 0.85, "PDBC": 0.59, "DJP": 0.70, "GSG": 0.48, "BIL": 0.14}

START = "2010-01-01"           # request start (USCI's own inception 2010-08 gates the sample)
AS_OF = "2026-06-30"           # last complete calendar month at build time

__all__ = [
    "TICKERS", "CASH", "OPTIMIZED", "FRONT", "SEMI", "BENCHMARKS", "EXPENSE_RATIOS",
    "START", "AS_OF", "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "monthly_returns", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for all tickers and cache them (network, run once)."""
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
        raise RuntimeError("yfinance returned no data for the commodity-roll tickers")
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide total-return close frame (one column per ticker), cache-first, sorted."""
    if not os.path.exists(path):
        return fetch(path=path)
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def monthly_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly simple total returns per ticker, sliced to the as-of month-end.

    Month-end sampling of the total-return closes; the partial current month is dropped
    at ``asof``. Columns are the tickers present in ``prices``.
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    m = m[m.index <= pd.Timestamp(asof)]
    return m.pct_change()


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(
    n_months: int = 190,
    roll_edge_annual: float = 0.03,
    seed: int = 908,
    spot_vol_annual: float = 0.16,
    contango_drag_annual: float = 0.05,
    idio_vol_annual: float = 0.02,
    cash_annual: float = 0.02,
) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED optimized-roll edge (the positive control).

    A shared commodity **spot** factor drives both indices. The **front-month** index eats
    a fixed contango drag; the **optimized** index recovers ``roll_edge_annual`` of that
    drag (plus small idiosyncratic tracking noise on each). ``cash`` is a low-vol positive
    (the BIL leg). By construction

        optimized_excess - front_excess  ~  roll_edge_annual/12 + noise

    so the excess-vs-excess Sharpe race must find a positive advantage when
    ``roll_edge_annual > 0`` and **nothing** when it is 0 (the null). Returned columns:
    ``optimized``, ``front``, ``cash`` (monthly simple returns). Index is a ``PeriodIndex``
    (kept as periods — never ``.to_timestamp`` with a large ``periods`` count).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2010-08", periods=n_months, freq="M")

    spot_mu = 0.02 / 12.0                       # a small positive spot drift
    spot = rng.normal(spot_mu, spot_vol_annual / np.sqrt(12.0), n_months)
    drag = contango_drag_annual / 12.0
    edge = roll_edge_annual / 12.0
    idio = idio_vol_annual / np.sqrt(12.0)

    front = spot - drag + rng.normal(0.0, idio, n_months)
    optimized = spot - drag + edge + rng.normal(0.0, idio, n_months)
    cash = np.full(n_months, cash_annual / 12.0) + rng.normal(0.0, 0.0005, n_months)

    return pd.DataFrame(
        {"optimized": optimized, "front": front, "cash": cash}, index=idx
    )
