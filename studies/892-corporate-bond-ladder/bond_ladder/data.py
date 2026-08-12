"""Data layer for Study 892 — Corporate-Bond Ladder.

The claim under test (a perennial retail / advisor talking point): **a held-to-maturity
bond LADDER beats a constant-maturity bond FUND**, because the ladder holds each rung to
par and reinvests the maturing principal at the new (higher) yield, whereas a fund
"forced to sell falling bonds" locks in mark-to-market losses — "the ladder shines
through a rate shock like 2022."

We approximate the two structures with liquid ETFs on total-return closes:

* **Ladder proxy** — a *duration-staggered* Treasury mix drawn from **SHY** (1-3y),
  **IEI** (3-7y), **IEF** (7-10y) and **TLT** (20y+), held with an annual roll. Two
  versions are studied: the naive **equal-weight** SHY/IEI/IEF/TLT basket a retail
  investor actually buys (blended duration ~7.5y), and a **duration-matched** ladder
  whose weights are tuned to the fund's ~6y duration (the fair, apples-to-apples
  control).
* **Constant-maturity fund** — **AGG** (iShares Core US Aggregate) and **BND** (Vanguard
  Total Bond), the flagship one-ticker funds that keep duration roughly constant by
  continuously rolling their holdings.
* **Cash leg** — **BIL** (1-3 month T-bill ETF) for the excess-vs-excess Sharpe race.
  **LQD** (IG corporate) is fetched for a credit-composition cross-check.

**The honest catch, front-loaded.** For a *default-free* bond, held-to-maturity vs
mark-to-market is an **accounting distinction, not an economic one**: the ladder's
"pull to par" is the exact mechanical reversal of the mark-to-market loss the fund
reports, so over a full horizon two portfolios of *equal duration* earn the *same* total
return. Any ladder-vs-fund gap is therefore a **duration** (and, vs AGG, a **credit /
MBS composition**) difference in disguise — which is exactly what the numbers show. And
note the ETF proxy is itself imperfect: SHY/IEI/IEF/TLT are *constant-maturity* funds, so
a basket of them is another constant-maturity portfolio, not a true HTM ladder — a real
ladder needs defined-maturity rungs (iBonds / BulletShares).

Synthetic world: a deterministic monthly generator with a TUNABLE *planted* ladder
premium (knob ``edge_annual``) over a shared duration factor plus idiosyncratic noise —
the positive control (and, at ``edge_annual = 0``, the null that must NOT fire). Index
built with ``period_range`` kept as a PeriodIndex (never ``.to_timestamp`` on a long span
— the pandas ns-Timestamp horizon overflows ~year 2262 on the CI).

Cache-first: ``fetch`` (network, yfinance, ``auto_adjust=True`` total-return) runs once
and writes ``_cache/ladder_prices.parquet``; everything else is offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "ladder_prices.parquet")

# Everything the study needs: Treasury-ladder rungs, the two aggregate funds, an IG-credit
# cross-check, and a T-bill cash leg. yfinance total-return closes (auto_adjust=True).
TICKERS = ["SHY", "IEI", "IEF", "TLT", "AGG", "BND", "LQD", "BIL"]

# Effective durations (years), from fund fact-sheets (iShares / Vanguard, 2025). Coarse and
# slowly time-varying, but good to ~0.3y — enough to duration-match the ladder to the fund.
# Sources cited in docs/references.md.
DURATION = {
    "SHY": 1.9, "IEI": 4.4, "IEF": 7.4, "TLT": 16.5,
    "AGG": 6.0, "BND": 5.9, "LQD": 8.3, "BIL": 0.1,
}

# The two ladder recipes.
#   * EW_LADDER   — the naive equal-weight basket (blended duration ~7.5y > the fund).
#   * DUR_LADDER  — weights tuned so the ladder's duration ~= AGG's ~6.0y (the fair control:
#                   0.165*1.9 + 0.165*4.4 + 0.67*7.4 = 6.00y, no TLT needed).
EW_LADDER = {"SHY": 0.25, "IEI": 0.25, "IEF": 0.25, "TLT": 0.25}
DUR_LADDER = {"SHY": 0.165, "IEI": 0.165, "IEF": 0.67}

FUND = "AGG"      # the constant-maturity comparison fund
CASH = "BIL"      # the risk-free / cash leg for excess-vs-excess

AS_OF = "2026-06-30"  # last complete calendar month at build time (the partial month is dropped)

__all__ = [
    "TICKERS", "DURATION", "EW_LADDER", "DUR_LADDER", "FUND", "CASH", "AS_OF",
    "CACHE_DIR", "PRICES_CACHE",
    "fetch", "have_real", "load_prices", "monthly_returns", "synthetic_world",
    "ladder_duration",
]


def ladder_duration(weights: dict[str, float]) -> float:
    """Blended effective duration (years) of a ladder recipe."""
    return float(sum(DURATION[k] * w for k, w in weights.items()))


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2002-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes (auto_adjust=True) and cache them (network, run once)."""
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
        raise RuntimeError("yfinance returned no data for the ladder tickers.")
    raw = raw[[c for c in TICKERS if c in raw.columns]].dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame, cache-first, sliced to the as-of month-end."""
    if not os.path.exists(path):
        raw = fetch(path=path)
    else:
        raw = pd.read_parquet(path)
    raw = raw.sort_index()
    return raw[raw.index <= pd.Timestamp(asof)]


def monthly_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly simple total returns on the joint (all-present) window.

    Month-end resample of the total-return closes, ``pct_change``, then the leading NaN
    row is dropped and the frame trimmed to rows where every ticker is present (the joint
    window starts when the youngest fund — BND/BIL, 2007 — begins). Sliced to ``asof``.
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    r = m.pct_change().dropna(how="all")
    return r.dropna()


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 228, edge_annual: float = 0.0, seed: int = 892,
                    dur_vol: float = 0.018, idio_vol: float = 0.0028,
                    carry_monthly: float = 0.0025) -> pd.DataFrame:
    """Deterministic monthly world: a ladder and a fund on a shared duration factor.

    Both legs share a common rate-driven return ``f`` (the duration factor) plus carry;
    the ladder additionally earns a PLANTED premium ``edge_annual / 12`` per month:

        fund   = carry + f + eps_fund
        ladder = carry + f + edge_annual/12 + eps_ladder
        cash   = carry * 0.4                       (a low, positive T-bill leg)

    So ``ladder - fund`` has mean ``edge_annual/12`` plus mean-zero noise — exactly the
    quantity the race's HAC t targets. ``edge_annual = 0`` is the null: matched-duration
    ladder and fund are the SAME portfolio up to noise, and the detector must find nothing
    (the real-tape finding). ``edge_annual > 0`` plants a recoverable premium — the
    machinery proof, never market evidence. Decorative monthly index is a plain
    ``RangeIndex`` (no Timestamp horizon to overflow).
    """
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, dur_vol, n_months)                 # shared duration/rate factor
    eps_f = rng.normal(0.0, idio_vol, n_months)
    eps_l = rng.normal(0.0, idio_vol, n_months)
    fund = carry_monthly + f + eps_f
    ladder = carry_monthly + f + edge_annual / 12.0 + eps_l
    cash = np.full(n_months, carry_monthly * 0.4)
    return pd.DataFrame(
        {"ladder": ladder, "fund": fund, "cash": cash},
        index=pd.RangeIndex(n_months, name="month"),
    )
