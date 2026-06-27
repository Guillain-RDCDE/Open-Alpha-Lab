"""Data layer for Study 538 (Industry-Relative-Reversal).

The Hameed-Mian (2015) / Da-Liu-Schaumburg (2014) refinement of the Jegadeesh (1990)
short-horizon reversal: the reversal lives in the **within-industry** component of last
month's return. Decompose a stock's month-t return into

    r[i,t] = industry_mean[g(i),t] + (r[i,t] - industry_mean[g(i),t])
             \\_______ across ______/   \\______ within (industry-relative) _____/

and the claim is that next month it is the *within* part that reverses; the *across*
(industry) part does not. The raw one-month reversal of Study 329 mixes both, which is
why -- the literature argues -- it is weaker and noisier than the industry-adjusted form.

Two tapes, one shape (a month x ticker panel of monthly closes), plus a fixed GICS sector
map so we can demean within industry:

- ``synthetic_panel`` -- a *deterministic, offline* generator with a tunable **within-
  industry** reversal knob. It plants a common industry factor (a real across-industry
  move that does NOT reverse) plus an idiosyncratic, industry-relative one-month reversal.
  ``within_reversal = 0`` is the null. This lets the synthetic control prove the engine
  isolates the within component from the across component.

- ``load_real`` / ``fetch_monthly`` -- the real Yahoo! monthly-close panel for a fixed
  ~52-name large-cap survivor basket spanning six GICS sectors, cache-first. The headline
  run reuses the **shared S&P 500 monthly panel** built by Study 196
  (``../../196-long-term-reversal/_cache/ltr_monthly.parquet``) when present, sliced to the
  basket, and falls back to this study's own ``_cache/irr_monthly.parquet``.

**Survivorship-bias caveat**: the basket is firms *still in the S&P 500 in 2026*. Every
name survived, so positive real-tape results are upper bounds. Named on the Signal axis.

**Industry-map caveat**: the GICS sector tags are the firms' *current* sector. Sector
membership is far stickier than price, so backfilling today's sector is a mild
approximation (a handful of names re-classified over 36 years); it does not manufacture a
reversal -- demeaning by the wrong group can only *weaken* the within signal.

No look-ahead lives here -- the signal for holding month t is built from month t-1's
return (known at the close of t-1); the discipline is enforced in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# This study's own cache target (populated by examples/verify.py --fetch).
MONTHLY_CACHE = os.path.join(DEFAULT_CACHE, "irr_monthly.parquet")

# The shared S&P 500 monthly panel built by Study 196 -- reused cache-first so the
# headline run does not duplicate a download.
SHARED_LTR_CACHE = os.path.abspath(
    os.path.join(REPO_ROOT, "studies", "196-long-term-reversal", "_cache", "ltr_monthly.parquet")
)

# Last fully-complete month pinned for the headline run (partial months are dropped).
AS_OF_LAST_MONTH = "2026-05-31"

# ---------------------------------------------------------------------------
# Fixed survivor basket with GICS sector tags -- the industry map.
# Six sectors, ~8-14 names each, so within-industry demeaning is well-conditioned.
# All names are current S&P 500 members present in the shared monthly panel from 1990.
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, str] = {
    # Information Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "CSCO": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "TXN": "Tech", "QCOM": "Tech",
    "INTC": "Tech", "AMD": "Tech", "ACN": "Tech", "IBM": "Tech",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "AXP": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "USB": "Financials",
    # Health Care
    "JNJ": "Health", "UNH": "Health", "PFE": "Health", "MRK": "Health", "ABBV": "Health",
    "ABT": "Health", "TMO": "Health", "LLY": "Health", "BMY": "Health", "AMGN": "Health",
    "MDT": "Health", "GILD": "Health",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy", "EOG": "Energy",
    "PSX": "Energy",
    # Consumer Staples
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "WMT": "Staples", "COST": "Staples",
    "TGT": "Staples",
    # Consumer Discretionary
    "MCD": "Discretionary", "HD": "Discretionary", "NKE": "Discretionary",
    "SBUX": "Discretionary", "LOW": "Discretionary", "DIS": "Discretionary",
}

BASKET: list[str] = list(SECTOR_MAP.keys())


def sector_series(columns) -> pd.Series:
    """A ticker -> sector Series aligned to the given column index (NaN if unmapped)."""
    return pd.Series({c: SECTOR_MAP.get(c, np.nan) for c in columns}, name="sector")


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_per_sector: int = 12,
    n_sectors: int = 6,
    n_months: int = 300,
    within_reversal: float = 0.06,
    industry_vol: float = 0.05,
    idio_vol: float = 0.06,
    base_ret: float = 0.008,
    seed: int = 538,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A firm x month panel with a known *within-industry* one-month reversal.

    Construction (the heart of the Hameed-Mian decomposition):

    - Each month, every sector ``g`` draws a **common industry move** ``I[g,t]`` (vol
      ``industry_vol``). This is the *across-industry* component -- a real return that
      does **not** reverse (it is an i.i.d. industry factor).
    - Each firm's *idiosyncratic* (industry-relative) return carries a planted **within-
      industry reversal**: next month's idiosyncratic return loads negatively on this
      month's idiosyncratic return, by ``within_reversal``. Losers *relative to their own
      sector* bounce; sector-wide moves do not.
    - The realised firm return is ``base_ret + I[g,t] + idio[i,t]``.

    So a raw one-month reversal (which sorts on total return) is *diluted* by the non-
    reversing industry component, while an industry-relative reversal (which sorts on
    ``idio``) sees the planted signal cleanly. ``within_reversal = 0`` is the null.

    Returns ``(price_df, sectors, truth)`` where ``price_df`` is a (month x firm) price
    frame (base 100) and ``sectors`` is the firm -> sector map.

    ``pd.period_range`` is used for the decorative monthly index so we never risk a
    ``date_range(periods=BIG)`` timestamp overflow on CI pandas.
    """
    rng = np.random.default_rng(seed)
    months = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp(how="end").normalize()

    sectors_list = [f"S{g}" for g in range(n_sectors)]
    firms: list[str] = []
    firm_sector: dict[str, str] = {}
    for g, sname in enumerate(sectors_list):
        for k in range(n_per_sector):
            f = f"{sname}_F{k:02d}"
            firms.append(f)
            firm_sector[f] = sname
    n_firms = len(firms)

    # Sector index for each firm column (to broadcast the industry move).
    sec_idx = np.array([sectors_list.index(firm_sector[f]) for f in firms])

    # Industry common move per (month, sector), broadcast to firms.
    ind_move_sec = rng.normal(0.0, industry_vol, (n_months, n_sectors))  # across component
    ind_move = ind_move_sec[:, sec_idx]  # (n_months, n_firms)

    # Idiosyncratic returns carry the planted within-industry reversal.
    idio_noise = rng.normal(0.0, idio_vol, (n_months, n_firms))
    idio = np.zeros_like(idio_noise)
    idio[0] = idio_noise[0]
    for t in range(1, n_months):
        prev = idio[t - 1]
        # cross-sectional z of the NEGATED prior idiosyncratic return -> losers ranked high
        neg = -prev
        z = (neg - neg.mean()) / (neg.std() + 1e-9)
        idio[t] = idio_noise[t] + within_reversal * z

    total_ret = base_ret + ind_move + idio  # (n_months, n_firms)
    cum_log = np.cumsum(total_ret, axis=0)
    price_arr = 100.0 * np.exp(cum_log)
    price_df = pd.DataFrame(price_arr, index=months, columns=firms)
    sectors = pd.Series(firm_sector, name="sector")

    truth = {
        "within_reversal": within_reversal,
        "industry_vol": industry_vol,
        "idio_vol": idio_vol,
        "n_firms": n_firms,
        "n_sectors": n_sectors,
        "n_months": n_months,
        "seed": seed,
    }
    return price_df, sectors, truth


# ---------------------------------------------------------------------------
# Real tape -- cache-first monthly closes for the fixed sector basket
# ---------------------------------------------------------------------------
def load_real(cache_path: str | None = None) -> pd.DataFrame:
    """Cache-first monthly close panel sliced to the fixed sector basket; offline.

    Resolution order (graceful fallback):
      1. an explicit ``cache_path`` if given and present,
      2. this study's own ``_cache/irr_monthly.parquet``,
      3. the shared Study-196 panel ``ltr_monthly.parquet``, sliced to ``BASKET``.

    The frame is trimmed to the last fully-complete month (``AS_OF_LAST_MONTH``) so the
    headline run never includes a partial month.

    **Survivorship bias**: the basket is current S&P 500 names projected backwards. All
    positive results are upper bounds.
    """
    candidates = [c for c in (cache_path, MONTHLY_CACHE, SHARED_LTR_CACHE) if c]
    for path in candidates:
        if os.path.exists(path):
            prices = pd.read_parquet(path)
            cols = [t for t in BASKET if t in prices.columns]
            prices = prices[cols]
            prices = prices.loc[:AS_OF_LAST_MONTH]
            if prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            return prices.dropna(how="all")
    raise FileNotFoundError(
        "No cached monthly panel found. Tried: "
        + ", ".join(candidates)
        + ". Run examples/verify.py --fetch once, or rely on the synthetic core."
    )


def fetch_monthly(
    fetch: bool = False,
    cache_path: str = MONTHLY_CACHE,
) -> pd.DataFrame:
    """Monthly adjusted-close panel for the fixed sector basket; cache-only unless fetch.

    With ``fetch=False`` this is a thin wrapper over :func:`load_real` (cache-first,
    offline). With ``fetch=True`` it goes to ``yfinance`` for the fixed ``BASKET`` and
    caches the result. Columns are tickers; index is month-end dates.

    **Survivorship bias**: the basket is current S&P 500 names projected backwards.
    """
    if not fetch:
        return load_real(cache_path=cache_path)

    import time

    import yfinance as yf  # lazy: network only on fetch=True

    raw = None
    for attempt in range(3):
        try:
            raw = yf.download(
                BASKET, start="1990-01-01", interval="1mo",
                auto_adjust=True, progress=False, group_by="column",
            )
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # transient yfinance flakiness
            print(f"  fetch attempt {attempt + 1} failed: {exc}")
        time.sleep(2.0 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no monthly data after retries")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = (
            raw["Close"] if "Close" in raw.columns.get_level_values(0)
            else raw.xs("Close", axis=1, level=0)
        )
    else:
        closes = raw
    closes.index = pd.to_datetime(closes.index) + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()
    closes = closes.loc[:, [t for t in BASKET if t in closes.columns]]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached monthly panel: {closes.shape[0]} months x {closes.shape[1]} tickers -> {cache_path}")
    return closes.loc[:AS_OF_LAST_MONTH]


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a prices frame (last row), for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]
