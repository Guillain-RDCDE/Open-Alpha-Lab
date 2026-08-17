"""Data layer for Study 960 — The Unstaked ETF (US spot ETH funds vs the coin).

Two tapes, one shape (a date-indexed daily close frame):

- ``fetch`` / ``load_prices`` — daily closes from Yahoo! Finance (``yfinance``,
  ``auto_adjust=True``) for the two US spot-Ethereum lines with usable history
  (**ETHA**, iShares Ethereum Trust; **ETHE**, Grayscale Ethereum Trust), the coin
  itself (**ETH-USD**) and a cash proxy (**BIL**, the 1-3 month T-bill ETF).
  ``fetch`` touches the network and caches parquet under the desk-wide
  ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache **offline**
  and never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic only), so CI is green on a fresh checkout.

- ``synthetic_panel`` — a *deterministic, offline* generator that rebuilds the
  measurement problem in miniature: one true coin path, a *mis-timed* observed coin
  close (the 24/7-vs-16:00 mismatch), and two fund NAV paths that bleed a planted
  drag and carry premium/discount noise. The ``signal_strength`` knob scales the
  *difference* between the two funds' drags: at 1 the cheap and dear wrappers are
  separated by the planted spread, at 0 they are identical (the null).

**Price-only vs total-return.** ETHA and ETHE make no distributions, so their
``auto_adjust`` close is both the price and the total return of the wrapper. ETH-USD
is a **price-only** series by construction — it is the market price of an *unstaked*
coin and contains no consensus reward. BIL's close is a genuine total return. Those
three labels do most of the analytical work in this study and are repeated wherever a
number is quoted.

**Non-tape inputs.** Sponsor fees and the Ethereum consensus reward rate are *not*
observed here; they are documented external constants, declared below as
``EXPENSE_RATIOS`` and ``STAKING_YIELD_ANN`` and swept in ``strategy.staking_sweep``.
They are ASSUMPTIONS, labelled as such everywhere they enter a number.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The desk-wide shared cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# ETHA / ETHE = the two US spot-ETH lines with a usable ETF-era history; ETH-USD =
# the coin (price-only, unstaked); BIL = the cash leg for the excess-of-cash race.
TICKERS = ("ETHA", "ETHE", "ETH-USD", "BIL")

FUNDS = ("ETHA", "ETHE")
COIN = "ETH-USD"
CASH = "BIL"

# COVERAGE, stated because the study's headline number depends on it. The US spot-ETH
# cohort that began trading together on 2024-07-23 has roughly nine lines (ETHA, FETH,
# ETHW, ETHV, QETH, EZET, CETH, the Grayscale mini trust, and ETHE). This study measures
# **two** of them — the two present in the desk-wide cache — and they are the *extremes*
# of the cohort's published fee schedule: the cheapest tier (ETHA, 0.25%/yr headline,
# partly waived) against the dearest line by a wide margin (ETHE, 2.50%/yr). That is not
# selection on an outcome (fees are published in advance and were never re-ranked after
# seeing a result), but it IS the widest fee pair available, so the measured 1.6%/yr
# spread is the cohort's upper bound and not a typical fund-to-fund number. Study 959
# runs the full ten-line version of this exercise on bitcoin; this ETH tape does not have
# that coverage here.
COHORT_NOT_MEASURED = ("FETH", "ETHW", "ETHV", "QETH", "EZET", "CETH")

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# The US spot-ETH funds began trading on 2024-07-23. Everything before that date is a
# different animal (ETHE was a closed-end trust swinging between a +100% premium and a
# -55% discount — that is Study 618's subject, not this one), so the ETF-era window
# starts here and is never widened.
ETF_ERA_START = "2024-07-23"

# --------------------------------------------------------------------------- #
# NON-TAPE CONSTANTS — declared, sourced, swept. None of these is measured here.
# --------------------------------------------------------------------------- #
# ASSUMPTION (documented sponsor fees, prospectus/issuer pages). ETHA's headline
# sponsor fee is 0.25%/yr; the issuer waived part of it early in the fund's life, so
# the *effective* fee over our window is somewhere in [0.12%, 0.25%] — swept.
EXPENSE_RATIOS = {"ETHA": 0.0025, "ETHE": 0.0250}
ETHA_FEE_RANGE = (0.0012, 0.0025)

# ASSUMPTION (external, not tape): the Ethereum consensus reward rate a *staked* coin
# earns, net of the validator/provider commission a regulated fund would pay. The
# network-average gross rate drifted roughly 3-4%/yr over the sample; a fund-level net
# pass-through in the high 2s to low 3s is the central case. Swept, always.
STAKING_YIELD_ANN = 0.030
STAKING_SWEEP = (0.020, 0.025, 0.030, 0.035, 0.045)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2019-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted. The ETH funds distribute
    nothing, so for them the adjustment is a no-op; for BIL it is the whole point.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    start: str | None = ETF_ERA_START,
) -> pd.DataFrame:
    """Read cached daily closes OFFLINE into one aligned frame, ETF-era only.

    Rows are the intersection of all four tapes (so every comparison is like-for-like
    on the same calendar), sliced to ``[start, asof]``. The coin trades on weekends;
    the funds do not, so the intersection is the NYSE session calendar. Raises
    ``FileNotFoundError`` if any ticker is missing — the offline core and the whole
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call unstaked.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    return df.dropna()


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 2,
    vol_ann: float = 0.70,            # ETH-scale annualised vol
    drift_ann: float = -0.10,         # the sample happened to be a down market
    drag_cheap: float = 0.0025,       # planted annual drag of the cheap wrapper
    drag_dear: float = 0.0250,        # planted annual drag of the dear wrapper
    signal_strength: float = 1.0,     # 1 = full planted spread, 0 = identical wrappers
    misalign_vol_ann: float = 0.30,   # the 24/7-vs-16:00 observation mismatch
    prem_disc_bps: float = 8.0,       # daily premium/discount noise on each fund, bps
    start: str = "2024-07-23",
    seed: int = 960,
    cash_rate_ann: float = 0.045,
) -> tuple[pd.DataFrame, dict]:
    """A miniature of the measurement problem: one coin, two wrappers, mis-timed closes.

    The world contains a single true coin path. Three things are then *observed*:

    - ``coin`` — the coin's close as a data vendor stamps it, which is taken a few
      hours after the fund close. The offset injects a transient, strongly
      mean-reverting error into any fund-minus-coin daily return difference: exactly
      the effect that makes the real fund-vs-coin tracking difference so imprecise.
    - ``cheap`` / ``dear`` — two fund NAV paths that compound the *true* coin return
      minus their own planted annual drag, plus i.i.d. premium/discount noise. Because
      both are stamped at the fund close, the mis-timing cancels in ``dear - cheap``.
    - ``cash`` — a cash accrual index for the excess-of-cash race.

    ``signal_strength`` scales the *difference* between the two drags around their
    mean: at 1 the planted spread is ``drag_dear - drag_cheap``; at 0 both wrappers
    carry the same drag and the fund-vs-fund estimator must read ~0 (the null).

    Returns ``(prices, truth)``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # bdate_range with a modest count stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mid = 0.5 * (drag_cheap + drag_dear)
    half = 0.5 * (drag_dear - drag_cheap) * signal_strength
    d_cheap = mid - half
    d_dear = mid + half

    s = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mu = drift_ann / TRADING_DAYS_PER_YEAR - 0.5 * s * s
    true_ret = rng.normal(mu, s, n_days)
    true_level = 100.0 * np.exp(np.cumsum(true_ret))

    # The vendor's coin close is the true level times a transient timing error: it
    # reverses the next day, so it telescopes out of a long-window comparison but
    # blows up the day-by-day variance.
    m = misalign_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    misalign = rng.normal(0.0, m, n_days)
    coin_obs = true_level * np.exp(misalign)

    pd_noise_cheap = rng.normal(0.0, prem_disc_bps * 1e-4, n_days)
    pd_noise_dear = rng.normal(0.0, prem_disc_bps * 1e-4, n_days)
    nav_cheap = true_level * np.exp(-d_cheap / TRADING_DAYS_PER_YEAR * np.arange(n_days))
    nav_dear = true_level * np.exp(-d_dear / TRADING_DAYS_PER_YEAR * np.arange(n_days))
    cheap_obs = nav_cheap * np.exp(pd_noise_cheap)
    dear_obs = nav_dear * np.exp(pd_noise_dear)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"coin": coin_obs, "cheap": cheap_obs, "dear": dear_obs,
         "coin_true": true_level, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "drag_cheap_planted": float(d_cheap),
        "drag_dear_planted": float(d_dear),
        "spread_planted_pct": float((d_dear - d_cheap) * 100.0),
        "vol_ann": vol_ann,
        "misalign_vol_ann": misalign_vol_ann,
        "prem_disc_bps": prem_disc_bps,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Alias of :func:`synthetic_panel` for the desk's usual generator name."""
    return synthetic_panel(**kwargs)
