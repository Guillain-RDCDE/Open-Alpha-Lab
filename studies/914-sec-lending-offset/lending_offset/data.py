"""Data layer for Study 914 — Securities-Lending Offset.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for five same-asset-class ETF pairs plus a cash
  proxy. ``fetch`` touches the network and caches parquet under the shared
  ``studies/_cache/`` (retry up to 4x); ``load_prices`` reads that cache **offline** and
  never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic-only), so CI is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  common index path is shared by both funds of a pair; each fund then subtracts its own
  fee accrual, adds its own (planted) lending-revenue accrual, and adds an idiosyncratic
  *composition wedge* — the return difference caused by tracking a slightly different
  basket. The ``signal_strength`` knob scales the **lending differential**: at
  ``signal_strength=0`` the two funds differ only by fees and composition noise (the null
  — the detector must be quiet); at ``signal_strength=1`` a real, planted passthrough
  differential exists and must be recovered. Seeds are fixed.

The claim under test: lending-heavy asset classes (small caps, emerging markets) earn
securities-lending revenue that partly or wholly **offsets the expense ratio**, so a
fund's realised tracking difference should be *better* than ``-ER``.

The honest measurement constraint, stated up front: **fund-level lending revenue is not in
the tape**, and neither is an index total-return series (Yahoo's index quotes are
price-only, and a licensed net-of-withholding index TR is not free). So this study cannot
measure a *level* tracking difference against its own benchmark. What it can measure is a
**relative** tracking difference between two funds in the same asset class, and ask
whether the realised return gap is explained by the fee gap alone. Everything downstream
is that one relative quantity.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# The five funds' peers plus a cash proxy (BIL) used only to report each leg's
# excess-of-cash Sharpe for context.
TICKERS = ("SPY", "IVV", "IWM", "IJR", "EEM", "IEMG", "EFA", "IEFA", "VEA", "BIL")

# --------------------------------------------------------------------------- #
# PROXY / ASSUMPTION — published net expense ratios, in percent per year.
# These are NOT tape. They are the sponsors' current published net ERs, held
# CONSTANT through history. Almost every fund here was cut during the sample, and the
# CHEAP legs were cut hardest (IVV 0.0945 -> 0.03, IEMG 0.18 -> 0.09, IEFA 0.14 ->
# 0.07, IJR 0.20 -> 0.06) while the dear legs barely moved (EEM 0.75 -> 0.70,
# IWM 0.20 -> 0.19, SPY 0.0945 unchanged). Holding today's values constant therefore
# OVERSTATES every fee gap and biases every residual UPWARD. See ``ER_HISTORY_BAND``
# below: the whole table is re-run under launch-era and time-averaged fees, and the
# SIGN of the tight-pair residuals does not survive it.
# --------------------------------------------------------------------------- #
EXPENSE_RATIOS = {
    "SPY": 0.0945,
    "IVV": 0.03,
    "IWM": 0.19,
    "IJR": 0.06,
    "EEM": 0.70,
    "IEMG": 0.09,
    "EFA": 0.33,
    "IEFA": 0.07,
    "VEA": 0.03,
    "BIL": 0.1356,
}

# Alternative ER values to sweep on the DEAR leg (the historical highs sponsors charged).
ER_ALTERNATIVES = {
    "EEM": (0.68, 0.70, 0.72, 0.75),
    "IWM": (0.19, 0.20),
    "EFA": (0.32, 0.33, 0.35),
    "SPY": (0.0945,),
}

# --------------------------------------------------------------------------- #
# PROXY / ASSUMPTION — and the one that decides this study's sign.
#
# Every CHEAP leg on this desk had its fee **cut** during the sample; none was ever
# raised. So holding TODAY's ER constant back to launch UNDERSTATES the cheap leg's
# historical fee, OVERSTATES the fee gap, and biases the residual UPWARD — in exactly
# the direction that makes a lending offset look absent. Sweeping only the dear leg
# (``ER_ALTERNATIVES`` above) is therefore a one-sided sweep, and the study refuses to
# rest its sign on it.
#
# ``ER_HISTORY_BAND[t] = (earliest in-sample, approx time-weighted mid, current)``, in %
# per year. The two endpoints are the sponsors' published headline ERs at the start of this
# study's window and today; the middle value is an APPROXIMATE time-weighting of the
# published cut schedule — NOT tape and not precise. It is used only as a bracketing
# scenario, never as the headline, and the endpoints bound it, so no conclusion depends on
# the middle number being exactly right.
# --------------------------------------------------------------------------- #
ER_HISTORY_BAND = {
    "SPY": (0.1200, 0.1000, 0.0945),
    "IVV": (0.0945, 0.0650, 0.0300),
    "IWM": (0.2000, 0.1950, 0.1900),
    "IJR": (0.2000, 0.1500, 0.0600),
    "EEM": (0.7500, 0.7100, 0.7000),
    "IEMG": (0.1800, 0.1300, 0.0900),
    "EFA": (0.3500, 0.3350, 0.3300),
    "IEFA": (0.1400, 0.0900, 0.0700),
    "VEA": (0.1200, 0.0600, 0.0300),
    "BIL": (0.1356, 0.1356, 0.1356),
}

# Scenario dicts drawn from the band, for a like-for-like re-run of the whole table.
EXPENSE_RATIOS_EARLY = {k: v[0] for k, v in ER_HISTORY_BAND.items()}
EXPENSE_RATIOS_TIMEAVG = {k: v[1] for k, v in ER_HISTORY_BAND.items()}

# --------------------------------------------------------------------------- #
# The pairs. ``a`` is always the OLDER / legacy fund and ``b`` its modern low-cost
# stablemate; the residual is measured for ``a`` relative to ``b``. ``a`` is also the
# dearer leg in four of the five pairs — the exception is VEA/IEFA, where the legacy leg
# (Vanguard's VEA, 0.03%) is today the *cheaper* of the two, so its fee gap is negative.
# ``note`` states the composition wedge honestly.
# --------------------------------------------------------------------------- #
PAIRS = (
    ("SPY", "IVV", "US large cap",
     "Identical index (S&P 500). SPY is a unit investment trust and is legally barred "
     "from lending its securities; IVV is an open-end fund that lends. The one pair with "
     "a structural lending asymmetry and a near-zero composition wedge."),
    ("EEM", "IEMG", "Emerging markets",
     "MSCI EM vs MSCI EM IMI (the IMI adds EM small caps). Both iShares, both lend into "
     "the hardest-to-borrow class on the desk."),
    ("EFA", "IEFA", "Developed ex-US",
     "MSCI EAFE vs MSCI EAFE IMI. Same sponsor, same withholding treatment, small "
     "composition wedge."),
    ("VEA", "IEFA", "Developed ex-US (cross-sponsor)",
     "FTSE Developed ex-North-America (Korea and Canada in) vs MSCI EAFE IMI. The widest "
     "composition wedge of the developed pairs — kept as the cautionary exhibit."),
    ("IWM", "IJR", "US small cap",
     "Russell 2000 vs S&P SmallCap 600 — different index families with a documented "
     "quality gap. The composition wedge dominates anything fees or lending could do."),
)

# The lending-heavy classes the claim is really about (used only for reporting).
LENDING_HEAVY = ("Emerging markets", "US small cap")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1999-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted **total return** — mandatory here,
    because the whole quantity under test is a few basis points a year of accrual and a
    price-only series would be swamped by dividend timing.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out = {}
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
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so the
    sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call lending_offset.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted lending passthrough)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 25,
    er_a: float = 0.70,            # dearer leg's expense ratio, % per year
    er_b: float = 0.09,            # cheaper leg's expense ratio, % per year
    lend_a: float = 0.60,          # planted lending revenue passed to A, % per year
    lend_b: float = 0.00,          # planted lending revenue passed to B, % per year
    index_vol: float = 0.18,       # annualised vol of the common index leg
    index_drift: float = 0.07,     # annualised drift of the common index leg
    wedge_vol: float = 0.005,      # annualised vol of each fund's composition wedge
    signal_strength: float = 1.0,  # 0 = no lending differential (null), 1 = full
    cash_rate_ann: float = 0.03,
    start: str = "2004-01-02",
    seed: int = 914,
):
    """One synthetic same-asset-class ETF pair with a planted lending differential.

    Both funds ride the *same* index path, so the only systematic wedge between them is
    ``(lend_a - lend_b) - (er_a - er_b)``. ``signal_strength`` scales only the **lending**
    part:

    - ``signal_strength = 1`` → a real passthrough differential of ``lend_a - lend_b``
      that the residual estimator must recover;
    - ``signal_strength = 0`` → the funds differ by fees and composition noise alone, so
      the fee-adjusted residual is *exactly zero in expectation* (the null).

    Returns ``(prices, truth)`` where ``prices`` has columns ``fund_a``, ``fund_b`` and
    ``cash`` (total-return close levels) and ``truth`` records the planted parameters and
    the implied residual in bp/yr. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mu = index_drift / TRADING_DAYS_PER_YEAR
    sd = index_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    index_lr = rng.normal(mu, sd, n_days)

    wedge_sd = wedge_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    wedge_a = rng.normal(0.0, wedge_sd, n_days)
    wedge_b = rng.normal(0.0, wedge_sd, n_days)

    la = signal_strength * lend_a / 100.0 / TRADING_DAYS_PER_YEAR
    lb = signal_strength * lend_b / 100.0 / TRADING_DAYS_PER_YEAR
    fa = er_a / 100.0 / TRADING_DAYS_PER_YEAR
    fb = er_b / 100.0 / TRADING_DAYS_PER_YEAR

    lr_a = index_lr - fa + la + wedge_a
    lr_b = index_lr - fb + lb + wedge_b

    fund_a = 100.0 * np.exp(np.cumsum(lr_a))
    fund_b = 100.0 * np.exp(np.cumsum(lr_b))
    cash = np.cumprod(np.full(n_days, (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)))

    prices = pd.DataFrame(
        {"fund_a": fund_a, "fund_b": fund_b, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "er_a": er_a, "er_b": er_b,
        "lend_a": lend_a, "lend_b": lend_b,
        "planted_residual_bp": float(signal_strength * (lend_a - lend_b) * 100.0),
        "index_vol": index_vol, "wedge_vol": wedge_vol,
        "n_years": n_years, "n_days": n_days, "seed": seed,
        "cash_rate_ann": cash_rate_ann,
    }
    return prices, truth


def synthetic_panel(
    n_pairs: int = 5,
    n_years: int = 25,
    signal_strength: float = 1.0,
    lend_spread: float = 0.60,
    seed: int = 914,
):
    """A panel of ``n_pairs`` independent synthetic ETF pairs sharing one design.

    Each pair gets its own index path, its own composition wedges and its own fee gap; the
    planted lending differential is the *same* ``lend_spread`` (scaled by
    ``signal_strength``) across pairs, so pooling should sharpen the estimate exactly as
    it would on the real desk. Returns ``(frames, truth)`` where ``frames`` is a dict
    ``{pair_name: prices}``.
    """
    frames = {}
    ers = [(0.70, 0.09), (0.33, 0.07), (0.19, 0.06), (0.0945, 0.03), (0.25, 0.05)]
    for i in range(n_pairs):
        er_a, er_b = ers[i % len(ers)]
        px, _ = synthetic_daily(
            n_years=n_years, er_a=er_a, er_b=er_b,
            lend_a=lend_spread, lend_b=0.0,
            signal_strength=signal_strength, seed=seed + 17 * i,
        )
        frames[f"pair_{i + 1}"] = px
    truth = {
        "n_pairs": n_pairs, "n_years": n_years, "seed": seed,
        "signal_strength": signal_strength,
        "planted_residual_bp": float(signal_strength * lend_spread * 100.0),
        "expense_ratios": ers[:n_pairs],
    }
    return frames, truth
