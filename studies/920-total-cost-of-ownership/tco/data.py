"""Data layer for Study 920 — Total Cost of Ownership.

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for three S&P 500 wrappers (SPY, IVV, VOO),
  two Nasdaq-100 wrappers (QQQ, QQQM) and a cash proxy (BIL, the 1-3 month T-bill
  ETF). ``fetch`` touches the network and caches parquet in the **shared** desk cache
  ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache **offline** and
  never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic-only), so CI is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  common index path is shared by two fund wrappers; each wrapper bleeds a constant
  annual **drag** and each carries a transient closing-print error plus a slow
  persistent wobble. The ``signal_strength`` knob scales the planted drag *gap*: at
  ``signal_strength=0`` the two wrappers are identical up to noise (the null — the
  estimator must not manufacture a gap); at ``signal_strength=1`` the gap is the full
  planted ``gap_bp_yr``. Seed is fixed → tests are deterministic.

**What total cost of ownership means here.** Two funds on the same index differ in
three places: the **expense ratio** (a prospectus number, not a tape number), the
**realised tracking difference** (what the fund actually delivered against its twin,
which folds fee, securities-lending revenue, sampling error, cash drag and the
wrapper's legal form together), and the **round-trip spread** you pay to get in and out
(a quote-level number that is not on a daily-close tape at all). Only the middle term
is measurable here; the other two are declared PROXIES / ASSUMPTIONS and swept.

**Total return, not price only.** ``auto_adjust=True`` reinvests distributions, so the
realised return difference between two wrappers on the same index *is* the realised
tracking-difference gap. On price-only closes a difference in dividend schedule would
masquerade as tracking difference; every number in this study is total-return.

Known tape limitation, stated up front: **SPLG cannot enter the race.** At build time
Yahoo's SPLG series carries a first-trade date of 2026-07-17 and exactly one session —
the history has been reset upstream. SPLG is left in ``TICKERS`` (so a later ``fetch``
picks it up if Yahoo restores it) but ``PAIRS`` uses VOO and IVV as the cheap S&P
wrappers instead. No number in this study depends on SPLG.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache: studies/_cache (two levels up from the package directory).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# Every wrapper this study pulls, plus the cash leg. SPLG is requested but its Yahoo
# history is currently a single session (see the module docstring); nothing depends on it.
TICKERS = ("SPY", "IVV", "VOO", "SPLG", "QQQ", "QQQM", "BIL")

# The wrappers that actually carry the race, plus the cash proxy.
USABLE = ("SPY", "IVV", "VOO", "QQQ", "QQQM", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# NON-TAPE INPUTS — every one of these is an ASSUMPTION, not a measurement.
# --------------------------------------------------------------------------- #
# Stated net expense ratios, in basis points per year, as published in the funds'
# prospectuses at the as-of date. These are PROSPECTUS numbers, not tape numbers: the
# study uses them only as a *prior* to compare the realised tracking difference against,
# and never as an input to a realised return. Fee schedules change (QQQM launched at
# 15 bp; SPY has been 9.45 bp since 2004), so treat the whole column as an assumption.
STATED_ER_BPS = {
    "SPY": 9.45,   # SPDR S&P 500 Trust — a 1993 unit investment trust
    "IVV": 3.0,    # iShares Core S&P 500
    "VOO": 3.0,    # Vanguard S&P 500
    "SPLG": 2.0,   # SPDR Portfolio S&P 500 (not on the tape here)
    "QQQ": 20.0,   # Invesco QQQ Trust — also a unit investment trust
    "QQQM": 15.0,  # Invesco Nasdaq-100 ETF — the open-end clone
}

# Indicative round-trip quoted spreads, in basis points of NAV. A daily-close tape
# carries no quotes, so these are a PROXY, declared here and **swept end to end** in
# ``strategy.breakeven_curve`` — no headline number rests on any single value. They are
# ordered the way the market orders them: the mega-liquid original quotes inside the
# cheap clone.
ASSUMED_RT_SPREAD_BPS = {
    "SPY": 0.3,
    "IVV": 0.6,
    "VOO": 0.7,
    "SPLG": 1.5,
    "QQQ": 0.3,
    "QQQM": 1.2,
}

# The races. (liquid, cheap, label): the liquid original against the cheaper wrapper on
# the same index. Named ex ante from the study brief — no pair was searched over.
#
# CONFOUND, NAMED: in all three pairs that have a fee gap, the expensive leg (SPY, QQQ) is
# a 1993/1999 *unit investment trust*, and the cheap leg is a modern open-end fund. A UIT
# cannot reinvest dividends between distribution dates and cannot lend securities, so it
# bleeds a cash drag in a rising market on top of its fee (Elton, Gruber, Comer & Li 2002).
# The realised tracking difference this study measures is therefore
#     fee gap  +  trust-form cash drag  +  lending pass-through,
# and it cannot separate them: there is no pair on the tape where a cheap open-end fund
# races an expensive open-end fund on the same index. The same-fee placebo (VOO vs IVV)
# proves the *ruler* reads zero when there is nothing to read, but both its legs are
# open-end funds, so it cannot separate the fee from the trust form either. Read every
# number below as "what the cheap wrapper delivered", never as "what the fee cost".
#
# SURVIVORSHIP, NAMED: these are the wrappers still trading at the as-of date. Cheap clones
# that closed never entered the sample, and a clone that closes is more likely to have been
# a poor tracker. The bias is small for a fee measurement — the fee gap is contractual, not
# performance-selected — but the sample is the survivors, and it is not a random draw.
PAIRS = (
    ("SPY", "IVV", "S&P 500 — the 1993 trust vs the iShares core fund"),
    ("SPY", "VOO", "S&P 500 — the 1993 trust vs the Vanguard fund"),
    ("IVV", "VOO", "S&P 500 — two 3 bp funds (the placebo pair)"),
    ("QQQ", "QQQM", "Nasdaq-100 — the trust vs its own cheaper clone"),
)


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
    skip_empty: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared desk cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted total return — mandatory here,
    because a tracking difference measured on price-only closes would be contaminated by
    the two wrappers' differing distribution schedules.

    ``skip_empty`` leaves an already-cached series alone when Yahoo answers with an empty
    or single-row history (which is exactly what it does for SPLG at the as-of date), so a
    refresh never destroys a good cached tape with an upstream outage.
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
        if raw is None or len(raw) < 2:
            if skip_empty:
                # Upstream outage (or a reset history, as for SPLG). Keep whatever is
                # already cached and carry on — never silently overwrite good data.
                continue
            raise RuntimeError(f"yfinance returned no usable data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=USABLE, cache_dir: str = DEFAULT_CACHE, min_rows: int = 250) -> bool:
    """True iff every ticker has a usable (>= ``min_rows``) cached parquet — offline test."""
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            return False
        try:
            if len(pd.read_parquet(path)) < min_rows:
                return False
        except Exception:
            return False
    return True


def load_prices(
    tickers=USABLE,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker (the adjusted close),
    sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` if any
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call tco.data.fetch() once to populate the shared desk cache."
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
# Synthetic tape — two wrappers on one index, with a planted annual drag gap
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 16,
    gap_bp_yr: float = 6.0,        # planted ANNUAL drag gap, liquid minus cheap, in bp
    base_drag_bp_yr: float = 3.0,  # the cheap wrapper's own drag against the index
    signal_strength: float = 1.0,  # 0 = identical wrappers (the null), 1 = full gap
    index_drift: float = 0.09,     # annualised index drift
    index_vol: float = 0.17,       # annualised index vol
    print_noise_bp: float = 3.0,   # transient closing-print error, bp of level, per fund
    wobble_bp: float = 4.0,        # slow AR(1) level wobble, bp, per fund
    wobble_phi: float = 0.995,     # persistence of the slow wobble
    start: str = "2010-01-04",
    seed: int = 920,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """Two total-return closes on one index, separated only by a planted annual drag.

    The generator reproduces the three things that make a real tracking difference hard
    to measure from public closes:

    1. a **constant drag gap** (the thing we want) of ``signal_strength * gap_bp_yr``
       basis points per year, charged to the ``liquid`` leg;
    2. a **transient closing-print error** on each leg's *level* (so it mean-reverts
       within a day or two and inflates the daily return difference without moving the
       long-run drift);
    3. a **slow AR(1) wobble** on each leg's level, which is what makes the *annual*
       realised tracking difference of two real S&P wrappers swing by tens of basis
       points around a fee gap of five.

    ``signal_strength=0`` sets the gap to zero: the two wrappers are then identical up to
    noise, and any tracking difference the estimator reports is spurious (the null).

    Returns ``(prices, truth)`` with columns ``liquid``, ``cheap``, ``cash``.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    gap = float(signal_strength) * float(gap_bp_yr) * 1e-4
    drag_cheap = float(base_drag_bp_yr) * 1e-4
    drag_liquid = drag_cheap + gap

    mu = index_drift / TRADING_DAYS_PER_YEAR
    sig = index_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    idx_log = np.cumsum(rng.normal(mu, sig, n_days))

    def _wobble() -> np.ndarray:
        e = rng.normal(0.0, wobble_bp * 1e-4 * np.sqrt(1 - wobble_phi ** 2), n_days)
        out = np.empty(n_days)
        acc = 0.0
        for i in range(n_days):
            acc = wobble_phi * acc + e[i]
            out[i] = acc
        return out

    t = np.arange(n_days) / TRADING_DAYS_PER_YEAR
    liquid_log = idx_log - drag_liquid * t + _wobble() + rng.normal(0.0, print_noise_bp * 1e-4, n_days)
    cheap_log = idx_log - drag_cheap * t + _wobble() + rng.normal(0.0, print_noise_bp * 1e-4, n_days)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {
            "liquid": 100.0 * np.exp(liquid_log),
            "cheap": 100.0 * np.exp(cheap_log),
            "cash": cash_idx,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength),
        "gap_bp_yr": float(gap_bp_yr),
        "planted_gap_bp_yr": float(gap * 1e4),
        "base_drag_bp_yr": float(base_drag_bp_yr),
        "print_noise_bp": float(print_noise_bp),
        "wobble_bp": float(wobble_bp),
        "n_years": int(n_years),
        "n_days": int(n_days),
        "seed": int(seed),
        "cash_rate_ann": float(cash_rate_ann),
    }
    return prices, truth


def synthetic_panel(
    gaps_bp_yr=(0.0, 3.0, 6.0, 12.0),
    n_years: int = 16,
    signal_strength: float = 1.0,
    seed: int = 920,
    **kwargs,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A panel of synthetic wrapper pairs, one per planted gap in ``gaps_bp_yr``.

    Used to check the estimator is *calibrated*, not merely non-zero: the recovered
    tracking difference should track the planted gap roughly one-for-one across the panel,
    and sit at zero for the zero-gap pair. Each pair gets its own seed offset so the noise
    draws are independent. Deterministic given ``seed``.
    """
    panel: dict[str, pd.DataFrame] = {}
    truths: dict[str, dict] = {}
    for i, g in enumerate(gaps_bp_yr):
        px, tr = synthetic_daily(
            n_years=n_years, gap_bp_yr=float(g), signal_strength=signal_strength,
            seed=seed + 101 * i, **kwargs
        )
        key = f"gap_{g:g}bp"
        panel[key] = px
        truths[key] = tr
    return panel, {"pairs": truths, "gaps_bp_yr": tuple(float(g) for g in gaps_bp_yr),
                   "signal_strength": float(signal_strength), "seed": int(seed)}
