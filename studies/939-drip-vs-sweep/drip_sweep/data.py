"""Data layer for Study 939 — DRIP or Sweep.

Two real tapes and one synthetic generator, all with the same shape (a date-indexed
daily frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the three dividend payers under test
  (SPY, VYM, SCHD) and the cash leg (BIL, the 1-3 month T-bill ETF). Cached in the
  **shared** desk cache ``studies/_cache`` under the house convention
  ``prices_<TICKER>_1d.parquet``.

- ``fetch`` / ``load_distributions`` — the **price-only** leg: ``auto_adjust=False``
  closes (split-adjusted, *not* dividend-adjusted) plus the per-share ``dividend``
  and ``capital_gains`` cash amounts, cached as ``divs_<TICKER>_1d.parquet``
  (the convention established by Study 916). Both legs are needed because this
  study simulates a *share count and a cash balance*, not an index: you cannot DRIP
  a total-return series, you DRIP a dividend into a price.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators.
  A geometric price with a quarterly dividend and a cash accrual leg. The
  ``signal_strength`` knob scales the **equity risk premium over cash**: at
  ``signal_strength=0`` the price drifts at exactly the cash rate, so parking a
  distribution in T-bills costs nothing and the DRIP-minus-sweep gap must be zero
  (the null); at ``signal_strength=1`` there is a fat premium and the gap must be
  positive and detectable. The generator's defaults are a deliberately loud **lab
  bench** (see its docstring) — at market-realistic parameters the same detector
  cannot resolve the true effect in twenty years, which is a *power* result the study
  reports rather than hides. Seeds are fixed → the whole test-suite is deterministic
  and runs with **no cache present**.

The dividend reconstruction
---------------------------
The study's headline input is the *reconstructed* dividend stream, built from the two
price legs alone and never from the ``dividend`` column:

    D_t  =  P_{t-1} · (TR_t / TR_{t-1})  −  P_t

where ``P`` is the price-only close and ``TR`` the total-return close. The identity
follows from the definition of the adjusted series: on an ex-dividend day the
total-return holder earns ``(P_t + D_t) / P_{t-1}`` while the price-only holder earns
``P_t / P_{t-1}``. On non-ex days the two legs move together and ``D_t`` collapses to
float noise, so the reconstruction is thresholded (see ``reconstruct_dividends``).
``dividend_reconstruction_check`` scores the reconstruction against the ``dividend``
column that yfinance reports — an *audit*, not an input.

NON-TAPE ASSUMPTIONS (all labelled, all swept in ``strategy.py``)
-----------------------------------------------------------------
* **The pay lag.** The tape gives the *ex-dividend* date. The cash reaches the account
  on the *pay* date, typically two to five weeks later; Yahoo does not publish pay
  dates. ``PAY_LAG_DAYS`` is a calendar-day ASSUMPTION, swept over 0/15/30/45.
* **Idle accrual between ex and pay.** Money in transit earns nothing, for both arms
  alike. That is a modelling choice, not tape.
* **Trading cost.** DRIP is free at every large US broker; a swept-cash reinvestment is
  a real ticket. ``DRIP_COST_BPS`` / ``SWEEP_COST_BPS`` are ASSUMPTIONS, swept.
* **Tax.** Ignored entirely. Both policies receive the *same* taxable distribution on
  the same date, so income tax is a wash; only the (tiny) difference in realised
  capital gains at eventual sale would differ, and that is outside the tape.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: a
distribution or a sweep decided at the close of day ``t`` is executed at the close of
day ``t+1`` (exactly one execution lag).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The three dividend payers under test: the market (SPY, ~1.3% yield), a broad
# high-yield tilt (VYM, ~2.8%) and a quality-screened dividend fund (SCHD, ~3.5%).
# The yield spread across them is the point: if the DRIP-vs-sweep gap scales with
# yield, it should be ~3x larger on SCHD than on SPY.
PAYERS = ("SPY", "VYM", "SCHD")
CASH = "BIL"
TICKERS = PAYERS + (CASH,)

# --- ASSUMPTIONS (not tape) ------------------------------------------------- #
# Calendar days from ex-date to pay date. US equity ETFs from the big issuers settle
# distributions in roughly 2-5 weeks; 30 is the central case. Swept 0/15/30/45.
PAY_LAG_DAYS = 30
# One-way cost in bps of the amount reinvested. DRIP is free at Schwab/Fidelity/
# Vanguard/IBKR; a swept-cash purchase pays at least the half-spread on the ETF.
DRIP_COST_BPS = 0.0
SWEEP_COST_BPS = 2.0

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# The reconstruction threshold: an implied per-share amount below this fraction of the
# price is float noise from the two legs' independent rounding, not a distribution.
DIV_NOISE_FRAC = 5e-4


# --------------------------------------------------------------------------- #
# Cache plumbing
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _tr_path(ticker: str, cache_dir: str) -> str:
    """Total-return close, house convention (shared with every other study)."""
    return os.path.join(cache_dir, f"prices_{_safe(ticker)}_1d.parquet")


def _div_path(ticker: str, cache_dir: str) -> str:
    """Price-only close + per-share dividend / capital-gain cash amounts."""
    return os.path.join(cache_dir, f"divs_{_safe(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download both legs for ``tickers`` and cache them as parquet.

    Leg 1 (``auto_adjust=True``) is the daily **total-return** close — the house
    convention every other study reads. Leg 2 (``auto_adjust=False, actions=True``)
    is the **price-only** close plus the per-share ``dividend`` and ``capital_gains``
    cash amounts. This study needs both: the gap between them *is* the dividend
    stream it reinvests.

    Network-only; run once to populate the shared cache. Retries up to ``retries``.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        # --- leg 1: total return -------------------------------------------- #
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(tk, start=start, end=end, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no total-return data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        tr = raw[["close"]].copy()
        tr.index = pd.to_datetime(tr.index)
        tr.index.name = "date"
        tr = tr.dropna(subset=["close"])
        tr.to_parquet(_tr_path(tk, cache_dir))

        # --- leg 2: price-only + distributions ------------------------------ #
        raw2 = None
        for _ in range(retries):
            try:
                raw2 = yf.download(tk, start=start, end=end, interval="1d",
                                   auto_adjust=False, actions=True, progress=False)
                if raw2 is not None and len(raw2) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw2 is None or len(raw2) == 0:
            raise RuntimeError(f"yfinance returned no price-only data for {tk}")
        if isinstance(raw2.columns, pd.MultiIndex):
            raw2.columns = raw2.columns.get_level_values(0)
        raw2 = raw2.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
        px = pd.DataFrame({"close": raw2["close"]})
        px["dividend"] = raw2["dividends"] if "dividends" in raw2 else 0.0
        px["capital_gains"] = raw2["capital_gains"] if "capital_gains" in raw2 else 0.0
        px.index = pd.to_datetime(px.index)
        px.index.name = "date"
        px = px.dropna(subset=["close"]).fillna({"dividend": 0.0, "capital_gains": 0.0})
        px.to_parquet(_div_path(tk, cache_dir))

        out[tk] = px
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff both legs of every ticker are cached (offline-testable)."""
    return all(
        os.path.exists(_tr_path(tk, cache_dir)) and os.path.exists(_div_path(tk, cache_dir))
        for tk in tickers
    )


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily **total-return** closes OFFLINE into one aligned frame.

    One column per ticker, sliced to ``asof`` so the sample never creeps. Raises
    ``FileNotFoundError`` if any leg is missing — the offline core and the whole
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _tr_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached total-return prices for {tk} at {path}. "
                f"Call drip_sweep.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_distributions(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> dict[str, pd.DataFrame]:
    """Read the cached **price-only** leg OFFLINE: close + dividend + capital_gains.

    Returns one frame per ticker, sliced to ``asof``. Never touches the network.
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _div_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached price-only leg for {tk} at {path}. "
                f"Call drip_sweep.data.fetch() once to populate the shared cache."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()
        out[tk] = df[df.index <= pd.Timestamp(asof)]
    return out


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a numeric frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# The dividend reconstruction (price leg vs total-return leg)
# --------------------------------------------------------------------------- #
def reconstruct_dividends(
    price: pd.Series,
    total_return: pd.Series,
    noise_frac: float = DIV_NOISE_FRAC,
) -> pd.Series:
    """Recover the per-share distribution stream from the two price legs.

    On an ex-dividend day the total-return holder earns ``(P_t + D_t) / P_{t-1}``
    while the price-only holder earns ``P_t / P_{t-1}``; equating the total-return
    leg's realised gross return to the former and solving gives

        D_t = P_{t-1} · (TR_t / TR_{t-1}) − P_t.

    Off ex-dates the two legs are proportional and ``D_t`` is float noise, so any
    implied amount below ``noise_frac × P_t`` (5 bp of price by default, an order of
    magnitude under the smallest real ETF distribution) is zeroed. Negative implied
    amounts — which occur only as rounding artefacts — are zeroed too.

    Returns a per-share cash series aligned to ``price.index`` (0.0 on non-ex days).
    """
    common = price.index.intersection(total_return.index)
    p = price.reindex(common).astype(float).sort_index()
    tr = total_return.reindex(common).astype(float).sort_index()
    implied = p.shift(1) * (tr / tr.shift(1)) - p
    implied = implied.fillna(0.0)
    implied[implied < noise_frac * p] = 0.0
    return implied.rename("dividend_ps")


def dividend_reconstruction_check(
    price: pd.Series,
    total_return: pd.Series,
    reported: pd.Series,
    noise_frac: float = DIV_NOISE_FRAC,
) -> dict:
    """Audit the reconstruction against yfinance's reported ``dividend`` column.

    The reported column is **never** an input to the simulation — this is a data-
    quality check only. Returns per-event counts, the total reconstructed vs reported
    cash per share, and the correlation of the two event streams.
    """
    d_rec = reconstruct_dividends(price, total_return, noise_frac=noise_frac)
    rep = reported.reindex(d_rec.index).fillna(0.0).astype(float)
    n_rec = int((d_rec > 0).sum())
    n_rep = int((rep > 0).sum())
    both = int(((d_rec > 0) & (rep > 0)).sum())
    tot_rec, tot_rep = float(d_rec.sum()), float(rep.sum())
    corr = float(pd.Series(d_rec).corr(rep)) if n_rec and n_rep else float("nan")
    return {
        "n_events_reconstructed": n_rec,
        "n_events_reported": n_rep,
        "n_events_matched": both,
        "total_ps_reconstructed": tot_rec,
        "total_ps_reported": tot_rep,
        "ratio_rec_over_rep": (tot_rec / tot_rep) if tot_rep > 0 else float("nan"),
        "event_corr": corr,
    }


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 20,
    equity_premium: float = 0.20,    # annualised excess of the fund over cash (LAB scale)
    cash_rate_ann: float = 0.03,     # the T-bill leg's yield
    vol_ann: float = 0.06,           # annualised vol (LAB scale — a quiet bench)
    div_yield_ann: float = 0.06,     # annual gross distribution yield (LAB scale)
    payments_per_year: int = 4,      # quarterly, like every ETF under test
    signal_strength: float = 1.0,    # 0 = no premium over cash (the null)
    start: str = "2004-01-02",
    seed: int = 939,
) -> tuple[pd.DataFrame, dict]:
    """A daily price / dividend / cash tape with a *known* equity premium over cash.

    ``signal_strength`` scales the premium the price leg earns over the cash leg:

    - ``signal_strength = 1`` → the full ``equity_premium``. Money parked in T-bills
      between the pay date and a quarterly (or annual) reinvestment date forgoes that
      premium, so DRIP must beat sweep by a positive, recoverable margin.
    - ``signal_strength = 0`` → the price leg drifts at exactly the cash rate. Parking
      the distribution costs nothing, so the DRIP-minus-sweep gap must be ~0 (the null,
      up to the noise of when the cash happens to be in or out of the market).

    **The defaults are a deliberately loud LAB BENCH, not a market.** A 20% premium, a
    6% distribution yield and a *quiet* 6% vol are chosen so that a planted effect worth
    a few basis points a year is resolvable in twenty years of daily data. Feed the
    realistic values (``equity_premium=0.055, div_yield_ann=0.03, vol_ann=0.16``) and
    the same detector can no longer separate the planted effect from zero — which is
    the study's *power* result, and is reported as such rather than hidden.

    Returns ``(frame, truth)``. ``frame`` has columns ``close`` (the **price-only**
    close, which drops on each ex-date), ``dividend`` (per-share cash on the ex-date)
    and ``cash`` (the T-bill accrual index). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe and fast: take weekdays out of a modest calendar range (bdate_range is
    # an order of magnitude slower and the test-suite calls this hundreds of times).
    cal = pd.date_range(start=start, periods=int(n_days * 1.5) + 10, freq="D")
    dates = cal[cal.dayofweek < 5][:n_days]

    drift_ann = cash_rate_ann + signal_strength * equity_premium
    # The diffusion carries the FULL total-return drift; the discrete ex-date drops
    # below are what turn that into a price-only path. (Between ex-dates the price
    # accrues the whole return — the standard accrual convention. Putting the drift
    # net of the yield in the diffusion *and* dropping the price on the ex-date would
    # double-count the payout and make the null arm drift below cash by construction.)
    d = (drift_ann - 0.5 * vol_ann ** 2) / TRADING_DAYS_PER_YEAR
    s = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    shocks = rng.normal(d, s, n_days)

    step = max(1, n_days // (n_years * payments_per_year))
    ex_idx = np.arange(step, n_days, step)
    div_per_event = div_yield_ann / payments_per_year

    # Vectorised: the ex-date drop is multiplicative (level *= 1 - q), so the price is
    # the diffusion times the cumulative product of the drops, and the per-share cash
    # is q times the price *before* the drop.
    is_ex = np.zeros(n_days, dtype=bool)
    is_ex[ex_idx] = True
    drop = np.where(is_ex, 1.0 - div_per_event, 1.0)
    price = 100.0 * np.exp(np.cumsum(shocks)) * np.cumprod(drop)
    div = np.where(is_ex, price / (1.0 - div_per_event) * div_per_event, 0.0)

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    frame = pd.DataFrame(
        {"close": price, "dividend": div, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "equity_premium": equity_premium,
        "effective_premium": signal_strength * equity_premium,
        "cash_rate_ann": cash_rate_ann,
        "vol_ann": vol_ann,
        "div_yield_ann": div_yield_ann,
        "payments_per_year": payments_per_year,
        "n_years": n_years,
        "n_days": n_days,
        "n_events": int(len(ex_idx)),
        "seed": seed,
    }
    return frame, truth


def synthetic_panel(
    tickers=PAYERS,
    div_yields=(0.03, 0.06, 0.09),
    signal_strength: float = 1.0,
    seed: int = 939,
    **kwargs,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A small panel of payers with *different* distribution yields, one cash leg.

    Mirrors the *ordering* of the real line-up (a low-yield market fund, a high-yield
    tilt, a dividend fund) at the lab scale of ``synthetic_daily``'s defaults. Because
    the DRIP-vs-sweep gap is mechanically proportional to the amount of cash in
    transit, the planted gap must be **monotone in yield** — a sharper test of the
    machinery than a single tape. Deterministic given ``seed``.
    """
    frames: dict[str, pd.DataFrame] = {}
    truths = {}
    for k, (tk, y) in enumerate(zip(tickers, div_yields)):
        f, t = synthetic_daily(div_yield_ann=y, signal_strength=signal_strength,
                               seed=seed + k, **kwargs)
        frames[tk] = f
        truths[tk] = t
    truth = {"per_ticker": truths, "tickers": tuple(tickers),
             "div_yields": tuple(div_yields), "signal_strength": signal_strength,
             "seed": seed}
    return frames, truth
