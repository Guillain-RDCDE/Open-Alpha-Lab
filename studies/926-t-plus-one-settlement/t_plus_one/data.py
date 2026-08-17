"""Data layer for Study 926 — T+1 (the May-2024 US settlement switch).

Two tapes, one shape (a date-indexed daily OHLC frame per ticker):

- ``fetch`` / ``load_ohlc`` — daily **total-return** OHLC from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the four ETFs the study races (SPY, IWM,
  EEM, EFA) plus a cash proxy (BIL, the 1-3 month T-bill ETF). ``fetch`` touches the
  network and caches parquet under the **shared** ``studies/_cache`` (retry up to 4x);
  ``load_ohlc`` / ``load_prices`` read that cache **offline** and never import
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI is
  green on a fresh checkout where ``_cache/`` is git-ignored and absent.

  Two files are written per ticker: the desk-standard ``prices_<T>_1d.parquet`` (a
  ``close`` column) and, because this study needs the *open* auction print to split a
  day into its night and day legs, ``ohlc_<T>_1d.parquet`` (``open``/``high``/``low``/
  ``close``). ``auto_adjust=True`` scales open and close by the *same* daily factor, so
  the exact night/day identity ``(1+r_on)(1+r_id) = (1+r_cc)`` survives adjustment.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. Two
  assets ("treated" and "control") whose night and day legs are drawn separately, with a
  **planted structural break** at the switch date that hits the treated asset only. The
  ``signal_strength`` knob scales the planted break: at ``signal_strength=0`` nothing
  changes at the switch (the null — the difference-in-difference must stay quiet); at
  ``signal_strength=1`` the treated asset's post-switch overnight drift, overnight
  volatility and turn-of-month overnight drift all shift by a known amount the estimator
  must recover. Seed is fixed → tests are deterministic.

**The event.** US cash equities, ETFs and corporate bonds moved from T+2 to **T+1**
settlement on **Tuesday 28 May 2024** (SEC Rule 15c6-1(a) as amended; Canada and Mexico
moved one business day earlier, on 27 May). Europe, the UK and most of Asia stayed on
T+2 through the whole post-period. That mismatch is the study's treatment: a fund whose
*shares* settle T+1 in New York while its *holdings* settle T+2 in London or Tokyo has to
pre-fund a day of FX and a day of stock, which is exactly the plumbing an EFA or EEM
sits on. SPY, whose shares and holdings both settle T+1, is the control.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the overlay's
signal is formed on data through day ``t`` and acted on at day ``t+1``).
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

# The tape. EFA (developed ex-US) and EEM (emerging) are the settlement-mismatched legs;
# SPY is the domestic control; IWM is a *domestic* placebo-treated leg (small caps, all
# T+1 both sides — if the "T+1 effect" shows up in IWM too, it is not settlement).
# BIL is the cash proxy for the excess-of-cash Sharpe race.
TICKERS = ("SPY", "IWM", "EEM", "EFA", "BIL")

TREATED = ("EFA", "EEM")
CONTROL = "SPY"
PLACEBO = "IWM"
CASH = "BIL"

# The event: US settlement cycle shortened from T+2 to T+1.
SWITCH_DATE = "2024-05-28"

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return OHLC, cache-only by default
# --------------------------------------------------------------------------- #
def _safe(ticker: str) -> str:
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _close_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"prices_{_safe(ticker)}_1d.parquet")


def _ohlc_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"ohlc_{_safe(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1990-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return OHLC for ``tickers`` and cache them as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` means the
    open and the close carry the *same* daily adjustment factor, so the exact night/day
    return identity is preserved after adjustment. Writes both the desk-standard
    close-only ``prices_<T>_1d.parquet`` and this study's ``ohlc_<T>_1d.parquet``.
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
        df = raw[["open", "high", "low", "close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["open", "close"])
        df.to_parquet(_ohlc_path(tk, cache_dir))
        df[["close"]].to_parquet(_close_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's OHLC parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_ohlc_path(tk, cache_dir)) for tk in tickers)


def load_ohlc(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> dict[str, pd.DataFrame]:
    """Read cached daily total-return OHLC OFFLINE as ``{ticker: DataFrame}``.

    Each frame is indexed by date with lowercase ``open``/``high``/``low``/``close``
    columns, sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError``
    if any ticker is missing — the offline core and the test-suite never touch the
    network.
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _ohlc_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached OHLC for {tk} at {path}. "
                f"Call t_plus_one.data.fetch() once to populate the shared cache."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[df.index <= pd.Timestamp(asof)]
        out[tk] = df
    return out


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return **closes** OFFLINE into one aligned frame.

    One column per ticker, sliced to ``asof``. Uses the desk-standard
    ``prices_<T>_1d.parquet`` files so this study reads the same cache every other
    study does.
    """
    cols = {}
    for tk in tickers:
        path = _close_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call t_plus_one.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted settlement break)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 2600,
    start: str = "2016-01-04",
    switch_frac: float = 0.8,          # where in the sample the planted switch lands
    signal_strength: float = 1.0,      # 0 = no break at all (the null), 1 = full break
    on_share: float = 0.55,            # overnight share of variance, both assets, pre
    vol_ann: float = 0.17,             # annualised close-to-close vol
    drift_ann: float = 0.07,           # annualised close-to-close drift
    d_on_drift_bps: float = 8.0,       # planted post-switch overnight drift shift (bps/day)
    d_on_vol_mult: float = 0.30,       # planted post-switch idiosyncratic overnight vol lift
    d_tom_bps: float = 45.0,           # planted post-switch turn-of-month overnight bump
    seed: int = 926,
    cash_rate_ann: float = 0.03,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A two-asset daily OHLC world with a planted settlement-style break.

    Both assets ("treated" and "control") are built leg by leg: an overnight return
    (previous close → today's open) and an intraday return (today's open → today's
    close), drawn independently, so the close-to-close return is exactly their compound.
    A common market factor is shared by both assets, which is what makes the
    difference-in-difference well-behaved.

    At the switch date the **treated** asset alone gets three planted changes, each
    scaled by ``signal_strength``:

    1. its overnight drift rises by ``signal_strength * d_on_drift_bps`` bps/day;
    2. its overnight volatility rises by ``signal_strength * d_on_vol_mult`` (relative);
    3. its overnight drift on turn-of-month days rises by a further
       ``signal_strength * d_tom_bps`` bps.

    At ``signal_strength = 0`` nothing whatsoever changes at the switch, in either asset
    — the null the difference-in-difference estimator must not fire on.

    Returns ``(panel, truth)`` where ``panel`` maps ``"treated"``/``"control"``/``"cash"``
    to date-indexed frames (OHLC for the two assets, a ``close`` accrual index for cash)
    and ``truth`` records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.DatetimeIndex(pd.bdate_range(start=start, periods=int(n_days)), name="date")
    n = len(dates)
    switch_i = int(round(switch_frac * n))
    post = np.zeros(n, dtype=bool)
    post[switch_i:] = True

    # Turn-of-month: the last trading day of a month and the first three of the next.
    month = np.asarray(dates.to_period("M").astype(str))
    is_last = np.zeros(n, dtype=bool)
    is_last[:-1] = month[:-1] != month[1:]
    is_last[-1] = True
    rank_in_month = np.zeros(n, dtype=int)
    counter = 0
    for i in range(n):
        if i > 0 and month[i] != month[i - 1]:
            counter = 0
        rank_in_month[i] = counter
        counter += 1
    tom = is_last | (rank_in_month < 3)

    sd_d = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    sd_on_base = sd_d * np.sqrt(on_share)
    sd_id_base = sd_d * np.sqrt(1.0 - on_share)
    mu_on = drift_ann / TRADING_DAYS_PER_YEAR * on_share
    mu_id = drift_ann / TRADING_DAYS_PER_YEAR * (1.0 - on_share)

    # A shared market factor (half the variance) plus idiosyncratic noise per asset.
    f_on = rng.normal(0.0, sd_on_base / np.sqrt(2.0), n)
    f_id = rng.normal(0.0, sd_id_base / np.sqrt(2.0), n)

    panel: dict[str, pd.DataFrame] = {}
    for name in ("control", "treated"):
        e_on = rng.normal(0.0, sd_on_base / np.sqrt(2.0), n)
        e_id = rng.normal(0.0, sd_id_base / np.sqrt(2.0), n)
        if name == "treated":
            bump = signal_strength * d_on_drift_bps * 1e-4
            tom_bump = signal_strength * d_tom_bps * 1e-4
            vol_mult = 1.0 + signal_strength * d_on_vol_mult
            # The lift hits the IDIOSYNCRATIC overnight noise only, so the shared market
            # factor still cancels inside the treated-minus-control difference and the
            # DiD stays an unbiased estimate of the planted drift shift.
            e_on = np.where(post, e_on * vol_mult, e_on)
            r_on = mu_on + f_on + e_on + np.where(post, bump, 0.0)
            r_on = r_on + np.where(post & tom, tom_bump, 0.0)
        else:
            r_on = mu_on + f_on + e_on
        r_id = mu_id + f_id + e_id
        close = np.empty(n)
        open_ = np.empty(n)
        lvl = 100.0
        for i in range(n):
            open_[i] = lvl * (1.0 + r_on[i])
            lvl = open_[i] * (1.0 + r_id[i])
            close[i] = lvl
        hi = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, sd_d / 3.0, n)))
        lo = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, sd_d / 3.0, n)))
        panel[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close}, index=dates
        )

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    panel["cash"] = pd.DataFrame(
        {"close": np.cumprod(np.full(n, cash_daily)) * 100.0}, index=dates
    )

    truth = {
        "signal_strength": signal_strength,
        "switch_date": str(dates[switch_i].date()),
        "switch_index": switch_i,
        "n_days": n,
        "seed": seed,
        "on_share": on_share,
        "vol_ann": vol_ann,
        "planted_on_drift_bps": signal_strength * d_on_drift_bps,
        "planted_on_vol_mult": signal_strength * d_on_vol_mult,
        "planted_tom_bps": signal_strength * d_tom_bps,
        "tom_frac": float(tom.mean()),
        "post_frac": float(post.mean()),
    }
    return panel, truth


def synthetic_daily(
    signal_strength: float = 1.0, seed: int = 926, **kw
) -> tuple[pd.DataFrame, dict]:
    """Single-asset convenience view of :func:`synthetic_panel` (the treated leg).

    Returns ``(ohlc, truth)`` for the treated asset only — handy for testing the
    night/day decomposition itself, which needs no control leg.
    """
    panel, truth = synthetic_panel(signal_strength=signal_strength, seed=seed, **kw)
    return panel["treated"], truth
