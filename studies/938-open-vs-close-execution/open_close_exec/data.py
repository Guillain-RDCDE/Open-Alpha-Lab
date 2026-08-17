"""Data layer for Study 938 — Open or Close (where you fill a timing rule's trades).

Two tapes, one shape (a date-indexed daily OHLC frame with adjusted ``open`` and ``close``):

- ``fetch`` / ``load_ohlc`` — daily **adjusted** open and close from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for four liquid index ETFs (SPY, IWM, EEM, EFA)
  plus a cash proxy (BIL, the 1-3 month T-bill ETF). ``fetch`` touches the network and
  caches parquet under the **shared** ``studies/_cache`` as ``ohlc_<TICKER>_1d.parquet``
  (retry up to 4x); ``load_ohlc`` reads that cache **offline** and never imports yfinance.
  The whole test-suite runs with NO cache present (synthetic-only), so CI is green on a
  fresh checkout where ``_cache/`` is git-ignored and absent.

  **What ``auto_adjust=True`` does to the open.** Yahoo applies one split- and
  dividend-adjustment *factor per day* to all four of O/H/L/C. Two consequences we lean on
  and label everywhere:

  * the same-day **intraday** return ``close/open - 1`` is unaffected by the factor — it is
    a **price-only** return, exactly what a real fill at the open and a mark at the close
    would have earned, ex-dividend;
  * the **overnight** return ``open_t / close_{t-1} - 1`` inherits the whole adjustment
    step, so on ex-dividend days the dividend is credited to the *overnight* leg. The
    close-to-close product ``(1 + overnight)(1 + intraday)`` is therefore the familiar
    **total return**, and the dividend sits at night. Every number in this study says which
    of the two it is.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  two-state Markov trend tape (so a 10-month moving-average rule actually trades) whose
  close-to-close path is then split into an overnight and an intraday leg. The
  ``signal_strength`` knob plants **intraday continuation**: the day's open-to-close move
  is tilted by the sign of the trailing month's return. Critically the split leaves the
  close-to-close tape *bit-identical* across ``signal_strength`` — so the close-executed
  arm is invariant by construction and only the open-executed arm can move. At
  ``signal_strength = 0`` the two execution venues must tie (the null); at
  ``signal_strength = 1`` filling at the open must win. Seeds are fixed.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal
formed on data through the close of day ``t`` is filled on day ``t+1``, at that day's open
for one arm and at that day's close for the other).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The four risk tapes: US large (SPY), US small (IWM), emerging (EEM), developed ex-US
# (EFA). BIL is the tradable cash leg the rule sits in when it is out of the market.
TICKERS = ("SPY", "IWM", "EEM", "EFA")
CASH = "BIL"
ALL_TICKERS = TICKERS + (CASH,)

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# The tradable window is gated by BIL's 2007-05-30 inception (the cash leg has to be
# a real, buyable total return, not an assumed rate).
START = "2007-05-30"

OHLC_COLS = ("open", "high", "low", "close")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily adjusted OHLC, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ohlc_{safe}_1d.parquet")


def fetch(
    tickers=ALL_TICKERS,
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily adjusted OHLC for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so all four
    of O/H/L/C carry the same split/dividend factor — see the module docstring for what
    that means for the intraday (price-only) and overnight (dividend-bearing) legs.
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
        df = raw[list(OHLC_COLS)].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["open", "close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=ALL_TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's OHLC parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_ohlc(
    tickers=ALL_TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    start: str | None = START,
) -> dict[str, pd.DataFrame]:
    """Read cached daily adjusted OHLC OFFLINE into ``{ticker: frame}``.

    Each frame is indexed by date with ``open``/``close`` columns, sliced to ``asof`` (and
    from ``start`` when given) so the sample never creeps. Raises ``FileNotFoundError`` if
    any ticker is missing — the offline core and the test-suite never touch the network.
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached OHLC for {tk} at {path}. "
                f"Call open_close_exec.data.fetch() once to populate the shared cache."
            )
        df = pd.read_parquet(path)[["open", "close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()
        df = df[df.index <= pd.Timestamp(asof)]
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        out[tk] = df.dropna()
    return out


def fingerprint(frames) -> str:
    """Short content fingerprint of one or more OHLC frames, for the as-of data stamp."""
    if isinstance(frames, pd.DataFrame):
        frames = {"_": frames}
    h = hashlib.sha1()
    for key in sorted(frames):
        arr = np.ascontiguousarray(frames[key].to_numpy(dtype=float))
        h.update(np.nan_to_num(arr, nan=0.0).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (planted intraday continuation)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 25,
    drift_bull: float = 0.16,          # annualised bull drift
    drift_bear: float = -0.22,         # annualised bear drift (so the MA rule trades)
    vol_bull: float = 0.13,
    vol_bear: float = 0.26,
    p_bull_to_bear: float = 0.025,     # monthly prob of entering a bear
    p_bear_to_bull: float = 0.09,      # monthly prob of leaving it
    intraday_share: float = 0.45,      # share of daily variance booked open->close
    kappa: float = 0.005,              # planted intraday continuation, per unit z, per day
    signal_strength: float = 1.0,      # 0 = venue-neutral null, 1 = full continuation
    start: str = "2004-01-02",
    seed: int = 938,
    cash_rate_ann: float = 0.025,
) -> tuple[pd.DataFrame, dict]:
    """A daily OHLC-like tape (open, close) plus a cash index, with a planted split.

    The close-to-close path is generated first, from a two-state Markov trend regime, and
    is **identical for every ``signal_strength``** — the knob only decides how that fixed
    daily move is *split* between the overnight and the intraday leg:

    ``intraday_log[t] = kappa * signal_strength * z[t] + noise``, where ``z[t]`` is the
    standardised trailing 21-day return **through ``t-1``**. So after a strong month the
    day's gain is booked disproportionately open-to-close (a trend-follower filling at the
    open captures it), and after a weak month the loss is. The overnight leg takes the
    residual so ``(1 + overnight)(1 + intraday)`` reproduces the close-to-close return
    exactly.

    Returns ``(frame, truth)`` where ``frame`` has ``open``, ``close`` and ``cash`` columns
    and ``truth`` records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    d_bull = drift_bull / TRADING_DAYS_PER_YEAR
    d_bear = drift_bear / TRADING_DAYS_PER_YEAR
    s_bull = vol_bull / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_bear = vol_bear / np.sqrt(TRADING_DAYS_PER_YEAR)

    MONTH = 21
    regime = np.zeros(n_days, dtype=int)   # 0 = bull, 1 = bear
    state = 0
    for i in range(n_days):
        if i % MONTH == 0 and i > 0:
            if state == 0:
                state = 1 if rng.random() < p_bull_to_bear else 0
            else:
                state = 0 if rng.random() < p_bear_to_bull else 1
        regime[i] = state

    # Close-to-close log returns: fixed given the seed, independent of signal_strength.
    r_cc = np.where(
        regime == 0,
        rng.normal(d_bull, s_bull, n_days),
        rng.normal(d_bear, s_bear, n_days),
    )
    close = 100.0 * np.exp(np.cumsum(r_cc))

    # Trailing 21-day momentum z-score through t-1 (strictly past — no same-day peek).
    csum = np.concatenate([[0.0], np.cumsum(r_cc)])
    mom = np.full(n_days, np.nan)
    mom[MONTH:] = csum[MONTH:-1] - csum[:-MONTH - 1]
    sd = np.nanstd(mom)
    z = np.nan_to_num(mom / sd if sd > 0 else mom, nan=0.0)

    # Split each day's fixed close-to-close move into intraday + overnight.
    sig_i = np.sqrt(intraday_share) * np.std(r_cc)
    noise_i = rng.normal(0.0, sig_i, n_days)
    r_id = kappa * float(signal_strength) * z + noise_i
    r_on = r_cc - r_id                     # residual -> the product is exact

    open_px = np.empty(n_days)
    open_px[0] = 100.0 * np.exp(r_on[0])
    open_px[1:] = close[:-1] * np.exp(r_on[1:])

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    frame = pd.DataFrame(
        {"open": open_px, "close": close, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength),
        "kappa": kappa,
        "intraday_share": intraday_share,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "bear_frac": float(regime.mean()),
        "cash_rate_ann": cash_rate_ann,
        "planted_intraday_tilt_bps": float(kappa * float(signal_strength) * 1e4),
    }
    return frame, truth


def synthetic_panel(
    n_assets: int = 4,
    signal_strength: float = 1.0,
    seed: int = 938,
    **kwargs,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """``n_assets`` synthetic tapes (SYN0…), each from ``synthetic_daily`` at a fresh seed.

    Mirrors the four-tape real panel: the same rule, the same planted continuation, four
    independent worlds — so the pooled test in ``strategy.py`` can be exercised offline.
    Deterministic given ``seed``.
    """
    frames: dict[str, pd.DataFrame] = {}
    truth = {"n_assets": n_assets, "signal_strength": float(signal_strength), "seed": seed}
    for i in range(n_assets):
        f, t = synthetic_daily(signal_strength=signal_strength, seed=seed + 17 * i, **kwargs)
        frames[f"SYN{i}"] = f
        truth[f"SYN{i}"] = t
    return frames, truth
