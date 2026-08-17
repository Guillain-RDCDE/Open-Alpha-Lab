"""Data layer for Study 947 — The Buffer Ladder (laddered buffer fund vs its own vintages).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the laddered buffer fund (**BUFR**), the four
  quarterly Innovator S&P 500 Power Buffer vintages (**PJAN, PAPR, PJUL, POCT**), the
  underlying market (**SPY**) and a cash proxy (**BIL**, the 1-3 month T-bill ETF).
  ``fetch`` touches the network and caches parquet under the shared ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI is
  green on a fresh checkout where ``_cache/`` is git-ignored and absent.

- ``synthetic_panel`` / ``synthetic_daily`` — *deterministic, offline* generators. A market
  factor, a cash accrual leg, and a panel of buffer "vintages" that each load on the market
  with a vintage-specific idiosyncratic *entry-point luck* term. A laddered wrapper is then
  built as the equal-weight basket of those vintages **plus a planted per-annum ladder
  alpha** and **minus a planted extra fee layer**. The ``signal_strength`` knob scales the
  planted alpha: at ``signal_strength=0`` the ladder is exactly the basket net of the fee
  (the null — the detector must report ~-fee and nothing more); at ``signal_strength=1``
  there is a real, recoverable laddering premium. Seeds are fixed → tests are deterministic.

The question this study asks: **BUFR** wraps the whole Power Buffer ladder in a single
ticker and charges an extra fee layer on top of the underlying funds' expense ratios. Does
that laddering do anything a private investor could not do by simply buying the vintages
himself and equal-weighting them — i.e. is there a laddering premium beyond averaging away
entry-point luck, and is the entry-point luck it averages away even large enough to matter?

No look-ahead is baked in here — that discipline lives in ``strategy.py``: every weight
(the beta match, the basket rebalance) is estimated on data through day ``t`` and applied
at day ``t+1``.
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

# The laddered wrapper, its four quarterly constituents, the underlying, and cash.
LADDER = "BUFR"
VINTAGES = ("PJAN", "PAPR", "PJUL", "POCT")
MARKET = "SPY"
CASH = "BIL"
TICKERS = (LADDER,) + VINTAGES + (MARKET, CASH)

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# ---------------------------------------------------------------------------
# NON-TAPE INPUTS — declared PROXIES / ASSUMPTIONS, swept in strategy.fee_sweep.
# ---------------------------------------------------------------------------
# Prospectus expense ratios, quoted (not measured on the tape). A single Power Buffer
# vintage charges 0.79%/yr. BUFR charges a 0.15%/yr management fee on top of the acquired
# funds' fees, so the *extra layer* an investor pays for the wrapper is roughly 0.15-0.26
# pp/yr depending on how the acquired-fund fee waiver is counted. Every published NAV
# return below is already NET of whatever was actually charged; these numbers are used
# only to build the "pre-extra-fee" counterfactual and are swept, never trusted.
FEE_SINGLE_VINTAGE_PCT = 0.79      # PROXY: prospectus ER of one Power Buffer vintage
FEE_LADDER_EXTRA_PCT = 0.20        # PROXY/ASSUMPTION: BUFR's incremental wrapper fee
FEE_EXTRA_GRID_PCT = (0.00, 0.10, 0.20, 0.26, 0.40)   # the sweep


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2018-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. Uses ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted total return. That matters here:
    the buffer vintages make annual capital-gain distributions and BIL's whole return is a
    distribution, so a price-only comparison would silently mis-rank every arm.
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
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
    dropna: bool = True,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker (the adjusted close),
    sliced to ``asof`` so the sample never creeps, and (by default) reduced to the common
    window on which *every* arm trades — BUFR's 2020-08 inception gates the race. Raises
    ``FileNotFoundError`` if any ticker is missing: the offline core and the whole
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call buffer_ladder.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    if dropna:
        df = df.dropna()
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 6,
    drift_mkt: float = 0.09,
    vol_mkt: float = 0.17,
    cash_rate_ann: float = 0.03,
    start: str = "2018-01-02",
    seed: int = 947,
) -> tuple[pd.DataFrame, dict]:
    """A plain daily (market, cash) tape — the factor the synthetic vintages load on.

    Returns ``(prices, truth)`` where ``prices`` has columns ``market`` (a total-return
    equity index) and ``cash`` (a cash accrual index). Deterministic given ``seed``.
    Used on its own by the beta-match tests and internally by ``synthetic_panel``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    d = drift_mkt / TRADING_DAYS_PER_YEAR
    s = vol_mkt / np.sqrt(TRADING_DAYS_PER_YEAR)
    r_mkt = rng.normal(d, s, n_days)
    market = 100.0 * np.cumprod(1.0 + r_mkt)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = 100.0 * np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"market": market, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "n_days": n_days, "n_years": n_years, "seed": seed,
        "drift_mkt": drift_mkt, "vol_mkt": vol_mkt, "cash_rate_ann": cash_rate_ann,
    }
    return prices, truth


def synthetic_panel(
    n_years: int = 6,
    n_vintages: int = 4,
    beta_vintage: float = 0.45,
    entry_luck_vol: float = 0.035,     # annualised vintage-idiosyncratic vol
    wrapper_noise_vol: float = 0.020,  # annualised wrapper-vs-basket tracking noise
    ladder_alpha_ann: float = 0.04,    # the planted laddering premium (before the fee)
    extra_fee_ann: float = 0.002,      # the planted extra wrapper fee layer
    signal_strength: float = 1.0,      # 0 = no laddering premium (the null), 1 = full
    drift_mkt: float = 0.09,
    vol_mkt: float = 0.17,
    cash_rate_ann: float = 0.03,
    start: str = "2018-01-02",
    seed: int = 947,
) -> tuple[pd.DataFrame, dict]:
    """A panel of buffer "vintages" plus a laddered wrapper built on top of them.

    Construction (all in simple daily returns, then compounded into close levels):

    - ``market`` / ``cash`` come from :func:`synthetic_daily`.
    - each vintage ``v1..vN`` earns ``beta_vintage * r_market`` plus an independent
      *entry-point luck* shock of annualised vol ``entry_luck_vol`` plus the cash rate on
      its un-invested notional — the stylised shape of a defined-outcome fund: damped
      market beta plus a vintage-specific path effect from when its outcome period began.
    - ``ladder`` = the equal-weight basket of those vintages, **plus**
      ``signal_strength * ladder_alpha_ann`` per annum, **minus** ``extra_fee_ann``, plus
      an independent tracking noise of annualised vol ``wrapper_noise_vol`` (the wrapper
      does not hold exactly these four vintages, so its gap is noisy, not deterministic —
      without this the detector's *t* would be meaninglessly large).

    The ``signal_strength`` knob is the control:

    - ``signal_strength = 1`` → a genuine laddering premium is present and the race must
      recover it (net of the planted fee).
    - ``signal_strength = 0`` → the ladder is *exactly* the DIY basket net of the fee, so
      the measured gap must sit at ``-extra_fee_ann`` and the machinery must not
      manufacture anything else (the null).

    Returns ``(prices, truth)`` where ``prices`` carries ``ladder``, ``v1..vN``,
    ``market`` and ``cash`` close levels, and ``truth`` records the planted parameters.
    Deterministic given ``seed``.
    """
    base, base_truth = synthetic_daily(
        n_years=n_years, drift_mkt=drift_mkt, vol_mkt=vol_mkt,
        cash_rate_ann=cash_rate_ann, start=start, seed=seed,
    )
    idx = base.index
    n_days = len(idx)
    r_mkt = base["market"].pct_change().fillna(0.0).to_numpy()
    r_cash = base["cash"].pct_change().fillna(0.0).to_numpy()

    rng = np.random.default_rng(seed + 1)
    idio_sd = entry_luck_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    vint_names = [f"v{i + 1}" for i in range(n_vintages)]

    cols: dict[str, np.ndarray] = {}
    vint_rets = np.zeros((n_vintages, n_days))
    for i, name in enumerate(vint_names):
        idio = rng.normal(0.0, idio_sd, n_days)
        r_v = beta_vintage * r_mkt + (1.0 - beta_vintage) * r_cash + idio
        vint_rets[i] = r_v
        cols[name] = 100.0 * np.cumprod(1.0 + r_v)

    alpha_d = signal_strength * ladder_alpha_ann / TRADING_DAYS_PER_YEAR
    fee_d = extra_fee_ann / TRADING_DAYS_PER_YEAR
    noise = rng.normal(0.0, wrapper_noise_vol / np.sqrt(TRADING_DAYS_PER_YEAR), n_days)
    r_ladder = vint_rets.mean(axis=0) + alpha_d - fee_d + noise
    cols["ladder"] = 100.0 * np.cumprod(1.0 + r_ladder)
    cols["market"] = base["market"].to_numpy()
    cols["cash"] = base["cash"].to_numpy()

    prices = pd.DataFrame(cols, index=idx)
    prices = prices[["ladder"] + vint_names + ["market", "cash"]]

    truth = dict(base_truth)
    truth.update({
        "n_vintages": n_vintages,
        "vintages": tuple(vint_names),
        "beta_vintage": beta_vintage,
        "entry_luck_vol": entry_luck_vol,
        "wrapper_noise_vol": wrapper_noise_vol,
        "ladder_alpha_ann": ladder_alpha_ann,
        "extra_fee_ann": extra_fee_ann,
        "signal_strength": signal_strength,
        # what an unbiased detector should report for (ladder - DIY basket), per annum:
        "expected_gap_ann": signal_strength * ladder_alpha_ann - extra_fee_ann,
    })
    return prices, truth
