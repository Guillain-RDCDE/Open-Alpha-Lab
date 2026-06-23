"""Data layer for Study 388 (Lumber-Gold-Ratio).

Two tapes, one shape (a tz-naive daily price frame with columns ``lumber``, ``gold``,
``equity``, ``bond`` and a constant/observed short rate ``rf``):

- ``synthetic_daily`` — a *deterministic, offline* generator with a single planted-edge knob,
  ``pred_r``. It controls the only thing a lumber/gold **timing switch** can possibly harvest:
  a forward link between the lagged lumber/gold ratio *level* (relative to its own trailing
  mean) and the next day's *equity-minus-bond* spread. ``pred_r = 0`` is the **null** — the
  asset tapes are random walks uncorrelated with the ratio, so a switch can do no better than
  chance and the inference must NOT manufacture significance. ``pred_r > 0`` plants the
  *folklore-consistent* control: a **high** lumber/gold ratio (risk-on) precedes equities
  *out*-performing bonds, so a "rotate into stocks when the ratio is high" rule should add
  value. This is the study's null and positive control in one bottle.
- ``load_real`` / ``fetch_daily`` — the real tape. Cache-first: it reads cached parquets under
  ``_cache/`` and only touches the network on an explicit ``fetch=True``. Tickers: WOOD (the
  timber/forestry ETF used as the **lumber proxy** — see the proxy note below), GLD (gold),
  SPY (equity), TLT (long Treasuries) and ``^IRX`` (13-week T-bill yield) for the cash leg.

**Proxy note (named on the Signal axis).** The classic "lumber/gold ratio" is built from
*lumber futures* (Yahoo ``LBS=F``). That contract was discontinued in **May 2023** (replaced by
the smaller ``LBR`` future with little free history), so a *current-window* study cannot use it
to 2026. We therefore use the **WOOD** timber/forestry-equity ETF as a transparent, continuous
**lumber proxy** — a basket of forestry/timber *equities*, not the cash lumber price. It tracks
the lumber complex but is itself an equity (so it co-moves with stocks), which we name as a
caveat: if anything it *flatters* a "risk-on" reading. No look-ahead is baked in here — that
discipline lives in ``strategy.py`` (the switch is set from the ratio known at the close of *t*
and earns the return of *t+1*, one ``shift``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers (Yahoo!): lumber proxy (timber ETF), gold ETF, equity ETF, long-bond ETF,
# 13-week T-bill yield (the cash leg).
TICKERS = ["WOOD", "GLD", "SPY", "TLT", "^IRX"]
START_DATE = "2008-06-25"  # WOOD inception (the binding constraint on the common window)

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4000,
    pred_r: float = 0.0,
    annual_vol: float = 0.16,
    rf_annual: float = 0.02,
    lookback: int = 60,
    start: str = "2008-06-25",
    seed: int = 388,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily lumber/gold/equity/bond tape with a controllable timing link.

    Four price series are simulated in daily log-returns:

    - ``lumber`` and ``gold`` — independent random walks (log-vol ≈ ``annual_vol``). Their
      ratio ``lumber/gold`` is the predictor; its deviation from a trailing ``lookback``-day
      mean is the *signal* the switch reads.
    - ``equity`` and ``bond`` — the next day's *equity-minus-bond* log spread is
      ``pred_r * z_{t-1} + noise``, where ``z_{t-1}`` is the lagged standardised
      lumber/gold-ratio deviation. ``rf`` is a constant daily risk-free rate (the cash leg).

    ``pred_r`` is the *only* forecasting structure in the tape:

    - ``pred_r = 0``   → no predictability; the switch is a fair coin (the **null**).
    - ``pred_r > 0``   → folklore-consistent: a high ratio (risk-on) precedes equities
      out-performing bonds, so "stocks when the ratio is high, bonds when it is low" adds
      value (the **positive control**).
    - ``pred_r < 0``   → the inverted link.

    Returns ``(daily, truth)`` where ``truth`` records the planted parameters. Dates are
    consecutive weekdays from ``start`` (well under any ns-Timestamp overflow horizon).
    """
    rng = np.random.default_rng(seed)
    dv = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

    lumber_r = rng.normal(0.0, dv * 1.5, n_days)  # lumber is the more volatile leg
    gold_r = rng.normal(0.0, dv, n_days)

    lumber_p = 50.0 * np.exp(np.cumsum(lumber_r))
    gold_p = 120.0 * np.exp(np.cumsum(gold_r))
    ratio = lumber_p / gold_p
    log_ratio = np.log(ratio)

    # Standardised deviation of the log ratio from its trailing mean (the signal).
    s = pd.Series(log_ratio)
    mu = s.rolling(lookback, min_periods=lookback).mean()
    sd = s.rolling(lookback, min_periods=lookback).std(ddof=1)
    z = ((s - mu) / sd).to_numpy()

    bond_r = rng.normal(0.0, dv * 0.6, n_days)     # bonds lower vol
    eq_noise = rng.normal(0.0, dv, n_days)
    equity_r = np.zeros(n_days)
    for t in range(1, n_days):
        zt = z[t - 1]
        if not np.isfinite(zt):
            zt = 0.0
        # plant the edge in the equity-minus-bond spread: high ratio -> equity beats bond
        equity_r[t] = bond_r[t] + pred_r * zt + eq_noise[t]

    equity_p = 100.0 * np.exp(np.cumsum(equity_r))
    bond_p = 90.0 * np.exp(np.cumsum(bond_r))

    dates = pd.bdate_range(start=start, periods=n_days)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR

    daily = pd.DataFrame(
        {
            "lumber": lumber_p,
            "gold": gold_p,
            "equity": equity_p,
            "bond": bond_p,
            "rf": np.full(n_days, rf_daily),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "pred_r": pred_r,
        "annual_vol": annual_vol,
        "rf_annual": rf_annual,
        "lookback": lookback,
        "n_days": n_days,
        "seed": seed,
    }
    return daily, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first, network only on explicit fetch
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"daily_{safe}.parquet")


def fetch_daily(
    ticker: str,
    start: str = "2007-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily price history for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). A missing cache raises a clear ``FileNotFoundError`` rather
    than silently hitting the network, so the test-suite and reproducible core stay offline.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.copy()
        bars.columns = [str(c).lower() for c in bars.columns]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def load_real(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    lumber_ticker: str = "WOOD",
) -> pd.DataFrame:
    """Assemble the real lumber/gold/equity/bond/rf frame, cache-first.

    Returns a daily frame with columns ``lumber``, ``gold``, ``equity``, ``bond``, ``rf``
    aligned on the common business-day index (forward-filled at most 3 days over holidays
    before being dropped). ``rf`` is the ^IRX 13-week T-bill *daily* rate (annual yield / 252);
    if the ^IRX cache is absent it falls back to a flat 2%/yr cash leg so the frame still
    assembles.

    The cache is consulted first (``_cache/`` parquets); the network is touched only on
    ``fetch=True``.
    """
    lumber = fetch_daily(lumber_ticker, fetch=fetch, cache_dir=cache_dir)["close"].rename("lumber")
    gold = fetch_daily("GLD", fetch=fetch, cache_dir=cache_dir)["close"].rename("gold")
    equity = fetch_daily("SPY", fetch=fetch, cache_dir=cache_dir)["close"].rename("equity")
    bond = fetch_daily("TLT", fetch=fetch, cache_dir=cache_dir)["close"].rename("bond")

    # Cash leg: ^IRX is an *annualised yield in percent*. Convert to a daily simple rate.
    try:
        irx = fetch_daily("^IRX", fetch=fetch, cache_dir=cache_dir)["close"]
        rf = (irx / 100.0 / TRADING_DAYS_PER_YEAR).rename("rf")
    except (FileNotFoundError, KeyError):
        rf = pd.Series(0.02 / TRADING_DAYS_PER_YEAR, index=equity.index, name="rf")

    df = pd.concat([lumber, gold, equity, bond, rf], axis=1)
    df["rf"] = df["rf"].ffill().fillna(0.02 / TRADING_DAYS_PER_YEAR)
    df = df.ffill(limit=3).dropna(subset=["lumber", "gold", "equity", "bond"])
    return df


def have_real_cache(cache_dir: str = DEFAULT_CACHE, lumber_ticker: str = "WOOD") -> bool:
    """True iff the minimal real tape (lumber, gold, equity, bond) is cached locally."""
    needed = [lumber_ticker, "GLD", "SPY", "TLT"]
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in needed)


# back-compat alias used by some notebooks/examples
def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    return have_real_cache(cache_dir)


def fingerprint(df: pd.DataFrame, col: str = "equity") -> str:
    """A short content fingerprint of the panel (one column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]
