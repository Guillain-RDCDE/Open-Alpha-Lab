"""Data layer for Study 306 (Crack-Spread).

The **crack spread** is the refiner's margin in a single number: the value of the
products a refinery makes (gasoline + distillate) minus the cost of the crude it cracks.
The classic textbook proxy is the **3-2-1 crack**: 3 barrels of crude in → 2 of gasoline
+ 1 of heating oil out. In dollars per barrel:

    crack_321 = (2 * RB * 42 + HO * 42) / 3 - CL

where ``RB`` (RBOB gasoline) and ``HO`` (heating oil) are quoted in $/gallon (×42 to get
$/barrel) and ``CL`` (WTI crude) is already in $/barrel.

The folklore we test: a **fat crack spread predicts refiner stocks** (VLO, MPC, PSX) —
i.e. you can *time* refiner equity from the margin. The honest tension is that the crack
spread is the refiner's revenue line *contemporaneously*, so a tradable edge needs the
spread to **lead** the stock, not merely move with it.

Two tapes, one shape (a tz-naive daily frame of crack-spread level + refiner-basket
total-return index):

- ``synthetic_tape`` — a *deterministic, offline* generator. A ``lead`` knob plants a
  genuine predictive lead-lag (yesterday's crack change predicts today's stock return);
  ``lead=0`` makes the crack purely *contemporaneous* (it moves with the stock but cannot
  forecast it) — the study's null in a bottle. A test asserts the timer beats buy-and-hold
  only when a real lead is planted, and not otherwise.
- ``load_real`` — the real Yahoo! tape (``yfinance``), cache-only by default, so the
  test-suite and the reproducible core never touch the network. RB=F, HO=F, CL=F build the
  crack; VLO/MPC/PSX an equal-weight refiner basket.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (one ``shift``,
signal known at the close of *t* earns the return of *t+1*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The futures legs and the refiner basket used to build the real tape.
CRACK_TICKERS = ("RB=F", "HO=F", "CL=F")  # RBOB gasoline, heating oil, WTI crude
REFINERS = ("VLO", "MPC", "PSX")          # Valero, Marathon, Phillips 66
GAL_PER_BBL = 42.0


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_tape(
    n_days: int = 2500,
    lead: float = 0.0,
    crack_vol: float = 0.02,
    stock_vol: float = 0.018,
    start: str = "2012-01-03",
    seed: int = 306,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily (crack_spread, stock_index) tape with a *known* lead-lag.

    The crack spread follows a mean-reverting level around 15 $/bbl. The refiner-stock
    log return is

        stock_ret_t = beta * dcrack_t  +  lead * dcrack_{t-1}  +  eps_t

    where ``dcrack_t`` is the (standardised) daily change in the crack spread.

    - ``beta`` (fixed) is the **contemporaneous** co-movement — the crack *is* today's
      margin, so the stock moves *with* it. This term carries **no** predictive power.
    - ``lead`` is the only **forecastable** part: yesterday's crack change predicting
      today's stock return. ``lead = 0`` → the crack is purely coincident; a timer that
      acts on yesterday's crack can harvest nothing. ``lead > 0`` → a real edge to harvest.

    Returns ``(frame, truth)`` with columns ``crack``, ``stock`` (a total-return index),
    and ``stock_ret`` (its log return). ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Mean-reverting crack-spread *level* (AR(1) in levels around 15 $/bbl).
    crack = np.empty(n_days)
    crack[0] = 15.0
    phi, mu = 0.97, 15.0
    crack_shocks = rng.normal(0.0, crack_vol * mu, n_days)
    for t in range(1, n_days):
        crack[t] = mu + phi * (crack[t - 1] - mu) + crack_shocks[t]

    dcrack = np.diff(crack, prepend=crack[0])
    # Standardise the crack change so beta/lead are in interpretable units.
    dcrack_std = dcrack / (dcrack.std() + 1e-12)

    beta = 0.6  # contemporaneous co-movement (NOT predictive)
    eps = rng.normal(0.0, stock_vol, n_days)
    stock_ret = beta * dcrack_std * stock_vol + lead * np.r_[0.0, dcrack_std[:-1]] * stock_vol + eps
    stock = 100.0 * np.exp(np.cumsum(stock_ret))

    frame = pd.DataFrame(
        {"crack": crack, "stock": stock, "stock_ret": stock_ret},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "lead": lead,
        "beta": beta,
        "crack_vol": crack_vol,
        "stock_vol": stock_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def _fetch_close(ticker: str, start: str, fetch: bool, cache_dir: str) -> pd.Series:
    """Adjusted-close (total-return for equities) for one ticker; cache-only unless fetch."""
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached tape for {ticker} at {path}. "
                f"Run examples/verify.py --fetch once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(ticker, start=start, interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["close"]]
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)
        s = df["close"]
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.name = ticker
    return s


def crack_321(rb: pd.Series, ho: pd.Series, cl: pd.Series) -> pd.Series:
    """The 3-2-1 crack spread in $/barrel from RBOB ($/gal), heating oil ($/gal), WTI ($/bbl).

    3-2-1: (2 gallons-of-gasoline-value + 1 heating-oil-value), per barrel of crude, minus
    crude. RB and HO are $/gallon, so ×42 to dollars per barrel; CL is already $/barrel.
    """
    return (2.0 * rb * GAL_PER_BBL + ho * GAL_PER_BBL) / 3.0 - cl


def load_real(
    start: str = "2012-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily (crack_spread, refiner-basket) tape; cache-only unless ``fetch=True``.

    Builds the 3-2-1 crack from RB=F/HO=F/CL=F and an **equal-weight** total-return refiner
    basket from VLO/MPC/PSX (auto-adjusted closes, so dividends are in — labeled
    total-return). Network is touched only on an explicit ``fetch=True``.

    Returns a frame with columns ``crack`` (level, $/bbl), ``stock`` (basket TR index,
    base 100), ``stock_ret`` (its daily log return), aligned on the common trading days.
    """
    rb = _fetch_close("RB=F", start, fetch, cache_dir)
    ho = _fetch_close("HO=F", start, fetch, cache_dir)
    cl = _fetch_close("CL=F", start, fetch, cache_dir)
    legs = pd.concat([rb, ho, cl], axis=1, sort=False).dropna()
    crack = crack_321(legs["RB=F"], legs["HO=F"], legs["CL=F"])

    rets = []
    for t in REFINERS:
        px = _fetch_close(t, start, fetch, cache_dir)
        rets.append(np.log(px / px.shift(1)))
    basket_ret = pd.concat(rets, axis=1, sort=False).dropna().mean(axis=1)

    frame = pd.concat(
        {"crack": crack, "stock_ret": basket_ret}, axis=1, sort=False
    ).dropna()
    frame["stock"] = 100.0 * np.exp(frame["stock_ret"].cumsum())
    frame.index.name = "date"
    return frame[["crack", "stock", "stock_ret"]]


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (crack column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["crack"].to_numpy()).tobytes())
    return h.hexdigest()[:12]
