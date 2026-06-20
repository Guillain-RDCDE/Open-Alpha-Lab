"""Data layer for Study 324 (Bitcoin-Treasury).

The question is whether MicroStrategy (ticker MSTR) is anything more than a *levered
Bitcoin proxy*. So the data layer produces two aligned daily close series — a "BTC" tape
and an "MSTR" tape — in one shape (a tz-naive daily frame with columns ``btc`` and
``mstr``), on two tapes:

- ``synthetic_pair`` — a *deterministic, offline* generator. BTC is a random walk with
  drift; MSTR is built **as** a levered function of BTC: ``r_mstr = beta * r_btc + alpha
  + premium_noise + idio``. The two knobs the whole study turns on are exposed:

  * ``alpha`` — a constant daily drift MSTR earns *on top of* its BTC exposure. This is
    the only thing that could make "hold MSTR instead of BTC" a real edge rather than a
    leverage bet. ``alpha = 0`` is the null: MSTR is *pure* levered beta, nothing more.
  * ``premium_vol`` — an AR(1) "NAV-premium" wobble (the market price of MSTR drifting
    above/below the value of the coins it holds). It adds idiosyncratic variance with
    **zero** expected drift, so it can fatten MSTR's vol and dent its Sharpe without ever
    being a source of return — exactly the trap a naive "MSTR beat BTC" comparison falls
    into.

  A test can therefore assert the harness recovers a real MSTR-over-levered-BTC edge
  *only* when ``alpha > 0`` is planted, and finds nothing on the null.

- ``fetch_pair`` — the real Yahoo! daily tape (``yfinance``) for MSTR and BTC-USD,
  cache-only by default so the test-suite and the reproducible core never touch the
  network. There is also a cache-first ``load_real`` convenience that falls back to the
  synthetic tape (clearly flagged) when no cache is present, for the notebooks.

No look-ahead is baked in here — the one execution lag lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 365  # BTC trades every day; MSTR is forward-filled to match.


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_pair(
    n_days: int = 1500,
    beta: float = 1.8,
    alpha: float = 0.0,
    btc_drift: float = 0.0008,
    btc_vol: float = 0.035,
    premium_vol: float = 0.0,
    premium_phi: float = 0.97,
    idio_vol: float = 0.010,
    start: str = "2019-01-01",
    seed: int = 324,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily (btc, mstr) close pair with a *known* MSTR construction.

    BTC simple returns are ``r_btc = btc_drift + eps_btc`` with ``eps_btc`` i.i.d. normal
    of standard deviation ``btc_vol``. MSTR is then built **as a levered function of BTC**,
    in simple-return space so the OLS recovers ``alpha`` exactly::

        r_mstr = beta * r_btc + alpha + (prem_t - prem_{t-1}) + idio_t

    where ``prem_t`` is an AR(1) "NAV-premium" path (``prem_t = premium_phi * prem_{t-1} +
    noise``, scaled to ``premium_vol``) and ``idio_t`` is i.i.d. idiosyncratic noise. The
    premium term enters as a *first difference*, so over any full sample it contributes
    ~zero drift — pure variance, no return — which is the whole point: a NAV-premium that
    swings around does **not** pay you, it just adds risk.

    Knobs that decide the study:

    - ``alpha = 0``  → MSTR is pure levered beta + noise; "hold MSTR not BTC" is a
      leverage bet with no genuine edge (the **null**).
    - ``alpha > 0``  → MSTR earns a real daily drift on top of its BTC exposure (the
      **positive control** the harness must detect).
    - ``premium_vol > 0`` → adds the NAV-premium wobble (return-free variance).

    Returns ``(frame, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # Decorative monthly-free daily index built safely (no huge date_range overflow).
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Work in SIMPLE-return space so the OLS of MSTR-on-BTC recovers alpha exactly:
    # MSTR is built directly AS beta×BTC + alpha + premium-wobble + idio, with no
    # log/simple convexity term sneaking a spurious intercept into the null.
    eps_btc = rng.normal(0.0, btc_vol, n_days)
    sr_btc = btc_drift + eps_btc  # simple daily BTC return

    # AR(1) NAV-premium path, scaled so its stationary std ≈ premium_vol.
    prem = np.zeros(n_days)
    if premium_vol > 0:
        innov_sd = premium_vol * np.sqrt(1.0 - premium_phi**2)
        for i in range(1, n_days):
            prem[i] = premium_phi * prem[i - 1] + rng.normal(0.0, innov_sd)
    d_prem = np.empty(n_days)
    d_prem[0] = 0.0
    d_prem[1:] = np.diff(prem)

    idio = rng.normal(0.0, idio_vol, n_days)
    sr_mstr = beta * sr_btc + alpha + d_prem + idio  # simple daily MSTR return

    # Compound simple returns into price paths (clip to keep prices strictly positive).
    btc_close = 100.0 * np.cumprod(1.0 + np.clip(sr_btc, -0.95, None))
    mstr_close = 100.0 * np.cumprod(1.0 + np.clip(sr_mstr, -0.95, None))

    frame = pd.DataFrame(
        {"btc": btc_close, "mstr": mstr_close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "beta": beta,
        "alpha": alpha,
        "btc_drift": btc_drift,
        "btc_vol": btc_vol,
        "premium_vol": premium_vol,
        "premium_phi": premium_phi,
        "idio_vol": idio_vol,
        "n_days": n_days,
        "seed": seed,
        "premium_path": prem,  # the latent NAV premium (for diagnostics/figures)
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace("-", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_close(
    ticker: str,
    start: str = "2014-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.Series:
    """Real daily adjusted close for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). Returns a tz-naive close Series named ``close``.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_close({ticker!r}, fetch=True) once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw.rename(columns=str.lower)["close"]
        s.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        s.to_frame("close").to_parquet(path)

    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.rename("close")


def fetch_pair(
    mstr_ticker: str = "MSTR",
    btc_ticker: str = "BTC-USD",
    start: str = "2014-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real aligned (btc, mstr) daily close frame; cache-only unless ``fetch=True``.

    Both legs are total-return (auto-adjusted) closes. They are inner-joined on the
    intersection of their trading calendars (MSTR trades weekdays, BTC every day), then
    forward-fill is *not* applied — we keep only days both quote, so returns line up.
    """
    mstr = fetch_close(mstr_ticker, start=start, fetch=fetch, cache_dir=cache_dir)
    btc = fetch_close(btc_ticker, start=start, fetch=fetch, cache_dir=cache_dir)
    frame = pd.concat({"mstr": mstr, "btc": btc}, axis=1, sort=True).dropna()
    return frame[["btc", "mstr"]]


def load_real(
    cache_dir: str = DEFAULT_CACHE,
    **synth_kwargs,
) -> tuple[pd.DataFrame, str]:
    """Cache-first loader for the notebooks: real (btc, mstr) if cached, else synthetic.

    Returns ``(frame, tape)`` where ``tape`` is ``"real"`` or ``"synthetic"`` so a cell
    can banner exactly which tape it is showing. Never touches the network.
    """
    try:
        frame = fetch_pair(fetch=False, cache_dir=cache_dir)
        if len(frame) > 100:
            return frame, "real"
    except (FileNotFoundError, KeyError, OSError):
        pass
    frame, _ = synthetic_pair(**synth_kwargs)
    return frame, "synthetic"


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a (btc, mstr) frame, for the as-of stamp."""
    arr = np.ascontiguousarray(frame[["btc", "mstr"]].to_numpy())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
