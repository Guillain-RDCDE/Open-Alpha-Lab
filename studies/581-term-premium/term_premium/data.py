"""Data layer for Study 581 (Term-Premium).

The **term premium** is the extra yield an investor demands for holding a long-maturity bond
instead of rolling short bills — the part of the long yield that is *not* explained by the
expected path of future short rates. Adrian, Crump & Moench (2013, "ACM") estimate it with a
five-factor affine term-structure model and publish a daily 10-year term-premium series at the
New York Fed. The claim this study tests: a **model term-premium estimate** — high when long
bonds are richly compensated for duration risk, low/negative when they are not — should *time*
long-duration (TLT) returns. Buy duration when the premium is fat; step aside when it is thin.

The ACM series itself is not reachable from a no-key retail stack (it is a NY-Fed CSV, not a
Yahoo ticker), so this study builds an **ACM-style proxy** from what yfinance *does* expose:

    term_premium_t  =  y10_t  -  expectations_t

where ``y10`` is the 10-year Treasury yield (``^TNX``) and ``expectations`` is an
exponentially-weighted moving average of the 3-month bill rate (``^IRX``) — a cheap stand-in for
"the average expected future short rate" that a full affine model estimates. Subtracting the
expectations component is exactly what makes a term-premium estimate *different from the raw
curve slope* (Study 132): the raw 10y-3m slope moves with the level and expected path of policy;
the term premium strips those out. The proxy is a simplification — named openly on the SIGNAL
axis, and the reason a synthetic-only certification is impossible here (there is no free ACM tape
to clear a robust *t* on). The proxy is the model estimate we *can* build.

Two tapes, one schema (a tz-naive daily frame with ``TLT_close``, ``TNX``, ``IRX``, ``tp``):

- ``synthetic_daily`` — a deterministic, offline generator. A single knob, ``tp_signal``, plants
  the effect: forward TLT returns load positively on the *lagged* term-premium rank
  (``tp_signal > 0`` = a genuine timing edge; ``= 0`` = the null). Tests never touch the network.
- ``fetch_daily`` — the real yfinance tape (TLT + ^TNX + ^IRX daily closes), cache-first into this
  study's own ``_cache/`` so the reproducible core never needs the network.

No look-ahead: the term premium measured at the close of day *t* forms a signal that trades at
the close of day *t+1* (a one-bar execution lag, applied identically on both tapes and in the
robustness sweep — see ``strategy.py``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers: the long-duration instrument + the 10y yield + the 3m bill.
TICKERS = ["TLT", "^TNX", "^IRX"]

# The expectations component is an EWMA of the short rate; this half-life (in trading days)
# controls how much of the short-rate path is treated as "expected". ~1 year of memory.
EXPECT_HALFLIFE = 252


# ---------------------------------------------------------------------------
# The ACM-style term-premium proxy (shared by both tapes)
# ---------------------------------------------------------------------------
def term_premium_proxy(
    y10: pd.Series, short: pd.Series, halflife: int = EXPECT_HALFLIFE
) -> pd.Series:
    """ACM-style term-premium proxy: 10y yield minus an EWMA-expectations component.

    ``expectations = EWMA(short_rate, halflife)`` proxies "the average expected future short
    rate"; the term premium is what the long yield pays *above* that. Uses only trailing data
    (the EWMA is causal), so the value at day *t* is public at the close of day *t*.
    """
    expectations = short.ewm(halflife=halflife, adjust=False, min_periods=1).mean()
    return (y10 - expectations).rename("tp")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 5000,
    tp_signal: float = 0.0,
    tlt_vol: float = 0.008,
    start: str = "2003-01-02",
    seed: int = 581,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape with a known amount of term-premium timing power.

    Structure:

    - ``TNX`` (10y yield) is an AR(1) in level space around ~4%; ``IRX`` (3m bill) is a smoother
      AR(1) that slowly tracks the 10y with a lag — so the curve inverts and re-steepens like the
      real one.
    - The **term premium** ``tp = TNX - EWMA(IRX)`` (the same proxy as the real tape) is formed,
      then ranked out-of-sample over a trailing 252-day window (``tp_rank`` in [0,1]).
    - Forward TLT returns load on the *lagged* term-premium rank::

          tlt_ret_t = tp_signal * (tp_rank_{t-1} - 0.5) + eps_t

      ``tp_signal = 0`` makes TLT a pure random walk (the null); ``tp_signal > 0`` plants a
      genuine edge — a fat term premium predicts higher forward duration returns.

    Returns ``(df, truth)`` with columns ``[TNX, IRX, tp, TLT_close, TLT_ret]``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # 10y yield: AR(1), range ~[1.5%, 8%]
    tnx = np.empty(n_days)
    tnx[0] = 4.0
    phi_tnx, sigma_tnx, mu_tnx = 0.97, 0.06, 4.0
    for i in range(1, n_days):
        tnx[i] = phi_tnx * tnx[i - 1] + (1 - phi_tnx) * mu_tnx + rng.normal(0, sigma_tnx)
    tnx = np.clip(tnx, 0.5, 9.0)

    # 3m bill: smoother AR(1) lagging the 10y (produces inversions when policy catches up)
    irx = np.empty(n_days)
    irx[0] = 2.0
    phi_irx, sigma_irx = 0.985, 0.04
    for i in range(1, n_days):
        irx[i] = phi_irx * irx[i - 1] + (1 - phi_irx) * (tnx[i - 1] - 1.5) + rng.normal(0, sigma_irx)
    irx = np.clip(irx, 0.01, 8.5)

    tnx_s = pd.Series(tnx, index=idx)
    irx_s = pd.Series(irx, index=idx)
    tp = term_premium_proxy(tnx_s, irx_s)
    tp_rank = tp.rolling(252, min_periods=63).rank(pct=True)

    eps = rng.normal(0.0, tlt_vol, n_days)
    tlt_ret = np.empty(n_days)
    tlt_ret[0] = eps[0]
    for i in range(1, n_days):
        lagged = tp_rank.iloc[i - 1]
        tlt_ret[i] = eps[i] if np.isnan(lagged) else tp_signal * (lagged - 0.5) + eps[i]

    tlt_close = 100.0 * np.exp(np.cumsum(tlt_ret))

    df = pd.DataFrame(
        {
            "TNX": tnx,
            "IRX": irx,
            "tp": tp.to_numpy(),
            "TLT_close": tlt_close,
            "TLT_ret": tlt_ret,
        },
        index=idx,
    )
    df.index.name = "date"
    truth = {
        "tp_signal": tp_signal,
        "tlt_vol": tlt_vol,
        "n_days": n_days,
        "seed": seed,
        "start": start,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "daily_term_premium.parquet")


def fetch_daily(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    start: str = "2002-07-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Real daily TLT / 10y-yield / 3m-bill tape with the ACM-style term premium; cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else raises
    ``FileNotFoundError``. Network is touched only on an explicit ``fetch=True`` (then cached).

    Returned frame columns: ``[TLT_close, TNX, IRX, tp, TLT_ret]`` with a tz-naive ``DatetimeIndex``
    named ``date``. ``tp`` is the term-premium proxy ``TNX - EWMA(IRX)`` in percentage points.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape at {path}. Call fetch_daily(fetch=True) once to populate."
            )
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

    import yfinance as yf  # lazy

    raw = yf.download(
        TICKERS, start=start, end=end, interval="1d", auto_adjust=True, progress=False
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily bars")
    closes = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].copy()
    closes.columns = [str(c).replace("^", "") for c in closes.columns]
    closes.index.name = "date"
    closes["TLT"] = closes["TLT"].ffill(limit=5)
    closes = closes.rename(columns={"TLT": "TLT_close"})
    closes = closes.dropna(subset=["TLT_close", "TNX", "IRX"])

    closes["tp"] = term_premium_proxy(closes["TNX"], closes["IRX"]).to_numpy()
    closes["TLT_ret"] = np.log(closes["TLT_close"]).diff()
    closes = closes.dropna(subset=["TLT_ret"])
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)

    os.makedirs(cache_dir, exist_ok=True)
    closes.to_parquet(path)
    return closes


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a daily tape (TLT_close), for the as-of stamp."""
    col = df["TLT_close"] if "TLT_close" in df.columns else df.iloc[:, 0]
    h = hashlib.sha1(np.ascontiguousarray(col.fillna(0).to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
