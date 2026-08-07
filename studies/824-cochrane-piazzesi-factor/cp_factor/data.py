"""Data layer for Study 824 — the Cochrane-Piazzesi return-forecasting factor.

The claim under test (John H. **Cochrane & Monika Piazzesi**, *"Bond Risk Premia"*,
American Economic Review 2005): a **single** tent-shaped linear combination of
**forward rates** predicts the one-year-ahead **excess return** of Treasury bonds
across *every* maturity. Regress each bond's next-year excess return on the whole
forward-rate vector, average the fitted values into one factor ``CP_t = gamma' f_t``,
and that single factor forecasts excess returns with an R^2 the yield-curve *slope*
(Study 132) or a single term-premium proxy (Study 581) cannot match. The loadings
``gamma`` trace a **tent**: negative on the short forward, rising to a peak in the
middle of the curve, falling back at the long end.

Cochrane & Piazzesi used the Fama-Bliss zero-coupon file (clean 1-…-5-year yields).
That tape is not on a no-key retail stack, so this study builds the factor from the
**constant-maturity yields yfinance does expose** and grades itself honestly on that
coarse grid:

* **Constant-maturity Treasury yield indices** — ``^IRX`` (13-week bill ≈ 0.25y),
  ``^FVX`` (5y), ``^TNX`` (10y), ``^TYX`` (30y), in annualised percent. Treating each
  as a continuously-compounded zero yield ``y(n)`` (log price ``p(n) = -n·y(n)``) lets
  us build the **implied forward rates** ``f(n1→n2) = (n2·y2 − n1·y1)/(n2 − n1)``
  spanning the four maturity nodes — the coarse analogue of Cochrane-Piazzesi's five
  one-year forwards.
* **Bond ETFs** — ``SHY`` (1-3y), ``IEF`` (7-10y), ``TLT`` (20y+), ``auto_adjust=True``
  (total-return). Their realised one-year total return minus the one-year risk-free
  (the ``^IRX`` bill yield) is the **excess return** the factor must forecast; we
  average across the three ETFs to mirror CP's "average (across maturity) excess
  return" left-hand side.
* **Synthetic world — the positive control.** ``synthetic_daily`` plants a *known*
  amount of forward-rate predictability in the average excess return via one knob
  ``edge`` (``edge = 0`` = the null: forwards move but forecast nothing; ``edge > 0``
  = a genuine CP factor). The regression machinery must recover it and stay silent on
  the null. It is a faithful-engine check only — never cited for the real-tape stamp.

The offline path is pure numpy + pandas + stdlib. ``fetch()`` (network, with retries)
runs once to build the cache; ``load_panel()`` / ``load_series()`` read the cached
parquet directly (no yfinance import). As-of **2026-06-30** — the last complete
calendar month; the partial current month is dropped.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE_PATH = os.path.join(CACHE_DIR, "cp_panel_2002-01-01.parquet")

# Constant-maturity yield indices (annualised %) + the bond ETFs (total-return prices).
YIELD_TICKERS = ["^IRX", "^FVX", "^TNX", "^TYX"]
ETF_TICKERS = ["SHY", "IEF", "TLT"]
TICKERS = YIELD_TICKERS + ETF_TICKERS

# Maturity (in years) attached to each constant-maturity yield node.
MATURITY = {"IRX": 0.25, "FVX": 5.0, "TNX": 10.0, "TYX": 30.0}

START = "2002-01-01"        # earliest the yield indices + ETFs overlap
AS_OF = "2026-06-30"        # last complete calendar month at publication
HORIZON = 252               # one-year-ahead excess return (trading days)

__all__ = [
    "YIELD_TICKERS", "ETF_TICKERS", "TICKERS", "MATURITY",
    "START", "AS_OF", "HORIZON", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_daily",
    "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> pd.DataFrame:
    """Download the yield indices + bond ETFs and cache a single wide parquet.

    Retries up to ``retries`` times with a short sleep on failure. Columns are the raw
    ticker names with the ``^`` stripped (``IRX, FVX, TNX, TYX, SHY, IEF, TLT``); the
    index is a tz-naive ``DatetimeIndex`` named ``date``. Yields are annualised percent;
    ETF columns are total-return (``auto_adjust=True``) prices.
    """
    import yfinance as yf  # lazy — never imported by the offline path

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            closes = raw["Close"].copy()
            closes.columns = [str(c).replace("^", "") for c in closes.columns]
            closes.index = pd.to_datetime(closes.index)
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            closes.index.name = "date"
            # ETF prices may have stray gaps; forward-fill a few days. Yields kept raw.
            for c in ETF_TICKERS:
                if c in closes.columns:
                    closes[c] = closes[c].ffill(limit=5)
            closes = closes.dropna(how="all")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as exc:  # pragma: no cover - network path
            last_err = exc
            time.sleep(2.0 + attempt)
    raise RuntimeError(f"fetch failed after {retries} tries: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Cached wide daily frame (yields + ETF prices), sliced to ``[START, asof]``.

    Reads the parquet directly — OFFLINE, no yfinance import. Drops the partial current
    month (rows after ``asof``).
    """
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call cp_factor.data.fetch() once."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index <= pd.Timestamp(asof)].sort_index()
    return df


def load_series(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached tape as ``{name: Series}`` for the yields and ETF prices (offline)."""
    df = load_panel(asof)
    return {c: df[c].dropna() for c in df.columns}


# --------------------------------------------------------------------------- #
# Synthetic world — a planted forward-rate -> excess-return relation (control)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    edge: float = 0.0,
    seed: int = 824,
    n_days: int = 5200,
    start: str = "2003-01-02",
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Deterministic seeded tape with a TUNABLE planted CP relation (positive control).

    Four latent yield levels (``IRX, FVX, TNX, TYX``) evolve as persistent AR(1)s that
    inherit a common level factor, so the curve steepens and inverts like the real one.
    From them we build the same implied forwards the real engine uses. A single hidden
    "risk-premium state" ``rp_t`` — a smooth function of the forward vector — drives the
    **one-year-ahead** average bond excess return only when ``edge > 0``::

        avg_rx_{t+h} = edge * rp_t + noise_{t+h}

    ``edge = 0`` is the null: the forwards still move, but carry **no** information about
    forward excess returns, and the predictive R^2 / NW t must be ~0. ``edge > 0`` plants
    a genuine Cochrane-Piazzesi factor. The frame mirrors the real cache's schema so the
    strategy code runs on it unchanged: yield columns ``IRX/FVX/TNX/TYX`` (percent) plus
    ETF total-return prices ``SHY/IEF/TLT`` whose realised one-year excess return equals
    the planted target (up to noise).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # Common level factor + maturity-specific AR(1) spreads -> four yield curves.
    level = np.empty(n_days)
    level[0] = 3.0
    for t in range(1, n_days):
        level[t] = 0.995 * level[t - 1] + 0.005 * 3.0 + rng.normal(0.0, 0.04)
    level = np.clip(level, 0.2, 8.0)

    def _spread(phi, sig, base):
        s = np.empty(n_days)
        s[0] = base
        for t in range(1, n_days):
            s[t] = phi * s[t - 1] + (1 - phi) * base + rng.normal(0.0, sig)
        return s

    irx = np.clip(level + _spread(0.99, 0.03, -0.4), 0.01, 9.0)
    fvx = np.clip(level + _spread(0.98, 0.02, 1.2), 0.05, 9.5)
    tnx = np.clip(level + _spread(0.98, 0.02, 1.8), 0.05, 9.8)
    tyx = np.clip(level + _spread(0.985, 0.02, 2.4), 0.05, 9.9)

    yields = pd.DataFrame(
        {"IRX": irx, "FVX": fvx, "TNX": tnx, "TYX": tyx}, index=idx
    )

    # Hidden risk-premium state: a tent-shaped contrast of the forwards (a function of
    # the day-t curve only), standardised. This is the linear combination the CP
    # regression should recover when the edge is planted.
    y = yields.to_numpy() / 100.0
    fwd_2_5 = (5.0 * y[:, 1] - 0.25 * y[:, 0]) / 4.75
    fwd_5_10 = (10.0 * y[:, 2] - 5.0 * y[:, 1]) / 5.0
    fwd_10_30 = (30.0 * y[:, 3] - 10.0 * y[:, 2]) / 20.0
    rp = -1.0 * y[:, 0] + 2.0 * fwd_2_5 - 1.0 * fwd_10_30  # tent-shaped contrast
    rp = (rp - np.nanmean(rp)) / (np.nanstd(rp) + 1e-12)

    # Build ETF total-return prices whose realised one-year excess return equals a
    # controlled target *exactly*. Trick: within each residue class (t mod horizon) the
    # log price is a cumulative sum, so ``log P[t+h] - log P[t] = D[t]`` is enforced
    # exactly and the strategy's reconstructed excess return equals ``target`` with no
    # smoothing artefact. Under ``edge = 0`` the target is white noise -> the regression
    # of persistent forwards on a *white* target has R^2 ~ 0 (the machinery stays
    # silent); under ``edge > 0`` the target loads on ``rp`` -> a genuine CP factor.
    rf_ann = irx / 100.0
    dur = np.array([1.8, 7.5, 18.0])  # SHY / IEF / TLT effective durations (years)
    noise = rng.normal(0.0, 0.06, size=(n_days, 3))

    out = yields.copy()
    for j, (name, d) in enumerate(zip(ETF_TICKERS, dur)):
        target = edge * rp * (d / dur.mean()) + noise[:, j]  # realised 1y excess ret
        logP = np.empty(n_days)
        # seed the first `horizon` rows with a gentle random walk
        seed_rw = np.cumsum(rng.normal((rf_ann[:horizon]) / horizon, 0.001, horizon))
        logP[:horizon] = np.log(100.0) + seed_rw
        for t in range(horizon, n_days):
            src = t - horizon
            gross = 1.0 + rf_ann[src] + target[src]
            logP[t] = logP[src] + np.log(max(gross, 1e-6))
        out[name] = np.exp(logP)
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------- #
# Content fingerprint for the as-of stamp
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the cached tape (all numeric columns)."""
    arr = np.ascontiguousarray(
        df.select_dtypes("number").fillna(0.0).to_numpy(dtype=float)
    )
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
