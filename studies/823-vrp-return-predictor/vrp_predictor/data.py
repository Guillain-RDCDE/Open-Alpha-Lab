"""Data layer for Study 823 — Variance-Risk-Premium Return Predictor.

The claim under test (Bollerslev, Tauchen & Zhou 2009, *"Expected Stock Returns and
Variance Risk Premia"*, RFS): the **variance risk premium** — the gap between the
*implied* (risk-neutral) variance the option market charges and the *realized*
(physical) variance the underlying actually delivers — is a **time-series predictor of
the aggregate equity market's forward excess return**. When the VRP is fat, investors
are paying up for variance protection (they are risk-averse), and the market goes on to
earn *more*; when it is thin, forward returns are lower. The R² of a predictive
regression of forward market return on the VRP peaks at the **quarterly** horizon.

This is the *predictive-regression* reading. It is distinct from Study 130
(vol-risk-premium), which asks whether the VRP *exists* / is positive on average (the
level of the premium, harvested by shorting variance); from Study 111 (vix-term-
structure), the shape of the VIX futures curve; and from Study 3 (fear-gauge), the VIX
*level* as a contrarian dip signal. Here the VRP is not harvested — it is used to
**time the direction of the market's next 1–3 months**.

Two ingredients, one monthly schema:

* **Real tape — SPY + ^VIX daily closes.** ``fetch()`` pulls both with yfinance
  (``auto_adjust=True``, total-return SPY) into this study's OWN ``_cache/`` as a single
  parquet (retry up to 4×). ``have_real()`` / ``load_panel()`` / ``load_series()`` read
  the cached parquet OFFLINE (no yfinance import). ``AS_OF="2026-06-30"`` — the last
  complete calendar month; the partial current month is dropped.

* **Synthetic world — the positive control.** ``synthetic_daily`` builds a deterministic,
  seeded daily SPY+VIX tape in which a persistent latent monthly variance premium
  ``vrp_m`` is planted *by construction* (implied variance = realized variance + vrp_m),
  and — only when ``edge > 0`` — the *next* month's mean return loads ``+edge`` on
  ``vrp_m``. ``edge = 0`` is the null (VRP present but carrying no forward information);
  ``edge > 0`` plants exactly the Bollerslev-Tauchen-Zhou predictive relation.

The offline path is pure numpy + pandas + stdlib.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE_PATH = os.path.join(CACHE_DIR, "spy_vix_daily.parquet")

TICKERS = ["SPY", "^VIX"]
START = "1993-01-29"        # SPY inception (VIX goes back to 1990; SPY is the binding start)
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_daily",
    "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4) -> pd.DataFrame:
    """Download SPY + ^VIX daily closes with yfinance; cache under this study's _cache/.

    Network; runs once to build the cache. Retries up to ``retries`` times with a short
    sleep on transient failure. Writes a single parquet with tz-naive daily index and
    columns ``SPY`` (auto-adjusted total-return close) and ``VIX`` (index level).
    """
    import yfinance as yf  # lazy — never imported on the offline path

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            closes = closes.copy()
            closes.columns = [str(c).replace("^", "") for c in closes.columns]
            closes = closes.rename(columns={"VIX": "VIX", "SPY": "SPY"})
            closes.index.name = "date"
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            closes = closes.dropna(subset=["SPY", "VIX"])
            if len(closes) < 500:
                raise RuntimeError(f"suspiciously short tape ({len(closes)} rows)")
            os.makedirs(CACHE_DIR, exist_ok=True)
            closes.to_parquet(CACHE_PATH)
            return closes
        except Exception as exc:  # noqa: BLE001 — retry on any transient failure
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"yfinance fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily SPY+VIX frame, sliced to ``<= asof``. OFFLINE (reads parquet).

    Returns a tz-naive daily ``DataFrame`` with columns ``SPY`` and ``VIX``.
    """
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call vrp_predictor.data.fetch() once."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index <= pd.Timestamp(asof)]
    return df.dropna(subset=["SPY", "VIX"]).sort_index()


def load_series(asof: str = AS_OF) -> tuple[pd.Series, pd.Series]:
    """Convenience: ``(SPY_close, VIX_close)`` daily series from the cache (offline)."""
    df = load_panel(asof)
    return df["SPY"], df["VIX"]


# --------------------------------------------------------------------------- #
# Synthetic world — planted Bollerslev-Tauchen-Zhou predictive relation
# --------------------------------------------------------------------------- #
def synthetic_daily(
    edge: float = 0.0,
    seed: int = 823,
    n_months: int = 400,
    days_per_month: int = 21,
    start: str = "1990-01-02",
    daily_vol: float = 0.010,
    drift_month: float = 0.005,
    vrp_base: float = 0.0020,
    vrp_rho: float = 0.80,
    vrp_sd: float = 0.0010,
) -> pd.DataFrame:
    """Deterministic seeded daily SPY+VIX tape with a TUNABLE planted VRP→return relation.

    A persistent monthly latent variance premium ``vrp_m`` (AR(1) around ``vrp_base``,
    floored at a small positive value so implied > realized on average) is planted
    **by construction**: within each month the implied variance is set to the month's
    realized variance *plus* ``vrp_m``, so a VIX²/12 − trailing-RV measurement recovers
    ``vrp_m`` exactly. The next month's daily mean return loads ``+edge`` on ``vrp_m``:

        RV_m   = Σ daily r²        (realized variance of month m)
        IV_m   = RV_m + vrp_m      (implied variance ⇒ measured VRP_m = vrp_m)
        VIX_m  = 100·√(12·IV_m)    (month-m VIX level, so (VIX/100)²/12 = IV_m)
        r̄_{m+1} = drift_month + edge·vrp_m        (forward mean loads on VRP_m)

    ``edge = 0`` is the null: the VRP still varies and is positive on average, but the
    forward market return does not respond to it. ``edge > 0`` plants exactly the
    predictive relation of Bollerslev-Tauchen-Zhou. Business-day index; span kept well
    below the pandas ns-timestamp horizon.

    Returns a tz-naive daily ``DataFrame`` with columns ``SPY`` and ``VIX``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_months * days_per_month
    idx = pd.bdate_range(start=start, periods=n_days)

    # Persistent monthly variance premium (AR(1) around vrp_base, kept positive).
    vrp = np.empty(n_months)
    vrp[0] = vrp_base
    for m in range(1, n_months):
        vrp[m] = vrp_base + vrp_rho * (vrp[m - 1] - vrp_base) + rng.normal(0.0, vrp_sd)
    vrp = np.maximum(vrp, 0.0002)

    spy_ret = np.empty(n_days)
    vix_level = np.empty(n_days)
    for m in range(n_months):
        sl = slice(m * days_per_month, (m + 1) * days_per_month)
        # This month's mean loads on LAST month's planted vrp (the lagged predictor).
        vrp_lag = vrp[m - 1] if m > 0 else vrp[0]
        mu_daily = (drift_month + edge * vrp_lag) / days_per_month
        # Slight month-to-month vol variation so RV is not a constant.
        sig = daily_vol * (0.75 + 0.5 * rng.random())
        r = rng.normal(mu_daily, sig, days_per_month)
        spy_ret[sl] = r
        rv_m = float(np.sum(r ** 2))               # realized variance this month
        iv_m = rv_m + vrp[m]                        # implied = realized + planted premium
        vix_level[sl] = 100.0 * np.sqrt(12.0 * iv_m)

    spy_close = 100.0 * np.exp(np.cumsum(spy_ret))
    return pd.DataFrame({"SPY": spy_close, "VIX": vix_level}, index=idx)


# --------------------------------------------------------------------------- #
# Content fingerprint (for the as-of stamp)
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the SPY close column (the as-of stamp)."""
    col = df["SPY"] if "SPY" in df.columns else df.iloc[:, 0]
    h = hashlib.sha1(np.ascontiguousarray(col.fillna(0).to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]
