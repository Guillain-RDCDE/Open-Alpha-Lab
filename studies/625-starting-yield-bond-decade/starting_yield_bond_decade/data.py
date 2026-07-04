"""Data layer for Study 625 (Starting-Yield-Bond-Decade).

Two tapes, one shape (a monthly yield/return panel):

- ``synthetic_world`` — a *deterministic, offline* generator. A seeded AR(1)
  10-year yield path drives a constant-maturity bond total-return index through
  the exact same pricing engine as the real tape. A tunable ``link`` parameter
  blends the mechanical (duration-arithmetic) return with pure noise:

    * ``link = 1.0`` — the planted effect: returns ARE the duration arithmetic
      (starting yield should explain ~all of the forward decade).
    * ``link = 0.0`` — the null: monthly returns are iid noise around a
      constant, independent of the starting yield (R² should collapse to ~0).

- ``load_gs10`` — the real tape. Shiller's monthly "Long Interest Rate"
  (GS10, monthly average, 1871→2023-09 in the staged file), extended to the
  present with the CBOE 10-year yield index ^TNX (yfinance, monthly mean of
  daily closes). Cache-first: everything is read from this study's own
  ``_cache/gs10_extended.csv``; the network is touched only on a cache miss.

- ``load_stock_decades`` — the third-axis tape. Shiller S&P 500 nominal total
  return (monthly dividend reinvestment) + CAPE, for the "does the same trick
  work for stocks?" myth check (sibling: study 120-excess-cape-yield).

No look-ahead: the signal at month *t* is the month-*t* average yield; the
forward decade is measured from the end-of-month-*t* index value onward
(one documented lag: signal averaged over month *t*, position entered at the
end of month *t*).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
CACHE_DIR = os.path.join(STUDY_ROOT, "_cache")

GS10_CACHE = os.path.join(CACHE_DIR, "gs10_extended.csv")
SHILLER_CACHE = os.path.join(CACHE_DIR, "shiller_full.csv")

# Repo-wide staged Shiller parquet (reused if present; never re-fetched live).
_REPO_SHILLER = [
    os.path.join(REPO_ROOT, "_cache", "shiller_sp500.parquet"),
    os.path.join(REPO_ROOT, "_cache", "_cache", "shiller_sp500.parquet"),
]
SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"

# As-of guard: the last complete calendar month for a 2026-07-03 run.
LAST_COMPLETE_MONTH = "2026-06-01"


# ---------------------------------------------------------------------------
# Bond pricing — the one engine shared by the real tape and the synthetic
# ---------------------------------------------------------------------------
def bond_price(coupon: float, ytm: float, maturity: float) -> float:
    """Price (per 1 face) of an annual-coupon bond; non-integer maturity OK.

    Closed form ``c/y (1 - (1+y)^-T) + (1+y)^-T``. At ``y = 0`` the limit
    ``c·T + 1`` is used.
    """
    if abs(ytm) < 1e-12:
        return coupon * maturity + 1.0
    disc = (1.0 + ytm) ** (-maturity)
    return coupon / ytm * (1.0 - disc) + disc


def constant_maturity_returns(y: pd.Series, maturity: float = 10.0) -> pd.Series:
    """Monthly total returns of a constant-maturity par-bond roll (Swinkels 2019).

    Each month-end, buy a par bond at yield ``y_t`` (coupon = y_t, maturity
    ``maturity`` years). One month later, sell it at yield ``y_{t+1}`` with
    ``maturity - 1/12`` years remaining, collecting one month of accrued
    coupon ``y_t / 12``. ``y`` is in decimal (0.05 = 5%). The return series is
    indexed on the *end* month (return earned over month t → t+1).
    """
    yv = y.to_numpy(dtype=float)
    n = len(yv)
    rets = np.full(n, np.nan)
    t_rem = maturity - 1.0 / 12.0
    for t in range(n - 1):
        p1 = bond_price(yv[t], yv[t + 1], t_rem)
        rets[t + 1] = p1 + yv[t] / 12.0 - 1.0
    return pd.Series(rets, index=y.index, name="bond_ret")


def total_return_index(rets: pd.Series) -> pd.Series:
    """Cumulative total-return index from monthly returns (base 1.0)."""
    return (1.0 + rets.fillna(0.0)).cumprod().rename("tr")


# ---------------------------------------------------------------------------
# Synthetic world — deterministic offline core with a TUNABLE planted effect
# ---------------------------------------------------------------------------
def synthetic_world(
    n_years: int = 150,
    link: float = 1.0,
    yield_mean: float = 0.045,
    seed: int = 625,
) -> tuple[pd.DataFrame, dict]:
    """A monthly yield → constant-maturity-return world, deterministic per seed.

    The 10-year yield follows a persistent AR(1) around ``yield_mean``,
    clipped to [0.1%, 16%]. Mechanical monthly returns come from the *same*
    ``constant_maturity_returns`` engine as the real tape; the observed return
    blends mechanics and noise:

        r_obs = link · r_mech + (1 − link) · (yield_mean/12 + ε)

    - ``link = 1`` → planted duration arithmetic (decade R² ≈ 0.9+).
    - ``link = 0`` → null world (returns independent of the starting yield).

    Span stays well under 250 years (pandas ns-Timestamp CI trap).
    Returns ``(frame, truth)``; frame columns: ``gs10`` (decimal), ``bond_ret``,
    ``tr``, ``fwd10`` (forward 10y annualised total return).
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1875-01-31", periods=n, freq="ME", name="date")

    yld = np.empty(n)
    yld[0] = yield_mean
    for t in range(1, n):
        yld[t] = yield_mean + 0.997 * (yld[t - 1] - yield_mean) + 0.0012 * rng.standard_normal()
    yld = np.clip(yld, 0.001, 0.16)
    y = pd.Series(yld, index=idx, name="gs10")

    r_mech = constant_maturity_returns(y)
    noise = yield_mean / 12.0 + 0.006 * rng.standard_normal(n)
    r_obs = link * r_mech + (1.0 - link) * pd.Series(noise, index=idx)
    r_obs.iloc[0] = np.nan

    tr = total_return_index(r_obs)
    fwd10 = (tr.shift(-120) / tr) ** (1.0 / 10.0) - 1.0

    frame = pd.DataFrame({"gs10": y, "bond_ret": r_obs, "tr": tr, "fwd10": fwd10})
    truth = {"link": link, "yield_mean": yield_mean, "n_years": n_years,
             "n_obs": n, "seed": seed}
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller GS10 spliced with ^TNX, cache-first
# ---------------------------------------------------------------------------
def _load_shiller_raw() -> pd.DataFrame:
    """Shiller monthly file: study cache → repo staged parquet → GitHub raw."""
    if os.path.exists(SHILLER_CACHE):
        df = pd.read_csv(SHILLER_CACHE, parse_dates=["Date"])
        return df
    for p in _REPO_SHILLER:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    else:  # pragma: no cover - network fallback
        df = pd.read_csv(SHILLER_URL)
    df["Date"] = pd.to_datetime(df["Date"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(SHILLER_CACHE, index=False)
    return df


def _fetch_tnx_monthly(start: str = "2023-01-01") -> pd.Series:
    """Monthly-average 10y yield (percent) from yfinance ^TNX daily closes."""
    import yfinance as yf  # imported lazily: only on a cache miss

    px = yf.download("^TNX", start=start, progress=False, auto_adjust=False)
    close = px["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance MultiIndex columns
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.median() > 20:  # CBOE convention = yield x 10; Yahoo usually not
        close = close / 10.0
    monthly = close.resample("MS").mean()
    monthly.name = "gs10"
    return monthly


def load_gs10(refresh: bool = False) -> pd.DataFrame:
    """The spliced monthly 10-year yield tape, 1871-01 → last complete month.

    Columns: ``gs10`` (decimal yield), ``source`` ('shiller' or 'tnx').
    Cache-first from ``_cache/gs10_extended.csv``; only a cache miss touches
    the network (^TNX). Partial current month always dropped.
    """
    if os.path.exists(GS10_CACHE) and not refresh:
        df = pd.read_csv(GS10_CACHE, parse_dates=["date"]).set_index("date")
        return df

    sh = _load_shiller_raw()
    sh = sh[sh["Long Interest Rate"] > 0].copy()
    shiller = pd.DataFrame(
        {"gs10": sh["Long Interest Rate"].to_numpy() / 100.0, "source": "shiller"},
        index=pd.DatetimeIndex(sh["Date"], name="date"),
    ).sort_index()

    tnx = _fetch_tnx_monthly()
    ext = tnx.loc[tnx.index > shiller.index.max()] / 100.0
    ext = ext.loc[ext.index <= pd.Timestamp(LAST_COMPLETE_MONTH)]
    extension = pd.DataFrame({"gs10": ext, "source": "tnx"})
    extension.index.name = "date"

    df = pd.concat([shiller, extension]).sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(GS10_CACHE)
    return df


def have_real() -> bool:
    """True iff the spliced GS10 cache (or its inputs) are available offline."""
    return os.path.exists(GS10_CACHE)


def load_bond_panel(maturity: float = 10.0) -> pd.DataFrame:
    """The full real-tape panel: yield, monthly return, TR index, forward decade.

    ``fwd10`` at month *t* = annualised nominal total return of the constant-
    maturity 10y roll over the 120 months *after* the end of month *t* (the
    documented one-lag convention: month-*t* average yield → enter end of
    month *t*).
    """
    g = load_gs10()
    y = g["gs10"]
    rets = constant_maturity_returns(y, maturity=maturity)
    tr = total_return_index(rets)
    fwd10 = (tr.shift(-120) / tr) ** (1.0 / 10.0) - 1.0
    return pd.DataFrame({"gs10": y, "source": g["source"], "bond_ret": rets,
                         "tr": tr, "fwd10": fwd10})


def load_bond_panel_real(maturity: float = 10.0) -> pd.DataFrame:
    """Honesty check: the SAME regression with CPI-deflated (real) returns.

    The claim locks the *nominal* decade only. This panel deflates the
    constant-maturity TR index by Shiller's CPI and recomputes ``fwd10_real``
    (annualised 10-year real total return), restricted to months where CPI is
    available (non-placeholder).
    """
    panel = load_bond_panel(maturity=maturity)
    sh = _load_shiller_raw()
    cpi = pd.Series(
        sh["Consumer Price Index"].to_numpy(),
        index=pd.DatetimeIndex(sh["Date"]), name="cpi",
    )
    cpi = cpi[cpi > 0]
    df = panel.join(cpi, how="inner")
    real_tr = df["tr"] / df["cpi"]
    df["fwd10_real"] = (real_tr.shift(-120) / real_tr) ** (1.0 / 10.0) - 1.0
    return df


# ---------------------------------------------------------------------------
# Third axis — the same trick on stocks (Shiller S&P nominal total return)
# ---------------------------------------------------------------------------
def load_stock_decades() -> pd.DataFrame:
    """Monthly S&P 500 panel: starting 'yield' (1/CAPE) vs forward 10y nominal TR.

    Nominal total return reinvests the (annual-rate) dividend monthly:
    ``ret = P.pct_change() + D / P / 12``. ``ey`` = 1/PE10 (CAPE earnings
    yield) — the natural stock analogue of "the yield you buy at".
    Rows before CAPE exists (pre-1881) or with placeholder zeros are dropped.
    """
    sh = _load_shiller_raw().set_index("Date").sort_index()
    sh = sh[(sh["SP500"] > 0) & (sh["Dividend"] > 0)]
    ret = sh["SP500"].pct_change() + sh["Dividend"] / sh["SP500"] / 12.0
    tr = (1.0 + ret.fillna(0.0)).cumprod()
    fwd10 = (tr.shift(-120) / tr) ** (1.0 / 10.0) - 1.0
    out = pd.DataFrame({"ey": np.where(sh["PE10"] > 0, 1.0 / sh["PE10"].replace(0, np.nan), np.nan),
                        "tr": tr, "fwd10": fwd10}, index=sh.index)
    out.index.name = "date"
    return out.dropna(subset=["ey"])


def fingerprint(df: pd.DataFrame, col: str = "gs10") -> str:
    """Short content fingerprint of the yield tape, for the as-of stamp."""
    arr = df[col].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
