"""Data layer for Study 593 — HFEA (UPRO/TMF 55/45).

Two sources, both offline-friendly once cached:

* **Real tape.** Daily *auto-adjusted (total-return)* closes from yfinance for
  ``SPY`` / ``TLT`` (the 1x underlyings), ``UPRO`` / ``TMF`` (the real 3x funds,
  live since mid-2009), and ``^IRX`` (13-week T-bill discount yield, the
  financing / risk-free leg — FRED is unreachable from this desk, ^IRX is the
  standard yfinance substitute). Cached wide in ``_cache/hfea_prices.csv``.

* **Pre-2009 extension (the study-100 recipe).** The daily constant-leverage
  identity ``r_3x = 3·r_index − 2·r_borrow − fee_daily`` synthesises each 3x fund
  from its 1x underlying. The all-in fee (expense ratio + swap-financing spread
  above T-bills) is **calibrated on the real-fund overlap** — chosen so the
  synthetic NAV matches the real fund's terminal NAV over 2009→as-of — and the
  synthesis is **validated** against the real funds (daily-return correlation
  must clear 0.99, as study 100's melting-ice engine did). The extended tape
  splices synthetic (2002-07, TLT inception → fund launch) onto real fund
  returns thereafter.

* **Synthetic world.** A deterministic, seeded two-asset daily generator with a
  TUNABLE stock–bond correlation ``rho`` and bond excess drift ``bond_mu_exc`` —
  the planted effect is the *diversification engine itself*: with ``rho < 0`` and
  positive bond carry a levered 55/45 must out-compound the 1x stock; with the
  2022 configuration (``rho > 0``, zero bond carry) it must NOT. Date spans stay
  well under 250 years (ns-Timestamp CI trap).

Pure numpy + pandas for the offline path. ``fetch_prices`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "hfea_prices.csv")

TICKERS = ["SPY", "TLT", "UPRO", "TMF", "^IRX"]

# The frozen as-of: June 2026 is the last complete month at publication time.
AS_OF = "2026-06-30"

LEV = 3.0                 # the funds' daily leverage
TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_prices(start: str = "2002-01-01", end: str | None = None,
                 path: str = PRICES_CACHE, retries: int = 3) -> pd.DataFrame:
    """Download the five series and cache them wide (network; run once)."""
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all").sort_index()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Cached wide close frame (index = date, columns = TICKERS)."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px


def build_tape(px: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Daily-return tape sliced to the as-of.

    Columns: ``spy`` / ``tlt`` (1x total-return daily returns), ``upro`` / ``tmf``
    (real 3x fund daily returns, NaN before launch), ``rf`` (daily T-bill return
    from ^IRX, annualised discount yield / 252 — the desk's standard simple
    proxy, forward-filled over missing quotes).
    """
    px = px[px.index <= pd.Timestamp(asof)].copy()
    out = pd.DataFrame(index=px.index)
    out["spy"] = px["SPY"].pct_change()
    out["tlt"] = px["TLT"].pct_change()
    out["upro"] = px["UPRO"].pct_change()
    out["tmf"] = px["TMF"].pct_change()
    out["rf"] = (px["^IRX"].ffill() / 100.0) / TRADING_DAYS
    # keep rows where both underlyings trade (TLT starts 2002-07)
    out = out.dropna(subset=["spy", "tlt"])
    return out


# --------------------------------------------------------------------------- #
# LETF synthesis + calibration (the study-100 recipe, applied per leg)
# --------------------------------------------------------------------------- #
def synth_letf(r_index: pd.Series, rf: pd.Series, fee_ann: float,
               lev: float = LEV) -> pd.Series:
    """Daily constant-leverage identity: ``lev·r − (lev−1)·rf − fee/252``.

    ``fee_ann`` is the ALL-IN annual drag (expense ratio + swap-financing spread
    above the T-bill leg already charged via ``(lev−1)·rf``).
    """
    return lev * r_index - (lev - 1.0) * rf - fee_ann / TRADING_DAYS


def calibrate_fee(r_index: pd.Series, rf: pd.Series, r_real: pd.Series,
                  lev: float = LEV) -> dict:
    """Choose the all-in fee so the synthetic terminal log-NAV matches the real
    fund's over the overlap, and report the validation stats.

    Closed form: the daily log-NAV gap between the zero-fee synthetic and the
    real fund, annualised. Returns fee, daily-return correlation, and the
    RMS daily tracking error at the calibrated fee.
    """
    df = pd.concat({"idx": r_index, "rf": rf, "real": r_real}, axis=1).dropna()
    syn0 = lev * df["idx"] - (lev - 1.0) * df["rf"]
    gap = np.log1p(syn0).sum() - np.log1p(df["real"]).sum()
    years = len(df) / TRADING_DAYS
    fee_ann = float(gap / years)          # log-drag per year; small, ~= simple fee
    syn = synth_letf(df["idx"], df["rf"], fee_ann, lev=lev)
    corr = float(np.corrcoef(syn, df["real"])[0, 1])
    te = float((syn - df["real"]).std() * np.sqrt(TRADING_DAYS))
    nav_syn = float(np.exp(np.log1p(syn).sum()))
    nav_real = float(np.exp(np.log1p(df["real"]).sum()))
    return {"fee_ann": fee_ann, "corr": corr, "te_ann": te,
            "nav_ratio": nav_syn / nav_real, "n_days": len(df),
            "start": str(df.index.min().date()), "end": str(df.index.max().date())}


def spliced_legs(tape: pd.DataFrame, fee_spy: float, fee_tlt: float) -> pd.DataFrame:
    """Extended daily 3x legs: real fund returns where the fund exists, the
    calibrated synthetic before. Adds ``s3x`` / ``b3x`` columns; a ``src`` column
    flags 'real' vs 'synth' rows (real starts at the LATER of the two launches so
    both legs switch on the same day)."""
    syn_s = synth_letf(tape["spy"], tape["rf"], fee_spy)
    syn_b = synth_letf(tape["tlt"], tape["rf"], fee_tlt)
    both_real = tape[["upro", "tmf"]].notna().all(axis=1)
    first_real = both_real[both_real].index.min()
    out = tape.copy()
    out["s3x"] = np.where(out.index >= first_real, out["upro"], syn_s)
    out["b3x"] = np.where(out.index >= first_real, out["tmf"], syn_b)
    out["src"] = np.where(out.index >= first_real, "real", "synth")
    return out.dropna(subset=["s3x", "b3x"])


# --------------------------------------------------------------------------- #
# Synthetic world — the planted diversification engine
# --------------------------------------------------------------------------- #
def synthetic_world(n_years: int = 20, rho: float = -0.4,
                    stock_mu_exc: float = 0.06, stock_sig: float = 0.16,
                    bond_mu_exc: float = 0.02, bond_sig: float = 0.11,
                    rf_ann: float = 0.02, seed: int = 593) -> pd.DataFrame:
    """Deterministic correlated stock/bond daily-return world.

    The PLANTED effect is the HFEA engine itself: with ``rho < 0`` and positive
    ``bond_mu_exc`` (bond carry), a 3x 55/45 quarterly-rebalanced mix must
    out-compound the 1x stock; the NULL configuration (``rho = +0.6``,
    ``bond_mu_exc = 0`` — the 2022 world made permanent) must show no such edge.
    Business-day index kept far under the 250-year ns-Timestamp cap.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS)
    z = rng.standard_normal((n, 2))
    zb = rho * z[:, 0] + np.sqrt(1.0 - rho ** 2) * z[:, 1]
    dt = 1.0 / TRADING_DAYS
    rf_d = rf_ann * dt
    r_s = (rf_ann + stock_mu_exc) * dt + stock_sig * np.sqrt(dt) * z[:, 0]
    r_b = (rf_ann + bond_mu_exc) * dt + bond_sig * np.sqrt(dt) * zb
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"spy": r_s, "tlt": r_b, "rf": rf_d,
                         "upro": np.nan, "tmf": np.nan}, index=idx)
