"""Data layer for Study 377 — Bid-Ask-Bounce (Roll 1984 microstructure + a real-tape proxy).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes for a basket of small/mid-cap US names plus the
  IWM small-cap ETF (yfinance, no key), cached under ``_cache/smallcap_prices.csv`` (a wide
  CSV indexed by date, one column per ticker). Small-caps trade on wider relative spreads, so
  their close-to-close returns carry the most visible **bid-ask bounce** — the cleanest place
  to ask whether one-day "mean reversion" is a tradable edge or just the bounce. From the basket
  we build an equal-weight 1-day cross-sectional reversal book and the lag-1 autocorrelation of
  the average constituent.

* **Synthetic (the offline CORE).** A deterministic, fixed-seed generator of the **Roll (1984)**
  microstructure model: a true (efficient) log-price that is a pure random walk — **zero**
  predictable mean reversion — plus a bid-ask bounce of half-spread ``s/2`` that flips the
  observed transaction price between the bid and the ask. The bounce alone manufactures a
  *negative* lag-1 autocovariance ``cov1 = -(s/2)**2`` in observed returns, i.e. **illusory
  mean reversion** with no edge behind it. A PLANTED-edge knob ``edge`` injects a *genuine*
  AR(1) reversion into the true price: ``edge = 0`` must reproduce the pure-bounce illusion
  (and a reversal book net of the spread must NOT make money); a large ``edge`` must light up a
  net-of-cost edge. This is the positive control that separates "real reversion" from "the
  bounce in disguise."

Pure numpy + pandas + stdlib for the offline path. ``fetch_basket`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "smallcap_prices.csv")

# A transparent, fixed basket of small/mid-cap US names (long-listed, wider relative spreads
# than mega-caps) used for the real-tape reversal book, plus IWM as the small-cap reference.
# This is a PROXY: yfinance serves daily adjusted closes, not the bid/ask quotes Roll's
# estimator wants, so the real-tape spread is *inferred* (Roll estimator) and named as such.
# Survivorship is acknowledged on the Signal axis: a fixed surviving-names basket is
# survivors-only, which if anything makes a reversal edge look BETTER, not worse.
BASKET = [
    "IWM",  "PBI", "GES", "SKX", "CRUS", "SLAB", "PLXS", "MTX",  "KFRC", "EGP",
    "UFPI", "WERN", "HUBG", "SAIA", "MGEE", "WDFC", "JJSF", "LANC", "SXT",  "CSL",
    "ATR",  "GGG",  "RLI",  "WTS",  "MSA",  "AOS",  "BMI",  "RBC",  "EXPO", "ESE",
    "POWI", "MMSI", "HEES", "CBT",  "AVT",  "BDC",  "ENS",  "HELE", "MTRN", "TTC",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_basket(start: str = "2005-01-01", end: str | None = None,
                 path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the small/mid-cap basket via yfinance and cache a wide adjusted-close CSV.

    Network-only; used once to build ``_cache/smallcap_prices.csv``. Never imported by the
    offline notebook cells. Keeps only columns with a long, mostly-complete history.
    """
    import yfinance as yf

    raw = yf.download(BASKET, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    keep = [c for c in raw.columns if raw[c].notna().mean() >= 0.80]
    out = raw[keep].copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide adjusted-close frame (index = date, columns = tickers)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Cached small-cap prices -> daily log-returns frame (index = date, columns = tickers).

    Drops the leading/trailing all-NaN rows and forward-fills short gaps so the panel is a
    clean wide return matrix for the reversal book and the autocovariance estimator.
    """
    px = load_prices(path).ffill().dropna(how="any")
    ret = np.log(px / px.shift(1)).dropna(how="all")
    return ret


# --------------------------------------------------------------------------- #
# Synthetic positive control — the Roll (1984) model (the offline CORE)
# --------------------------------------------------------------------------- #
def synthetic_roll(n_days: int = 6_000, spread: float = 0.010, edge: float = 0.0,
                   seed: int = 377, sig_daily: float = 0.012,
                   mu_daily: float = 0.0002) -> pd.DataFrame:
    """Deterministic Roll (1984) transaction-price series with a planted-edge knob.

    The **true (efficient)** log-price ``m_t`` evolves as an AR(1) in its increments:

        Δm_t = mu + edge * Δm_{t-1} + e_t,   e_t ~ N(0, sig_daily)

    With ``edge = 0`` the true price is a pure random walk — **no** predictable reversion.
    With ``edge < 0`` we plant a *genuine* one-day mean reversion (the thing a real reversal
    strategy would harvest). The **observed** transaction price adds a **bid-ask bounce**: each
    trade prints at the bid or the ask, ``q_t in {-1, +1}`` i.i.d. fair coin, so

        p_t = m_t + (spread / 2) * q_t.

    Roll's result: the bounce alone (edge = 0) injects a negative lag-1 autocovariance
    ``cov1 = -(spread/2)**2`` into observed returns — *illusory* mean reversion that carries no
    tradable edge, because you must cross the very spread that creates it. The observed return
    series ``r_obs`` is what a close-to-close backtest sees; the true return ``r_true`` is the
    counterfactual with no microstructure.

    Returns a frame indexed by business days with columns:
      * ``p_obs``  — observed transaction (close) price
      * ``m_true`` — true efficient price
      * ``r_obs``  — observed log-return  (Δ log p_obs)
      * ``r_true`` — true log-return      (Δ log m_true)
    """
    rng = np.random.default_rng(seed)

    # true efficient log-returns as an AR(1): edge=0 -> pure random walk (no reversion).
    e = rng.normal(0.0, sig_daily, size=n_days)
    dm = np.empty(n_days)
    dm[0] = mu_daily + e[0]
    for t in range(1, n_days):
        dm[t] = mu_daily + edge * (dm[t - 1] - mu_daily) + e[t]
    log_m = np.cumsum(dm)
    m_true = 100.0 * np.exp(log_m)

    # bid-ask bounce: i.i.d. fair coin between bid (-1) and ask (+1), half-spread = spread/2.
    q = rng.choice([-1.0, 1.0], size=n_days)
    p_obs = m_true * np.exp((spread / 2.0) * q)

    r_obs = np.concatenate([[np.nan], np.diff(np.log(p_obs))])
    r_true = np.concatenate([[np.nan], np.diff(np.log(m_true))])

    idx = pd.bdate_range("1990-01-02", periods=n_days)
    return pd.DataFrame(
        {"p_obs": p_obs, "m_true": m_true, "r_obs": r_obs, "r_true": r_true},
        index=idx,
    )
