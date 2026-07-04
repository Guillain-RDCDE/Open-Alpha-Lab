"""Data layer for Study 620 — A-H Share Premium.

Two sources, both offline-friendly once cached:

* **Real tape.** Daily closes for a fixed panel of **dual-listed Chinese companies** — the same
  legal entity with an **A-share** line in Shanghai (CNY) and an **H-share** line in Hong Kong
  (HKD). A and H shares of these companies are ordinary shares of identical par value with
  identical per-share dividends and voting rights — economically the *same claim*, quoted on two
  exchanges in two currencies. We fetch both the **raw close** (for the premium *level*, matching
  the Hang Seng AH Premium index construction: raw prices × FX) and the **adjusted close**
  (total-return, for the relative-return races), plus the two Yahoo FX crosses ``CNY=X`` and
  ``HKD=X`` (USD/CNY and USD/HKD) whose ratio gives the HKD→CNY rate. Cached wide under
  ``_cache/`` (``ahp_close.csv``, ``ahp_adj.csv``).

* **Synthetic.** A deterministic, fixed-seed generator producing an A/H pair panel with a
  **planted premium level** (knob ``premium``) that mean-reverts around the plant with an AR(1)
  and a common cross-pair factor. With ``premium = 0`` the two lines are the same price plus
  noise: the level detector must NOT manufacture a persistent premium. It is the positive
  control — machinery proof only, never market evidence.

Pure numpy + pandas + stdlib on the offline path. ``fetch_panel`` (network) is used only once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
CLOSE_CACHE = os.path.join(CACHE_DIR, "ahp_close.csv")
ADJ_CACHE = os.path.join(CACHE_DIR, "ahp_adj.csv")

# The frozen as-of: the sample never creeps as new sessions arrive. 2026-06 is the last
# complete calendar month at publication.
AS_OF = "2026-06-30"

# Fixed panel of dual-listed A/H pairs: (name, A-share ticker Shanghai, H-share ticker HK).
# All are 1:1 par ordinary shares (identical per-share dividend on both lines), all still
# dual-listed in 2026 — a SURVIVOR panel, named on the Signal axis: pairs that delisted or
# merged away (e.g. Haitong Securities, absorbed 2025) cannot appear, and the panel skews to
# mega-cap SOEs where the premium is best documented.
PAIRS = [
    ("Ping An Insurance",       "601318.SS", "2318.HK"),
    ("PetroChina",              "601857.SS", "0857.HK"),
    ("Sinopec",                 "600028.SS", "0386.HK"),
    ("ICBC",                    "601398.SS", "1398.HK"),
    ("Bank of China",           "601988.SS", "3988.HK"),
    ("China Construction Bank", "601939.SS", "0939.HK"),
    ("Agricultural Bank",       "601288.SS", "1288.HK"),
    ("China Life",              "601628.SS", "2628.HK"),
    ("China Merchants Bank",    "600036.SS", "3968.HK"),
    ("CITIC Securities",        "600030.SS", "6030.HK"),
    ("China Shenhua",           "601088.SS", "1088.HK"),
    ("Air China",               "601111.SS", "0753.HK"),
    ("Tsingtao Brewery",        "600600.SS", "0168.HK"),
    ("Great Wall Motor",        "601633.SS", "2333.HK"),
]
FX_TICKERS = ["CNY=X", "HKD=X"]  # USD/CNY and USD/HKD; HKD->CNY = CNY=X / HKD=X


def all_tickers() -> list[str]:
    out: list[str] = []
    for _n, a, h in PAIRS:
        out.extend([a, h])
    return out + FX_TICKERS


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_panel(start: str = "2010-01-01", end: str | None = None,
                close_path: str = CLOSE_CACHE, adj_path: str = ADJ_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download raw + adjusted closes for every pair leg and the FX crosses; cache wide CSVs.

    Network-only; used once to build the cache. Raw ``Close`` feeds the premium *level*
    (the Hang Seng AHP-index convention); ``Adj Close`` (dividend-adjusted, total-return)
    feeds every relative-return race.
    """
    import yfinance as yf

    tickers = all_tickers()
    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                              progress=False)
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    close = raw["Close"].dropna(how="all")
    adj = raw["Adj Close"].dropna(how="all")
    os.makedirs(os.path.dirname(close_path), exist_ok=True)
    close.to_csv(close_path)
    adj.to_csv(adj_path)
    return close, adj


def have_real(close_path: str = CLOSE_CACHE, adj_path: str = ADJ_CACHE) -> bool:
    return os.path.exists(close_path) and os.path.exists(adj_path)


def load_real(close_path: str = CLOSE_CACHE, adj_path: str = ADJ_CACHE,
              asof: str | None = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (raw close, adjusted close) wide frames, sliced to the frozen as-of.

    Data hygiene: Yahoo backfills **pre-listing placeholder prices of 0.01** on some HK lines
    (6030.HK carries 435 rows of 0.01 before its 2011-10-06 listing). Any close ≤ 0.05 is
    masked to NaN on both frames — those rows are not prices.
    """
    close = pd.read_csv(close_path, index_col=0, parse_dates=True).sort_index()
    adj = pd.read_csv(adj_path, index_col=0, parse_dates=True).sort_index()
    junk = close <= 0.05
    close, adj = close.mask(junk), adj.mask(junk)
    if asof is not None:
        ts = pd.Timestamp(asof)
        close, adj = close[close.index <= ts], adj[adj.index <= ts]
    return close, adj


def hkd_cny(close: pd.DataFrame) -> pd.Series:
    """HKD->CNY cross from the two Yahoo USD crosses, forward-filled over FX holidays."""
    fx = (close["CNY=X"] / close["HKD=X"]).ffill()
    fx.name = "HKDCNY"
    return fx


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(n_pairs: int = 12, n_months: int = 198, premium: float = 0.30,
                    phi: float = 0.97, seed: int = 620,
                    sig_h: float = 0.08, sig_prem: float = 0.025,
                    common_share: float = 0.6) -> dict:
    """Deterministic monthly A/H world with a PLANTED premium level.

    Each pair's H line follows a lognormal random walk; its A line is
    ``A = H_in_CNY * (1 + premium_t)`` where the pair's premium follows an AR(1) with
    persistence ``phi`` around the planted mean ``premium``, driven by a mix of a COMMON
    factor (share ``common_share`` of shock variance — the one-factor co-movement) and an
    idiosyncratic shock. With ``premium = 0`` the two lines are the same price up to the
    mean-zero AR(1) noise: the mean-level detector must stay quiet.

    Returns ``{"prem": DataFrame (months x pairs) of premium levels,
               "rel": DataFrame of monthly H-minus-A total-return diffs}``
    on a decorative monthly PeriodIndex (period_range: no ns-overflow, span << 250 years).
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2010-01", periods=n_months, freq="M")
    names = [f"P{i:02d}" for i in range(n_pairs)]

    prem_cols, rel_cols = {}, {}
    z_common = rng.normal(0.0, 1.0, size=n_months)
    for j in range(n_pairs):
        z_idio = rng.normal(0.0, 1.0, size=n_months)
        shock = (np.sqrt(common_share) * z_common
                 + np.sqrt(1.0 - common_share) * z_idio) * sig_prem
        p = np.empty(n_months)
        if phi >= 1.0:                       # random-walk world: start AT the plant, then wander
            p[0] = premium
            for t in range(1, n_months):
                p[t] = p[t - 1] + shock[t]
        else:
            p[0] = premium + shock[0] / np.sqrt(1.0 - phi ** 2)
            for t in range(1, n_months):
                p[t] = premium + phi * (p[t - 1] - premium) + shock[t]
        r_h = rng.normal(0.004, sig_h, size=n_months)          # H total return, CNY terms
        h = 10.0 * np.exp(np.cumsum(r_h))
        a = h * (1.0 + p)
        # monthly H-minus-A total-return diff implied by the premium path
        r_a = np.empty(n_months)
        r_a[0] = r_h[0]
        r_a[1:] = np.diff(np.log(a))
        rel = np.exp(r_h) - np.exp(r_a)                        # ~ simple-return diff
        rel[0] = 0.0
        prem_cols[names[j]] = p
        rel_cols[names[j]] = rel
    prem = pd.DataFrame(prem_cols, index=pidx)
    rel = pd.DataFrame(rel_cols, index=pidx)
    return {"prem": prem, "rel": rel}
