"""Data layer for Study 725 — "Eggflation" (trade the egg-price spike via CALM).

Three sources, all offline-friendly:

* **The retail egg-price series (hardcoded, cited, approximate).** The USDA/BLS average
  retail price of Grade-A large eggs (per dozen, US city average — the number that lands
  in the "eggflation" news headline, BLS series ``APU0000708111``) is public but only
  *published with a ~two-week lag* and is not conveniently API-available in a single call.
  So we hardcode a small **monthly** series (Jan-2015 → Jun-2026) reconstructed from public
  BLS/USDA reporting and clearly labelled **approximate**. It is anchored on figures the
  desk could cite: the **2015** HPAI spike (~$2.97/dozen, Sep-2015), the **Jan-2023**
  record (~$4.82), and the **Feb/Mar-2025** all-time high (~$6.23). Sources are in
  ``docs/references.md``. This is a PROXY for the real tape, never the tape.

* **Cal-Maine + benchmark (yfinance).** **CALM** — Cal-Maine Foods, the largest US
  shell-egg producer, the closest thing to a listed "egg price" — and **SPY** as the
  benchmark. Month-end Adj Close, cached under ``_cache/`` so the notebooks run offline;
  on a cache miss *with* network we fetch via yfinance.

* **Synthetic positive control.** A deterministic fixed-seed world in which this month's
  egg-price change *genuinely* drives next month's CALM return (a *planted* predictive
  lead), used to prove the inference engine recovers a real lead when one exists. No
  network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(STUDY_DIR, "_cache")

CALM = "CALM"   # Cal-Maine Foods — largest US shell-egg producer (the "egg stock")
BENCH = "SPY"
TICKERS = [CALM, BENCH]
MONTHS_PER_YEAR = 12

# --------------------------------------------------------------------------- #
# The retail egg-price series — HARDCODED, CITED, APPROXIMATE (a proxy, not a feed)
# --------------------------------------------------------------------------- #
# Monthly USDA/BLS average retail price, Grade-A large eggs, $/dozen, US city average
# (BLS APU0000708111). Reconstructed from public BLS/USDA reporting; LABELLED APPROXIMATE.
# The shape is the load-bearing fact: quiet ~$1.30-1.80 base punctuated by three
# avian-flu (HPAI) spikes — 2015 (~$2.97), the Jan-2023 record (~$4.82) and the
# Feb/Mar-2025 all-time high (~$6.23) — each a sharp spike-and-mean-revert.
_EGG = {
    "2015-01": 2.11, "2015-02": 2.10, "2015-03": 2.06, "2015-04": 2.05, "2015-05": 2.06,
    "2015-06": 2.57, "2015-07": 2.88, "2015-08": 2.94, "2015-09": 2.97, "2015-10": 2.72,
    "2015-11": 2.44, "2015-12": 2.36,
    "2016-01": 2.20, "2016-02": 1.98, "2016-03": 1.72, "2016-04": 1.53, "2016-05": 1.43,
    "2016-06": 1.37, "2016-07": 1.42, "2016-08": 1.44, "2016-09": 1.47, "2016-10": 1.48,
    "2016-11": 1.43, "2016-12": 1.34,
    "2017-01": 1.30, "2017-02": 1.32, "2017-03": 1.42, "2017-04": 1.53, "2017-05": 1.58,
    "2017-06": 1.55, "2017-07": 1.58, "2017-08": 1.62, "2017-09": 1.60, "2017-10": 1.55,
    "2017-11": 1.44, "2017-12": 1.47,
    "2018-01": 1.42, "2018-02": 1.66, "2018-03": 1.75, "2018-04": 1.81, "2018-05": 1.78,
    "2018-06": 1.62, "2018-07": 1.55, "2018-08": 1.44, "2018-09": 1.42, "2018-10": 1.48,
    "2018-11": 1.51, "2018-12": 1.40,
    "2019-01": 1.55, "2019-02": 1.44, "2019-03": 1.40, "2019-04": 1.44, "2019-05": 1.40,
    "2019-06": 1.38, "2019-07": 1.40, "2019-08": 1.39, "2019-09": 1.42, "2019-10": 1.44,
    "2019-11": 1.41, "2019-12": 1.44,
    "2020-01": 1.46, "2020-02": 1.45, "2020-03": 1.50, "2020-04": 1.62, "2020-05": 1.62,
    "2020-06": 1.52, "2020-07": 1.48, "2020-08": 1.42, "2020-09": 1.40, "2020-10": 1.41,
    "2020-11": 1.45, "2020-12": 1.48,
    "2021-01": 1.47, "2021-02": 1.59, "2021-03": 1.62, "2021-04": 1.64, "2021-05": 1.63,
    "2021-06": 1.59, "2021-07": 1.62, "2021-08": 1.63, "2021-09": 1.62, "2021-10": 1.66,
    "2021-11": 1.72, "2021-12": 1.79,
    "2022-01": 1.93, "2022-02": 2.00, "2022-03": 2.05, "2022-04": 2.52, "2022-05": 2.86,
    "2022-06": 2.71, "2022-07": 2.94, "2022-08": 3.12, "2022-09": 2.90, "2022-10": 3.42,
    "2022-11": 3.59, "2022-12": 4.25,
    "2023-01": 4.82, "2023-02": 4.21, "2023-03": 3.45, "2023-04": 3.27, "2023-05": 2.67,
    "2023-06": 2.22, "2023-07": 2.09, "2023-08": 2.04, "2023-09": 2.07, "2023-10": 2.07,
    "2023-11": 2.14, "2023-12": 2.51,
    "2024-01": 2.52, "2024-02": 3.00, "2024-03": 2.99, "2024-04": 2.86, "2024-05": 2.70,
    "2024-06": 2.72, "2024-07": 3.08, "2024-08": 3.20, "2024-09": 3.82, "2024-10": 3.37,
    "2024-11": 3.65, "2024-12": 4.15,
    "2025-01": 4.95, "2025-02": 5.90, "2025-03": 6.23, "2025-04": 5.12, "2025-05": 4.55,
    "2025-06": 3.78, "2025-07": 3.20, "2025-08": 3.10, "2025-09": 3.05, "2025-10": 3.15,
    "2025-11": 3.20, "2025-12": 3.00,
    "2026-01": 2.85, "2026-02": 2.80, "2026-03": 2.75, "2026-04": 2.90, "2026-05": 3.00,
    "2026-06": 3.10,
}
# The three reported avian-flu peaks (label, month-end, $/dozen) — the news anchors.
EGG_PEAKS = [("2015 HPAI", "2015-09", 2.97),
             ("2022-23 HPAI (record)", "2023-01", 4.82),
             ("2024-25 HPAI (all-time high)", "2025-03", 6.23)]


def load_egg_index() -> pd.Series:
    """Month-end levels of the (approximate, cited) retail egg price ($/dozen).

    Returns a ``pd.Series`` indexed by month-end ``Timestamp``. LABELLED A PROXY:
    reconstructed from public BLS/USDA reporting, not a live data feed.
    """
    idx = pd.to_datetime([f"{k}-01" for k in _EGG]) + pd.offsets.MonthEnd(0)
    return pd.Series(list(_EGG.values()), index=idx, name="egg").sort_index()


def egg_peaks() -> pd.DataFrame:
    """The three reported avian-flu blow-off peaks (label, date, $/dozen)."""
    rows = [(lab, pd.Timestamp(m) + pd.offsets.MonthEnd(0), lvl) for lab, m, lvl in EGG_PEAKS]
    return pd.DataFrame(rows, columns=["episode", "date", "price"]).set_index("date")


# --------------------------------------------------------------------------- #
# CALM + SPY via yfinance (cached, offline-friendly)
# --------------------------------------------------------------------------- #
def _cache_path() -> str:
    return os.path.join(CACHE, "eggflation_market.parquet")


def have_market() -> bool:
    return os.path.exists(_cache_path())


def fetch_market(fetch: bool = False, start: str = "2014-12-01",
                 end: str = "2026-07-01") -> pd.DataFrame:
    """Month-end Adj Close for CALM + SPY, cache-first.

    **Cache-only** unless ``fetch=True`` (empty frame on a cache miss with ``fetch=False``).
    Built from Yahoo's month-end feed, the in-progress month dropped so the last bar is
    complete. Columns ``CALM``, ``SPY``. Network only on an explicit cache miss + ``fetch``.
    """
    cache = _cache_path()
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy — the offline core never imports this

    raw = yf.download(TICKERS, start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    close.index = pd.DatetimeIndex(close.index).tz_localize(None) + pd.offsets.MonthEnd(0)
    close = close[~close.index.duplicated(keep="last")]
    last_complete = pd.Timestamp.today().to_period("M") - 1
    close = close[close.index.to_period("M") <= last_complete]
    out = close[TICKERS].dropna()
    out.index.name = "date"
    os.makedirs(CACHE, exist_ok=True)
    out.to_parquet(cache)
    return out


def build_panel(asof: str | None = "2026-06-30") -> pd.DataFrame:
    """Merge the egg-price proxy with CALM/SPY into one monthly returns panel.

    Columns: ``egg`` (level $/dozen), ``calm``, ``spy`` (Adj Close levels), and the
    monthly simple returns ``egg_chg``, ``calm_ret``, ``spy_ret``. Sliced to ``asof``
    (never a future/partial month). Returns an **empty** frame if the market cache is
    missing (so offline callers can fall back to frozen numbers).
    """
    mkt = fetch_market(fetch=False)
    if mkt.empty:
        return pd.DataFrame()
    egg = load_egg_index()
    df = pd.DataFrame({"egg": egg, "calm": mkt[CALM], "spy": mkt[BENCH]}).dropna()
    if asof is not None:
        df = df[df.index <= pd.Timestamp(asof)]
    df["egg_chg"] = df["egg"].pct_change()
    df["calm_ret"] = df["calm"].pct_change()
    df["spy_ret"] = df["spy"].pct_change()
    return df.dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 137, lead_beta: float = 0.6,
                    egg_vol: float = 0.09, calm_vol: float = 0.10,
                    seed: int = 725) -> pd.DataFrame:
    """A world where the egg-price change **genuinely leads** next-month CALM returns.

    ``calm_ret[t] = lead_beta * egg_chg[t-1] + noise``. With ``lead_beta > 0`` the
    predictive regression *should* recover a significant positive slope and the timer
    *should* pay — the machinery proof. ``lead_beta = 0`` is the null. Deterministic
    given ``seed``; returns the same columns as :func:`build_panel`. No network.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-02-28", periods=n_months, freq="ME", name="date")
    egg_chg = egg_vol * rng.standard_normal(n_months)
    calm_ret = lead_beta * np.concatenate([[0.0], egg_chg[:-1]]) + calm_vol * rng.standard_normal(n_months)
    spy_ret = 0.04 / 12 + 0.04 * rng.standard_normal(n_months)
    egg = 2.0 * np.cumprod(1 + egg_chg)
    calm = 40.0 * np.cumprod(1 + calm_ret)
    spy = 200.0 * np.cumprod(1 + spy_ret)
    return pd.DataFrame({"egg": egg, "calm": calm, "spy": spy,
                         "egg_chg": egg_chg, "calm_ret": calm_ret, "spy_ret": spy_ret},
                        index=idx)
