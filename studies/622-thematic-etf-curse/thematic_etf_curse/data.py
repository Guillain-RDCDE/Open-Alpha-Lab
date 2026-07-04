"""Data layer for Study 622 — Thematic ETF Curse.

The claim (Ben-David, Franzoni, Kim & Moussawi, RFS 2023, *Competition for Attention in the
ETF Space*): **specialized / thematic ETFs launch at peak hype and lose ~5%/yr risk-adjusted
over their first years** — the launch date itself is the sell signal. We rebuild the test on
a transparent panel of US-listed thematic ETFs (yfinance, no key):

* **Real tape.** Daily auto-adjusted (total-return) closes for ~45 thematic ETFs, ~14
  broad-index ETF launches (the placebo), SPY (the market) and ^IRX (13-week T-bill yield,
  the risk-free proxy). The **launch = the first candle** on the tape. Each candidate ticker
  carries a hardcoded *expected inception month* (from the issuer's fund page / prospectus)
  and is **dropped if its first candle disagrees by more than a few months** — the guard
  against Yahoo ticker reuse (a recycled symbol would silently splice an unrelated fund's
  history onto the launch date). Prices cached wide (``tec_prices.csv``), risk-free cached
  as a single series (``tec_rf.csv``).

* **Survivorship — named, and it works AGAINST the claim.** Delisted thematic ETFs (the
  worst performers: they bled, shrank and closed) return nothing on yfinance and drop out of
  the panel. Excluding the dead funds biases post-launch performance **up** — so a negative
  post-launch alpha found on survivors alone *understates* the curse. The bias strengthens,
  never manufactures, a negative finding.

* **Synthetic.** A deterministic, seeded generator of a monthly fund panel with a TUNABLE
  planted post-launch alpha drag (knob ``drag_ann``): funds launch at staggered dates, carry
  beta ≈ 1.2 to a synthetic market, and (if ``drag_ann != 0``) bleed the planted alpha over
  their first 36 event months. ``drag_ann = 0`` is the null — the calendar-time machinery
  must NOT manufacture a negative alpha out of noise. Monthly period_range only (spans well
  under 250 years — the ns-Timestamp CI trap).

Pure numpy + pandas + stdlib on the offline path; ``fetch_panel`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "tec_prices.csv")
RF_CACHE = os.path.join(CACHE_DIR, "tec_rf.csv")

MARKET = "SPY"
RF_TICKER = "^IRX"          # 13-week T-bill discount yield, annualized %

# --------------------------------------------------------------------------- #
# The thematic panel — (ticker, expected inception "YYYY-MM")
# Expected inceptions from the issuers' fund pages / prospectuses (see
# docs/references.md). Used ONLY as a ticker-reuse guard: if the first candle
# on Yahoo is far from the expected launch, the symbol was recycled and the
# name is dropped. Launches span 2005-2021 — solar/clean energy (2008 wave),
# robotics/cyber (2014-16 wave), and the 2018-2021 hype vintage (cloud, genomics,
# cannabis, e-sports, metaverse, crypto-adjacent, social sentiment).
# --------------------------------------------------------------------------- #
THEMATIC = [
    ("PBW",  "2005-03"),   # Invesco WilderHill Clean Energy
    ("QCLN", "2007-02"),   # First Trust NASDAQ Clean Edge Green Energy
    ("TAN",  "2008-04"),   # Invesco Solar
    ("FAN",  "2008-06"),   # First Trust Global Wind Energy
    ("ICLN", "2008-06"),   # iShares Global Clean Energy
    ("LIT",  "2010-07"),   # Global X Lithium & Battery Tech
    ("REMX", "2010-10"),   # VanEck Rare Earth/Strategic Metals
    ("URA",  "2010-11"),   # Global X Uranium
    ("SKYY", "2011-07"),   # First Trust Cloud Computing
    ("SOCL", "2011-11"),   # Global X Social Media
    ("KWEB", "2013-07"),   # KraneShares CSI China Internet
    ("IPO",  "2013-10"),   # Renaissance IPO
    ("ROBO", "2013-10"),   # ROBO Global Robotics & Automation
    ("ARKW", "2014-09"),   # ARK Next Generation Internet
    ("ARKQ", "2014-09"),   # ARK Autonomous Tech & Robotics
    ("ARKK", "2014-10"),   # ARK Innovation
    ("ARKG", "2014-10"),   # ARK Genomic Revolution
    ("HACK", "2014-11"),   # Amplify Cybersecurity (ex-ETFMG Prime Cyber)
    ("XT",   "2015-03"),   # iShares Exponential Technologies
    ("JETS", "2015-04"),   # U.S. Global Jets
    ("CIBR", "2015-07"),   # First Trust NASDAQ Cybersecurity
    ("GAMR", "2016-03"),   # Wedbush ETFMG Video Game Tech
    ("IBUY", "2016-04"),   # Amplify Online Retail
    ("MILN", "2016-05"),   # Global X Millennial Consumer
    ("PRNT", "2016-07"),   # 3D Printing ETF
    ("BOTZ", "2016-09"),   # Global X Robotics & AI
    ("FINX", "2016-09"),   # Global X FinTech
    ("IZRL", "2017-12"),   # ARK Israel Innovative Technology
    ("KARS", "2018-01"),   # KraneShares Electric Vehicles & Future Mobility
    ("ROBT", "2018-02"),   # First Trust Nasdaq AI & Robotics
    ("DRIV", "2018-04"),   # Global X Autonomous & Electric Vehicles
    ("AIQ",  "2018-05"),   # Global X Artificial Intelligence & Tech
    ("IRBO", "2018-06"),   # iShares Robotics & AI Multisector
    ("ONLN", "2018-07"),   # ProShares Online Retail
    ("QTUM", "2018-09"),   # Defiance Quantum
    ("ESPO", "2018-10"),   # VanEck Video Gaming & eSports
    ("EBIZ", "2018-11"),   # Global X E-commerce
    ("ARKF", "2019-02"),   # ARK Fintech Innovation
    ("CLOU", "2019-04"),   # Global X Cloud Computing
    ("WCLD", "2019-09"),   # WisdomTree Cloud Computing
    ("GNOM", "2019-04"),   # Global X Genomics & Biotechnology
    ("HERO", "2019-10"),   # Global X Video Games & Esports
    ("BETZ", "2020-06"),   # Roundhill Sports Betting & iGaming
    ("MSOS", "2020-09"),   # AdvisorShares Pure US Cannabis
    ("BUZZ", "2021-03"),   # VanEck Social Sentiment
    ("BITQ", "2021-05"),   # Bitwise Crypto Industry Innovators
    ("METV", "2021-06"),   # Roundhill Metaverse (ex-Ball Metaverse)
    ("HYDR", "2021-07"),   # Global X Hydrogen
    ("BKCH", "2021-07"),   # Global X Blockchain
]

# Broad, plain-vanilla index-ETF launches — the placebo. Same "first candle =
# launch" treatment; if launch timing per se (not the hype) drove returns, these
# should bleed too. They track total-market / S&P-style cap-weighted baskets.
BROAD = [
    ("IWB",  "2000-05"),   # iShares Russell 1000
    ("IVV",  "2000-05"),   # iShares Core S&P 500
    ("SPTM", "2000-10"),   # SPDR Portfolio S&P 1500 (ex-total market)
    ("VTI",  "2001-05"),   # Vanguard Total Stock Market
    ("ITOT", "2004-01"),   # iShares Core S&P Total US
    ("VV",   "2004-01"),   # Vanguard Large-Cap
    ("SPLG", "2005-11"),   # SPDR Portfolio S&P 500 (ex-large cap)
    ("MGC",  "2007-12"),   # Vanguard Mega Cap
    ("SCHB", "2009-11"),   # Schwab US Broad Market
    ("SCHX", "2009-11"),   # Schwab US Large-Cap
    ("VOO",  "2010-09"),   # Vanguard S&P 500
    ("VONE", "2010-09"),   # Vanguard Russell 1000
    ("SCHK", "2017-10"),   # Schwab 1000
    ("BBUS", "2019-03"),   # JPMorgan BetaBuilders US
]

# first candle must land within [-45, +100] days of the expected inception
# month start, else the symbol is treated as recycled/bad and dropped.
LAUNCH_TOL_BEFORE = 45
LAUNCH_TOL_AFTER = 100


# --------------------------------------------------------------------------- #
# Real tape (network — once, to build the cache)
# --------------------------------------------------------------------------- #
def _validated(prices: pd.DataFrame, expected: list[tuple[str, str]]) -> list[str]:
    """Tickers whose first candle sits within tolerance of the expected inception."""
    keep = []
    for tk, exp in expected:
        if tk not in prices.columns:
            continue
        s = prices[tk].dropna()
        if len(s) < 200:                     # need a usable post-launch tape
            continue
        first = s.index.min()
        exp_start = pd.Timestamp(exp + "-01")
        gap = (first - exp_start).days
        if -LAUNCH_TOL_BEFORE <= gap <= LAUNCH_TOL_AFTER:
            keep.append(tk)
        else:
            print(f"  [drop] {tk}: first candle {first.date()} vs expected {exp} "
                  f"(gap {gap}d) — recycled ticker or bad history")
    return keep


def fetch_panel(prices_path: str = PRICES_CACHE, rf_path: str = RF_CACHE,
                retries: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Download the full panel (max history) and cache it. Network-only, run once.

    Auto-adjusted closes = total-return prices. Launch-date validation drops any
    ticker whose first candle disagrees with the issuer's inception month.
    """
    import yfinance as yf

    tickers = [t for t, _ in THEMATIC] + [t for t, _ in BROAD] + [MARKET]
    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(tickers, period="max", auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    raw = raw.dropna(how="all")

    keep = _validated(raw, THEMATIC) + _validated(raw, BROAD)
    prices = raw[keep + [MARKET]].copy()
    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    prices.to_csv(prices_path)

    rf_raw = None
    for _ in range(retries):
        try:
            rf_raw = yf.download(RF_TICKER, period="max", auto_adjust=False,
                                 progress=False)["Close"]
            if rf_raw is not None and len(rf_raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    rf = rf_raw.iloc[:, 0] if isinstance(rf_raw, pd.DataFrame) else rf_raw
    rf = rf.dropna()
    rf.name = "IRX"
    rf.to_frame().to_csv(rf_path)
    return prices, rf


def have_real(prices_path: str = PRICES_CACHE, rf_path: str = RF_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(rf_path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide auto-adjusted (total-return) close frame, index = date."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def load_rf(path: str = RF_CACHE) -> pd.Series:
    """^IRX daily level (annualized T-bill discount yield, %)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()["IRX"]


# --------------------------------------------------------------------------- #
# Monthly universe construction (shared by real + synthetic paths)
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame, as_of: str | None = None) -> pd.DataFrame:
    """Month-end-to-month-end simple returns; the trailing partial month is dropped.

    ``as_of`` slices the daily tape first, so the published sample never creeps.
    A month-end bucket built from a tape that stops mid-month is removed.
    """
    px = prices if as_of is None else prices[prices.index <= pd.Timestamp(as_of)]
    m = px.resample("ME").last()
    ret = m.pct_change()
    last_px = px.index.max()
    last_bucket = ret.index.max()
    if last_px < last_bucket:            # tape stops before the bucket's month-end
        ret = ret.iloc[:-1]
    return ret


def rf_monthly(rf_daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Monthly risk-free simple return from the ^IRX level.

    ^IRX quotes the annualized 13-week T-bill discount yield in %; the monthly
    risk-free return is approximated as the within-month mean level / 100 / 12.
    Reindexed onto the monthly return index (early-1990s gaps forward-filled).
    """
    lvl = rf_daily.resample("ME").mean() / 100.0 / 12.0
    return lvl.reindex(index).ffill().fillna(0.0)


def build_universe(prices: pd.DataFrame, rf_daily: pd.Series,
                   as_of: str | None = None) -> dict:
    """The analysis object every strategy function consumes.

    Returns ``{mret, mkt, rf, launch, thematic, broad}`` where ``mret`` is the
    wide monthly return frame (funds only), ``mkt``/``rf`` are monthly series,
    and ``launch[tk]`` is the launch month (Period M) = month of the first
    candle. The launch month itself is a partial month: its stub return never
    exists in ``mret`` (pct_change needs the prior month-end), so event month 1
    is automatically the first COMPLETE calendar month after launch.
    """
    mret = monthly_returns(prices, as_of=as_of)
    mkt = mret[MARKET]
    funds = [c for c in mret.columns if c != MARKET]
    launch = {}
    px = prices if as_of is None else prices[prices.index <= pd.Timestamp(as_of)]
    for tk in funds:
        s = px[tk].dropna()
        if len(s) == 0:
            continue
        launch[tk] = s.index.min().to_period("M")
    rf = rf_monthly(rf_daily, mret.index)
    thematic = [t for t, _ in THEMATIC if t in launch]
    broad = [t for t, _ in BROAD if t in launch]
    return {"mret": mret[funds], "mkt": mkt, "rf": rf, "launch": launch,
            "thematic": thematic, "broad": broad}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_world(n_funds: int = 40, n_months: int = 240, drag_ann: float = 0.0,
                    seed: int = 622, beta_mean: float = 1.2,
                    idio_sig: float = 0.055) -> dict:
    """Deterministic monthly fund panel with a PLANTED post-launch alpha drag.

    A synthetic market (monthly mean 0.7%, vol 4.5%) plus ``n_funds`` funds
    launching at staggered months; each fund has beta ~ ``beta_mean`` and idio
    noise. If ``drag_ann != 0`` every fund bleeds ``drag_ann / 12`` per month of
    PLANTED alpha over event months 1..36 — exactly the effect the calendar-time
    machinery must recover. ``drag_ann = 0`` is the null: the pipeline must not
    manufacture a negative alpha from noise.

    Returns the same dict shape as ``build_universe`` (plus every fund tagged
    thematic; ``broad`` is empty). Monthly period index only — the span is 20
    years, far under the 250-year ns-Timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    pidx = pd.period_range("2005-01", periods=n_months, freq="M")
    stamps = pidx.to_timestamp(how="end").normalize()
    mkt = pd.Series(rng.normal(0.007, 0.045, n_months), index=stamps, name="MKT")
    rf = pd.Series(0.002, index=stamps, name="rf")

    cols, launch = {}, {}
    drag_m = drag_ann / 12.0
    for j in range(n_funds):
        name = f"F{j:02d}"
        lm = int(rng.integers(12, n_months - 60))     # leave >= 5y of post-launch tape
        beta = float(rng.normal(beta_mean, 0.15))
        r = rf.values + beta * (mkt.values - rf.values) \
            + rng.normal(0.0, idio_sig, n_months)
        k = np.arange(n_months) - lm                  # event month index
        r = r + np.where((k >= 1) & (k <= 36), drag_m, 0.0)
        r = np.where(k >= 1, r, np.nan)               # no tape before event month 1
        cols[name] = r
        launch[name] = pidx[lm]
    mret = pd.DataFrame(cols, index=stamps)
    return {"mret": mret, "mkt": mkt, "rf": rf, "launch": launch,
            "thematic": list(cols.keys()), "broad": []}
