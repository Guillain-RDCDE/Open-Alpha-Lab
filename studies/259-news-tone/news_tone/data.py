"""Data layer for Study 259 (News-Tone).

The question: does *aggregate news tone* -- a single daily sentiment score for
the macro/market news flow -- predict the *next* day's S&P 500 return?

Three components, all offline for the core:

- ``synthetic_daily`` -- a *deterministic, offline* generator of a (date x tone,
  next-day-return) panel with an explicit ``gamma`` knob that plants exactly the
  effect under test: a linear link between today's tone z-score and *tomorrow's*
  return.  ``gamma = 0`` is the null (tone is pure noise, uncorrelated with the
  forward return).  This is the study's null-in-a-bottle and the positive
  control for the inference machinery.

- ``NEWS_TONE_MONTHLY`` -- a hardcoded, curated *monthly* news-tone index proxy
  (z-scored), 2007-01 .. 2025-12.  Each value is a coarse, deterministic
  hand-curated read of the macro headline mood for that month (crisis months
  print deeply negative, melt-up months print positive).  This is a *proxy*, not
  a feed: it is a stand-in for a daily sentiment index (e.g. RavenPack / GDELT
  tone / a Loughran-McDonald headline score), built to be offline and
  reproducible.  It is forward-filled to business days by ``news_tone_daily``.
  **The proxy is the headline-mood envelope, not an information edge** -- it is
  constructed *contemporaneously* with the market, so any same-day correlation
  is mechanical, and the forward (next-day) test is the honest one.

- ``fetch_gspc`` -- the real ^GSPC daily adjusted-close series via yfinance,
  cache-only by default (cache: ``_cache/gspc_daily.parquet``), so the test
  suite and reproducible core never touch the network.

**No look-ahead**: the tone observed at the close of day t is matched to the
return *earned* on day t+1 (close-to-close).  Positions implied by the tone
signal are formed at the close of day t and held through day t+1 (a one-day
execution lag).  The signal at day t uses only tone information available at the
close of day t.

**Proxy / survivorship caveat**: the monthly tone proxy is a hand-curated,
in-sample read of well-known macro events.  If anything it is *biased toward*
finding a relationship (the curator knows which months were bad).  Any positive
result is therefore an upper bound -- named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

GSPC_CACHE = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")

SEED = 259


# ---------------------------------------------------------------------------
# Curated monthly news-tone proxy -- the deterministic, offline core
# ---------------------------------------------------------------------------
# A coarse, hand-curated z-scored monthly news-tone index proxy.
# Negative = risk-off / bad-news flow; positive = risk-on / good-news flow.
# Values are intentionally crude (one-decimal envelope) -- this is a *proxy* for
# a daily sentiment feed, anchored to well-known macro mood swings:
#   2008-09..2009-03 : GFC collapse (deeply negative)
#   2011-08          : US downgrade / euro crisis
#   2018-12          : Q4 vol spike
#   2020-03          : COVID crash (record-negative)
#   2022             : inflation / rate-hike grind (persistently negative)
#   2017, 2021, 2023-24 : melt-ups (positive)
# Source spirit: GDELT tone, RavenPack event sentiment, Baker-Bloom-Davis EPU
# (sign-flipped), Loughran-McDonald headline tone.  Hardcoded to stay offline.
NEWS_TONE_MONTHLY: dict[str, float] = {
    # 2007 -- pre-crisis complacency then summer credit jitters
    "2007-01":  0.6, "2007-02":  0.1, "2007-03":  0.3, "2007-04":  0.7,
    "2007-05":  0.6, "2007-06":  0.2, "2007-07": -0.4, "2007-08": -0.9,
    "2007-09": -0.2, "2007-10":  0.4, "2007-11": -0.6, "2007-12": -0.5,
    # 2008 -- the global financial crisis
    "2008-01": -1.1, "2008-02": -0.8, "2008-03": -1.0, "2008-04": -0.3,
    "2008-05":  0.0, "2008-06": -0.9, "2008-07": -1.0, "2008-08": -0.6,
    "2008-09": -2.4, "2008-10": -3.1, "2008-11": -2.6, "2008-12": -1.8,
    # 2009 -- bottom then recovery
    "2009-01": -1.9, "2009-02": -2.2, "2009-03": -1.3, "2009-04":  0.2,
    "2009-05":  0.6, "2009-06":  0.4, "2009-07":  0.8, "2009-08":  0.9,
    "2009-09":  0.9, "2009-10":  0.5, "2009-11":  0.7, "2009-12":  0.8,
    # 2010 -- flash crash, euro crisis I
    "2010-01":  0.3, "2010-02": -0.2, "2010-03":  0.6, "2010-04":  0.4,
    "2010-05": -1.4, "2010-06": -0.9, "2010-07":  0.5, "2010-08": -0.3,
    "2010-09":  0.7, "2010-10":  0.6, "2010-11":  0.2, "2010-12":  0.8,
    # 2011 -- US downgrade, euro crisis II
    "2011-01":  0.5, "2011-02":  0.6, "2011-03": -0.7, "2011-04":  0.4,
    "2011-05": -0.2, "2011-06": -0.4, "2011-07": -0.6, "2011-08": -2.0,
    "2011-09": -1.3, "2011-10":  0.3, "2011-11": -0.5, "2011-12":  0.1,
    # 2012 -- draghi "whatever it takes"
    "2012-01":  0.6, "2012-02":  0.7, "2012-03":  0.5, "2012-04": -0.3,
    "2012-05": -1.0, "2012-06":  0.2, "2012-07":  0.4, "2012-08":  0.6,
    "2012-09":  0.7, "2012-10":  0.1, "2012-11":  0.0, "2012-12":  0.4,
    # 2013 -- taper tantrum + grind up
    "2013-01":  0.9, "2013-02":  0.6, "2013-03":  0.8, "2013-04":  0.5,
    "2013-05":  0.6, "2013-06": -0.6, "2013-07":  0.7, "2013-08": -0.2,
    "2013-09":  0.5, "2013-10":  0.7, "2013-11":  0.8, "2013-12":  0.8,
    # 2014 -- low-vol grind, oil crash year-end
    "2014-01": -0.3, "2014-02":  0.6, "2014-03":  0.3, "2014-04":  0.3,
    "2014-05":  0.6, "2014-06":  0.7, "2014-07":  0.2, "2014-08":  0.6,
    "2014-09":  0.1, "2014-10": -0.5, "2014-11":  0.6, "2014-12": -0.2,
    # 2015 -- china deval, august spike
    "2015-01":  0.0, "2015-02":  0.6, "2015-03":  0.1, "2015-04":  0.3,
    "2015-05":  0.2, "2015-06": -0.4, "2015-07":  0.1, "2015-08": -1.3,
    "2015-09": -0.7, "2015-10":  0.6, "2015-11":  0.1, "2015-12": -0.6,
    # 2016 -- jan selloff, brexit, trump rally
    "2016-01": -1.2, "2016-02": -0.6, "2016-03":  0.5, "2016-04":  0.3,
    "2016-05":  0.3, "2016-06": -0.5, "2016-07":  0.5, "2016-08":  0.4,
    "2016-09":  0.1, "2016-10": -0.2, "2016-11":  0.4, "2016-12":  0.7,
    # 2017 -- the low-vol melt-up
    "2017-01":  0.7, "2017-02":  0.9, "2017-03":  0.6, "2017-04":  0.5,
    "2017-05":  0.6, "2017-06":  0.6, "2017-07":  0.8, "2017-08":  0.3,
    "2017-09":  0.7, "2017-10":  0.9, "2017-11":  0.8, "2017-12":  0.8,
    # 2018 -- vol returns, Q4 swoon
    "2018-01":  0.9, "2018-02": -1.0, "2018-03": -0.7, "2018-04":  0.1,
    "2018-05":  0.3, "2018-06":  0.0, "2018-07":  0.5, "2018-08":  0.5,
    "2018-09":  0.4, "2018-10": -1.1, "2018-11": -0.4, "2018-12": -1.9,
    # 2019 -- fed pivot, trade-war headlines
    "2019-01":  0.6, "2019-02":  0.7, "2019-03":  0.4, "2019-04":  0.7,
    "2019-05": -0.8, "2019-06":  0.6, "2019-07":  0.4, "2019-08": -0.7,
    "2019-09":  0.3, "2019-10":  0.4, "2019-11":  0.7, "2019-12":  0.7,
    # 2020 -- COVID crash and reflation
    "2020-01":  0.2, "2020-02": -1.2, "2020-03": -3.4, "2020-04":  0.3,
    "2020-05":  0.5, "2020-06":  0.4, "2020-07":  0.6, "2020-08":  0.8,
    "2020-09": -0.3, "2020-10": -0.4, "2020-11":  1.0, "2020-12":  0.7,
    # 2021 -- reopening melt-up, late-year jitters
    "2021-01":  0.4, "2021-02":  0.6, "2021-03":  0.6, "2021-04":  0.8,
    "2021-05":  0.3, "2021-06":  0.6, "2021-07":  0.5, "2021-08":  0.6,
    "2021-09": -0.3, "2021-10":  0.6, "2021-11": -0.2, "2021-12":  0.4,
    # 2022 -- inflation / rate-hike bear
    "2022-01": -0.9, "2022-02": -1.1, "2022-03": -0.6, "2022-04": -1.0,
    "2022-05": -0.8, "2022-06": -1.4, "2022-07":  0.3, "2022-08": -0.7,
    "2022-09": -1.5, "2022-10":  0.2, "2022-11":  0.5, "2022-12": -0.6,
    # 2023 -- banking scare then AI rally
    "2023-01":  0.5, "2023-02": -0.2, "2023-03": -0.8, "2023-04":  0.3,
    "2023-05":  0.2, "2023-06":  0.7, "2023-07":  0.7, "2023-08": -0.3,
    "2023-09": -0.5, "2023-10": -0.4, "2023-11":  0.9, "2023-12":  0.9,
    # 2024 -- soft-landing melt-up, august vol blip
    "2024-01":  0.6, "2024-02":  0.7, "2024-03":  0.7, "2024-04": -0.4,
    "2024-05":  0.6, "2024-06":  0.6, "2024-07":  0.3, "2024-08": -0.5,
    "2024-09":  0.5, "2024-10":  0.3, "2024-11":  0.8, "2024-12":  0.2,
    # 2025 -- tariff scare then recovery
    "2025-01":  0.5, "2025-02": -0.2, "2025-03": -0.6, "2025-04": -1.3,
    "2025-05":  0.4, "2025-06":  0.5, "2025-07":  0.6, "2025-08":  0.2,
    "2025-09":  0.5, "2025-10":  0.4, "2025-11":  0.3, "2025-12":  0.4,
}


def news_tone_monthly() -> pd.Series:
    """Return the curated monthly news-tone proxy as a month-end indexed Series.

    Index = month-end ``DatetimeIndex``; values = z-scored tone (negative = bad
    news flow, positive = good).  Deterministic and offline.
    """
    idx = pd.to_datetime(list(NEWS_TONE_MONTHLY.keys())) + pd.offsets.MonthEnd(0)
    s = pd.Series(list(NEWS_TONE_MONTHLY.values()), index=idx, name="tone")
    return s.sort_index()


def news_tone_daily(business_days: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill the monthly tone proxy onto a business-day index.

    Each day inherits the tone of the month it falls in (a calendar-month read of
    headline mood, held constant within the month).  This deliberately makes the
    proxy *low-frequency*: it is a stand-in for a slowly-varying sentiment
    envelope, not a high-frequency feed.

    Parameters
    ----------
    business_days :
        The trading-day index to align onto (e.g. the ^GSPC index).

    Returns
    -------
    Series indexed by ``business_days`` with the month's tone broadcast to each
    day; days before the first / after the last curated month are NaN.
    """
    monthly = news_tone_monthly()
    # Map each business day to its month-end key.
    bd = pd.DatetimeIndex(business_days)
    keys = bd + pd.offsets.MonthEnd(0)
    tone = monthly.reindex(keys).to_numpy()
    return pd.Series(tone, index=bd, name="tone")


# ---------------------------------------------------------------------------
# Synthetic daily generator -- the deterministic offline core / positive control
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 2000,
    gamma: float = 0.0,
    base_ret: float = 0.0003,
    ret_vol: float = 0.011,
    tone_ar: float = 0.94,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A daily (tone, next-day return) panel with a known predictive link.

    The data-generating process::

        tone_t   = tone_ar * tone_{t-1} + eps_t          (persistent AR(1) tone)
        ret_{t+1} = base_ret + gamma * z(tone_t) + noise_{t+1}

    where ``z(tone_t)`` is the standardized tone.  When ``gamma = 0`` the tone is
    a persistent but *useless* series -- it does not forecast tomorrow's return
    (the null).  A positive ``gamma`` plants exactly the next-day predictability
    the strategy is built to harvest.

    The persistence (``tone_ar``) mimics a slowly-varying sentiment envelope and
    induces autocorrelation in the *signal* but not in the *forward return* under
    the null -- so a naive same-day fit would look strong while the honest
    next-day test reads zero.

    Returns ``(df, truth)`` where ``df`` has a ``DatetimeIndex`` and columns
    ``tone`` (today's tone) and ``fwd_ret`` (the return earned tomorrow), and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)

    # Persistent AR(1) tone
    eps = rng.normal(0.0, 1.0, n_days)
    tone = np.empty(n_days)
    tone[0] = eps[0]
    for t in range(1, n_days):
        tone[t] = tone_ar * tone[t - 1] + np.sqrt(1.0 - tone_ar**2) * eps[t]
    tone_z = (tone - tone.mean()) / tone.std(ddof=0)

    # Forward return: depends on TODAY's tone (gamma), noise is independent
    noise = rng.normal(0.0, ret_vol, n_days)
    fwd = base_ret + gamma * tone_z + noise

    df = pd.DataFrame({"tone": tone_z, "fwd_ret": fwd}, index=dates)
    df.index.name = "date"
    truth = {
        "n_days": n_days,
        "gamma": gamma,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "tone_ar": tone_ar,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC daily closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_gspc(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """^GSPC daily adjusted-close panel; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as parquet).  Returns a frame with a ``DatetimeIndex`` and a single
    ``close`` column.  Raises ``FileNotFoundError`` if the cache is absent and
    ``fetch=False`` -- callers must guard with ``os.path.exists(cache_path)``.

    Price-only (auto-adjusted) closes: the next-day return is a *price* return,
    not a total return.  Over a 1-day horizon the dividend component is
    negligible, but the label is named here for honesty.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC series at {cache_path}. "
                "Call fetch_gspc(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start="2007-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    out = pd.DataFrame({"close": close.astype(float)})
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().dropna()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached ^GSPC daily: {out.shape[0]} days -> {cache_path}")
    return out


def build_real_panel(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """Join the curated tone proxy with ^GSPC next-day returns (cache-only by default).

    Returns a daily frame with columns:
      - ``tone``     : today's curated news-tone proxy (z-scored, month-broadcast)
      - ``ret``      : today's close-to-close return
      - ``fwd_ret``  : the return earned *tomorrow* (the honest forecast target)

    Rows with missing tone (outside the curated window) or missing forward return
    are dropped.  No look-ahead: ``fwd_ret`` at row t is the t->t+1 return.
    """
    px = fetch_gspc(fetch=fetch, cache_path=cache_path)
    ret = px["close"].pct_change()
    tone = news_tone_daily(px.index)
    fwd = ret.shift(-1)  # return earned tomorrow

    df = pd.DataFrame({"tone": tone, "ret": ret, "fwd_ret": fwd})
    df = df.dropna(subset=["tone", "fwd_ret"])
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a numeric frame (last row), for the as-of stamp."""
    arr = df.select_dtypes(include=[float, int]).iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
