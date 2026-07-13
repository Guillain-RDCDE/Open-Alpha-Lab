"""Data for the guacamole-bowl study — a cited avocado-price seasonal, a cached equity tape, and a synthetic world.

The folklore: America eats a mountain of guacamole for the Super Bowl (early February), so the
"avocado / produce trade" should carry a January–February seasonal — buy ahead of the big game,
ride the surge. We test the strongest tradable version of that on three offline-friendly sources:

  * :func:`load_avocado_seasonal` — a **hardcoded, cited, APPROXIMATE** 12-month seasonal-price index
    for wholesale Hass avocados (base 100 ≈ annual average), reconstructed from public USDA AMS Market
    News terminal-market averages and Hass Avocado Board volume seasonality. A **labelled proxy** used
    only to show the *shape* of avocado seasonality — it NEVER backs a Signal stamp. Its own shape is
    already awkward for the folklore: winter (Jan–Feb) is the *soft* part of the price year (peak
    Mexican Hass supply), and the price peak is the late-summer supply gap, not the Super Bowl.

  * :func:`fetch_data` — monthly returns for the tradable leg (``PEP`` — PepsiCo, whose Frito-Lay arm
    is the Super-Bowl chip-and-dip complex: Tostitos, Fritos and the branded dips guacamole rides on),
    the benchmark (``SPY``) and the 13-week T-bill cash leg (``^IRX``), **cache-first**, built from
    *daily* closes resampled to month-end. The pure-play avocado equity (Calavo Growers, ``CVGW``) is
    the *intended* proxy, but its Yahoo daily history is currently truncated to a single bar
    (a documented feed outage — see ``docs/references.md``), so we test the strongest **available**
    tradable expression of the trade, labelled a proxy throughout. Fingerprinted run in ``docs/results.md``.

  * :func:`synthetic_world` — **offline, deterministic**. Monthly equity returns with a tunable
    ``jan_feb_premium`` injected into the Jan–Feb window. ``jan_feb_premium = 0`` is the null. Pins the
    seasonality machinery; can never back a Signal stamp (METHODOLOGY → the inference bar).

Cache lives in the study-local ``_cache/`` directory (gitignored) to avoid races with other agents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# Tradable leg + benchmark + cash leg. PEP/SPY are LABELLED PROXIES for the "Super Bowl guac trade".
PEP, SPY, TBILL = "PEP", "SPY", "^IRX"
TICKERS = [PEP, SPY, TBILL]
MONTHS_PER_YEAR = 12

# The "guacamole surge" window the folklore points at: January build-up + February game month.
# The Super Bowl is the first Sunday of February; guac/avocado demand ramps through late January.
GUAC_MONTHS = [1, 2]


# --------------------------------------------------------------------------- #
# The avocado seasonal — hardcoded, cited, APPROXIMATE (a proxy, not a live feed)
# --------------------------------------------------------------------------- #
# Monthly *wholesale Hass avocado price* seasonal index, base 100 ≈ annual average, reconstructed
# from public USDA AMS Market News terminal-market monthly averages and Hass Avocado Board volume
# seasonality (see docs/references.md). The SHAPE is the load-bearing fact, and it is already awkward
# for the "Super Bowl surge" story: Jan–Feb is a *soft* stretch (heavy Mexican Hass supply lands in
# winter and is pre-positioned for the game), while the true price peak is the late-summer supply gap
# between the Mexican and Californian crops. The Super Bowl is a huge *volume* event, not a *price* one.
_AVOCADO_SEASONAL = {
    1: 95.0,   # Jan — heavy Mexican supply, soft price despite the demand build-up
    2: 97.0,   # Feb — Super Bowl volume spike, but well-supplied → price barely moves
    3: 100.0,
    4: 103.0,
    5: 102.0,  # a Cinco de Mayo volume bump, again well-supplied
    6: 105.0,
    7: 110.0,
    8: 114.0,  # late-summer supply gap → the real annual price peak
    9: 111.0,
    10: 103.0,
    11: 93.0,
    12: 90.0,  # new-crop pressure, the annual price trough
}


def load_avocado_seasonal() -> pd.Series:
    """The (approximate, cited) 12-month wholesale-Hass seasonal price index, base 100 ≈ annual mean.

    Returns a ``pd.Series`` indexed 1..12 (calendar month). LABELLED A PROXY: reconstructed from public
    USDA AMS / Hass Avocado Board reporting, not a live data feed, and used only to show the *shape* of
    avocado seasonality — never to certify a Signal stamp.
    """
    return pd.Series(_AVOCADO_SEASONAL, name="avocado_seasonal").rename_axis("month")


def avocado_window_vs_year(window: list[int] = GUAC_MONTHS) -> dict:
    """Where the guac window sits in the avocado price year: window mean vs annual mean (index points).

    A *positive* gap would support a Super-Bowl price surge; a *negative* gap undercuts it before we
    even look at the tape. Returns ``window_mean``, ``year_mean``, ``gap``.
    """
    s = load_avocado_seasonal()
    wm = float(s[s.index.isin(window)].mean())
    ym = float(s.mean())
    return {"window_mean": wm, "year_mean": ym, "gap": wm - ym}


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorldTruth:
    jan_feb_premium: float

    @property
    def has_seasonality(self) -> bool:
        return self.jan_feb_premium != 0.0


def synthetic_world(
    n_years: int = 30,
    jan_feb_premium: float = 0.03,
    equity_vol: float = 0.22,
    seed: int = 723,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly equity world — deterministic given ``seed``.

    Monthly returns are i.i.d. with annual vol ``equity_vol`` plus a ``jan_feb_premium`` added to the
    Jan–Feb window and spread symmetrically as a small discount across the other ten months (so the
    annual mean is unchanged). ``jan_feb_premium = 0`` is the null. Returns a frame with columns
    ``pep`` (the planted asset), ``spy`` (a plain market with no seasonal), ``tbill``.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1996-01-31", periods=n, freq="ME", name="date")
    months = idx.month

    base = (equity_vol / np.sqrt(12)) * rng.standard_normal(n)
    in_win = np.isin(months, GUAC_MONTHS)
    # add premium to the 2 window months; refund it across the other 10 so E[year] is unchanged
    seasonal = np.where(in_win, jan_feb_premium, -jan_feb_premium * len(GUAC_MONTHS) / (12 - len(GUAC_MONTHS)))
    pep = base + seasonal

    spy = (0.16 / np.sqrt(12)) * rng.standard_normal(n) + 0.10 / 12  # plain market, no seasonal
    tbill = pd.Series(0.02 / 12, index=idx)  # flat 2%/yr cash leg
    return (
        pd.DataFrame({"pep": pep, "spy": spy, "tbill": tbill.values}, index=idx),
        WorldTruth(jan_feb_premium),
    )


# --------------------------------------------------------------------------- #
# The tradable tape via yfinance (cache-first, offline-friendly)
# --------------------------------------------------------------------------- #
def _assert_continuous_monthly(df: pd.DataFrame) -> None:
    """Fail loudly if the monthly grid has interior holes."""
    months = pd.PeriodIndex(df.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    missing = expected.difference(months)
    if len(missing):
        raise AssertionError(
            f"monthly grid has {len(missing)} interior hole(s) out of {len(expected)} months "
            f"(first few: {list(missing[:5])}) — refetch with fetch_data(fetch=True)"
        )


def fetch_data(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Monthly returns for the tradable leg (PEP), benchmark (SPY) and T-bill (^IRX), cache-first.

    **Cache-only** unless ``fetch=True``. Columns ``pep, spy, tbill`` (empty DataFrame on a cache miss
    with ``fetch=False``). Built from **daily** closes resampled to month-end (last close of each
    calendar month) — not Yahoo's native monthly feed, which has holes. The in-progress month is
    dropped so the last bar is always complete. The window starts once SPY exists (1993). Grid is
    asserted hole-free on every read. PEP/SPY are auto-adjusted (total-return-ish); labelled as such.
    """
    cache = os.path.join(cache_dir, "guacamole_bowl.parquet")
    if os.path.exists(cache):
        out = pd.read_parquet(cache)
        _assert_continuous_monthly(out)
        return out
    if not fetch:
        return pd.DataFrame()
    import yfinance as yf  # lazy

    px = yf.download(TICKERS, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    me = px.resample("ME").last()

    out = pd.DataFrame(
        {
            "pep": me[PEP].pct_change(),
            "spy": me[SPY].pct_change(),
            "tbill": (me[TBILL].ffill() / 100.0) / MONTHS_PER_YEAR,
        }
    ).dropna(subset=["pep", "spy"])
    out = out[out.index >= "1993-02-01"]  # both PEP and SPY live from here

    last_complete = pd.Timestamp.today().to_period("M") - 1
    out = out[out.index.to_period("M") <= last_complete]
    out.index.name = "date"
    _assert_continuous_monthly(out)
    os.makedirs(cache_dir, exist_ok=True)
    out.to_parquet(cache)
    return out
