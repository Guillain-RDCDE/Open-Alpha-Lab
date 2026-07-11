"""Data layer for Study 708 — Eurovision-Effect.

The claim under test: winning (or hosting) the Eurovision Song Contest gives a
country's stock market a "feel-good" bump — the folklore Edmans-style national-pride
effect (real for football World Cup elimination shocks; here applied to a much smaller,
much sillier stage.

Three ingredients:

* **The calendar, hardcoded.** Every Eurovision Grand Final 2000->2025 (2020 cancelled,
  COVID), its winning country, and its host country (= the PRIOR year's winner, except
  2021 which is the rolled-over 2020 host, and 2023 where the United Kingdom hosted "on
  behalf of" 2022 winner Ukraine because of the war — both named quirks). Source:
  Wikipedia "List of Eurovision Song Contest winners" / "...host cities" (cross-checked
  against the individual per-year contest pages for the exact Saturday final date).
* **A country -> single-country ETF map.** Only a minority of winner/host countries have
  ever had a US-listed single-country ETF (``COUNTRY_ETF``); this is itself part of the
  honest story (Ukraine has never had one; Russia's ERUS was effectively wiped from the
  data record after the 2022 sanctions; small Baltics/Balkans/Caucasus states never had
  one at all) — a country that wins the contest but has no tradable equity vehicle is a
  finding, not a gap to paper over.
* **A Europe benchmark**, VGK (Vanguard FTSE Europe — spans euro AND non-euro Europe:
  UK, Switzerland, Sweden, Norway, Denmark are all in our event list and NOT in the
  euro-only FEZ, which is why VGK and not FEZ is the study's benchmark). VGK's inception
  (2005-03-10) is itself a hard floor on how far back the test can reach.
* **Synthetic world.** A deterministic, seeded pair of (asset, benchmark) log-return
  series with a TUNABLE planted "national-pride bump" on a synthetic event calendar.
  ``bump = 0`` is the null world; the one-sample-t machinery must not manufacture
  significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)
EUROPE_BENCHMARK = "VGK"    # Vanguard FTSE Europe -- spans euro AND non-euro Europe

# --------------------------------------------------------------------------- #
# The Eurovision calendar, hardcoded: year, Grand Final date (Saturday),
# winning country, host country. 2020 is COVID-cancelled (no final, no winner).
# Host country = the PRIOR year's winner, with two named quirks:
#   * 2021 hosted by the Netherlands again (rolled over from the cancelled 2020 edition,
#     which the Netherlands had earned by winning 2019).
#   * 2023 hosted by the United Kingdom "on behalf of" 2022 winner Ukraine, which could
#     not host during the war -- the market tested for that year's "host effect" is the
#     UK's (the actual venue/market), not Ukraine's (which has no tradable ETF anyway).
# Source: Wikipedia "List of Eurovision Song Contest winners" and "...host cities",
# cross-checked against each year's Eurovision Song Contest <YEAR> page for the exact
# Grand Final date.
# --------------------------------------------------------------------------- #
EVENTS = [
    # year, final_date,   winner_country,    host_country
    (2000, "2000-05-13", "Denmark",          "Sweden"),
    (2001, "2001-05-12", "Estonia",          "Denmark"),
    (2002, "2002-05-25", "Latvia",           "Estonia"),
    (2003, "2003-05-24", "Turkey",           "Latvia"),
    (2004, "2004-05-15", "Ukraine",          "Turkey"),
    (2005, "2005-05-21", "Greece",           "Ukraine"),
    (2006, "2006-05-20", "Finland",          "Greece"),
    (2007, "2007-05-12", "Serbia",           "Finland"),
    (2008, "2008-05-24", "Russia",           "Serbia"),
    (2009, "2009-05-16", "Norway",           "Russia"),
    (2010, "2010-05-29", "Germany",          "Norway"),
    (2011, "2011-05-14", "Azerbaijan",       "Germany"),
    (2012, "2012-05-26", "Sweden",           "Azerbaijan"),
    (2013, "2013-05-18", "Denmark",          "Sweden"),
    (2014, "2014-05-10", "Austria",          "Denmark"),
    (2015, "2015-05-23", "Sweden",           "Austria"),
    (2016, "2016-05-14", "Ukraine",          "Sweden"),
    (2017, "2017-05-13", "Portugal",         "Ukraine"),
    (2018, "2018-05-12", "Israel",           "Portugal"),
    (2019, "2019-05-18", "Netherlands",      "Israel"),
    (2020, None,          None,               "Netherlands"),   # cancelled, COVID-19
    (2021, "2021-05-22", "Italy",            "Netherlands"),
    (2022, "2022-05-14", "Ukraine",          "Italy"),
    (2023, "2023-05-13", "Sweden",           "United Kingdom"),  # UK hosted for Ukraine
    (2024, "2024-05-11", "Switzerland",      "Sweden"),
    (2025, "2025-05-17", "Austria",          "Switzerland"),
]

# --------------------------------------------------------------------------- #
# Country -> single-country US-listed ETF ticker (None = no tradable vehicle exists).
# Inception dates are NOT hardcoded here -- they fall out of the actual cached price
# history, so a country whose ETF postdates its Eurovision moment is excluded by the
# data itself, not by a judgment call.
# --------------------------------------------------------------------------- #
COUNTRY_ETF: dict[str, str | None] = {
    "Denmark":        "EDEN",   # Global X MSCI Denmark, inception 2012-01
    "Estonia":        None,     # no US-listed Estonia ETF, ever
    "Latvia":         None,     # no US-listed Latvia ETF, ever
    "Turkey":         "TUR",    # iShares MSCI Turkey, inception 2008-03
    "Ukraine":        None,     # no US-listed Ukraine ETF, ever
    "Greece":         "GREK",   # Global X MSCI Greece, inception 2011-12
    "Finland":        "EFNL",   # iShares MSCI Finland Capped, inception 2012-01
    "Serbia":         None,     # no US-listed Serbia ETF, ever
    "Russia":         None,     # iShares MSCI Russia (ERUS) delisted / no data post-2022 sanctions
    "Norway":         "NORW",   # Global X MSCI Norway, inception 2009-08
    "Germany":        "EWG",    # iShares MSCI Germany, inception 1996-03
    "Azerbaijan":     None,     # no US-listed Azerbaijan ETF, ever
    "Sweden":         "EWD",    # iShares MSCI Sweden, inception 1996-03
    "Austria":        "EWO",    # iShares MSCI Austria, inception 1996-03
    "Portugal":       "PGAL",   # Global X FTSE Portugal 20, inception 2013-11, delisted 2024-03
    "Israel":         "EIS",    # iShares MSCI Israel, inception 2008-03
    "Netherlands":    "EWN",    # iShares MSCI Netherlands, inception 1996-03
    "Italy":          "EWI",    # iShares MSCI Italy, inception 1996-03
    "United Kingdom": "EWU",    # iShares MSCI United Kingdom, inception 1996-03
    "Switzerland":    "EWL",    # iShares MSCI Switzerland, inception 1996-03
}


def all_tickers() -> list[str]:
    """Every distinct ticker this study ever needs: the ETF map plus the benchmark."""
    ts = sorted({t for t in COUNTRY_ETF.values() if t is not None})
    return ts + [EUROPE_BENCHMARK]


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"eurovision_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1996-01-01", end: str = "2026-07-01") -> None:
    """Download adjusted (total-return) daily closes for every ticker; cache them.

    ``auto_adjust=True`` -- these are equity ETFs, so total-return (dividends
    reinvested) is the honest comparison, unlike the VIX/SPY-range studies where the
    index has no adjustment concept.
    """
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in all_tickers():
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Close"]].dropna()
        d.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in all_tickers())


def load_real(asof: str = AS_OF) -> dict[str, pd.Series]:
    """Cached {ticker: adjusted-close Series}, each sliced to <= asof."""
    out = {}
    for t in all_tickers():
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        s = df["Close"]
        out[t] = s[s.index <= pd.Timestamp(asof)]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world -- planted national-pride bump (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 708, n_events: int = 20,
                    n_days: int = 6000, spacing: int = 260,
                    ) -> tuple[pd.Series, pd.Series, list[int]]:
    """Deterministic paired (asset, benchmark) log-return world with a planted bump.

    Both series are correlated (rho ~ 0.75, like a country ETF vs a regional
    benchmark) zero-mean noise; on a synthetic "event day" (every ``spacing``-th
    business day) the asset gets an EXTRA ``bump`` log-return (in addition to its
    normal correlated draw) while the benchmark does not. ``bump = 0`` is the null
    world -- event days statistically identical to the rest.

    Business-day integer index (positions 0..n_days), far below the 250-year
    ns-timestamp trap (no calendar dates are generated at all here).
    Returns (asset_logret, bench_logret, event_positions).
    """
    rng = np.random.default_rng(seed)
    rho = 0.75
    common = rng.normal(0.0, 0.010, n_days)
    idio_a = rng.normal(0.0, 0.010, n_days)
    idio_b = rng.normal(0.0, 0.010, n_days)
    a = rho * common + np.sqrt(1 - rho**2) * idio_a
    b = rho * common + np.sqrt(1 - rho**2) * idio_b

    event_pos = list(range(spacing, n_days - 30, spacing))[:n_events]
    for p in event_pos:
        a[p] += bump

    idx = pd.RangeIndex(n_days)
    return pd.Series(a, index=idx), pd.Series(b, index=idx), event_pos
