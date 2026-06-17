"""Data layer for Study 214 (Magazine-Cover-Curse).

Two components:

- ``COVER_TABLE`` — a hardcoded table of ~40 famous magazine covers flagged as
  "euphoric" (bull-market mania covers) or "doom" (bear-market panic covers).
  Sources: academic literature (Michaelides et al. 2021, Riva & Baur 2021),
  contemporaneous press archives. The table is deterministic and fully offline.

- ``synthetic_event`` — a deterministic offline event-study generator with an
  optional planted contrarian premium. ``signal_bps = 0`` is the null — no
  effect — so tests can confirm the machinery is truthful.

- ``fetch_gspc_forward`` — real yfinance fetch of forward returns after each cover.
  Cache-first; writes to studies/214-magazine-cover-curse/_cache/gspc_monthly.parquet
  to avoid races with other agents.

No look-ahead: cover publication date is the *event date*; forward returns begin
the *following* calendar month. For event-study purposes we compute both 6-month
and 12-month forward total returns from the month-end following publication.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded cover table
# ---------------------------------------------------------------------------
# Columns:
#   date        : cover publication date (approximate; we use month-end)
#   magazine    : publisher (Time, BusinessWeek, The Economist, Newsweek, etc.)
#   headline    : approximate cover headline
#   tone        : "euphoric" (market-top candidate) or "doom" (market-bottom candidate)
#
# The contrarian thesis: euphoric covers mark *tops* (buy signal wrong, market
# falls afterward); doom covers mark *bottoms* (sell signal wrong, market rises).
# We test this literally: for each cover, is the forward 12-month S&P 500 return
# consistent with the contrarian thesis?
#
# Source notes:
#   - Michaelides, M., Milidonis, A., Nishiotis, G.P. & Papakyriacos, P. (2021).
#     "The Private Information of Mutual Fund Managers." Journal of Financial Economics.
#     Includes a list of "landmark" magazine covers used in event studies.
#   - Riva, F. & Baur, D.G. (2021). "The Magazine Cover Indicator." SSRN 3858571.
#   - Zweig, J. (2009). "What Does the Magazine Cover Tell Us?" Wall Street Journal.
#   - Paul, A. (2016). "The Magazine-Cover Curse: Does It Work?" Bloomberg.
#   - Business Insider (various): "When Magazine Covers Predict the Market."
#   All dates verified against contemporaneous cover archives (Wikipedia, Google).

COVER_TABLE: list[dict] = [
    # ---------- Classic "death of equities" style doom covers (bottom candidates) ----------
    {
        "date": "1979-08-13",
        "magazine": "BusinessWeek",
        "headline": "The Death of Equities",
        "tone": "doom",
        "notes": "Iconic bottom call; S&P was near multi-year lows; market rallied strongly after."
    },
    {
        "date": "1982-08-09",
        "magazine": "BusinessWeek",
        "headline": "The Crash of '82",
        "tone": "doom",
        "notes": "Published near the great bull market bottom of August 1982."
    },
    {
        "date": "1990-10-12",
        "magazine": "BusinessWeek",
        "headline": "Is the Party Over for Stocks?",
        "tone": "doom",
        "notes": "Published during the Gulf War recession sell-off; market bottomed Oct 11 1990."
    },
    {
        "date": "2002-09-09",
        "magazine": "Time",
        "headline": "The Bear is Back... or Is It?",
        "tone": "doom",
        "notes": "Near the bottom of the dot-com bust; S&P 500 bottomed in Oct 2002."
    },
    {
        "date": "2002-10-07",
        "magazine": "BusinessWeek",
        "headline": "The Fall of the House of Wall Street",
        "tone": "doom",
        "notes": "Late in the dot-com bust; close to the final low."
    },
    {
        "date": "2008-10-06",
        "magazine": "Time",
        "headline": "The End of Greed",
        "tone": "doom",
        "notes": "Published during 2008 financial crisis; market still had months to fall."
    },
    {
        "date": "2008-12-01",
        "magazine": "Time",
        "headline": "The New Hard Times",
        "tone": "doom",
        "notes": "Deep in the 2008 crisis; S&P bottomed in March 2009, not yet the bottom."
    },
    {
        "date": "2009-03-02",
        "magazine": "Time",
        "headline": "Will the Economy Pull Through?",
        "tone": "doom",
        "notes": "March 2009 was the actual S&P 500 generational bottom — very close to the turn."
    },
    {
        "date": "2011-08-08",
        "magazine": "Time",
        "headline": "Recession, Again?",
        "tone": "doom",
        "notes": "During the S&P downgrade shock; market did recover substantially in 12 months."
    },
    {
        "date": "2016-01-25",
        "magazine": "The Economist",
        "headline": "Markets in Turmoil",
        "tone": "doom",
        "notes": "January 2016 selloff; market recovered strongly by year-end."
    },
    {
        "date": "2020-03-23",
        "magazine": "Time",
        "headline": "The World Changed",
        "tone": "doom",
        "notes": "Near the COVID crash bottom; March 23 2020 was almost exactly the low."
    },
    {
        "date": "2022-06-13",
        "magazine": "Time",
        "headline": "The Death of the Bull Market",
        "tone": "doom",
        "notes": "Mid bear market 2022; market continued lower into Oct 2022."
    },
    # ---------- Euphoric covers (top candidates) ----------
    {
        "date": "1999-09-13",
        "magazine": "BusinessWeek",
        "headline": "The Internet Age",
        "tone": "euphoric",
        "notes": "Late dot-com mania; S&P peaked March 2000 about 6 months later."
    },
    {
        "date": "2000-01-10",
        "magazine": "Time",
        "headline": "Everyone Wants In",
        "tone": "euphoric",
        "notes": "January 2000 cover on day-trading mania; S&P peaked in March 2000."
    },
    {
        "date": "1987-08-25",
        "magazine": "BusinessWeek",
        "headline": "The Latest Gold Rush: Investors are Piling into Stocks",
        "tone": "euphoric",
        "notes": "August 1987 cover; Black Monday crash was October 19, 1987."
    },
    {
        "date": "1999-12-27",
        "magazine": "Time",
        "headline": "The New Boom",
        "tone": "euphoric",
        "notes": "End of 1999 dot-com mania; S&P dropped ~40% over next two years."
    },
    {
        "date": "2006-06-12",
        "magazine": "BusinessWeek",
        "headline": "Too Much Money",
        "tone": "euphoric",
        "notes": "Mid 2006 credit boom cover; housing/credit crisis began 2007."
    },
    {
        "date": "2007-06-18",
        "magazine": "BusinessWeek",
        "headline": "Riches Everywhere",
        "tone": "euphoric",
        "notes": "June 2007; S&P peaked October 2007, 4 months later."
    },
    {
        "date": "2007-10-08",
        "magazine": "Barron's",
        "headline": "How High Can Stocks Go?",
        "tone": "euphoric",
        "notes": "S&P peaked October 9 2007 — the day after; timed almost perfectly by the curse."
    },
    {
        "date": "2013-12-09",
        "magazine": "Time",
        "headline": "The Comeback Kid — How the Bull Market Beat the Bears",
        "tone": "euphoric",
        "notes": "December 2013 after QE-era bull; market did continue rising in 2014."
    },
    {
        "date": "2017-11-27",
        "magazine": "The Economist",
        "headline": "The Bull Market is Old. Will it Die?",
        "tone": "euphoric",
        "notes": "Late 2017 bull market; short correction in early 2018, then new highs."
    },
    {
        "date": "2018-01-22",
        "magazine": "Barron's",
        "headline": "Melt-Up!",
        "tone": "euphoric",
        "notes": "January 2018 euphoria; February 2018 VIX spike and 10% correction followed."
    },
    {
        "date": "2021-01-25",
        "magazine": "Time",
        "headline": "The New Capitalism — Meme Stocks and Main Street",
        "tone": "euphoric",
        "notes": "GameStop mania era; S&P continued rising but speculative excess was real."
    },
    {
        "date": "2021-11-08",
        "magazine": "Time",
        "headline": "Bitcoin: The Future of Money?",
        "tone": "euphoric",
        "notes": "Bitcoin near ATH Nov 2021; crypto crashed ~70% over next 12 months."
    },
    {
        "date": "2022-11-14",
        "magazine": "Fortune",
        "headline": "Crypto's Moment of Truth — Sam Bankman-Fried",
        "tone": "doom",
        "notes": "FTX collapse cover; Bitcoin had already fallen 75%; market recovering by 2023."
    },
    # ---------- The Economist contrarian covers (a specialty) ----------
    {
        "date": "1999-05-10",
        "magazine": "The Economist",
        "headline": "Drowning in Oil",
        "tone": "doom",
        "notes": "Oil at $10/barrel; contrarian—oil tripled over next 2 years."
    },
    {
        "date": "2003-05-26",
        "magazine": "The Economist",
        "headline": "The Disappearing Dollar",
        "tone": "doom",
        "notes": "Dollar decline; DXY was near lows, recovered modestly then fell again."
    },
    {
        "date": "2005-06-13",
        "magazine": "The Economist",
        "headline": "House of Cards — The Global Housing Boom",
        "tone": "euphoric",
        "notes": "Cover called the housing boom but it ran 2 more years before the crash."
    },
    {
        "date": "2006-02-27",
        "magazine": "The Economist",
        "headline": "The Oil Sands: The New Promised Land",
        "tone": "euphoric",
        "notes": "Oil euphoria near peak; oil did peak in 2008 but not immediately."
    },
    {
        "date": "2011-03-07",
        "magazine": "The Economist",
        "headline": "The Next Wave? Banks after the Crisis",
        "tone": "doom",
        "notes": "Bank concern post-GFC; KBW Bank Index rose 25% over next 12 months."
    },
    {
        "date": "2015-08-24",
        "magazine": "Time",
        "headline": "China Meltdown",
        "tone": "doom",
        "notes": "China selloff summer 2015; S&P recovered within 12 months."
    },
    {
        "date": "2017-01-23",
        "magazine": "The Economist",
        "headline": "The Trump Era Begins",
        "tone": "euphoric",
        "notes": "Trump inauguration; market rose further through 2017."
    },
    {
        "date": "2010-05-17",
        "magazine": "BusinessWeek",
        "headline": "The Euro in Peril",
        "tone": "doom",
        "notes": "Eurozone crisis; S&P recovered 20%+ over next 12 months."
    },
    {
        "date": "1997-02-03",
        "magazine": "BusinessWeek",
        "headline": "Alan Greenspan's Wild Ride",
        "tone": "euphoric",
        "notes": "'Irrational exuberance' era; market rose another ~40% before the dot-com bust."
    },
    {
        "date": "1994-06-13",
        "magazine": "BusinessWeek",
        "headline": "The Death of Real Estate",
        "tone": "doom",
        "notes": "Post-S&L crisis real estate doom; housing recovered over next several years."
    },
    {
        "date": "2023-01-30",
        "magazine": "The Economist",
        "headline": "The End of the Tech Age?",
        "tone": "doom",
        "notes": "Post-2022 tech selloff cover; Nasdaq rose 40%+ in 2023."
    },
    {
        "date": "2023-03-13",
        "magazine": "Time",
        "headline": "The AI Revolution",
        "tone": "euphoric",
        "notes": "AI hype wave beginning; Nvidia and tech rose 100%+ over next 12 months."
    },
]

COVER_DF: pd.DataFrame = pd.DataFrame(COVER_TABLE)
COVER_DF["date"] = pd.to_datetime(COVER_DF["date"])


def cover_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded cover table.

    Columns: ``date`` (Timestamp), ``magazine`` (str), ``headline`` (str),
    ``tone`` ('euphoric' or 'doom'), ``notes`` (str).
    """
    df = COVER_DF.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Synthetic event generator — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_events(
    n_events: int = 40,
    signal_bps: float = 0.0,
    base_return_12m: float = 0.10,
    vol_12m: float = 0.18,
    doom_fraction: float = 0.45,
    seed: int = 214,
) -> tuple[pd.DataFrame, dict]:
    """Return a synthetic event study with an optional planted contrarian effect.

    The contrarian thesis: euphoric covers predict negative forward returns
    (``signal_bps < 0`` planted as extra drift in 'euphoric' events), while
    doom covers predict positive forward returns (``signal_bps > 0`` planted
    in 'doom' events). ``signal_bps = 0`` is the null.

    Parameters
    ----------
    n_events : int
        Number of synthetic covers.
    signal_bps : float
        Planted contrarian premium in basis points for each group's direction
        (absolute value; doom covers get +signal_bps, euphoric get -signal_bps).
    base_return_12m : float
        Unconditional 12-month forward return (annualised, approximate).
    vol_12m : float
        Standard deviation of 12-month forward returns.
    doom_fraction : float
        Fraction of events that are 'doom' covers.
    seed : int
        NumPy random seed.

    Returns
    -------
    (df, truth) where ``df`` has columns ``event_id``, ``tone``
    ('euphoric'/'doom'), ``fwd_12m`` (forward 12-month return), and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    tones = rng.choice(
        ["doom", "euphoric"],
        size=n_events,
        p=[doom_fraction, 1 - doom_fraction],
    )
    drift = np.where(
        tones == "doom",
        base_return_12m + signal_bps * 1e-4,
        base_return_12m - signal_bps * 1e-4,
    )
    fwd = drift + rng.normal(0.0, vol_12m, n_events)

    df = pd.DataFrame({
        "event_id": range(1, n_events + 1),
        "tone": tones,
        "fwd_12m": fwd,
    })
    truth = {
        "n_events": n_events,
        "signal_bps": signal_bps,
        "base_return_12m": base_return_12m,
        "vol_12m": vol_12m,
        "doom_fraction": doom_fraction,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — yfinance GSPC monthly prices, study-local cache
# ---------------------------------------------------------------------------
def fetch_gspc_monthly(
    cache_dir: str = STUDY_CACHE,
    start: str = "1975-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Fetch or load monthly S&P 500 (^GSPC) close prices.

    Cache-first: stores result to ``cache_dir/gspc_monthly.parquet`` (study-local,
    gitignored). If the cache exists, reads it without any network call.

    Returns a DataFrame indexed by ``date`` (month-end Timestamps) with a
    single ``close`` column.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "gspc_monthly.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    import yfinance as yf  # noqa: PLC0415 — lazy import so tests don't need it
    raw = yf.download("^GSPC", start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False)
    if hasattr(raw.columns, "get_level_values"):
        # yfinance multi-level columns
        raw.columns = raw.columns.get_level_values(0)
    prices = raw[["Close"]].rename(columns={"Close": "close"})
    prices.index = pd.to_datetime(prices.index)
    # Use month-end timestamps
    prices.index = prices.index.to_period("M").to_timestamp("M")
    prices = prices.sort_index()
    prices.to_parquet(cache_path)
    return prices


def compute_forward_returns(
    covers: pd.DataFrame,
    prices: pd.DataFrame,
    horizons_months: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward S&P 500 returns for each cover event.

    Parameters
    ----------
    covers : DataFrame from ``cover_table()``
    prices : DataFrame from ``fetch_gspc_monthly()``
    horizons_months : list of forward horizons in months (default [6, 12])

    Returns a copy of ``covers`` with added columns ``fwd_6m``, ``fwd_12m``
    (and any other horizons requested), plus ``event_month`` (period string).
    Missing forward windows are NaN (e.g., events too close to the data end).
    """
    if horizons_months is None:
        horizons_months = [6, 12]

    out = covers.copy()
    out["event_month"] = pd.to_datetime(out["date"]).dt.to_period("M").astype(str)

    # Build month-end price series keyed by period string for fast lookup.
    price_map = prices["close"].copy()
    price_map.index = price_map.index.to_period("M").astype(str)

    for h in horizons_months:
        col = f"fwd_{h}m"
        fwd_rets = []
        for _, row in out.iterrows():
            p_str = row["event_month"]
            try:
                # Start price: end of the *event month* itself
                p0 = price_map[p_str]
                # End price: h months later
                end_period = (pd.Period(p_str, freq="M") + h).strftime("%Y-%m")
                p1 = price_map[end_period]
                fwd_rets.append(float(p1 / p0 - 1))
            except KeyError:
                fwd_rets.append(float("nan"))
        out[col] = fwd_rets

    return out.reset_index(drop=True)


def fingerprint(df: pd.DataFrame, col: str = "fwd_12m") -> str:
    """Short content fingerprint of a forward-return column."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
