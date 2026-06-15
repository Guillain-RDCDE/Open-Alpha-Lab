"""Data layer for Study 158 (Super-Bowl).

Two components, both fully offline for the core:

- ``SUPER_BOWL_RESULTS`` — a hardcoded table of every Super Bowl (I through LVIII,
  1967–2026), recording the winning team, their conference, and a canonical
  "original-NFL" flag that captures the Krueger-Kennedy (1990) indicator's
  original spirit. The conference assignment is determined by the team's
  pre-merger league affiliation (all AFL franchises that joined the AFC in 1970
  and NFC = original NFL). Source: Pro Football Reference; Wikipedia; Krueger &
  Kennedy (1990). This table is deterministic, small, and well-known — hardcoded
  here to keep the core fully offline and reproducible.

- ``synthetic_annual`` — a deterministic, offline annual-return generator with an
  optional planted "Super Bowl effect" knob. A ``signal_bps = 0`` setting is the
  null (no effect), so tests can confirm the machinery is truthful before looking
  at real data.

- ``fetch_sp500_annual`` — real S&P 500 calendar-year returns from the Shiller
  dataset (``_cache/shiller_sp500.parquet``), already staged in the repo-level
  cache. Cache-only; no network call needed.

No look-ahead: the Super Bowl is played in late January or early February; we
match it to the *subsequent* calendar-year return (the ``feb_year`` convention used
by Krueger-Kennedy) OR the same calendar year the game is played in — both
conventions are available so the user can see that the label choice is itself a
degrees-of-freedom concern.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

# ---------------------------------------------------------------------------
# The hardcoded Super Bowl table — the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Columns:
#   game_number : Roman-numeral game number (I = 1, ..., LVIII = 58)
#   game_year   : calendar year in which the game was *played*
#                 (Super Bowl I was played January 1967; games through 2006
#                  were played in January, from 2007 onward in February)
#   winner      : winning team name
#   conference  : "NFC" or "AFC" (post-merger conference alignment)
#   orig_nfl    : True = winner was an *original NFL franchise* (pre-1970 merger);
#                 False = AFL/expansion franchise. This is the original
#                 Krueger-Kennedy signal axis.
#
# The Krueger-Kennedy (1990) rule: if an original-NFL team wins → bull year;
# if an AFL team (now AFC) wins → bear year.  Post-expansion teams complicate
# the picture (Dallas Cowboys joined the NFC but were an expansion club;
# Seattle Seahawks and Tampa Bay Buccaneers flip-flopped conferences).
# We follow the most widely cited version: conference (NFC/AFC) as the
# primary signal, orig_nfl as an alternative.
#
# Sources:
#   - Krueger, A. B. & Kennedy, J. M. (1990). "An Analysis of the Super Bowl
#     Stock Market Predictor." The Journal of Finance, 45(2), 691–697.
#   - Pro Football Reference: https://www.pro-football-reference.com/super-bowl/
#   - Wikipedia: List of Super Bowl champions
#
# As-of: 2026-06-15. Super Bowl LVIII (February 2024) won by Kansas City Chiefs (AFC).
# Super Bowl LIX (February 2025) won by Philadelphia Eagles (NFC).

SUPER_BOWL_RESULTS: list[dict] = [
    # game_number, game_year, winner, conference, orig_nfl
    # Super Bowl I: played Jan 15, 1967 — Green Bay Packers (NFL) def. Kansas City Chiefs (AFL)
    {"game_number": 1,  "game_year": 1967, "winner": "Green Bay Packers",       "conference": "NFC", "orig_nfl": True},
    # Super Bowl II: played Jan 14, 1968 — Green Bay Packers (NFL) def. Oakland Raiders (AFL)
    {"game_number": 2,  "game_year": 1968, "winner": "Green Bay Packers",       "conference": "NFC", "orig_nfl": True},
    # Super Bowl III: played Jan 12, 1969 — New York Jets (AFL) def. Baltimore Colts (NFL)
    {"game_number": 3,  "game_year": 1969, "winner": "New York Jets",           "conference": "AFC", "orig_nfl": False},
    # Super Bowl IV: played Jan 11, 1970 — Kansas City Chiefs (AFL) def. Minnesota Vikings (NFL)
    {"game_number": 4,  "game_year": 1970, "winner": "Kansas City Chiefs",      "conference": "AFC", "orig_nfl": False},
    # Super Bowl V: played Jan 17, 1971 — Baltimore Colts (NFL→AFC) def. Dallas Cowboys (NFL/NFC)
    # Note: Colts moved to AFC in 1970 merger despite being an original NFL franchise.
    {"game_number": 5,  "game_year": 1971, "winner": "Baltimore Colts",         "conference": "AFC", "orig_nfl": True},
    # Super Bowl VI: played Jan 16, 1972 — Dallas Cowboys (NFC, expansion '60) def. Miami Dolphins
    {"game_number": 6,  "game_year": 1972, "winner": "Dallas Cowboys",          "conference": "NFC", "orig_nfl": False},
    # Super Bowl VII: played Jan 14, 1973 — Miami Dolphins (AFC, AFL) def. Washington Redskins
    {"game_number": 7,  "game_year": 1973, "winner": "Miami Dolphins",          "conference": "AFC", "orig_nfl": False},
    # Super Bowl VIII: played Jan 13, 1974 — Miami Dolphins (AFC) def. Minnesota Vikings (NFC)
    {"game_number": 8,  "game_year": 1974, "winner": "Miami Dolphins",          "conference": "AFC", "orig_nfl": False},
    # Super Bowl IX: played Jan 12, 1975 — Pittsburgh Steelers (AFC, NFL→AFC) def. Minnesota Vikings
    {"game_number": 9,  "game_year": 1975, "winner": "Pittsburgh Steelers",     "conference": "AFC", "orig_nfl": True},
    # Super Bowl X: played Jan 18, 1976 — Pittsburgh Steelers (AFC) def. Dallas Cowboys (NFC)
    {"game_number": 10, "game_year": 1976, "winner": "Pittsburgh Steelers",     "conference": "AFC", "orig_nfl": True},
    # Super Bowl XI: played Jan 9, 1977 — Oakland Raiders (AFC, AFL) def. Minnesota Vikings (NFC)
    {"game_number": 11, "game_year": 1977, "winner": "Oakland Raiders",         "conference": "AFC", "orig_nfl": False},
    # Super Bowl XII: played Jan 15, 1978 — Dallas Cowboys (NFC) def. Denver Broncos (AFC)
    {"game_number": 12, "game_year": 1978, "winner": "Dallas Cowboys",          "conference": "NFC", "orig_nfl": False},
    # Super Bowl XIII: played Jan 21, 1979 — Pittsburgh Steelers (AFC) def. Dallas Cowboys (NFC)
    {"game_number": 13, "game_year": 1979, "winner": "Pittsburgh Steelers",     "conference": "AFC", "orig_nfl": True},
    # Super Bowl XIV: played Jan 20, 1980 — Pittsburgh Steelers (AFC) def. LA Rams (NFC)
    {"game_number": 14, "game_year": 1980, "winner": "Pittsburgh Steelers",     "conference": "AFC", "orig_nfl": True},
    # Super Bowl XV: played Jan 25, 1981 — Oakland Raiders (AFC) def. Philadelphia Eagles (NFC)
    {"game_number": 15, "game_year": 1981, "winner": "Oakland Raiders",         "conference": "AFC", "orig_nfl": False},
    # Super Bowl XVI: played Jan 24, 1982 — San Francisco 49ers (NFC) def. Cincinnati Bengals (AFC)
    {"game_number": 16, "game_year": 1982, "winner": "San Francisco 49ers",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XVII: played Jan 30, 1983 — Washington Redskins (NFC) def. Miami Dolphins (AFC)
    {"game_number": 17, "game_year": 1983, "winner": "Washington Redskins",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XVIII: played Jan 22, 1984 — LA Raiders (AFC) def. Washington Redskins (NFC)
    {"game_number": 18, "game_year": 1984, "winner": "Los Angeles Raiders",     "conference": "AFC", "orig_nfl": False},
    # Super Bowl XIX: played Jan 20, 1985 — San Francisco 49ers (NFC) def. Miami Dolphins (AFC)
    {"game_number": 19, "game_year": 1985, "winner": "San Francisco 49ers",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XX: played Jan 26, 1986 — Chicago Bears (NFC) def. New England Patriots (AFC)
    {"game_number": 20, "game_year": 1986, "winner": "Chicago Bears",           "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXI: played Jan 25, 1987 — NY Giants (NFC) def. Denver Broncos (AFC)
    {"game_number": 21, "game_year": 1987, "winner": "New York Giants",         "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXII: played Jan 31, 1988 — Washington Redskins (NFC) def. Denver Broncos (AFC)
    {"game_number": 22, "game_year": 1988, "winner": "Washington Redskins",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXIII: played Jan 22, 1989 — San Francisco 49ers (NFC) def. Cincinnati Bengals (AFC)
    {"game_number": 23, "game_year": 1989, "winner": "San Francisco 49ers",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXIV: played Jan 28, 1990 — San Francisco 49ers (NFC) def. Denver Broncos (AFC)
    {"game_number": 24, "game_year": 1990, "winner": "San Francisco 49ers",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXV: played Jan 27, 1991 — NY Giants (NFC) def. Buffalo Bills (AFC)
    {"game_number": 25, "game_year": 1991, "winner": "New York Giants",         "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXVI: played Jan 26, 1992 — Washington Redskins (NFC) def. Buffalo Bills (AFC)
    {"game_number": 26, "game_year": 1992, "winner": "Washington Redskins",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXVII: played Jan 31, 1993 — Dallas Cowboys (NFC) def. Buffalo Bills (AFC)
    {"game_number": 27, "game_year": 1993, "winner": "Dallas Cowboys",          "conference": "NFC", "orig_nfl": False},
    # Super Bowl XXVIII: played Jan 30, 1994 — Dallas Cowboys (NFC) def. Buffalo Bills (AFC)
    {"game_number": 28, "game_year": 1994, "winner": "Dallas Cowboys",          "conference": "NFC", "orig_nfl": False},
    # Super Bowl XXIX: played Jan 29, 1995 — San Francisco 49ers (NFC) def. San Diego Chargers (AFC)
    {"game_number": 29, "game_year": 1995, "winner": "San Francisco 49ers",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXX: played Jan 28, 1996 — Dallas Cowboys (NFC) def. Pittsburgh Steelers (AFC)
    {"game_number": 30, "game_year": 1996, "winner": "Dallas Cowboys",          "conference": "NFC", "orig_nfl": False},
    # Super Bowl XXXI: played Jan 26, 1997 — Green Bay Packers (NFC) def. New England Patriots (AFC)
    {"game_number": 31, "game_year": 1997, "winner": "Green Bay Packers",       "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXXII: played Jan 25, 1998 — Denver Broncos (AFC) def. Green Bay Packers (NFC)
    {"game_number": 32, "game_year": 1998, "winner": "Denver Broncos",          "conference": "AFC", "orig_nfl": False},
    # Super Bowl XXXIII: played Jan 31, 1999 — Denver Broncos (AFC) def. Atlanta Falcons (NFC)
    {"game_number": 33, "game_year": 1999, "winner": "Denver Broncos",          "conference": "AFC", "orig_nfl": False},
    # Super Bowl XXXIV: played Jan 30, 2000 — St. Louis Rams (NFC) def. Tennessee Titans (AFC)
    {"game_number": 34, "game_year": 2000, "winner": "St. Louis Rams",          "conference": "NFC", "orig_nfl": True},
    # Super Bowl XXXV: played Jan 28, 2001 — Baltimore Ravens (AFC) def. NY Giants (NFC)
    {"game_number": 35, "game_year": 2001, "winner": "Baltimore Ravens",        "conference": "AFC", "orig_nfl": True},
    # Super Bowl XXXVI: played Feb 3, 2002 — New England Patriots (AFC) def. St. Louis Rams (NFC)
    {"game_number": 36, "game_year": 2002, "winner": "New England Patriots",    "conference": "AFC", "orig_nfl": False},
    # Super Bowl XXXVII: played Jan 26, 2003 — Tampa Bay Buccaneers (NFC) def. Oakland Raiders (AFC)
    {"game_number": 37, "game_year": 2003, "winner": "Tampa Bay Buccaneers",    "conference": "NFC", "orig_nfl": False},
    # Super Bowl XXXVIII: played Feb 1, 2004 — New England Patriots (AFC) def. Carolina Panthers (NFC)
    {"game_number": 38, "game_year": 2004, "winner": "New England Patriots",    "conference": "AFC", "orig_nfl": False},
    # Super Bowl XXXIX: played Feb 6, 2005 — New England Patriots (AFC) def. Philadelphia Eagles (NFC)
    {"game_number": 39, "game_year": 2005, "winner": "New England Patriots",    "conference": "AFC", "orig_nfl": False},
    # Super Bowl XL: played Feb 5, 2006 — Pittsburgh Steelers (AFC) def. Seattle Seahawks (NFC)
    {"game_number": 40, "game_year": 2006, "winner": "Pittsburgh Steelers",     "conference": "AFC", "orig_nfl": True},
    # Super Bowl XLI: played Feb 4, 2007 — Indianapolis Colts (AFC) def. Chicago Bears (NFC)
    {"game_number": 41, "game_year": 2007, "winner": "Indianapolis Colts",      "conference": "AFC", "orig_nfl": True},
    # Super Bowl XLII: played Feb 3, 2008 — NY Giants (NFC) def. New England Patriots (AFC)
    {"game_number": 42, "game_year": 2008, "winner": "New York Giants",         "conference": "NFC", "orig_nfl": True},
    # Super Bowl XLIII: played Feb 1, 2009 — Pittsburgh Steelers (AFC) def. Arizona Cardinals (NFC)
    {"game_number": 43, "game_year": 2009, "winner": "Pittsburgh Steelers",     "conference": "AFC", "orig_nfl": True},
    # Super Bowl XLIV: played Feb 7, 2010 — New Orleans Saints (NFC) def. Indianapolis Colts (AFC)
    {"game_number": 44, "game_year": 2010, "winner": "New Orleans Saints",      "conference": "NFC", "orig_nfl": True},
    # Super Bowl XLV: played Feb 6, 2011 — Green Bay Packers (NFC) def. Pittsburgh Steelers (AFC)
    {"game_number": 45, "game_year": 2011, "winner": "Green Bay Packers",       "conference": "NFC", "orig_nfl": True},
    # Super Bowl XLVI: played Feb 5, 2012 — NY Giants (NFC) def. New England Patriots (AFC)
    {"game_number": 46, "game_year": 2012, "winner": "New York Giants",         "conference": "NFC", "orig_nfl": True},
    # Super Bowl XLVII: played Feb 3, 2013 — Baltimore Ravens (AFC) def. San Francisco 49ers (NFC)
    {"game_number": 47, "game_year": 2013, "winner": "Baltimore Ravens",        "conference": "AFC", "orig_nfl": True},
    # Super Bowl XLVIII: played Feb 2, 2014 — Seattle Seahawks (NFC) def. Denver Broncos (AFC)
    {"game_number": 48, "game_year": 2014, "winner": "Seattle Seahawks",        "conference": "NFC", "orig_nfl": False},
    # Super Bowl XLIX: played Feb 1, 2015 — New England Patriots (AFC) def. Seattle Seahawks (NFC)
    {"game_number": 49, "game_year": 2015, "winner": "New England Patriots",    "conference": "AFC", "orig_nfl": False},
    # Super Bowl 50: played Feb 7, 2016 — Denver Broncos (AFC) def. Carolina Panthers (NFC)
    {"game_number": 50, "game_year": 2016, "winner": "Denver Broncos",          "conference": "AFC", "orig_nfl": False},
    # Super Bowl LI: played Feb 5, 2017 — New England Patriots (AFC) def. Atlanta Falcons (NFC)
    {"game_number": 51, "game_year": 2017, "winner": "New England Patriots",    "conference": "AFC", "orig_nfl": False},
    # Super Bowl LII: played Feb 4, 2018 — Philadelphia Eagles (NFC) def. New England Patriots (AFC)
    {"game_number": 52, "game_year": 2018, "winner": "Philadelphia Eagles",     "conference": "NFC", "orig_nfl": True},
    # Super Bowl LIII: played Feb 3, 2019 — New England Patriots (AFC) def. LA Rams (NFC)
    {"game_number": 53, "game_year": 2019, "winner": "New England Patriots",    "conference": "AFC", "orig_nfl": False},
    # Super Bowl LIV: played Feb 2, 2020 — Kansas City Chiefs (AFC) def. San Francisco 49ers (NFC)
    {"game_number": 54, "game_year": 2020, "winner": "Kansas City Chiefs",      "conference": "AFC", "orig_nfl": False},
    # Super Bowl LV: played Feb 7, 2021 — Tampa Bay Buccaneers (NFC) def. Kansas City Chiefs (AFC)
    {"game_number": 55, "game_year": 2021, "winner": "Tampa Bay Buccaneers",    "conference": "NFC", "orig_nfl": False},
    # Super Bowl LVI: played Feb 13, 2022 — LA Rams (NFC) def. Cincinnati Bengals (AFC)
    {"game_number": 56, "game_year": 2022, "winner": "Los Angeles Rams",        "conference": "NFC", "orig_nfl": True},
    # Super Bowl LVII: played Feb 12, 2023 — Kansas City Chiefs (AFC) def. Philadelphia Eagles (NFC)
    {"game_number": 57, "game_year": 2023, "winner": "Kansas City Chiefs",      "conference": "AFC", "orig_nfl": False},
    # Super Bowl LVIII: played Feb 11, 2024 — Kansas City Chiefs (AFC) def. San Francisco 49ers (NFC)
    {"game_number": 58, "game_year": 2024, "winner": "Kansas City Chiefs",      "conference": "AFC", "orig_nfl": False},
    # Super Bowl LIX: played Feb 9, 2025 — Philadelphia Eagles (NFC) def. Kansas City Chiefs (AFC)
    {"game_number": 59, "game_year": 2025, "winner": "Philadelphia Eagles",     "conference": "NFC", "orig_nfl": True},
]

SUPER_BOWL_DF: pd.DataFrame = pd.DataFrame(SUPER_BOWL_RESULTS)


def superbowl_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Super Bowl results table.

    Columns: ``game_number`` (int), ``game_year`` (int), ``winner`` (str),
    ``conference`` ('NFC' or 'AFC'), ``orig_nfl`` (bool).

    The ``game_year`` is the calendar year in which the game was *played*
    (January or February). The Krueger-Kennedy convention is to match this to
    the S&P return for the *same* calendar year (or sometimes to the return for
    the year following the January game — see ``fetch_sp500_annual`` for both).
    """
    return SUPER_BOWL_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic annual return generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 58,
    signal_bps: float = 0.0,
    base_return: float = 0.07,
    vol_ann: float = 0.17,
    nfc_fraction: float = 0.52,
    seed: int = 158,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual-return series with an optional planted Super Bowl signal.

    Each year is drawn i.i.d. from N(base_return, vol_ann^2). In 'NFC' years
    (selected randomly with probability ``nfc_fraction``), an extra drift of
    ``signal_bps`` bps is added. ``signal_bps = 0`` is the null — no effect,
    the conference label is uncorrelated with the return. This is the study's
    null in a bottle.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int), ``return``
    (annual simple return), ``conference`` ('NFC' or 'AFC'), ``orig_nfl`` (bool),
    and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1967, 1967 + n_years))
    confs = rng.choice(["NFC", "AFC"], size=n_years,
                       p=[nfc_fraction, 1 - nfc_fraction])
    orig = (confs == "NFC")  # simplified: treat NFC = orig_nfl for synthetic data
    drift = np.where(confs == "NFC", base_return + signal_bps * 1e-4, base_return)
    rets = drift + rng.normal(0.0, vol_ann, n_years)

    df = pd.DataFrame({
        "year": years,
        "return": rets,
        "conference": confs,
        "orig_nfl": orig,
    })
    truth = {
        "n_years": n_years,
        "signal_bps": signal_bps,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "nfc_fraction": nfc_fraction,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — Shiller S&P 500 annual returns (repo-level cache, read-only)
# ---------------------------------------------------------------------------
def fetch_sp500_annual(
    cache_dir: str = REPO_CACHE,
    start_year: int = 1967,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Load S&P 500 calendar-year total real returns from the Shiller dataset.

    The Shiller parquet at ``_cache/shiller_sp500.parquet`` is staged in the
    repo-level cache and is read-only. We compute the calendar-year price return
    (using the nominal SP500 column) and join with the Super Bowl table.

    The function uses the ``game_year`` convention: the Super Bowl game played
    in year Y is paired with the S&P return *for year Y*. This is the convention
    used in Krueger-Kennedy (1990) and most subsequent commentary. The return for
    year Y is the December/December price change (monthly Shiller data).

    Returns a DataFrame with columns:
      ``year``, ``sp500_return``, ``game_number``, ``winner``, ``conference``,
      ``orig_nfl``, ``mkt_up`` (bool: SP500 return > 0).

    Raises ``FileNotFoundError`` if the Shiller cache is not present.
    """
    shiller_path = os.path.join(cache_dir, "shiller_sp500.parquet")
    if not os.path.exists(shiller_path):
        raise FileNotFoundError(
            f"Shiller cache not found at {shiller_path}. "
            "The repo-level _cache/shiller_sp500.parquet must be present."
        )
    sh = pd.read_parquet(shiller_path)

    # The Shiller dataset has a 'Date' column (string 'YYYY-MM-DD') and a RangeIndex.
    if "Date" in sh.columns:
        sh.index = pd.to_datetime(sh["Date"], errors="coerce")
        sh = sh.drop(columns=["Date"])
    elif not isinstance(sh.index, pd.DatetimeIndex):
        sh.index = pd.to_datetime(sh.index)

    # Keep only December observations to get year-end prices.
    dec = sh[sh.index.month == 12].copy()
    dec["year"] = dec.index.year

    # Drop rows where SP500 is 0 or NaN (Shiller placeholder rows).
    dec = dec[dec["SP500"].gt(0) & dec["SP500"].notna()].copy()
    dec = dec.sort_values("year")

    # Annual price return: (P_dec_Y / P_dec_{Y-1}) - 1
    dec["sp500_return"] = dec["SP500"].pct_change()
    dec = dec[["year", "sp500_return"]].dropna()

    # Filter to the study window.
    dec = dec[(dec["year"] >= start_year) & (dec["year"] <= end_year)].copy()

    # Join with the Super Bowl table (on game_year = year).
    sb = superbowl_table()
    sb = sb[(sb["game_year"] >= start_year) & (sb["game_year"] <= end_year)].copy()
    sb = sb.rename(columns={"game_year": "year"})

    merged = dec.merge(sb, on="year", how="inner")
    merged["mkt_up"] = merged["sp500_return"] > 0
    return merged.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the sp500_return column, for the as-of stamp."""
    vals = df["sp500_return"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
