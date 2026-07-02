"""Data layer for Study 543 (Western-Zodiac-CEO) — does a CEO's sun sign predict returns?

The folklore, western-astrology flavour (the sibling of [Study 165 — Chinese-Zodiac], which
tests the *zodiac year*): a company run by a CEO of the "right" star sign — say a decisive Aries
or a visionary Aquarius — supposedly outperforms. It is the horoscope column applied to the
C-suite. We test it the honest, underpowered way it deserves.

Three components:

- ``CEO_BIRTHDAYS`` — a small, **hand-curated** table of large-cap CEOs whose *birth dates* are a
  matter of public record (Wikipedia / SEC filings / company bios), mapped to their company
  ticker and their **sun sign** (derived from the birth date by the standard western tropical
  cutoffs). This is a *tiny curated cross-section* — a few dozen names spread over twelve signs —
  so it is structurally underpowered for a twelve-way test, exactly as the Chinese-zodiac study's
  "12 animals × ~3 years" is. That is named on the SIGNAL axis: **a hand-curated ~30-name table
  can never certify a REAL signal** (that needs a robust *t* ≥ 2 on a real, powered tape).

- ``synthetic_panel`` — a deterministic, offline generator with a single knob, ``sign_alpha``,
  that plants a *sun-sign effect*: one favoured sign is given an alpha bump. ``sign_alpha = 0`` is
  the null (returns unrelated to sign). The positive control and the null in one bottle; tests
  never touch the network.

- ``fetch_prices`` — real yfinance daily adjusted closes for the curated tickers, cache-first into
  this study's own ``_cache/`` so the reproducible core never needs the network. The *forward
  return* per CEO is then real; only the *assignment of a name to a sign* is curated.

Why this can only ever be WEAK/NONE: the sign is a birthday, i.e. a coin flip with respect to how
a company trades. There is no mechanism. The table is tiny; twelve signs over ~30 names leaves
2-3 names per sign — hopelessly underpowered (a genuine 12-way test needs hundreds of names). And
the birthday is fixed for a CEO's whole tenure, so there is no time-series to average. We include
the honest real-tape read purely to show the folklore evaporates, and cap the SIGNAL axis
accordingly.

No look-ahead: the sun sign is fixed at birth (decades before any price), and the forward return
is measured over a clearly delimited window *after* the as-of date. The signal is trivially public
in advance; the only convention is the single as-of scoring date and the same-close entry (one
documented execution lag — a one-bar-later entry moves nothing material on a multi-year hold).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# The twelve western (tropical) sun signs and their standard date cutoffs (month, day) of the
# START of each sign. Aries opens the zodiac at the spring equinox (~Mar 21).
SIGN_CUTOFFS = [
    ("Capricorn", (12, 22)),   # wraps year-end
    ("Aquarius", (1, 20)),
    ("Pisces", (2, 19)),
    ("Aries", (3, 21)),
    ("Taurus", (4, 20)),
    ("Gemini", (5, 21)),
    ("Cancer", (6, 21)),
    ("Leo", (7, 23)),
    ("Virgo", (8, 23)),
    ("Libra", (9, 23)),
    ("Scorpio", (10, 23)),
    ("Sagittarius", (11, 22)),
    ("Capricorn2", (12, 22)),  # sentinel
]
SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def sun_sign(month: int, day: int) -> str:
    """Western tropical sun sign for a (month, day) birth date."""
    # ordered cutoffs by (month, day); find the last cutoff <= the date
    md = (month, day)
    name = "Capricorn"  # default for late Dec / very early Jan handled below
    for label, cut in SIGN_CUTOFFS:
        if md >= cut:
            name = label
    name = name.replace("2", "")
    # early January (before Jan 20) is Capricorn (carried over from Dec)
    if month == 1 and day < 20:
        name = "Capricorn"
    return name


# ---------------------------------------------------------------------------
# The hand-curated CEO birth-date table (public record) — the offline curated core
# ---------------------------------------------------------------------------
# Each row: ticker -> (CEO name, birth year, birth month, birth day). Birth dates are drawn from
# public sources (Wikipedia / company bios / SEC / press). Where only a birth MONTH is public we
# do not include the name (we need the day for the sign cutoff). This is a CURATED, STYLISED table:
# it is deliberately small and its only purpose is to show the folklore has nothing to find. Some
# entries are founders/long-tenured chiefs so the ticker maps cleanly to one person over the
# window. (If a date is later corrected, the sign — not the price — is what would change.)
CEO_BIRTHDAYS: dict[str, tuple[str, int, int, int]] = {
    "TSLA": ("Elon Musk",            1971, 6, 28),   # Cancer
    "AMZN": ("Jeff Bezos",           1964, 1, 12),   # Capricorn (Bezos era; carried for window)
    "META": ("Mark Zuckerberg",      1984, 5, 14),   # Taurus
    "MSFT": ("Satya Nadella",        1967, 8, 19),   # Leo
    "AAPL": ("Tim Cook",             1960, 11, 1),   # Scorpio
    "NVDA": ("Jensen Huang",         1963, 2, 17),   # Aquarius
    "BRK-B": ("Warren Buffett",      1930, 8, 30),   # Virgo
    "JPM": ("Jamie Dimon",           1956, 3, 13),   # Pisces
    "ORCL": ("Larry Ellison",        1944, 8, 17),   # Leo (chairman/CTO, founder era)
    "DELL": ("Michael Dell",         1965, 2, 23),   # Pisces
    "SBUX": ("Howard Schultz",       1953, 7, 19),   # Cancer (Schultz era)
    "DIS": ("Bob Iger",              1951, 2, 10),   # Aquarius
    "GS": ("David Solomon",          1962, 2, 24),   # Pisces
    "CRM": ("Marc Benioff",          1964, 9, 25),   # Libra
    "NFLX": ("Reed Hastings",        1960, 10, 8),   # Libra (Hastings era)
    "T": ("Randall Stephenson",      1960, 4, 22),   # Taurus (Stephenson era)
    "KO": ("James Quincey",          1965, 1, 8),    # Capricorn
    "PEP": ("Ramon Laguarta",        1963, 8, 6),    # Leo
    "GM": ("Mary Barra",             1961, 12, 24),  # Capricorn
    "INTC": ("Pat Gelsinger",        1961, 3, 5),    # Pisces (Gelsinger era)
    "WMT": ("Doug McMillon",         1966, 10, 17),  # Libra
    "BA": ("David Calhoun",          1957, 4, 18),   # Aries (Calhoun era)
    "PLTR": ("Alex Karp",            1967, 10, 2),   # Libra
    "COIN": ("Brian Armstrong",      1983, 1, 25),   # Aquarius
    "SNAP": ("Evan Spiegel",         1990, 6, 4),    # Gemini
    "UBER": ("Dara Khosrowshahi",    1969, 5, 28),   # Gemini
    "ABNB": ("Brian Chesky",         1981, 8, 29),   # Virgo
    "SQ": ("Jack Dorsey",            1976, 11, 19),  # Scorpio (Block/Square era)
    "F": ("Jim Farley",              1962, 6, 10),   # Gemini
    "GE": ("Larry Culp",            1963, 3, 23),    # Aries
    "PG": ("Jon Moeller",            1964, 3, 21),   # Aries
    "HD": ("Ted Decker",             1963, 9, 15),   # Virgo
}


def curated_signs() -> pd.DataFrame:
    """The curated CEO table as a frame: ticker, ceo, birthdate, sign. Fully offline."""
    rows = {}
    for tk, (name, y, m, d) in CEO_BIRTHDAYS.items():
        rows[tk] = {
            "ceo": name,
            "birth": f"{y:04d}-{m:02d}-{d:02d}",
            "sign": sun_sign(m, d),
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    sign_alpha: float          # alpha bump for the favoured sign (0 = null)
    favoured_sign: str

    @property
    def has_effect(self) -> bool:
        return self.sign_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (positive control + null)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 120,
    sign_alpha: float = 0.10,
    favoured_sign: str = "Aries",
    idio_vol: float = 0.20,
    seed: int = 543,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible cross-section of firms, each with a random sun sign and a forward return.

    Every firm is assigned a sun sign uniformly at random (a birthday is unrelated to the firm).
    The forward return is::

        forward_ret = drift + sign_alpha * 1[sign == favoured_sign] + idio

    With ``sign_alpha > 0`` the favoured sign earns a genuine bump (the planted effect the engine
    must detect); ``sign_alpha = 0`` is the null (return is independent of sign). Returns a frame
    with one row per firm — columns ``sign`` and ``forward_ret`` — matching the schema the real
    tape produces, plus ``(panel, truth)``.
    """
    rng = np.random.default_rng(seed)
    signs = rng.choice(SIGNS, size=n_stocks, replace=True)
    drift = 0.12
    idio = idio_vol * rng.standard_normal(n_stocks)
    bump = sign_alpha * (signs == favoured_sign).astype(float)
    forward_ret = drift + bump + idio
    tickers = [f"S{j:03d}" for j in range(n_stocks)]
    panel = pd.DataFrame(
        {"sign": signs, "forward_ret": forward_ret},
        index=pd.Index(tickers, name="ticker"),
    )
    return panel, WorldTruth(sign_alpha=sign_alpha, favoured_sign=favoured_sign)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "zodiac_ceo_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2018-01-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Daily adjusted-close prices for the curated tickers, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    tickers = list(CEO_BIRTHDAYS.keys())
    raw = _retry(
        lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    cov = raw.notna().mean()
    raw = raw.loc[:, cov >= 0.5]
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def forward_return(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Simple total return per name over the forward window [start, end]."""
    px = prices.loc[start:end]
    if len(px) < 2:
        return pd.Series(dtype=float)
    return px.iloc[-1] / px.iloc[0] - 1.0


def build_panel(prices: pd.DataFrame, split_date: str, end_date: str) -> pd.DataFrame:
    """Assemble the real cross-section the sort consumes: sign (curated) + forward return (real).

    Only names present in both the curated table and the price frame (with a full forward window)
    are kept. Returns a frame with columns ``sign`` and ``forward_ret``.
    """
    signs = curated_signs()
    names = [t for t in signs.index if t in prices.columns]
    if not names:
        return pd.DataFrame()
    fr = forward_return(prices[names], split_date, end_date)
    rows = {}
    for t in names:
        if t not in fr.index or not np.isfinite(fr[t]):
            continue
        rows[t] = {"sign": signs.loc[t, "sign"], "forward_ret": float(fr[t])}
    panel = pd.DataFrame.from_dict(rows, orient="index")
    panel.index.name = "ticker"
    return panel.dropna()


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        h = hashlib.sha1(arr.tobytes())
        # fold in the non-numeric labels (signs) so the fp reflects the whole panel
        h.update(json.dumps(
            {c: obj[c].astype(str).tolist() for c in obj.columns if c not in num.columns},
            sort_keys=True).encode())
        h.update("|".join(map(str, obj.index)).encode())
        return h.hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
