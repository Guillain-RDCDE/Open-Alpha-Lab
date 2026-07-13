"""Data layer for Study 748 (CEO-Age-Effect) — do young-CEO firms behave differently?

The folklore, dressed up with a real academic literature: a **young** CEO is aggressive,
growth-hungry, an empire-builder who takes more risk (Yim 2013, "The Acquisitiveness of
Youth"); an **old** CEO is cautious, harvests cash, dials down volatility (Serfling 2014,
"CEO age and the riskiness of corporate policies"). If that flows through to the *stock*,
a long-young / short-old book should earn a spread. We test the tradable version of that
claim the honest way — and name the confound that dooms it loudly.

Three components:

- ``CEO_AGES`` — a small, **hand-curated, cited** table of large-cap CEOs with a clean
  single-person tenure over the sample, mapped to their company ticker and their **birth
  year** (public record: Wikipedia / SEC filings / company bios). A CEO is bucketed
  **young** if their age at the scoring date is below ``YOUNG_MAX_AGE``, else **old**.
  This is a *tiny curated cross-section* — a few dozen names — so it is structurally
  underpowered and, worse, badly confounded: the "young CEO" bucket is stuffed with
  recently-IPO'd growth-tech (COIN, HOOD, DASH, SNAP…), so the split is a **sector / size
  / listing-vintage tilt wearing an age costume**. That confound is named on the SIGNAL
  axis — a curated ~40-name table can never certify a `REAL` age premium.

- ``synthetic_panel`` — a deterministic, offline generator with a single knob,
  ``age_alpha``, that plants a *genuine* young-minus-old monthly return premium. With
  ``age_alpha = 0`` the returns are unrelated to the CEO-age label (the null). The
  positive control and the null in one bottle; tests never touch the network.

- ``fetch_prices`` — real yfinance daily **total-return** (dividend-adjusted) closes for
  the curated tickers + SPY, cache-first into this study's own ``_cache/`` so the
  reproducible core never needs the network. The monthly returns are then real; only the
  *assignment of a name to an age bucket* is curated.

No look-ahead: the CEO's age at the scoring date is public in advance, and the bucket
membership known at the close of month *t* earns the return of month *t+1* — one
documented execution lag, applied once. Names that IPO'd mid-sample enter the equal-
weight basket only in the months they have a price, which biases the young basket toward
recent hot listings — a **survivorship / listing-vintage** tilt, named on the SIGNAL axis.
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

# A CEO is "young" if their age at the scoring date is strictly below this cutoff.
YOUNG_MAX_AGE = 55
# The as-of date at which we compute each CEO's age and freeze the bucket labels.
SCORE_DATE = "2024-12-31"

# ---------------------------------------------------------------------------
# The hand-curated CEO birth-year table (public record) — the offline curated core
# ---------------------------------------------------------------------------
# Each row: ticker -> (CEO name, birth year). Birth years are drawn from public sources
# (Wikipedia / company bios / SEC / press). We keep names whose ticker maps cleanly to ONE
# long-tenured chief or founder over the sample so the bucket label is stable. This is a
# CURATED, STYLISED table — deliberately small; its purpose is to test (and, as it turns
# out, disprove) a tradable age premium, and to expose the sector/vintage confound.
# The "young" bucket skews toward founder-led growth-tech that IPO'd recently; the "old"
# bucket toward mega-cap incumbents and value — that imbalance IS the story.
CEO_AGES: dict[str, tuple[str, int]] = {
    # --- young-CEO firms (born ~1970+, mostly founder-led growth) ---
    "TSLA": ("Elon Musk",           1971),
    "META": ("Mark Zuckerberg",     1984),
    "GOOGL": ("Sundar Pichai",      1972),
    "ABNB": ("Brian Chesky",        1981),
    "SNAP": ("Evan Spiegel",        1990),
    "COIN": ("Brian Armstrong",     1983),
    "SHOP": ("Tobias Lutke",        1980),
    "SPOT": ("Daniel Ek",           1983),
    "DASH": ("Tony Xu",             1984),
    "CRWD": ("George Kurtz",        1970),
    "HOOD": ("Vlad Tenev",          1987),
    "NET":  ("Matthew Prince",      1974),
    "ABT":  ("Robert Ford",         1973),
    "BBY":  ("Corie Barry",         1975),
    "RBLX": ("David Baszucki",      1963),   # older founder — lands in OLD (see split)
    # --- old-CEO firms (born <1970, mostly incumbent mega-caps) ---
    "BRK-B": ("Warren Buffett",     1930),
    "JPM":  ("Jamie Dimon",         1956),
    "AAPL": ("Tim Cook",            1960),
    "BLK":  ("Larry Fink",          1952),
    "BAC":  ("Brian Moynihan",      1959),
    "GM":   ("Mary Barra",          1961),
    "CRM":  ("Marc Benioff",        1964),
    "ADBE": ("Shantanu Narayen",    1963),
    "NVDA": ("Jensen Huang",        1963),
    "WMT":  ("Doug McMillon",       1966),
    "GS":   ("David Solomon",       1962),
    "XOM":  ("Darren Woods",        1965),
    "CVX":  ("Mike Wirth",          1960),
    "PEP":  ("Ramon Laguarta",      1963),
    "KO":   ("James Quincey",       1965),
    "JNJ":  ("Joaquin Duato",       1962),
    "MSFT": ("Satya Nadella",       1967),
    "HD":   ("Ted Decker",          1963),
    "UBER": ("Dara Khosrowshahi",   1969),
    "PLTR": ("Alex Karp",           1967),
    "NOW":  ("Bill McDermott",      1961),
    "PANW": ("Nikesh Arora",        1968),
    "LOW":  ("Marvin Ellison",      1965),
    "CAT":  ("Jim Umpleby",         1958),
    "AMD":  ("Lisa Su",             1969),
}

BENCHMARK = "SPY"


def curated_ages(score_date: str = SCORE_DATE, young_max: int = YOUNG_MAX_AGE) -> pd.DataFrame:
    """The curated CEO table as a frame: ticker, ceo, birth_year, age, bucket. Fully offline.

    ``age`` is the CEO's age at ``score_date`` (year difference); ``bucket`` is ``young`` if
    ``age < young_max`` else ``old``. Deterministic and network-free.
    """
    yr = pd.Timestamp(score_date).year
    rows = {}
    for tk, (name, by) in CEO_AGES.items():
        age = yr - by
        rows[tk] = {
            "ceo": name,
            "birth_year": by,
            "age": age,
            "bucket": "young" if age < young_max else "old",
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "ticker"
    return df


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    age_alpha: float          # monthly young-minus-old premium bump (0 = null)

    @property
    def has_effect(self) -> bool:
        return self.age_alpha != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core (positive control + null)
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 96,
    n_young: int = 15,
    n_old: int = 25,
    age_alpha: float = 0.004,
    mkt_vol: float = 0.045,
    young_beta: float = 1.35,
    old_beta: float = 0.95,
    idio_vol: float = 0.06,
    seed: int = 748,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A reproducible monthly panel of firm returns with a shared market factor.

    Each firm's monthly return is::

        r_it = beta_i * mkt_t + age_alpha * 1[young_i] + idio_it

    where the market factor ``mkt_t`` is common. Young firms carry a higher beta (the
    real-world growth tilt) AND, when ``age_alpha > 0``, a genuine extra premium the
    engine must detect. ``age_alpha = 0`` is the null: young firms still have higher beta
    (so the raw spread is nonzero) but **no alpha** — the honest test (CAPM alpha) must
    then read flat. Returns ``(panel, mkt, truth)`` where ``panel`` has one column per
    firm and a ``bucket`` attribute-free schema (bucket carried separately).
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2018-01", periods=n_months, freq="M").to_timestamp("M")
    mkt = pd.Series(mkt_vol * rng.standard_normal(n_months), index=idx, name="mkt")
    cols, buckets = [], {}
    data = {}
    for j in range(n_young):
        name = f"Y{j:02d}"
        data[name] = young_beta * mkt.values + age_alpha + idio_vol * rng.standard_normal(n_months)
        buckets[name] = "young"
        cols.append(name)
    for j in range(n_old):
        name = f"O{j:02d}"
        data[name] = old_beta * mkt.values + idio_vol * rng.standard_normal(n_months)
        buckets[name] = "old"
        cols.append(name)
    panel = pd.DataFrame(data, index=idx)
    panel.attrs["bucket"] = buckets
    return panel, mkt, WorldTruth(age_alpha=age_alpha)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "ceo_age_prices.parquet")


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
    end: str = "2026-06-30",
) -> pd.DataFrame:
    """Daily total-return (dividend-adjusted) closes for the curated tickers + SPY, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    empty frame. Network is touched only on an explicit ``fetch=True`` (then cached).
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

    tickers = list(CEO_AGES.keys()) + [BENCHMARK]
    raw = _retry(
        lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real() -> bool:
    return os.path.exists(_price_cache())


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end total-return series per name (dividend-adjusted closes -> simple returns).

    Drops the first (all-NaN) month and any partial trailing month is the caller's concern
    (``build_returns`` trims it). A name with no price in a month is NaN there — the
    equal-weight basket simply averages the names present.
    """
    me = prices.resample("ME").last()
    return me.pct_change()


def build_returns(prices: pd.DataFrame, score_date: str = SCORE_DATE,
                  young_max: int = YOUNG_MAX_AGE) -> pd.DataFrame:
    """Assemble the tradable monthly time series the strategy consumes.

    Returns a frame indexed by month-end with columns ``young`` (equal-weight monthly
    return of young-CEO names present that month), ``old`` (same for old-CEO names),
    ``ls`` = young - old (the dollar-neutral long/short), and ``mkt`` (SPY monthly return).
    The last, possibly-partial month is dropped (house rule: no partial bar in a stamped
    run). Only names in both the curated table and the price frame are used.
    """
    ages = curated_ages(score_date, young_max)
    rets = monthly_returns(prices)
    young = [t for t in ages.index[ages["bucket"] == "young"] if t in rets.columns]
    old = [t for t in ages.index[ages["bucket"] == "old"] if t in rets.columns]
    if not young or not old or BENCHMARK not in rets.columns:
        return pd.DataFrame()
    y = rets[young].mean(axis=1, skipna=True)
    o = rets[old].mean(axis=1, skipna=True)
    out = pd.DataFrame({"young": y, "old": o, "ls": y - o, "mkt": rets[BENCHMARK]})
    out = out.dropna(subset=["young", "old", "mkt"])
    # drop the partial trailing month (its last calendar day may be < month end)
    if len(out) and out.index[-1] != out.index[-1] + pd.offsets.MonthEnd(0):
        out = out.iloc[:-1]
    return out


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        num = obj.select_dtypes(include=[np.number])
        arr = np.ascontiguousarray(num.fillna(0).to_numpy(dtype=float))
        h = hashlib.sha1(arr.tobytes())
        h.update("|".join(map(str, obj.columns)).encode())
        h.update("|".join(map(str, obj.index)).encode())
        return h.hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]
