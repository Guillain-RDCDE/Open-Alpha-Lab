"""Data layer for Study 272 (Champagne-Index).

The "Champagne Indicator" is a piece of market folklore: when the corks pop —
when champagne shipments boom — the party is nearly over, and equity markets are
near a euphoric top. The mirror image: when champagne sales slump, gloom has
peaked and a bottom is in. It is a sentiment / "exuberance" proxy in the same
family as the Hemline Index, the Skyscraper Curse, and the Magazine-Cover
Indicator. We test it honestly.

Two components, both fully offline for the core:

- ``CHAMPAGNE_SHIPMENTS`` — a hardcoded table of total worldwide champagne
  shipments (millions of bottles) per calendar year, compiled from the Comité
  Champagne (CIVC) annual figures and trade-press reporting. This is a small,
  well-known, slow-moving series; hardcoded here to keep the core fully offline
  and reproducible. The CIVC reports the global total each January for the prior
  year. Source: Comité Interprofessionnel du Vin de Champagne (CIVC) annual
  shipment statistics; trade press (The Drinks Business, Reuters). As-of
  2026-06-17. Figures are rounded to one decimal (millions of bottles) and are
  approximate — the *shape* (booms, the 2009 crash, the 2020 COVID dip, the
  2021 rebound) is what the indicator trades on, not the third significant digit.

- ``synthetic_annual`` — a deterministic, offline annual generator that produces
  a champagne-growth series and an equity-return series with an *optional* planted
  link (high champagne growth -> low forward equity return, the "euphoria marks
  the top" mechanism). A ``signal_bps = 0`` setting is the null (no link), so the
  tests can confirm the machinery is truthful before it ever looks at real data.

- ``fetch_sp500_annual`` — real S&P 500 (^GSPC) calendar-year price returns from
  the repo-level cache (``_cache/^GSPC_split_only.parquet``), cache-only by
  default. Network is touched only on an explicit ``fetch=True`` (lazy yfinance
  import). We compute December/December calendar-year price returns.

No look-ahead: the CIVC publishes year Y's shipment total in *January of year
Y+1*. So the champagne signal for year Y is only actionable from early Y+1
onward. We therefore pair the champagne *growth* observed over year Y with the
S&P *forward* return earned in year Y+1 (the ``forward`` convention, which is
the only one that is tradable). A same-year ``contemporaneous`` convention is
also exposed so the reader can see that the label choice is itself a
degrees-of-freedom concern (and that the contemporaneous version is not
tradable — you would need the figure before it is published).

PRICE-ONLY / SURVIVORSHIP: the ^GSPC series is a *price* index (no dividends);
returns are total-return-understated by the dividend yield. The S&P 500 itself
is survivorship-managed (the index committee drops failures). Both are named on
the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))
GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")

# ---------------------------------------------------------------------------
# The hardcoded champagne-shipment table — the study's deterministic core
# ---------------------------------------------------------------------------
# Columns:
#   year            : calendar year of the shipment total
#   shipments_mn    : total worldwide champagne shipments that year, in millions
#                     of bottles (France + export combined).
#
# The CIVC ("Comité Champagne") reports the global total each January for the
# prior calendar year. The headline narrative the indicator trades on:
#   - the long 1990s/2000s expansion peaking near ~339 mn bottles in 2007;
#   - the Great-Financial-Crisis collapse to ~293 mn in 2009;
#   - a grinding recovery through the 2010s (~300-310 mn);
#   - the COVID shock of 2020 (~245 mn, a 60-year low) and the euphoric 2021
#     rebound to a record ~322 mn bottles (and record *value*);
#   - a 2023-2024 normalisation back toward ~270-300 mn as the post-COVID
#     splurge faded.
#
# These are rounded approximate figures (one decimal, millions of bottles).
# Sources: CIVC annual shipment statistics; The Drinks Business; Reuters.
# As-of 2026-06-17.

CHAMPAGNE_SHIPMENTS: list[dict] = [
    {"year": 1999, "shipments_mn": 327.0},  # pre-millennium party stockpiling
    {"year": 2000, "shipments_mn": 253.0},  # millennium hangover, sharp drop
    {"year": 2001, "shipments_mn": 263.0},
    {"year": 2002, "shipments_mn": 287.0},
    {"year": 2003, "shipments_mn": 293.0},
    {"year": 2004, "shipments_mn": 301.0},
    {"year": 2005, "shipments_mn": 307.0},
    {"year": 2006, "shipments_mn": 322.0},
    {"year": 2007, "shipments_mn": 339.0},  # pre-GFC peak
    {"year": 2008, "shipments_mn": 322.0},
    {"year": 2009, "shipments_mn": 293.0},  # GFC collapse
    {"year": 2010, "shipments_mn": 319.0},
    {"year": 2011, "shipments_mn": 323.0},
    {"year": 2012, "shipments_mn": 309.0},
    {"year": 2013, "shipments_mn": 305.0},
    {"year": 2014, "shipments_mn": 307.0},
    {"year": 2015, "shipments_mn": 313.0},
    {"year": 2016, "shipments_mn": 306.0},
    {"year": 2017, "shipments_mn": 307.0},
    {"year": 2018, "shipments_mn": 302.0},
    {"year": 2019, "shipments_mn": 298.0},
    {"year": 2020, "shipments_mn": 245.0},  # COVID shock, 60-year low
    {"year": 2021, "shipments_mn": 322.0},  # euphoric rebound, record value
    {"year": 2022, "shipments_mn": 326.0},
    {"year": 2023, "shipments_mn": 299.0},
    {"year": 2024, "shipments_mn": 271.0},  # post-splurge normalisation
]

CHAMPAGNE_DF: pd.DataFrame = pd.DataFrame(CHAMPAGNE_SHIPMENTS)


def champagne_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded champagne-shipment table.

    Columns: ``year`` (int), ``shipments_mn`` (float, millions of bottles),
    ``ship_growth`` (float, year-over-year log growth of shipments).

    ``ship_growth`` for year Y is ``log(shipments[Y] / shipments[Y-1])`` — the
    YoY growth observed *over* year Y, which the CIVC publishes in January Y+1.
    The first row (no prior year) has ``NaN`` growth and is dropped by callers
    that need a complete observation.
    """
    df = CHAMPAGNE_DF.copy()
    df = df.sort_values("year").reset_index(drop=True)
    df["ship_growth"] = np.log(df["shipments_mn"] / df["shipments_mn"].shift(1))
    return df


# ---------------------------------------------------------------------------
# Synthetic annual generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 40,
    signal_bps: float = 0.0,
    base_return: float = 0.07,
    vol_ann: float = 0.17,
    ship_vol: float = 0.06,
    seed: int = 272,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual series with an optional planted champagne->equity link.

    Each year draws a champagne YoY growth ``g_t ~ N(0, ship_vol)`` and a forward
    equity return::

        fwd_return_t = base_return - (signal_bps * 1e-4) * z(g_t) + noise

    where ``z(g_t)`` is the standardized champagne growth. A *positive*
    ``signal_bps`` plants the folklore mechanism: high champagne growth
    (euphoria) -> *low* forward equity return (the top is in). ``signal_bps = 0``
    is the null — champagne growth carries no forward information.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int),
    ``shipments_mn`` (float), ``ship_growth`` (float YoY log growth),
    ``sp500_fwd_return`` (float, next-year equity return), and ``truth`` records
    the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1980, 1980 + n_years))
    g = rng.normal(0.0, ship_vol, n_years)
    # standardize champagne growth for a clean signal scale
    g_z = (g - g.mean()) / (g.std() + 1e-12)
    drift = base_return - (signal_bps * 1e-4) * g_z
    fwd = drift + rng.normal(0.0, vol_ann, n_years)

    # reconstruct a level series from the growths, starting at 300 mn bottles
    levels = 300.0 * np.exp(np.cumsum(g))

    df = pd.DataFrame({
        "year": years,
        "shipments_mn": levels,
        "ship_growth": g,
        "sp500_fwd_return": fwd,
    })
    truth = {
        "n_years": n_years,
        "signal_bps": signal_bps,
        "base_return": base_return,
        "vol_ann": vol_ann,
        "ship_vol": ship_vol,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — ^GSPC annual price returns (repo-level cache, cache-only default)
# ---------------------------------------------------------------------------
def _load_gspc_daily(fetch: bool = False, cache_path: str = GSPC_CACHE) -> pd.DataFrame:
    """Load the ^GSPC daily OHLC frame from the repo cache (or fetch once)."""
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"^GSPC cache not found at {cache_path}. "
                "The repo-level _cache/^GSPC_split_only.parquet must be present, "
                "or call with fetch=True once to populate it."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download("^GSPC", start="1927-01-01", auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw.to_parquet(cache_path)
    return raw


def fetch_sp500_annual(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start_year: int = 1999,
    end_year: int = 2025,
    convention: str = "forward",
) -> pd.DataFrame:
    """Pair champagne-shipment growth with S&P 500 (^GSPC) annual price returns.

    The ^GSPC daily parquet is staged in the repo-level cache and is read-only by
    default (cache-only). We compute December-to-December calendar-year *price*
    returns (no dividends — price index only).

    ``convention`` controls the label alignment:

    - ``"forward"`` (default, the only *tradable* version): champagne growth
      observed over year Y (published Jan Y+1) is paired with the S&P return
      earned in year **Y+1**. This respects the one-year publication lag and is
      the honest, look-ahead-free specification.
    - ``"contemporaneous"``: champagne growth over year Y paired with the S&P
      return of the **same** year Y. This is *not tradable* (you would need the
      figure before the year ends) and is shown only to expose the
      degrees-of-freedom in the label choice.

    Returns a DataFrame with columns:
      ``year`` (the champagne-signal year), ``shipments_mn``, ``ship_growth``,
      ``sp500_return`` (the paired forward or contemporaneous annual return),
      ``mkt_up`` (bool: that return > 0).

    Raises ``FileNotFoundError`` if the ^GSPC cache is absent and ``fetch=False``.
    """
    if convention not in ("forward", "contemporaneous"):
        raise ValueError("convention must be 'forward' or 'contemporaneous'")

    px = _load_gspc_daily(fetch=fetch, cache_path=cache_path)
    px = px.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)

    # Year-end close = last trading day of each calendar year.
    close = px["Close"].dropna()
    yr_end = close.groupby(close.index.year).last()
    yr_end.index = yr_end.index.astype(int)

    # Calendar-year price return for year Y: last close Y / last close Y-1 - 1.
    sp_ret = yr_end.pct_change().rename("ret")
    sp_ret = sp_ret.dropna()

    champ = champagne_table().dropna(subset=["ship_growth"]).copy()
    champ = champ[(champ["year"] >= start_year) & (champ["year"] <= end_year)].copy()

    rows = []
    for _, r in champ.iterrows():
        sig_year = int(r["year"])
        target_year = sig_year + 1 if convention == "forward" else sig_year
        if target_year in sp_ret.index:
            rows.append({
                "year": sig_year,
                "shipments_mn": float(r["shipments_mn"]),
                "ship_growth": float(r["ship_growth"]),
                "sp500_return": float(sp_ret.loc[target_year]),
            })

    out = pd.DataFrame(rows)
    out["mkt_up"] = out["sp500_return"] > 0
    return out.reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the sp500_return column, for the as-of stamp."""
    vals = df["sp500_return"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
