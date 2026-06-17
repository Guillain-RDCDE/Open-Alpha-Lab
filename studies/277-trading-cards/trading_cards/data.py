"""Data layer for Study 277 (Trading-Cards).

Three components, the first two fully offline for the core:

- ``CARD_INDEX`` -- a hardcoded, curated *annual* graded-card price index
  (year-end level, base 100 = 1990). It is a stylised composite of the
  best-documented public proxies for the graded collectible-card market:
  the PSA-graded Pokemon and vintage-sports card auction trade. There is no
  clean machine-readable total-return index for trading cards (the asset class
  has no exchange, no daily marks, and survivorship-riddled auction data), so
  this curated series is the honest best effort: a small, well-known, hand-built
  table that captures the documented shape -- a long flat stretch through the
  1990s-2010s, the explosive 2020-2021 pandemic boom, the brutal 2022-2023
  drawdown, and the partial stabilisation since. Hardcoded here to keep the core
  fully offline and reproducible. See ``docs/references.md`` for the proxy sources.

- ``synthetic_index`` -- a deterministic, offline annual-index generator with an
  optional planted "excess return vs equities" knob. A ``signal_bps = 0`` setting
  is the null (cards track equities plus noise, no structural edge), so tests can
  confirm the machinery is truthful before looking at the curated tape.

- ``fetch_gspc_annual`` -- real S&P 500 calendar-year price returns from the
  cached ``^GSPC_split_only.parquet`` (repo-level cache). Cache-only; no network
  call needed for the headline run. ``fetch=True`` would lazily import yfinance,
  but the default is offline.

No look-ahead: each card-index year-end level is paired with the *same* calendar
year's S&P return, so the comparison is contemporaneous (we are asking whether
cards are an attractive *asset class*, not a market-timing signal).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

STUDY_SEED = 277

# ---------------------------------------------------------------------------
# The hardcoded curated annual graded-card price index -- the offline core
# ---------------------------------------------------------------------------
# Columns:
#   year        : calendar year (year-end observation)
#   card_index  : composite graded-card price index level, base 100 = 1990
#
# Construction notes (curated, stylised -- NOT a vendored vendor index):
#   The level path is hand-built to match the documented shape of the graded
#   collectible-card market, anchored to publicly reported moves:
#     * 1990-2014: a long, slow grind. Cards roughly tracked inflation with
#       occasional pops (the 1990s junk-wax glut kept a lid on modern issues;
#       blue-chip vintage drifted up slowly).
#     * 2015-2019: a gentle uptrend as PSA/BGS grading professionalised the
#       market and online marketplaces (eBay, PWCC, Goldin) improved liquidity.
#     * 2020-2021: the pandemic mania. Stimulus, lockdown boredom, influencer
#       hype and a flood of new buyers roughly TRIPLED blue-chip graded prices;
#       PWCC's "500 index" and widely-cited auction comps rose ~+90% in 2020 and
#       a further ~+55% in 2021.
#     * 2022-2023: the hangover. As rates rose and the speculative froth left,
#       the same comps fell ~ -30% then ~ -20%, a brutal round-trip.
#     * 2024-2025: partial stabilisation, well below the 2021 peak.
#
# This is a single curated path, not a claim of vendor-grade precision. It is
# the study's deterministic, offline anchor; the verdict does NOT hinge on the
# exact levels (see the synthetic positive control and the bootstrap in
# docs/results.md). Sources: see docs/references.md.
#
# As-of: 2026-06-17.

CARD_INDEX: list[dict] = [
    # year, card_index (base 100 = 1990)
    {"year": 1990, "card_index": 100.0},
    {"year": 1991, "card_index": 103.0},
    {"year": 1992, "card_index": 101.0},
    {"year": 1993, "card_index": 99.0},
    {"year": 1994, "card_index": 100.0},
    {"year": 1995, "card_index": 104.0},
    {"year": 1996, "card_index": 108.0},
    {"year": 1997, "card_index": 112.0},
    {"year": 1998, "card_index": 118.0},
    {"year": 1999, "card_index": 124.0},
    {"year": 2000, "card_index": 129.0},
    {"year": 2001, "card_index": 131.0},
    {"year": 2002, "card_index": 134.0},
    {"year": 2003, "card_index": 140.0},
    {"year": 2004, "card_index": 147.0},
    {"year": 2005, "card_index": 153.0},
    {"year": 2006, "card_index": 160.0},
    {"year": 2007, "card_index": 168.0},
    {"year": 2008, "card_index": 158.0},   # GFC dent
    {"year": 2009, "card_index": 165.0},
    {"year": 2010, "card_index": 174.0},
    {"year": 2011, "card_index": 182.0},
    {"year": 2012, "card_index": 191.0},
    {"year": 2013, "card_index": 203.0},
    {"year": 2014, "card_index": 214.0},
    {"year": 2015, "card_index": 230.0},
    {"year": 2016, "card_index": 248.0},
    {"year": 2017, "card_index": 270.0},
    {"year": 2018, "card_index": 291.0},
    {"year": 2019, "card_index": 315.0},
    {"year": 2020, "card_index": 600.0},   # pandemic mania, ~+90%
    {"year": 2021, "card_index": 930.0},   # peak, ~+55%
    {"year": 2022, "card_index": 650.0},   # crash, ~-30%
    {"year": 2023, "card_index": 520.0},   # ~-20%
    {"year": 2024, "card_index": 560.0},   # partial recovery
    {"year": 2025, "card_index": 585.0},
]

CARD_INDEX_DF: pd.DataFrame = pd.DataFrame(CARD_INDEX)


def card_index_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded curated card index.

    Columns: ``year`` (int), ``card_index`` (float, base 100 = 1990),
    ``card_return`` (annual simple return of the index, NaN for the first year).
    """
    df = CARD_INDEX_DF.copy()
    df["card_return"] = df["card_index"].pct_change()
    return df


# ---------------------------------------------------------------------------
# Synthetic annual index generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_index(
    n_years: int = 36,
    signal_bps: float = 0.0,
    equity_mu: float = 0.08,
    equity_vol: float = 0.16,
    card_extra_vol: float = 0.22,
    beta: float = 0.6,
    seed: int = STUDY_SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible paired (cards, equities) annual-return series.

    Cards are modelled as ``card_ret = beta * equity_ret + alpha + idio`` where
    ``alpha = signal_bps * 1e-4`` is a planted structural excess return and
    ``idio`` is independent card-specific noise (illiquid collectibles carry
    large idiosyncratic vol). ``signal_bps = 0`` is the null -- cards offer no
    structural edge over equities, only beta plus noise. This is the study's
    null in a bottle.

    Returns ``(df, truth)`` where ``df`` has columns ``year`` (int),
    ``card_return``, ``gspc_return`` (both annual simple returns), and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    years = list(range(1990, 1990 + n_years))
    eq = equity_mu + rng.normal(0.0, equity_vol, n_years)
    alpha = signal_bps * 1e-4
    idio = rng.normal(0.0, card_extra_vol, n_years)
    card = beta * eq + alpha + idio

    df = pd.DataFrame({
        "year": years,
        "card_return": card,
        "gspc_return": eq,
    })
    truth = {
        "n_years": n_years,
        "signal_bps": signal_bps,
        "equity_mu": equity_mu,
        "equity_vol": equity_vol,
        "card_extra_vol": card_extra_vol,
        "beta": beta,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- S&P 500 annual price returns (repo-level cache, read-only)
# ---------------------------------------------------------------------------
def fetch_gspc_annual(
    cache_dir: str = REPO_CACHE,
    start_year: int = 1990,
    end_year: int = 2025,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load S&P 500 calendar-year *price* returns and join with the card index.

    By default (``fetch=False``) this is cache-only: it reads
    ``^GSPC_split_only.parquet`` from the repo-level cache. With ``fetch=True``
    it would lazily import yfinance to download ^GSPC -- but the headline run and
    CI never set this, so the core stays offline and deterministic.

    The S&P series is price-only (no dividends) to be label-consistent with the
    card index, which is a *price* index (auction hammer prices, no yield). This
    is the honest comparison: collectibles pay no dividend, so we compare against
    the S&P's price return, and separately note the total-return gap in the docs.

    Returns a DataFrame with columns:
      ``year``, ``card_index``, ``card_return``, ``gspc_return``,
      ``card_up`` (bool), ``gspc_up`` (bool).

    Raises ``FileNotFoundError`` if the cache is missing and ``fetch`` is False.
    """
    gspc_path = os.path.join(cache_dir, "^GSPC_split_only.parquet")
    if not os.path.exists(gspc_path):
        if not fetch:
            raise FileNotFoundError(
                f"GSPC cache not found at {gspc_path}. "
                "Pass fetch=True to download (network), or stage the cache."
            )
        import yfinance as yf  # lazy, only on explicit fetch
        raw = yf.download("^GSPC", start="1989-01-01", end=f"{end_year + 1}-01-31",
                          auto_adjust=False, progress=False)
        px = raw["Close"].copy()
        px.index = pd.to_datetime(px.index)
    else:
        raw = pd.read_parquet(gspc_path)
        px = raw["Close"].copy()
        px.index = pd.to_datetime(px.index)

    px = px.dropna()
    # Year-end level = last observation of each calendar year.
    yr_end = px.groupby(px.index.year).last()
    yr_end.index.name = "year"
    gspc = yr_end.pct_change().rename("gspc_return").to_frame().reset_index()
    gspc = gspc[(gspc["year"] >= start_year) & (gspc["year"] <= end_year)].copy()

    cards = card_index_table()
    cards = cards[(cards["year"] >= start_year) & (cards["year"] <= end_year)].copy()

    merged = cards.merge(gspc, on="year", how="inner")
    # First study year has no prior-year return for the card index.
    merged = merged.dropna(subset=["card_return", "gspc_return"]).reset_index(drop=True)
    merged["card_up"] = merged["card_return"] > 0
    merged["gspc_up"] = merged["gspc_return"] > 0
    return merged


def fingerprint(df: pd.DataFrame, col: str = "card_return") -> str:
    """A short content fingerprint of ``col``, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(np.round(vals, 8)).tobytes())
    return h.hexdigest()[:12]
