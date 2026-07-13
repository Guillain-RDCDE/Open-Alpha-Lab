"""Data layer for Study 747 — Founder-Led-Premium (a long/short characteristic sort).

Two sources, both offline-friendly:

* **Real tape.** Two hardcoded, transparent baskets of large-cap US firms as of a fixed
  formation date (2016-01): a **founder-led** basket (the founder was still CEO / in
  control around formation) and a **professional-CEO** peer basket (orderly managerial
  succession, no founder at the helm). We pull monthly *adjusted* closes (yfinance, no
  key; splits + dividends folded in, so this is a **total-return** proxy) for every name
  plus SPY, cached under ``_cache/`` as one parquet per ticker. From those we build two
  equal-weighted basket return series and the **founder − professional long/short**, then
  a market-model (CAPM) **abnormal return** (Jensen alpha) with a Newey-West HAC *t*.

  The membership lists are the transparent stand-in for a survivorship-clean founder
  panel (Fahlenbrach's hand-collected S&P 500 founder-CEO set is not redistributable).
  The honest catch — named loudly on the Signal axis — is that a basket of the founder
  firms we *remember in 2024*, applied back to 2016, is **hindsight-selected**: the
  founder firms that cratered or went bankrupt (Theranos never listed, WeWork, Pets.com,
  countless de-SPACs) are simply absent, so the basket is biased *up* before a single
  return is computed.

* **Synthetic.** A deterministic, fixed-seed generator that builds two basket return
  panels with a *plantable* monthly founder alpha (``alpha_bps``). It is the positive
  control: with the edge set to zero the inference must NOT manufacture a significant
  long/short alpha; with a large planted edge it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only
used once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# --------------------------------------------------------------------------- #
# Hardcoded baskets — membership fixed as of the 2016-01 formation date.
# Columns: ticker, leader (name), note.
# "Founder-led" = the founder (or a co-founder) was CEO / executive chair / in voting
# control at formation; "Professional" = a matched large-cap run by a hired manager with
# no founder at the helm. Compiled from company filings and contemporaneous press. The
# founder tag is the believers' own framing (Fahlenbrach 2009; the Bain "founder's
# mentality" thesis) and is deliberately transparent so a reader can re-tag and re-run.
# CRUCIALLY: this is the set of founder firms one REMEMBERS in 2024 — a hindsight,
# survivor-biased sample. Named on the Signal axis; see the docstring above.
# --------------------------------------------------------------------------- #
_FOUNDER_RAW = [
    # (ticker, leader, note)
    ("AMZN",  "Jeff Bezos",        "founder-CEO through 2021"),
    ("NVDA",  "Jensen Huang",      "co-founder-CEO"),
    ("NFLX",  "Reed Hastings",     "co-founder-CEO"),
    ("CRM",   "Marc Benioff",      "founder-CEO (Salesforce)"),
    ("TSLA",  "Elon Musk",         "early investor, CEO / product architect"),
    ("META",  "Mark Zuckerberg",   "founder-CEO, voting control"),
    ("GOOGL", "Page / Brin",       "co-founders, super-voting control"),
    ("SHOP",  "Tobi Lütke",        "founder-CEO (Shopify)"),
    ("SQ",    "Jack Dorsey",       "co-founder-CEO (Square / Block)"),
    ("GPRO",  "Nick Woodman",      "founder-CEO (GoPro) — post-IPO collapse"),
    ("YELP",  "Jeremy Stoppelman", "co-founder-CEO — perennial laggard"),
    ("W",     "Niraj Shah",        "co-founder-CEO (Wayfair)"),
    ("FIT",   "James Park",        "founder-CEO (Fitbit) — acquired by Google 2021"),
]

_PRO_RAW = [
    # (ticker, leader, note) — matched large-caps, professional (non-founder) CEO in 2016
    ("WMT",  "Doug McMillon",   "career-manager CEO (Walmart)"),
    ("KO",   "Muhtar Kent",     "professional succession (Coca-Cola)"),
    ("PG",   "David Taylor",    "insider succession (P&G)"),
    ("JNJ",  "Alex Gorsky",     "professional CEO (J&J)"),
    ("IBM",  "Ginni Rometty",   "professional CEO (IBM)"),
    ("INTC", "Brian Krzanich",  "insider-engineer CEO (Intel)"),
    ("CSCO", "Chuck Robbins",   "professional succession (Cisco)"),
    ("PFE",  "Ian Read",        "professional CEO (Pfizer)"),
    ("VZ",   "Lowell McAdam",   "professional CEO (Verizon)"),
    ("MCD",  "Steve Easterbrook", "professional CEO (McDonald's)"),
    ("DIS",  "Bob Iger",        "professional CEO (Disney)"),
    ("HD",   "Craig Menear",    "insider CEO (Home Depot)"),
    ("JPM",  "Jamie Dimon",     "professional CEO (JPMorgan)"),
]


def _mk(rows: list, founder: bool) -> list[dict]:
    return [{"ticker": t, "leader": lead, "note": note, "founder": founder}
            for (t, lead, note) in rows]


FOUNDER = _mk(_FOUNDER_RAW, True)
PRO = _mk(_PRO_RAW, False)
FIRMS = FOUNDER + PRO
FOUNDER_TICKERS = [r["ticker"] for r in FOUNDER]
PRO_TICKERS = [r["ticker"] for r in PRO]
TICKERS = sorted({r["ticker"] for r in FIRMS})

FORMATION = pd.Timestamp("2016-01-01")   # membership frozen here; no look-ahead re-selection


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker, plus SPY
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_747_{safe}_1mo.parquet")


def fetch_prices(start: str = "2015-10-01", end: str | None = "2025-01-01",
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download monthly adjusted closes for every basket ticker + SPY and cache parquet.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. Adjusted closes fold in splits + dividends, so the series is a **total-return**
    proxy (labelled as such everywhere). One parquet per ticker (column ``close``, index
    ``date`` = month start).
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    for ticker in TICKERS + ["SPY"]:
        raw = yf.download(ticker, start=start, end=end, interval="1mo",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.empty or "close" not in raw.columns:
            continue
        out = raw[["close"]].copy()
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out.index.name = "date"
        out = out[~out.index.duplicated(keep="last")]
        out.to_parquet(_cache_path(ticker, cache_dir))


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff SPY and at least most basket tickers are cached."""
    if not os.path.exists(_cache_path("SPY", cache_dir)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in TICKERS)
    return have >= max(1, int(0.7 * len(TICKERS)))


def load_prices(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load cached monthly closes into a wide frame (index = month, columns = tickers)."""
    series = {}
    for ticker in TICKERS + ["SPY"]:
        p = _cache_path(ticker, cache_dir)
        if not os.path.exists(p):
            continue
        s = pd.read_parquet(p)["close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        s = s[~s.index.duplicated(keep="last")]
        series[ticker] = s
    df = pd.DataFrame(series).sort_index()
    return df


def monthly_returns(cache_dir: str = DEFAULT_CACHE,
                    end: str | None = "2024-12-31") -> pd.DataFrame:
    """Simple monthly total-returns for every cached ticker (drop the partial last bar).

    Adjusted-close ``pct_change``; restricted to closed months on/before ``end`` so a
    stamped run never includes a partial month.
    """
    px = load_prices(cache_dir)
    if end is not None:
        px = px.loc[px.index <= pd.Timestamp(end)]
    rets = px.pct_change().iloc[1:]
    # keep months from the formation date onward
    rets = rets.loc[rets.index >= FORMATION]
    return rets


def load_real(cache_dir: str = DEFAULT_CACHE,
              end: str | None = "2024-12-31") -> tuple[pd.DataFrame, list[dict]]:
    """Convenience: monthly-return frame + the firm table (only firms with data)."""
    rets = monthly_returns(cache_dir, end=end)
    firms = [f for f in FIRMS if f["ticker"] in rets.columns]
    return rets, firms


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_baskets(n_months: int = 108, n_founder: int = 13, n_pro: int = 13,
                      alpha_bps: float = 0.0, seed: int = 747,
                      mkt_mu: float = 0.008, mkt_sd: float = 0.045,
                      beta_founder: float = 1.20, beta_pro: float = 0.85,
                      idio_sd: float = 0.075) -> dict:
    """Deterministic monthly basket panels with a *plantable* founder alpha.

    Both baskets are ``beta * market + idiosyncratic``; the founder basket additionally
    earns ``alpha_bps`` basis points of monthly abnormal return. With ``alpha_bps = 0``
    there is no founder edge and the long/short CAPM alpha must NOT come out significant;
    with a large planted alpha it must. The higher founder beta mimics the tech tilt of a
    real founder basket, so the control also checks the alpha is separated from beta.

    Returns a dict with:
      ``founder``, ``pro``  — (n_months,) equal-weighted basket monthly returns
      ``mkt``               — (n_months,) market monthly returns
      ``ls``                — founder − pro
      ``truth``             — the planted parameters.
    """
    rng = np.random.default_rng(seed)
    a = alpha_bps * 1e-4
    mkt = rng.normal(mkt_mu, mkt_sd, n_months)

    def basket(beta: float, alpha: float, k: int) -> np.ndarray:
        # k names, each beta*mkt + idio; equal-weight average -> idio shrinks by ~1/sqrt(k)
        idio = rng.normal(0.0, idio_sd, (n_months, k))
        stock = beta * mkt[:, None] + idio + alpha
        return stock.mean(axis=1)

    founder = basket(beta_founder, a, n_founder)
    pro = basket(beta_pro, 0.0, n_pro)
    return {
        "founder": founder, "pro": pro, "mkt": mkt, "ls": founder - pro,
        "truth": {"n_months": n_months, "alpha_bps": alpha_bps, "seed": seed,
                  "beta_founder": beta_founder, "beta_pro": beta_pro},
    }


def fingerprint(firms: list[dict]) -> str:
    """Short content fingerprint of the basket membership, for as-of stamps."""
    key = "|".join(sorted(f"{f['ticker']}:{int(f['founder'])}" for f in firms))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
