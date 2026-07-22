"""Data layer for Study 801 — Employee Satisfaction ("100 Best Companies" alpha).

Two ingredients, both offline-friendly once cached.

* **Real tape.** A hand-coded, transparent, *cited* basket (``BASKET``) of publicly-listed
  companies that are **perennial** members of Fortune's *"100 Best Companies to Work For"*
  (published annually with Great Place to Work since 1998) and its sibling *World's Best
  Workplaces* list — plus SPY as the market and a ``CONTROL`` universe of large-cap
  survivors **not** selected for workplace prestige, used as a random-basket placebo pool.
  Daily total-return (auto-adjusted) closes come from yfinance (no key), cached as one
  parquet per ticker under this study's own ``_cache/``. From those we build **monthly
  total returns**, an equal-weight basket, and a **market-model (CAPM) alpha** of the
  basket vs SPY.

  **SURVIVORSHIP, said out loud (and named on the Signal axis).** This basket is
  *hand-picked from today's known winners*: companies that stayed listed, stayed public,
  and stayed great-to-work-for long enough for us to recognise them in 2026. That is a
  textbook survivorship selection and it biases the raw comparison **toward** the basket.
  Two famous perennial members left the tape entirely and are **excluded** (:data:`DELISTED`):
  Whole Foods (acquired by Amazon, 2017) and Ultimate Software (taken private, 2019) —
  both at acquisition *premiums*, so dropping them is a mild bias in the *other* direction,
  but it does not undo the dominant winners-only selection. We therefore lean the verdict
  toward WEAK unless the risk-adjusted result is overwhelming, and we size the survivorship
  directly with a **random large-cap-survivor basket placebo** (does a random basket of
  known survivors beat the market just as well?).

  **This is a CAPM (market-model) alpha, not Edmans' 4-factor alpha.** We control only for
  the market. A tech-tilted survivor basket also loads on size/value/momentum/quality
  factors we do *not* strip out — so a positive market-model alpha is an **upper bound** on
  any true "satisfaction" alpha, stated as such.

* **Synthetic.** A deterministic, fixed-seed generator that builds a monthly market series
  and an equal-weight basket of ``beta*mkt + idio`` names with a **plantable monthly alpha**
  (``alpha_bps``). It is the positive control: at ``alpha_bps = 0`` the market-model detector
  must NOT manufacture a significant alpha; with a large planted alpha it must light up.

Pure numpy + pandas + stdlib for the offline path. ``fetch_prices`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

START = "2015-06-01"        # fetch start (all basket names trade well before 2016-01)
BASKET_START = "2016-01-01"  # first month of the equal-weight basket (clean full history)
AS_OF = "2026-06-30"        # last complete calendar month at publication (2026-07-22)

# --------------------------------------------------------------------------- #
# The hand-coded basket — publicly-listed PERENNIAL "100 Best Companies to Work
# For" / "World's Best Workplaces" members (Fortune x Great Place to Work).
# The comment on each line is the (public, citable) reason it is a perennial member.
# All names trade continuously from before 2016-01, so the equal-weight basket has a
# clean common history. Deliberately balanced ~half tech / half non-tech so the result
# is not purely a 2016-2026 megacap-tech run.
# --------------------------------------------------------------------------- #
BASKET = [
    # --- non-tech perennials ---
    ("MAR",   "Marriott — on the '100 Best' list every year since the 1998 inception"),
    ("AXP",   "American Express — repeat '100 Best' / World's Best member"),
    ("AFL",   "Aflac — 20+ consecutive years on the '100 Best' list"),
    ("GS",    "Goldman Sachs — repeat '100 Best' member"),
    ("HLT",   "Hilton — #1 World's Best Workplace 2019; perennial top ranks"),
    ("CPT",   "Camden Property Trust — perennial '100 Best' REIT (top-10 repeatedly)"),
    ("ACN",   "Accenture — repeat World's Best Workplaces member"),
    ("SBUX",  "Starbucks — repeat '100 Best' member (benefits/culture)"),
    # --- tech perennials ---
    ("CSCO",  "Cisco — #1 World's Best Workplace 2019-2021; perennial"),
    ("CRM",   "Salesforce — top-10 '100 Best' repeatedly since ~2010"),
    ("GOOGL", "Alphabet/Google — #1 '100 Best' 2007-2012; long top-ranked"),
    ("INTU",  "Intuit — perennial '100 Best' member"),
    ("ADBE",  "Adobe — repeat '100 Best' member"),
    ("NVDA",  "NVIDIA — repeat '100 Best' member (recent years)"),
    ("WDAY",  "Workday — repeat '100 Best' member since its 2012 IPO era"),
    ("NOW",   "ServiceNow — repeat 'best places to work' member"),
    ("CDNS",  "Cadence Design Systems — repeat '100 Best' member (recent years)"),
]

# Famous perennial members that LEFT the tape (delisted / taken private) and are excluded.
# Named for the survivorship caveat. Both left at acquisition PREMIUMS, so excluding them
# drops gains (a mild bias AGAINST the basket) — but the dominant bias is winners-only.
DELISTED = [
    "Whole Foods Market (WFM) — long-time '100 Best' member; acquired by Amazon 2017 (+27% pop)",
    "Ultimate Software (ULTI) — perennial top-ranked; taken private by Hellman & Friedman 2019",
    "Nordstrom (JWN) — many consecutive '100 Best' years; taken private by the family + Liverpool 2025",
    "SAS Institute, Wegmans, Publix, Edward Jones, Kimpton — perennial members but PRIVATE (no ticker)",
]

# Large-cap survivors NOT selected for workplace prestige — the random-basket placebo pool.
# Every one is still listed (they, too, are survivors), so a random basket drawn from here
# measures the pure survivorship yardstick: does ANY basket of known large-cap survivors beat
# the market as well as our satisfaction basket does? All trade continuously from before 2016.
CONTROL = [
    "JPM", "BAC", "WFC", "C", "XOM", "CVX", "WMT", "PG", "KO", "PEP",
    "JNJ", "PFE", "MRK", "HD", "LOW", "MCD", "VZ", "T", "IBM", "ORCL",
    "INTC", "CAT", "BA", "HON", "UNH", "CVS", "DIS", "NKE", "UPS", "FDX",
    "DE", "LMT", "MMM", "GE",
]

BASKET_TICKERS = [t for t, _ in BASKET]
MARKET = "SPY"
ALL_TICKERS = sorted(set(BASKET_TICKERS) | set(CONTROL) | {MARKET})


# --------------------------------------------------------------------------- #
# Real tape (network) — one parquet per ticker
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_801_{safe}_1d.parquet")


def fetch_prices(start: str = START, end: str | None = None,
                 cache_dir: str = DEFAULT_CACHE) -> None:
    """Download daily total-return (auto-adjusted) closes for every ticker; cache parquet.

    Network-only; used once to build ``_cache/``. Never imported by the offline notebook
    cells. One parquet per ticker (column ``close``, index ``date``).
    """
    import yfinance as yf

    os.makedirs(cache_dir, exist_ok=True)
    for ticker in ALL_TICKERS:
        raw = yf.download(ticker, start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        if raw.empty or "close" not in raw.columns:
            continue
        out = raw[["close"]].copy()
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out.index.name = "date"
        out.to_parquet(_cache_path(ticker, cache_dir))


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff SPY and most basket tickers are cached."""
    if not os.path.exists(_cache_path(MARKET, cache_dir)):
        return False
    have = sum(os.path.exists(_cache_path(t, cache_dir)) for t in BASKET_TICKERS)
    return have >= max(1, int(0.8 * len(BASKET_TICKERS)))


def load_prices(tickers: list[str], cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load cached daily closes for ``tickers`` into a wide frame (index=date)."""
    series = {}
    for ticker in tickers:
        p = _cache_path(ticker, cache_dir)
        if not os.path.exists(p):
            continue
        s = pd.read_parquet(p)["close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None)
        series[ticker] = s
    return pd.DataFrame(series).sort_index()


def monthly_total_returns(prices: pd.DataFrame, start: str = BASKET_START,
                          asof: str = AS_OF) -> pd.DataFrame:
    """Month-end total returns per ticker over [start, asof], last complete month only.

    Daily auto-adjusted closes -> month-end last -> pct_change. The first (NaN) month is
    dropped; the sample ends at ``asof`` (a real past month-end), so no partial month.
    """
    px = prices.loc[(prices.index >= pd.Timestamp(start) - pd.Timedelta(days=40)) &
                    (prices.index <= pd.Timestamp(asof))]
    m = px.resample("ME").last()
    r = m.pct_change().dropna(how="all")
    return r.loc[r.index >= pd.Timestamp(start)]


def load_real(cache_dir: str = DEFAULT_CACHE):
    """Convenience: monthly returns for (basket names present, market, control names present).

    Returns ``(monthly, basket_present, control_present)`` where ``monthly`` is the wide
    month-end total-return frame including SPY.
    """
    prices = load_prices(ALL_TICKERS, cache_dir)
    monthly = monthly_total_returns(prices)
    basket_present = [t for t in BASKET_TICKERS if t in monthly.columns]
    control_present = [t for t in CONTROL if t in monthly.columns]
    return monthly, basket_present, control_present


# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
def synthetic_world(alpha_bps: float = 0.0, seed: int = 801, n_months: int = 126,
                    n_names: int = 18, beta: float = 1.05,
                    mkt_mu: float = 0.007, mkt_sd: float = 0.042,
                    idio_sd: float = 0.055) -> dict:
    """Deterministic monthly market + equal-weight basket with a PLANTABLE monthly alpha.

    The market series is i.i.d.-drift monthly returns. Each of ``n_names`` basket names is
    ``alpha + beta*mkt + idio`` with a common **planted monthly alpha** of ``alpha_bps``
    basis points added to every name (the believers' "satisfaction lifts returns" effect the
    market-model detector must recover). ``alpha_bps = 0`` is the null world: the basket is
    pure beta plus noise, and the CAPM-alpha t must NOT clear the bar.

    Returns a dict with the market series, the equal-weight basket series, and the truth.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-31", periods=n_months, freq="ME")
    mkt = rng.normal(mkt_mu, mkt_sd, n_months)
    a = alpha_bps * 1e-4
    names = np.empty((n_months, n_names))
    for j in range(n_names):
        names[:, j] = a + beta * mkt + rng.normal(0.0, idio_sd, n_months)
    basket = names.mean(axis=1)
    return {
        "market": pd.Series(mkt, index=idx, name="SPY"),
        "basket": pd.Series(basket, index=idx, name="basket"),
        "truth": {"alpha_bps": alpha_bps, "beta": beta, "seed": seed,
                  "n_months": n_months, "n_names": n_names},
    }


def fingerprint(monthly: pd.DataFrame, cols: list[str]) -> str:
    """Short content fingerprint of the monthly return matrix over ``cols`` (as-of stamp)."""
    sub = monthly[sorted(c for c in cols if c in monthly.columns)]
    h = hashlib.sha256()
    for ts in sub.index:
        h.update(str(pd.Timestamp(ts).date()).encode())
        h.update(b"\x1f")
    for c in sub.columns:
        h.update(str(c).encode())
        for v in sub[c].to_numpy():
            token = "nan" if pd.isna(v) else f"{round(float(v), 6):.6f}"
            h.update(token.encode())
            h.update(b",")
    return h.hexdigest()[:12]
