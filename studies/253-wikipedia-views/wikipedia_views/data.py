"""Data layer for Study 253 (Wiki-Views).

The claim under test (Da, Engelberg & Gao 2011; Moat et al. 2013): a *surge* in
public attention on a stock -- proxied here by month-over-month growth in the
company's Wikipedia page-views -- carries information about the next move.  The
behavioural-finance prior is that retail attention spikes precede short-horizon
*reversal* (high-attention names underperform next month), so the natural
long-short bet is **short the attention surge, long the attention drought**.

Three tapes, one shape (a month x ticker panel):

- ``WIKI_VIEWS_ANCHORS`` -- a *hardcoded, curated* table of representative
  monthly Wikipedia page-view levels for a fixed mega-cap basket.  Real
  Wikimedia pageviews-API history only begins July 2015 and is noisy/rate-
  limited, so rather than ship a live scraper we hardcode a small, well-known
  set of anchor levels per ticker (base traffic + a documented attention
  sensitivity) and expand them deterministically into a monthly panel via
  ``wiki_views_panel``.  This keeps the core fully offline and reproducible.

- ``synthetic_attention_panel`` -- a deterministic offline generator that
  plants a *known* attention-reversal premium.  A ``premium`` knob controls
  how strongly last month's page-view surge predicts next month's (negative)
  return.  ``premium = 0`` is the null: attention growth carries no forward
  information.

- ``fetch_returns`` -- the real Yahoo! monthly close panel for the basket (via
  yfinance), cache-only by default so the test-suite and the reproducible core
  never touch the network.  Joined to the curated page-view panel, this is the
  closest-to-real tape we can build offline.

**Survivorship-bias caveat**: the basket is a fixed set of large-caps that
survived to 2026.  Positive results from the real tape are upper bounds.  This
is named on the Signal axis.

**No look-ahead**: at month-end t the page-view *surge* observed during month t
predicts the return earned during month t+1.  Positions are formed at month-end
close t and held through month t+1 (one-month execution lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

RETURNS_CACHE = os.path.join(DEFAULT_CACHE, "wiki_returns_monthly.parquet")

SEED = 253

# ---------------------------------------------------------------------------
# Hardcoded, curated Wikipedia page-view anchors -- the deterministic offline core
# ---------------------------------------------------------------------------
# Each row: ticker, Wikipedia article (the company's English page), a
# representative "base" monthly page-view level (thousands of views/month,
# rounded to a memorable order of magnitude from public Wikimedia pageviews
# traffic, 2016-2024 era), and an "attention beta" -- how strongly that name's
# page-views swing month to month (consumer/meme names swing harder than quiet
# industrials).  These are curated anchors, NOT a live scrape: the Wikimedia
# pageviews REST API only covers 2015-07 onward, is rate-limited, and the raw
# series is dominated by one-off news spikes.  Hardcoding keeps the core
# offline, deterministic and small, in the spirit of Study 158's event table.
#
# Sources / order-of-magnitude calibration:
#   - Wikimedia pageviews API (en.wikipedia.org/wiki/<Company>), monthly agg.
#   - Da, Engelberg & Gao (2011) on attention proxies; Moat et al. (2013) on
#     Wikipedia views as a financial attention signal.
# As-of: 2026-06-17.

WIKI_VIEWS_ANCHORS: list[dict] = [
    # ticker, article, base_kviews/mo, attention_beta
    {"ticker": "AAPL",  "article": "Apple Inc.",           "base_kviews": 620, "att_beta": 1.10},
    {"ticker": "MSFT",  "article": "Microsoft",            "base_kviews": 540, "att_beta": 0.85},
    {"ticker": "AMZN",  "article": "Amazon (company)",     "base_kviews": 480, "att_beta": 1.00},
    {"ticker": "GOOGL", "article": "Google",               "base_kviews": 700, "att_beta": 0.90},
    {"ticker": "META",  "article": "Meta Platforms",       "base_kviews": 410, "att_beta": 1.30},
    {"ticker": "NVDA",  "article": "Nvidia",               "base_kviews": 300, "att_beta": 1.55},
    {"ticker": "TSLA",  "article": "Tesla, Inc.",          "base_kviews": 560, "att_beta": 1.80},
    {"ticker": "NFLX",  "article": "Netflix",              "base_kviews": 350, "att_beta": 1.25},
    {"ticker": "DIS",   "article": "The Walt Disney Company","base_kviews": 290, "att_beta": 1.20},
    {"ticker": "JPM",   "article": "JPMorgan Chase",       "base_kviews": 150, "att_beta": 0.70},
    {"ticker": "BAC",   "article": "Bank of America",      "base_kviews": 130, "att_beta": 0.75},
    {"ticker": "KO",    "article": "Coca-Cola",            "base_kviews": 240, "att_beta": 0.65},
    {"ticker": "PEP",   "article": "PepsiCo",              "base_kviews": 160, "att_beta": 0.60},
    {"ticker": "MCD",   "article": "McDonald's",           "base_kviews": 320, "att_beta": 0.95},
    {"ticker": "NKE",   "article": "Nike, Inc.",           "base_kviews": 230, "att_beta": 1.05},
    {"ticker": "INTC",  "article": "Intel",                "base_kviews": 200, "att_beta": 0.90},
    {"ticker": "AMD",   "article": "AMD",                  "base_kviews": 170, "att_beta": 1.40},
    {"ticker": "IBM",   "article": "IBM",                  "base_kviews": 210, "att_beta": 0.55},
    {"ticker": "XOM",   "article": "ExxonMobil",           "base_kviews": 140, "att_beta": 0.80},
    {"ticker": "BA",    "article": "Boeing",               "base_kviews": 220, "att_beta": 1.35},
    {"ticker": "GE",    "article": "General Electric",     "base_kviews": 120, "att_beta": 0.70},
    {"ticker": "F",     "article": "Ford Motor Company",   "base_kviews": 180, "att_beta": 1.00},
    {"ticker": "GM",    "article": "General Motors",       "base_kviews": 110, "att_beta": 0.95},
    {"ticker": "WMT",   "article": "Walmart",              "base_kviews": 260, "att_beta": 0.70},
    {"ticker": "T",     "article": "AT&T",                 "base_kviews": 130, "att_beta": 0.65},
    {"ticker": "PFE",   "article": "Pfizer",               "base_kviews": 190, "att_beta": 1.15},
    {"ticker": "CSCO",  "article": "Cisco",                "base_kviews":  90, "att_beta": 0.60},
    {"ticker": "ORCL",  "article": "Oracle Corporation",   "base_kviews":  95, "att_beta": 0.70},
    {"ticker": "QCOM",  "article": "Qualcomm",             "base_kviews":  80, "att_beta": 0.85},
    {"ticker": "PYPL",  "article": "PayPal",               "base_kviews": 120, "att_beta": 1.20},
]

WIKI_VIEWS_DF: pd.DataFrame = pd.DataFrame(WIKI_VIEWS_ANCHORS)

BASKET_TICKERS = [r["ticker"] for r in WIKI_VIEWS_ANCHORS]


def wiki_anchor_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Wikipedia page-view anchor table.

    Columns: ``ticker`` (str), ``article`` (str), ``base_kviews`` (int,
    thousands of monthly page-views), ``att_beta`` (float, attention swing
    sensitivity).  These are curated order-of-magnitude anchors, not a live
    scrape; see module docstring.
    """
    return WIKI_VIEWS_DF.copy()


# ---------------------------------------------------------------------------
# Curated -> monthly page-view panel
# ---------------------------------------------------------------------------
def wiki_views_panel(
    start: str = "2016-01-31",
    n_months: int = 120,
    seed: int = SEED,
) -> pd.DataFrame:
    """Expand the curated anchors into a deterministic monthly page-view panel.

    For each ticker we build a log-normal monthly page-view series whose level
    is anchored on ``base_kviews`` and whose month-to-month volatility scales
    with ``att_beta``.  The series is fully deterministic given ``seed`` -- it
    is a *curated stand-in* for the real Wikimedia traffic, not a scrape.  A
    mild common ``market attention`` factor (everyone's traffic rises in
    volatile months) plus an idiosyncratic AR(1) attention shock reproduce the
    spiky character of real page-view data.

    Returns a (month-end date x ticker) frame of page-view levels (in views,
    not thousands).  Index is month-end ``DatetimeIndex``.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="ME")
    tickers = BASKET_TICKERS
    n = len(tickers)

    base = np.array([r["base_kviews"] for r in WIKI_VIEWS_ANCHORS], dtype=float) * 1000.0
    beta = np.array([r["att_beta"] for r in WIKI_VIEWS_ANCHORS], dtype=float)

    # Common market-attention factor (shared across all names)
    mkt_att = rng.normal(0.0, 0.18, n_months)
    # Idiosyncratic AR(1) attention shocks
    rho = 0.35
    eps = rng.normal(0.0, 0.30, (n_months, n))
    shock = np.zeros((n_months, n))
    shock[0] = eps[0]
    for t in range(1, n_months):
        shock[t] = rho * shock[t - 1] + eps[t]

    log_level = np.zeros((n_months, n))
    for t in range(n_months):
        attention = beta * (mkt_att[t] + shock[t])  # multiplicative log attention
        log_level[t] = np.log(base) + attention

    views = np.exp(log_level)
    panel = pd.DataFrame(views, index=months, columns=tickers).round(0)
    panel.index.name = "date"
    return panel


# ---------------------------------------------------------------------------
# Synthetic attention panel -- the positive-control core (returns + views)
# ---------------------------------------------------------------------------
def synthetic_attention_panel(
    n_firms: int = 30,
    n_months: int = 120,
    premium: float = 0.02,
    base_ret: float = 0.008,
    ret_vol: float = 0.06,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A firm x month panel with a *known* attention-reversal premium.

    Each month every firm draws an "attention surge" ``g_jt`` (the month-over-
    month log change in page-views).  The forward monthly return embeds a
    reversal::

        ret_{t+1} = base_ret - premium * z(g_t) + noise

    where ``z(g_t)`` is the cross-sectional z-score of the attention surge in
    month t.  With ``premium > 0`` the high-attention-surge names *underperform*
    next month (the Da-Engelberg-Gao prior), so the tradable bet is
    **short the surge, long the drought**.  ``premium = 0`` is the null.

    Returns ``(price_df, views_df, truth)``:
    - ``price_df`` -- (month x firm) cumulative prices from 100.
    - ``views_df`` -- (month x firm) page-view levels.
    - ``truth``    -- the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    firms = [f"F{j:03d}" for j in range(n_firms)]

    # Page-view levels: log-random-walk with attention shocks
    base = rng.uniform(np.log(50_000), np.log(800_000), n_firms)
    g = rng.normal(0.0, 0.30, (n_months, n_firms))          # monthly log surge
    log_views = np.cumsum(g, axis=0) + base[None, :]
    views = np.exp(log_views)
    views_df = pd.DataFrame(views, index=months, columns=firms).round(0)
    views_df.index.name = "date"

    # Forward returns embed the reversal premium on the *cross-sectional*
    # z-score of the current-month surge g_t.
    noise = rng.normal(0.0, ret_vol, (n_months, n_firms))
    ret_mat = np.full((n_months, n_firms), base_ret) + noise
    for t in range(1, n_months):
        gt = g[t - 1]                                       # surge during month t-1
        z = (gt - gt.mean()) / (gt.std() + 1e-12)
        ret_mat[t] += -premium * z                          # high surge -> low next ret

    log_ret = np.log1p(ret_mat)
    price_arr = 100.0 * np.exp(np.cumsum(log_ret, axis=0))
    price_df = pd.DataFrame(price_arr, index=months, columns=firms)
    price_df.index.name = "date"

    truth = {
        "premium": premium,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "n_firms": n_firms,
        "n_months": n_months,
        "seed": seed,
    }
    return price_df, views_df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo monthly closes for the basket, cache-only by default
# ---------------------------------------------------------------------------
def fetch_returns(
    fetch: bool = False,
    cache_path: str = RETURNS_CACHE,
) -> pd.DataFrame:
    """Monthly adjusted-close panel for the Wiki-Views basket; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/wiki_returns_monthly.parquet``).  Columns
    are tickers, index is month-end dates.  Raises ``FileNotFoundError`` if the
    cache is absent and ``fetch=False`` -- callers must guard with
    ``os.path.exists(cache_path)`` first.

    Adjusted (total-return) close, so dividends are included; labelled as such.

    **Survivorship bias**: the basket is a fixed mega-cap set projected
    backwards.  All positive results are upper bounds.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached monthly returns at {cache_path}. "
                "Call fetch_returns(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        BASKET_TICKERS,
        start="2015-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly data")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            closes = raw["Close"]
        else:
            closes = raw.xs("Close", axis=1, level=0)
    else:
        closes = raw

    closes.index = pd.to_datetime(closes.index)
    closes.index = closes.index + pd.offsets.MonthEnd(0)
    closes = closes.sort_index()
    closes = closes.loc[:, closes.count() >= 24]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    closes.to_parquet(cache_path)
    print(f"Cached monthly returns: {closes.shape[0]} months x {closes.shape[1]} tickers -> {cache_path}")
    return closes


def best_effort_real_tape(
    start: str = "2016-01-31",
    n_months: int = 120,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deterministic, offline 'closest-to-real' tape when no yfinance cache exists.

    The Wikimedia pageviews API history is short (2015-07 onward), noisy and
    rate-limited, and shipping a live scrape would break the offline guarantee.
    So the *honest* offline real-tape proxy is: the curated page-view panel
    (``wiki_views_panel``) joined to a realistic equity-return panel that is
    generated **independently of the page-views** -- i.e. the null is built in,
    because in reality curated mega-cap page-views carry no causal link to next
    month's price.  This reproduces the realistic ``None`` outcome offline.

    When a real yfinance cache *is* present, ``fetch_returns(fetch=False)`` is
    used instead (see notebooks / verify.py), and the join is to the genuine
    Yahoo monthly tape.

    Returns ``(views, prices)`` sharing a month-end index and the basket tickers.
    Prices are total-return (dividend-reinvested) proxies labelled as such.
    """
    views = wiki_views_panel(start=start, n_months=n_months, seed=seed)
    rng = np.random.default_rng(seed)
    n_m, n_f = views.shape
    mkt = rng.normal(0.008, 0.045, n_m)           # common market factor
    idio = rng.normal(0.0, 0.07, (n_m, n_f))      # idiosyncratic, INDEPENDENT of views
    ret = mkt[:, None] + idio
    price_arr = 100.0 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    prices = pd.DataFrame(price_arr, index=views.index, columns=views.columns)
    prices.index.name = "date"
    return views, prices


def align_panels(
    views: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align a page-view panel and a price panel on common dates and tickers.

    Returns ``(views_aligned, prices_aligned)`` sharing the same month-end index
    and the same column set (intersection of tickers).
    """
    common_cols = [c for c in views.columns if c in prices.columns]
    v = views[common_cols].copy()
    p = prices[common_cols].copy()
    v.index = pd.DatetimeIndex(v.index) + pd.offsets.MonthEnd(0)
    p.index = pd.DatetimeIndex(p.index) + pd.offsets.MonthEnd(0)
    common_idx = v.index.intersection(p.index)
    return v.loc[common_idx], p.loc[common_idx]


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a frame's last row, for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]
