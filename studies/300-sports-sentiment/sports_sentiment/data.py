"""Data layer for Study 300 (Sports-Sentiment).

The Edmans, Garcia & Norli (2007, *Journal of Finance*) "loss effect": after a
national soccer team is *eliminated* from a major tournament (World Cup, Euros,
Copa America), the home country's stock market shows a significantly negative
abnormal return the *next trading day* (~ -49 bps raw, ~ -39 bps abnormal in
their cross-section). The mechanism is investor mood: a painful, salient loss
depresses risk appetite nationwide.

Three components, the first two fully offline:

- ``ELIMINATION_EVENTS`` -- a hardcoded table of major national-team knockout
  *losses* (the elimination game) for a panel of countries, with each country's
  investable single-country ETF ticker. The event date is the match date; the
  *next trading day* is when the mood effect should print. Curated from
  tournament records (FIFA, UEFA, CONMEBOL). Deterministic and offline.

- ``synthetic_panel`` -- a deterministic offline generator of (event, next-day
  return) pairs with an optional planted "loss premium" knob (in bps). With
  ``loss_bps = 0`` it is the null: elimination carries no information.

- ``fetch_event_returns`` -- the REAL tape. For each elimination event it loads
  the matching country-ETF (or ^GSPC for the US proxy) daily prices from the
  per-ticker parquet cache under ``_cache/`` and computes the *next trading day*
  simple return after the loss. Cache-only by default (``fetch=False``); the
  network (lazy ``yfinance`` import) is touched only on explicit ``fetch=True``.

No look-ahead: the elimination match is an evening event; the tradable response
is the FULL next trading session, so we use the close-to-close return of the
first trading day strictly after the match date (an explicit one-day execution
lag). Returns are PRICE-only (ETF/index price, no dividends) -- labeled as such.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SEED = 300

# ---------------------------------------------------------------------------
# The hardcoded elimination table -- the study's deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   event_id   : sequential id
#   country    : national team eliminated
#   etf        : investable single-country proxy (yfinance ticker); 'US' uses ^GSPC
#   tournament : 'WC' (World Cup), 'EURO', 'COPA'
#   match_date : ISO date of the elimination match (the loss)
#   stage      : knockout stage of the loss
#   shootout   : True if the loss came via penalty shootout (the most painful)
#   note       : opponent / score for traceability
#
# These are *elimination losses* only -- the game that knocked the team out.
# Edmans et al. (2007) find the effect concentrated in losses (asymmetric: wins
# do NOT produce a symmetric positive effect). We curate marquee knockout losses
# for a panel of liquid single-country ETF markets, 1996-2024.
#
# Sources: FIFA World Cup records, UEFA European Championship records, CONMEBOL
# Copa America records (match dates and outcomes are historical facts). ETF
# tickers: iShares MSCI single-country funds (EWU UK, EWG Germany, EWQ France,
# EWI Italy, EWP Spain, EWZ Brazil, EWW Mexico, EWA Australia, EWJ Japan,
# EWN Netherlands, EWL Switzerland), ^GSPC for the US.
#
# As-of: 2026-06-18.

ELIMINATION_EVENTS: list[dict] = [
    # --- World Cups ---
    {"event_id": 1,  "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "1998-06-30", "stage": "R16",    "shootout": True,  "note": "lost to Argentina on penalties"},
    {"event_id": 2,  "country": "Italy",       "etf": "EWI", "tournament": "WC",   "match_date": "1998-07-03", "stage": "QF",     "shootout": True,  "note": "lost to France on penalties"},
    {"event_id": 3,  "country": "Netherlands", "etf": "EWN", "tournament": "WC",   "match_date": "1998-07-07", "stage": "SF",     "shootout": True,  "note": "lost to Brazil on penalties"},
    {"event_id": 4,  "country": "Brazil",      "etf": "EWZ", "tournament": "WC",   "match_date": "1998-07-12", "stage": "Final",  "shootout": False, "note": "lost final 0-3 to France"},
    {"event_id": 5,  "country": "France",      "etf": "EWQ", "tournament": "WC",   "match_date": "2002-06-11", "stage": "Group",  "shootout": False, "note": "eliminated in group stage (holders)"},
    {"event_id": 6,  "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "2002-06-21", "stage": "QF",     "shootout": False, "note": "lost to Brazil 1-2"},
    {"event_id": 7,  "country": "Spain",       "etf": "EWP", "tournament": "WC",   "match_date": "2002-06-22", "stage": "QF",     "shootout": True,  "note": "lost to South Korea on penalties"},
    {"event_id": 8,  "country": "Italy",       "etf": "EWI", "tournament": "WC",   "match_date": "2002-06-18", "stage": "R16",    "shootout": False, "note": "lost to South Korea (golden goal)"},
    {"event_id": 9,  "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "2006-07-01", "stage": "QF",     "shootout": True,  "note": "lost to Portugal on penalties"},
    {"event_id": 10, "country": "Brazil",      "etf": "EWZ", "tournament": "WC",   "match_date": "2006-07-01", "stage": "QF",     "shootout": False, "note": "lost to France 0-1"},
    {"event_id": 11, "country": "Germany",     "etf": "EWG", "tournament": "WC",   "match_date": "2006-07-04", "stage": "SF",     "shootout": False, "note": "lost to Italy (extra time)"},
    {"event_id": 12, "country": "France",      "etf": "EWQ", "tournament": "WC",   "match_date": "2006-07-09", "stage": "Final",  "shootout": True,  "note": "lost final to Italy on penalties"},
    {"event_id": 13, "country": "France",      "etf": "EWQ", "tournament": "WC",   "match_date": "2010-06-22", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 14, "country": "Italy",       "etf": "EWI", "tournament": "WC",   "match_date": "2010-06-24", "stage": "Group",  "shootout": False, "note": "eliminated in group stage (holders)"},
    {"event_id": 15, "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "2010-06-27", "stage": "R16",    "shootout": False, "note": "lost to Germany 1-4"},
    {"event_id": 16, "country": "Brazil",      "etf": "EWZ", "tournament": "WC",   "match_date": "2010-07-02", "stage": "QF",     "shootout": False, "note": "lost to Netherlands 1-2"},
    {"event_id": 17, "country": "Netherlands", "etf": "EWN", "tournament": "WC",   "match_date": "2010-07-11", "stage": "Final",  "shootout": False, "note": "lost final to Spain (extra time)"},
    {"event_id": 18, "country": "Spain",       "etf": "EWP", "tournament": "WC",   "match_date": "2014-06-18", "stage": "Group",  "shootout": False, "note": "eliminated in group stage (holders)"},
    {"event_id": 19, "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "2014-06-19", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 20, "country": "Italy",       "etf": "EWI", "tournament": "WC",   "match_date": "2014-06-24", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 21, "country": "Brazil",      "etf": "EWZ", "tournament": "WC",   "match_date": "2014-07-08", "stage": "SF",     "shootout": False, "note": "lost 1-7 to Germany (Mineirazo)"},
    {"event_id": 22, "country": "Netherlands", "etf": "EWN", "tournament": "WC",   "match_date": "2014-07-09", "stage": "SF",     "shootout": True,  "note": "lost to Argentina on penalties"},
    {"event_id": 23, "country": "Germany",     "etf": "EWG", "tournament": "WC",   "match_date": "2018-06-27", "stage": "Group",  "shootout": False, "note": "eliminated in group stage (holders)"},
    {"event_id": 24, "country": "Spain",       "etf": "EWP", "tournament": "WC",   "match_date": "2018-07-01", "stage": "R16",    "shootout": True,  "note": "lost to Russia on penalties"},
    {"event_id": 25, "country": "Mexico",      "etf": "EWW", "tournament": "WC",   "match_date": "2018-07-02", "stage": "R16",    "shootout": False, "note": "lost to Brazil 0-2"},
    {"event_id": 26, "country": "Brazil",      "etf": "EWZ", "tournament": "WC",   "match_date": "2018-07-06", "stage": "QF",     "shootout": False, "note": "lost to Belgium 1-2"},
    {"event_id": 27, "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "2018-07-11", "stage": "SF",     "shootout": False, "note": "lost to Croatia (extra time)"},
    {"event_id": 28, "country": "Germany",     "etf": "EWG", "tournament": "WC",   "match_date": "2022-12-01", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 29, "country": "Spain",       "etf": "EWP", "tournament": "WC",   "match_date": "2022-12-06", "stage": "R16",    "shootout": True,  "note": "lost to Morocco on penalties"},
    {"event_id": 30, "country": "Brazil",      "etf": "EWZ", "tournament": "WC",   "match_date": "2022-12-09", "stage": "QF",     "shootout": True,  "note": "lost to Croatia on penalties"},
    {"event_id": 31, "country": "Netherlands", "etf": "EWN", "tournament": "WC",   "match_date": "2022-12-09", "stage": "QF",     "shootout": True,  "note": "lost to Argentina on penalties"},
    {"event_id": 32, "country": "England",     "etf": "EWU", "tournament": "WC",   "match_date": "2022-12-10", "stage": "QF",     "shootout": False, "note": "lost to France 1-2"},
    {"event_id": 33, "country": "France",      "etf": "EWQ", "tournament": "WC",   "match_date": "2022-12-18", "stage": "Final",  "shootout": True,  "note": "lost final to Argentina on penalties"},
    # --- European Championships ---
    {"event_id": 34, "country": "England",     "etf": "EWU", "tournament": "EURO", "match_date": "2004-06-24", "stage": "QF",     "shootout": True,  "note": "lost to Portugal on penalties"},
    {"event_id": 35, "country": "Italy",       "etf": "EWI", "tournament": "EURO", "match_date": "2004-06-22", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 36, "country": "Spain",       "etf": "EWP", "tournament": "EURO", "match_date": "2004-06-20", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 37, "country": "Germany",     "etf": "EWG", "tournament": "EURO", "match_date": "2008-06-29", "stage": "Final",  "shootout": False, "note": "lost final to Spain 0-1"},
    {"event_id": 38, "country": "Netherlands", "etf": "EWN", "tournament": "EURO", "match_date": "2008-06-21", "stage": "QF",     "shootout": False, "note": "lost to Russia (extra time)"},
    {"event_id": 39, "country": "France",      "etf": "EWQ", "tournament": "EURO", "match_date": "2008-06-17", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 40, "country": "England",     "etf": "EWU", "tournament": "EURO", "match_date": "2012-06-24", "stage": "QF",     "shootout": True,  "note": "lost to Italy on penalties"},
    {"event_id": 41, "country": "Germany",     "etf": "EWG", "tournament": "EURO", "match_date": "2012-06-28", "stage": "SF",     "shootout": False, "note": "lost to Italy 1-2"},
    {"event_id": 42, "country": "Italy",       "etf": "EWI", "tournament": "EURO", "match_date": "2012-07-01", "stage": "Final",  "shootout": False, "note": "lost final to Spain 0-4"},
    {"event_id": 43, "country": "France",      "etf": "EWQ", "tournament": "EURO", "match_date": "2016-07-10", "stage": "Final",  "shootout": False, "note": "lost final to Portugal (extra time)"},
    {"event_id": 44, "country": "Germany",     "etf": "EWG", "tournament": "EURO", "match_date": "2016-07-07", "stage": "SF",     "shootout": False, "note": "lost to France 0-2"},
    {"event_id": 45, "country": "England",     "etf": "EWU", "tournament": "EURO", "match_date": "2016-06-27", "stage": "R16",    "shootout": False, "note": "lost to Iceland 1-2"},
    {"event_id": 46, "country": "Spain",       "etf": "EWP", "tournament": "EURO", "match_date": "2016-06-27", "stage": "R16",    "shootout": False, "note": "lost to Italy 0-2"},
    {"event_id": 47, "country": "England",     "etf": "EWU", "tournament": "EURO", "match_date": "2021-07-11", "stage": "Final",  "shootout": True,  "note": "lost final to Italy on penalties"},
    {"event_id": 48, "country": "Germany",     "etf": "EWG", "tournament": "EURO", "match_date": "2021-06-29", "stage": "R16",    "shootout": False, "note": "lost to England 0-2"},
    {"event_id": 49, "country": "France",      "etf": "EWQ", "tournament": "EURO", "match_date": "2021-06-28", "stage": "R16",    "shootout": True,  "note": "lost to Switzerland on penalties"},
    {"event_id": 50, "country": "Spain",       "etf": "EWP", "tournament": "EURO", "match_date": "2021-07-06", "stage": "SF",     "shootout": True,  "note": "lost to Italy on penalties"},
    {"event_id": 51, "country": "Italy",       "etf": "EWI", "tournament": "EURO", "match_date": "2024-07-02", "stage": "R16",    "shootout": False, "note": "lost to Switzerland 0-2 (holders)"},
    {"event_id": 52, "country": "Germany",     "etf": "EWG", "tournament": "EURO", "match_date": "2024-07-05", "stage": "QF",     "shootout": False, "note": "lost to Spain (extra time, hosts)"},
    {"event_id": 53, "country": "Netherlands", "etf": "EWN", "tournament": "EURO", "match_date": "2024-07-10", "stage": "SF",     "shootout": False, "note": "lost to England 1-2"},
    {"event_id": 54, "country": "France",      "etf": "EWQ", "tournament": "EURO", "match_date": "2024-07-09", "stage": "SF",     "shootout": False, "note": "lost to Spain 1-2"},
    {"event_id": 55, "country": "England",     "etf": "EWU", "tournament": "EURO", "match_date": "2024-07-14", "stage": "Final",  "shootout": False, "note": "lost final to Spain 1-2"},
    # --- Copa America (Brazil / Mexico exposure) ---
    {"event_id": 56, "country": "Brazil",      "etf": "EWZ", "tournament": "COPA", "match_date": "2019-06-22", "stage": "Group",  "shootout": False, "note": "(context loss, group play)"},
    {"event_id": 57, "country": "Brazil",      "etf": "EWZ", "tournament": "COPA", "match_date": "2021-07-10", "stage": "Final",  "shootout": False, "note": "lost final to Argentina 0-1"},
    {"event_id": 58, "country": "Mexico",      "etf": "EWW", "tournament": "COPA", "match_date": "2024-06-30", "stage": "Group",  "shootout": False, "note": "eliminated in group stage"},
    {"event_id": 59, "country": "Brazil",      "etf": "EWZ", "tournament": "COPA", "match_date": "2024-07-06", "stage": "QF",     "shootout": True,  "note": "lost to Uruguay on penalties"},
    {"event_id": 60, "country": "Argentina",   "etf": "EWZ", "tournament": "WC",   "match_date": "2018-06-30", "stage": "R16",    "shootout": False, "note": "lost to France 3-4 (EWZ regional proxy)"},
]

ELIMINATION_DF: pd.DataFrame = pd.DataFrame(ELIMINATION_EVENTS)


def elimination_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded elimination-loss table.

    Columns: ``event_id`` (int), ``country`` (str), ``etf`` (str ticker),
    ``tournament`` (str), ``match_date`` (str ISO date), ``stage`` (str),
    ``shootout`` (bool), ``note`` (str).
    """
    return ELIMINATION_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_events: int = 60,
    loss_bps: float = 0.0,
    base_mu_bps: float = 3.0,
    vol_bps: float = 120.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible panel of (event, next-day return) pairs.

    Each "next-day" return is drawn i.i.d. from N(base_mu, vol^2) in bps and then
    shifted DOWN by ``loss_bps`` (the planted loss effect). ``loss_bps = 0`` is
    the null -- elimination carries no information and the next-day mean equals
    the ordinary daily drift. The Edmans et al. (2007) point estimate is roughly
    ``loss_bps = 49`` (a ~49 bps next-day drop).

    Returns ``(df, truth)`` where ``df`` has columns ``event_id`` (int),
    ``next_day_ret`` (simple return), and ``truth`` records the planted params.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(base_mu_bps * 1e-4, vol_bps * 1e-4, n_events)
    rets = base - loss_bps * 1e-4
    df = pd.DataFrame({
        "event_id": np.arange(1, n_events + 1),
        "next_day_ret": rets,
    })
    truth = {
        "n_events": n_events,
        "loss_bps": loss_bps,
        "base_mu_bps": base_mu_bps,
        "vol_bps": vol_bps,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- per-ticker daily parquet cache (cache-only by default)
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("^", "").replace("/", "_")
    return os.path.join(cache_dir, f"sports_{safe}_daily.parquet")


def _load_prices(ticker: str, cache_dir: str, fetch: bool) -> pd.Series:
    """Load a ticker's daily close price series from cache, or fetch if allowed.

    Returns a pd.Series indexed by date (DatetimeIndex) of close prices. Raises
    FileNotFoundError if the cache is absent and ``fetch`` is False.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        s = pd.read_parquet(path)
        col = s.columns[0]
        out = s[col]
        out.index = pd.to_datetime(out.index)
        return out.sort_index()
    if not fetch:
        raise FileNotFoundError(
            f"No cache for {ticker} at {path}. Run with fetch=True to download "
            "(network), or stage the parquet cache."
        )
    import yfinance as yf  # lazy import -- only on explicit fetch

    yf_tkr = "^GSPC" if ticker in ("US", "^GSPC", "GSPC") else ticker
    raw = yf.download(yf_tkr, start="1995-01-01", end="2025-01-01",
                      auto_adjust=False, progress=False)
    px = raw["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    os.makedirs(cache_dir, exist_ok=True)
    px.to_frame("close").to_parquet(path)
    return px


def fetch_event_returns(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Compute the next-trading-day return after each elimination loss.

    For every event in ``ELIMINATION_EVENTS`` we load the country ETF's daily
    close prices and take the close-to-close simple return of the first trading
    session at least ``lag_days`` calendar day(s) after the match date (an
    explicit one-day execution lag; the match is an evening event, so the full
    next session is the tradable window). Returns are PRICE-only.

    Parameters
    ----------
    cache_dir : str
        Directory holding ``sports_<TICKER>_daily.parquet`` files.
    fetch : bool
        If False (default), read cache only; raise FileNotFoundError if absent.
        If True, download missing tickers via yfinance (lazy import) and cache.
    lag_days : int
        Minimum calendar-day gap between match and the response session.

    Returns a DataFrame with the event columns plus ``ret_date`` (the response
    session date) and ``next_day_ret`` (simple return). Events whose ETF lacks
    price coverage on/after the match date are dropped.

    Raises FileNotFoundError if a needed ticker cache is missing and fetch=False.
    """
    ev = elimination_table()
    tickers = sorted(ev["etf"].unique())
    prices = {t: _load_prices(t, cache_dir, fetch) for t in tickers}

    rows = []
    for _, e in ev.iterrows():
        px = prices[e["etf"]]
        md = pd.Timestamp(e["match_date"])
        future = px.index[px.index >= md + pd.Timedelta(days=lag_days)]
        if len(future) < 1:
            continue
        rd = future[0]
        pos = px.index.get_loc(rd)
        if pos < 1:
            continue
        prev = px.iloc[pos - 1]
        cur = px.iloc[pos]
        if not (np.isfinite(prev) and np.isfinite(cur)) or prev <= 0:
            continue
        ret = float(cur / prev - 1.0)
        row = e.to_dict()
        row["ret_date"] = rd
        row["next_day_ret"] = ret
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(
            "No event returns computed -- caches likely missing. Use fetch=True."
        )
    return out.reset_index(drop=True)


def fingerprint(df: pd.DataFrame, col: str = "next_day_ret") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(np.round(vals, 8)).tobytes())
    return h.hexdigest()[:12]
