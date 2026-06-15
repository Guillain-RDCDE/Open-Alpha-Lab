"""Data layer for Study 177 (Megacap-Concentration).

Two tapes, one shape (a dict of annual market-cap-ranked DataFrames):

- ``synthetic_annual`` — a *deterministic, offline* generator. An ``effect`` knob
  controls whether the top-N portfolio outperforms the equal-weight baseline. When
  ``effect=0`` the generator is skill-free and top-N is just as good as any other
  selection. This is the study's null in a bottle.
- ``load_real`` — loads the EDGAR-based cross-section (shares × year-end price) from
  the shared caches (``_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet``
  and yfinance annual closes), returning a DataFrame of annual returns ranked by market
  cap at the start of each year. Cache-only by default; the real fetch path writes to
  the study's own ``_cache/``.

No look-ahead: market caps are formed from *prior year-end* prices and shares, so the
rank used for year-T is based on data known at the start of year-T.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

# ---------------------------------------------------------------------------
# Shared EDGAR & price helpers
# ---------------------------------------------------------------------------
_SHARES_FILE = "_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet"
_MCAP_CACHE  = "megacap_annual_mcap.parquet"
_RET_CACHE   = "megacap_annual_returns.parquet"


def _edgar_shares() -> pd.DataFrame:
    """Load diluted shares (rows = fiscal years, cols = tickers) from the repo cache."""
    path = os.path.join(REPO_CACHE, _SHARES_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"EDGAR shares not found at {path}")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 18,
    n_stocks: int = 100,
    top_n: int = 10,
    effect: float = 0.0,
    annual_vol: float = 0.25,
    market_drift: float = 0.10,
    start_year: int = 2007,
    seed: int = 177,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual cross-section of stocks with a tunable concentration effect.

    Each year, stocks are given a rank (1 = largest by market cap). When ``effect > 0``,
    top-N stocks receive an extra annual excess return of ``effect`` vs the rest — i.e.
    the size-concentration claim is planted. When ``effect = 0`` the ranking is noise and
    top-N is just as good as any randomly selected N stocks.

    Returns a DataFrame with columns:
      ``year, ticker, rank, ret_gross, is_top_n``
    and a ``truth`` dict recording the planted parameters.

    Useful for tests: with ``effect=0`` the top-N mean return must be statistically
    indistinguishable from the equal-weight basket; with ``effect > 0`` it should exceed it.
    """
    rng = np.random.default_rng(seed)
    years = list(range(start_year, start_year + n_years))
    tickers = [f"S{i:03d}" for i in range(n_stocks)]

    rows = []
    for yr in years:
        # Random market-cap rank (1 = largest) — re-shuffled each year.
        ranks = rng.permutation(n_stocks) + 1  # 1..n_stocks
        # Annual returns: market-wide drift + idiosyncratic noise.
        base = rng.normal(market_drift, annual_vol, n_stocks)
        # Plant the effect: top-N stocks get an extra alpha bump.
        boost = np.where(ranks <= top_n, effect, 0.0)
        rets = base + boost
        for i, t in enumerate(tickers):
            rows.append(
                {
                    "year": yr,
                    "ticker": t,
                    "rank": int(ranks[i]),
                    "ret_gross": float(rets[i]),
                    "is_top_n": bool(ranks[i] <= top_n),
                }
            )

    df = pd.DataFrame(rows)
    truth = {
        "effect": effect,
        "n_years": n_years,
        "n_stocks": n_stocks,
        "top_n": top_n,
        "annual_vol": annual_vol,
        "market_drift": market_drift,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — EDGAR shares × yfinance year-end prices
# ---------------------------------------------------------------------------
def _build_mcap_table(fetch: bool, cache_dir: str) -> pd.DataFrame:
    """Build a (year × ticker) market-cap table and cache it.

    Market cap at end of year-T = shares_{T} × close_{T-12-31} (last trading day).
    We use EDGAR diluted shares and yfinance daily closes fetched via cache.
    Survivorship-bias note: we use *current* S&P 500 membership (name-survivorship),
    which is disclosed in results and README.
    """
    mcap_path = os.path.join(cache_dir, _MCAP_CACHE)
    if os.path.exists(mcap_path) and not fetch:
        return pd.read_parquet(mcap_path)

    shares_df = _edgar_shares()  # rows=year, cols=ticker
    tickers = list(shares_df.columns)
    years = list(shares_df.index)

    if fetch:
        import yfinance as yf  # lazy import

        # Download all tickers in one batch — year-end closes, annual frequency.
        # We need the last close of each calendar year.
        start_dt = f"{min(years) - 1}-01-01"
        end_dt   = f"{max(years) + 1}-01-01"
        raw = yf.download(
            tickers,
            start=start_dt,
            end=end_dt,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise RuntimeError("yfinance returned empty price data")
        # Extract Close panel
        if isinstance(raw.columns, pd.MultiIndex):
            close_panel = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
        else:
            close_panel = raw[["Close"]]

        # Year-end close: last trading day of each year
        close_panel.index = pd.to_datetime(close_panel.index)
        close_panel["_year"] = close_panel.index.year
        year_end = close_panel.groupby("_year").last().drop(columns="_year", errors="ignore")
        year_end.index.name = "year"

        # Align to EDGAR years
        common_years = sorted(set(years) & set(year_end.index))
        common_tickers = [t for t in tickers if t in year_end.columns]
        mcap = shares_df.loc[common_years, common_tickers].multiply(
            year_end.loc[common_years, common_tickers]
        )
        os.makedirs(cache_dir, exist_ok=True)
        mcap.to_parquet(mcap_path)
        return mcap
    else:
        raise FileNotFoundError(
            f"No cached market-cap table at {mcap_path}. "
            f"Call load_real(fetch=True) once to build the cache."
        )


def _build_return_table(fetch: bool, cache_dir: str) -> pd.DataFrame:
    """Build a long DataFrame of annual returns for S&P 500 members.

    Uses yfinance year-end closes to compute annual price returns (excluding dividends;
    we use SPY total-return as the benchmark which does include divs, so this is
    conservative for the concentration strategy).

    Returns: year, ticker, ret_annual (long format)
    """
    ret_path = os.path.join(cache_dir, _RET_CACHE)
    if os.path.exists(ret_path) and not fetch:
        return pd.read_parquet(ret_path)

    shares_df = _edgar_shares()
    tickers = list(shares_df.columns)
    years = sorted(shares_df.index)

    if fetch:
        import yfinance as yf

        start_dt = f"{min(years) - 1}-01-01"
        end_dt   = f"{max(years) + 1}-01-01"
        raw = yf.download(
            tickers,
            start=start_dt,
            end=end_dt,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if raw.empty:
            raise RuntimeError("yfinance returned empty price data")

        if isinstance(raw.columns, pd.MultiIndex):
            close_panel = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=0)
        else:
            close_panel = raw[["Close"]]

        close_panel.index = pd.to_datetime(close_panel.index)
        close_panel["_year"] = close_panel.index.year
        year_end = close_panel.groupby("_year").last().drop(columns="_year", errors="ignore")
        year_end.index.name = "year"

        # Annual returns: (price_T / price_{T-1}) - 1
        ret_panel = year_end.pct_change()  # rows = year, cols = tickers

        # Align to EDGAR years
        common_tickers = [t for t in tickers if t in ret_panel.columns]
        ret_panel = ret_panel.loc[:, common_tickers]

        rows = []
        for yr in years:
            if yr not in ret_panel.index:
                continue
            row = ret_panel.loc[yr]
            valid = row.dropna()
            for tk, r in valid.items():
                rows.append({"year": yr, "ticker": tk, "ret_annual": float(r)})

        df = pd.DataFrame(rows)
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(ret_path)
        return df
    else:
        raise FileNotFoundError(
            f"No cached return table at {ret_path}. "
            f"Call load_real(fetch=True) once to build the cache."
        )


def load_real(
    top_n: int = 10,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> tuple[pd.DataFrame, dict]:
    """Load the real annual cross-section ranked by market cap.

    Returns a long DataFrame with columns:
      ``year, ticker, mcap, mcap_rank, ret_annual, is_top_n``

    Market-cap rank uses *prior year-end* market cap (no look-ahead). Returns are the
    *current year* price return (year-end to year-end).

    ⚠ Survivorship bias: universe = current S&P 500 members (EDGAR coverage). Named
    prominently in all results and README.

    Parameters
    ----------
    top_n : int
        Concentration size (default 10; study also tests 7).
    fetch : bool
        If True, touches yfinance to build the caches.
    cache_dir : str
        Study-level cache directory.
    """
    mcap = _build_mcap_table(fetch=fetch, cache_dir=cache_dir)
    rets = _build_return_table(fetch=fetch, cache_dir=cache_dir)

    # mcap: rows = year, cols = ticker
    # We rank *prior year-end* market cap → the rank used for year-T is mcap_{T-1}.
    mcap_lagged = mcap.shift(1)  # shift rows (years) by 1

    rows = []
    for yr in sorted(set(rets["year"])):
        if yr not in mcap_lagged.index:
            continue
        mcap_row = mcap_lagged.loc[yr].dropna()
        if len(mcap_row) < top_n:
            continue
        # Rank: 1 = largest market cap
        ranks = mcap_row.rank(ascending=False, method="first").astype(int)
        # Filter to tickers with both mcap and return data
        year_rets = rets[rets["year"] == yr].set_index("ticker")["ret_annual"]
        common = ranks.index.intersection(year_rets.index)
        if len(common) < top_n:
            continue
        for tk in common:
            rows.append(
                {
                    "year": yr,
                    "ticker": tk,
                    "mcap": float(mcap_row.get(tk, np.nan)),
                    "mcap_rank": int(ranks[tk]),
                    "ret_annual": float(year_rets[tk]),
                    "is_top_n": bool(ranks[tk] <= top_n),
                }
            )

    df = pd.DataFrame(rows)
    meta = {
        "top_n": top_n,
        "years": sorted(df["year"].unique().tolist()),
        "n_years": int(df["year"].nunique()),
        "n_tickers_total": int(df["ticker"].nunique()),
        "source": "EDGAR shares × yfinance year-end close; survivorship-biased universe",
    }
    return df, meta


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    vals = df["ret_annual"].dropna().to_numpy()
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]
