"""Data layer for the shareholder-yield study.

Three sources:
  1. Synthetic panel (offline, deterministic) — used by tests.
  2. EDGAR shared cache — diluted shares (net buyback yield) + annual returns.
  3. yfinance (cache-first, study-local) — dividend yields per ticker-year.

Shareholder yield = dividend yield + net buyback yield.
Net buyback yield = -(share_count change) / prior share count, times price (proxy: use the YoY
  fractional decrease in shares as buyback yield, sign-flipped from net issuance).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
LOCAL_CACHE = os.path.join(_HERE, "..", "_cache")


@dataclass(frozen=True)
class WorldTruth:
    yield_premium: float

    @property
    def has_premium(self) -> bool:
        return self.yield_premium != 0.0


def synthetic_panel(n_firms: int = 200, n_years: int = 18, yield_premium: float = 0.05,
                    base_ret: float = 0.10, ret_vol: float = 0.28, seed: int = 233
                    ) -> tuple[pd.DataFrame, pd.DataFrame, WorldTruth]:
    """Deterministic synthetic shareholder-yield panel.

    Each firm-year draws a total yield ~N(0.04, 0.03) (mimics real distributions).
    Forward return = base_ret + yield_premium * z(yield) + noise, so high-yield
    firms outperform when the premium is on. yield_premium=0 is the null.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2008, 2008 + n_years)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    # total yield per firm-year: mean 4%, std 3%
    yields = 0.04 + 0.03 * rng.standard_normal((n_years, n_firms))
    yields = np.clip(yields, -0.15, 0.25)  # realistic bounds
    z = (yields - yields.mean(axis=1, keepdims=True)) / (yields.std(axis=1, keepdims=True) + 1e-9)
    fwd = base_ret + yield_premium * z + ret_vol * rng.standard_normal((n_years, n_firms))
    sig_df = pd.DataFrame(yields, index=pd.Index(years, name="year"), columns=firms)
    fwd_df = pd.DataFrame(fwd,   index=pd.Index(years, name="year"), columns=firms)
    return sig_df, fwd_df, WorldTruth(yield_premium)


def _net_buyback_yield(shares: pd.DataFrame) -> pd.DataFrame:
    """Net buyback yield: fractional decrease in diluted shares YoY.

    Positive = shares declined (net buyback). Negative = net issuance.
    This is the sign-flip of net issuance from study 64.
    """
    pct_change = shares.pct_change()  # positive = more shares (issuance)
    return -pct_change                # flip: positive = fewer shares (buyback)


def _div_yield_from_cache(local_cache_dir: str = LOCAL_CACHE) -> pd.DataFrame:
    """Load pre-fetched dividend yield (year x ticker) from study-local cache, if present."""
    path = os.path.join(local_cache_dir, "div_yield.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _fetch_div_yield_yfinance(tickers: list[str], years: range,
                               local_cache_dir: str = LOCAL_CACHE) -> pd.DataFrame:
    """Fetch annual dividend yield via yfinance (sum of dividends / year-end price) and cache it."""
    os.makedirs(local_cache_dir, exist_ok=True)
    path = os.path.join(local_cache_dir, "div_yield.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    import yfinance as yf
    records = {}
    for yr in years:
        start, end = f"{yr}-01-01", f"{yr}-12-31"
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=False,
                              progress=False, threads=True)
        except Exception:
            continue
        if raw.empty:
            continue
        # dividends and close
        if "Dividends" in raw.columns.get_level_values(0):
            divs = raw["Dividends"].sum()
        elif "Dividends" in raw.columns:
            divs = raw["Dividends"].sum()
        else:
            divs = pd.Series(0.0, index=tickers)
        if "Close" in raw.columns.get_level_values(0):
            price_yr_end = raw["Close"].iloc[-1]
        else:
            price_yr_end = raw["Close"].iloc[-1] if "Close" in raw.columns else pd.Series(np.nan, index=tickers)
        dy = divs / price_yr_end.replace(0, np.nan)
        records[yr] = dy
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).T
    df.index.name = "year"
    df.to_parquet(path)
    return df


def fetch_panel(cache_dir: str = DEFAULT_CACHE, local_cache_dir: str = LOCAL_CACHE,
                fetch: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (shareholder_yield, fwd_ret, components).

    shareholder_yield = net_buyback_yield + div_yield (aligned, year x ticker).
    fwd_ret = next-year total return.
    components = DataFrame with columns ['buyback_yield', 'div_yield', 'total_yield'] stacked,
                 returned as a single-year cross-section summary dict for diagnostics.

    Returns empty DataFrames if cache is absent (CI / offline).
    """
    sh_p = os.path.join(cache_dir, "_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet")
    r_p  = os.path.join(cache_dir, "_edgar_yrret.parquet")
    if not (os.path.exists(sh_p) and os.path.exists(r_p)):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    shares  = pd.read_parquet(sh_p)
    yr_ret  = pd.read_parquet(r_p)

    # --- net buyback yield (year x ticker) ---
    bby = _net_buyback_yield(shares)   # first row is NaN (no prior year)

    # --- dividend yield (year x ticker) ---
    tickers = sorted(set(bby.columns) & set(yr_ret.columns))
    years_avail = sorted(bby.index)

    div_yield = _div_yield_from_cache(local_cache_dir)
    if div_yield.empty and fetch:
        div_yield = _fetch_div_yield_yfinance(tickers, range(years_avail[0], years_avail[-1] + 1),
                                               local_cache_dir)

    # --- combine ---
    bby = bby[tickers]
    if not div_yield.empty:
        div_yield = div_yield.reindex(index=bby.index, columns=tickers)
        total_yield = bby.add(div_yield, fill_value=0.0)
    else:
        # fallback: buyback yield only (still meaningful)
        total_yield = bby.copy()

    # align fwd return (year t signal -> year t+1 return)
    fwd = yr_ret[tickers].reindex(total_yield.index + 1)
    fwd.index = total_yield.index

    # drop rows where signal has <20 observations
    mask = total_yield.notna().sum(axis=1) >= 20
    total_yield = total_yield[mask]
    fwd = fwd[mask]
    bby_out = bby[mask]
    dvy_out = div_yield.reindex(index=total_yield.index, columns=tickers) if not div_yield.empty else pd.DataFrame()

    return total_yield, fwd, pd.DataFrame({"bby_mean": bby_out.mean(axis=1),
                                            "dvy_mean": dvy_out.mean(axis=1) if not dvy_out.empty else np.nan})
