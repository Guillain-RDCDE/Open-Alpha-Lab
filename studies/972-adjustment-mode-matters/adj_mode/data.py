"""Data layer for Study 972 — Adjusted or Not.

Two caches per ticker, fetched in the same pass: the **total-return** close
(``auto_adjust=True``) and the **price-only** close (``auto_adjust=False``, no dividend
adjustment). Everything in this study is the difference between those two columns, so they
have to come from the same vintage or the comparison measures the provider's revisions
instead.

Two flavours per ticker, both from the same call pattern, cached under this study's own
prefixes so they cannot collide with the desk's shared ``prices_`` files:

- ``tr``      — ``auto_adjust=True``: split- and dividend-adjusted, the wealth of a holder
  who reinvested every distribution;
- ``pxonly``  — ``auto_adjust=False``: split-adjusted only, the line on a price chart.

The pair is the whole study, so ``fetch`` writes both in one pass and ``load_pair`` refuses to
return a ticker whose two files are not both present.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# A yield ladder, on purpose: from a fund that pays almost nothing to one that pays
# most of its return as income. The bias this study measures is proportional to yield,
# so a universe with no yield dispersion would find nothing.
TICKERS = ("QQQ", "SPY", "IWM", "EFA", "XLU", "VYM", "HYG", "TLT")

AS_OF = "2026-06-30"
START = "1998-01-01"

MODES = ("tr", "pxonly")
MODE_LABEL = {"tr": "total return (dividends reinvested)", "pxonly": "price only"}


def _cache_path(ticker: str, mode: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"adjmode_{safe}_{mode}.parquet")


def fetch(tickers=TICKERS, start: str = START, end: str | None = None,
          cache_dir: str = DEFAULT_CACHE, retries: int = 4) -> dict:
    """Download both flavours for every ticker in one pass (network-only)."""
    import yfinance as yf  # lazy: only when we actually go to the network

    os.makedirs(cache_dir, exist_ok=True)
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for tk in tickers:
        out[tk] = {}
        for mode in MODES:
            raw = None
            for _ in range(retries):
                try:
                    raw = yf.download(tk, start=start, end=end, interval="1d",
                                      auto_adjust=(mode == "tr"), progress=False)
                    if raw is not None and len(raw) > 0:
                        break
                except Exception:
                    time.sleep(2.0)
            if raw is None or len(raw) == 0:
                raise RuntimeError(f"yfinance returned no data for {tk} ({mode})")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
            df = raw[["close"]].copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df = df.dropna(subset=["close"])
            df.to_parquet(_cache_path(tk, mode, cache_dir))
            out[tk][mode] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff both flavours are cached for every ticker."""
    return all(os.path.exists(_cache_path(tk, m, cache_dir))
               for tk in tickers for m in MODES)


def load_panel(mode: str = "tr", tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
               asof: str = AS_OF) -> pd.DataFrame:
    """One aligned close panel for the requested adjustment mode, offline."""
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, mode, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached {mode} prices for {tk} at {path}. "
                f"Call adj_mode.data.fetch() once to populate the cache.")
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def load_prices(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
                asof: str = AS_OF) -> pd.DataFrame:
    """The desk's standard loader name, pointing at the total-return panel.

    Every other study on the bench exposes ``load_prices``; keeping the name here means the
    shared data-layer tests apply unchanged, and it makes the default explicit: when this
    study says "prices" without qualification it means **total return**.
    """
    return load_panel("tr", tickers, cache_dir, asof)


def load_pair(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE,
              asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """``{'tr': panel, 'pxonly': panel}`` on a common index — the study's whole input."""
    panels = {m: load_panel(m, tickers, cache_dir, asof) for m in MODES}
    common = panels["tr"].index.intersection(panels["pxonly"].index)
    return {m: p.loc[common] for m, p in panels.items()}


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]




# --------------------------------------------------------------------------- #
# Synthetic tape — a panel with KNOWN, dispersed dividend yields
# --------------------------------------------------------------------------- #
def synthetic_pair(
    n_assets: int = 6,
    n_years: int = 20,
    vol_ann: float = 0.18,
    total_return_ann: float = 0.09,
    yields=None,                      # per-asset dividend yield; None = a spread 0% .. 6%
    signal_strength: float = 1.0,     # scales every yield; 0 = no dividends at all (the null)
    start: str = "2004-01-02",
    seed: int = 972,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A panel where every asset earns the SAME total return but pays a different yield.

    That construction is the point: on the total-return panel the assets are
    indistinguishable in the long run, and on the price-only panel they are ordered exactly by
    their yields — so any ranking difference between the two views is attributable to the
    convention and nothing else. ``signal_strength = 0`` sets every yield to zero, where the
    two panels must coincide.

    Returns ``(tr_panel, px_panel, truth)``. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n)
    if yields is None:
        yields = np.linspace(0.0, 0.06, n_assets)
    y = np.asarray(yields, dtype=float) * float(signal_strength)

    sd = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    mkt = rng.normal(0.0, sd * 0.8, n)
    tr_cols, px_cols = {}, {}
    for i in range(len(y)):
        idio = rng.normal(0.0, sd * 0.6, n)
        r_total = total_return_ann / TRADING_DAYS_PER_YEAR + mkt + idio
        r_price = r_total - y[i] / TRADING_DAYS_PER_YEAR
        tr_cols[f"A{i}"] = 100.0 * np.cumprod(1.0 + r_total)
        px_cols[f"A{i}"] = 100.0 * np.cumprod(1.0 + r_price)
    idx = pd.DatetimeIndex(dates, name="date")
    truth = {"n_days": n, "n_years": n_years, "seed": seed, "yields": y.tolist(),
             "total_return_ann": total_return_ann, "vol_ann": vol_ann,
             "signal_strength": float(signal_strength)}
    return pd.DataFrame(tr_cols, index=idx), pd.DataFrame(px_cols, index=idx), truth


def synthetic_panel(n_assets: int = 6, n_years: int = 20, signal_strength: float = 1.0,
                    seed: int = 972, cash_rate_ann: float = 0.02):
    """Panel wrapper (prices, cash, truth) so the shared data-layer tests apply unchanged."""
    tr, _, truth = synthetic_pair(n_assets=n_assets, n_years=n_years,
                                  signal_strength=signal_strength, seed=seed)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(np.cumprod(np.full(len(tr), cash_daily)), index=tr.index, name="cash")
    truth = dict(truth, n_assets=n_assets, alpha_vol_eff=0.05 * float(signal_strength),
                 cash_rate_ann=cash_rate_ann)
    return tr, cash, truth


def synthetic_daily(n_years: int = 20, signal_strength: float = 1.0, seed: int = 972):
    """Single-name convenience wrapper, matching the desk's shared shape."""
    prices, cash, truth = synthetic_panel(n_assets=1, n_years=n_years,
                                          signal_strength=signal_strength, seed=seed)
    return pd.DataFrame({"asset": prices.iloc[:, 0], "cash": cash}), truth

