"""Cross-sectional breadth — is the overnight effect a firm-level fact?

Ten ETFs could be a fluke or a data quirk. The decisive credibility test is the
**firm level**: does the overnight-vs-intraday split hold across hundreds of
individual stocks (cf. Lou, Polk, and Skouras 2019)? This module reproduces that
on the current S&P 500 constituents.

Honest caveat, stated not hidden: it uses *current* membership, so it carries
**survivorship bias** (today's index excludes firms that were delisted). The
qualitative breadth result is robust to this; precise magnitudes are not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decompose import decompose, summary

_CACHE = Path("_cache")
_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_symbols(use_cache: bool = True) -> list[str]:
    """Current S&P 500 tickers (Yahoo format). Wikipedia, cached to JSON.

    Class shares use '-' on Yahoo (e.g. BRK-B), so '.' is translated.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CACHE / "sp500_symbols.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    # Wikipedia 403s the default urllib UA; fetch with a browser UA, then parse.
    import io
    import urllib.request

    req = urllib.request.Request(_WIKI, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (trusted URL)
        html = r.read().decode("utf-8", "ignore")
    table = pd.read_html(io.StringIO(html))[0]  # needs lxml
    syms = [str(s).replace(".", "-").strip() for s in table["Symbol"].tolist()]
    cache.write_text(json.dumps(syms))
    return syms


def download_panel(symbols: list[str], start: str = "2010-01-01", use_cache: bool = True) -> dict:
    """Batched OHLC download for many tickers (one threaded call), cached.

    Uses ``auto_adjust=True`` (total-return prices) for a clean, simple breadth
    study — documented choice. Returns ``{ticker: DataFrame[Open, High, Low,
    Close]}``, skipping tickers Yahoo returns empty.
    """
    import yfinance as yf  # lazy

    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CACHE / f"panel_{len(symbols)}_{start}.parquet"
    if use_cache and cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = yf.download(
            symbols, start=start, auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        if use_cache:
            raw.to_parquet(cache)

    panel = {}
    for s in symbols:
        if s not in raw.columns.get_level_values(0):
            continue
        df = raw[s][["Open", "High", "Low", "Close"]].dropna()
        if not df.empty:
            panel[s] = df
    return panel


def cross_section_decompose(panel: dict, min_days: int = 750) -> dict:
    """Per-symbol night/day stats + equal-weight aggregate across the cross-section.

    Returns a dict with:
      ``per_symbol``   DataFrame indexed by ticker (overnight/intraday cum & Sharpe, n)
      ``ew_overnight`` / ``ew_intraday``  equal-weight daily cross-sectional mean returns
      ``frac_overnight_wins``  share of stocks whose overnight Sharpe > intraday Sharpe
      ``n_stocks``
    """
    rows, on_cols, id_cols = {}, {}, {}
    for s, ohlc in panel.items():
        dec = decompose(ohlc)
        if len(dec) < min_days:
            continue
        sm = summary(dec)
        rows[s] = {
            "overnight_cum": sm.loc["overnight", "cum_return"],
            "intraday_cum": sm.loc["intraday", "cum_return"],
            "overnight_sharpe": sm.loc["overnight", "sharpe"],
            "intraday_sharpe": sm.loc["intraday", "sharpe"],
            "n": int(len(dec)),
        }
        on_cols[s] = dec["r_overnight"]
        id_cols[s] = dec["r_intraday"]

    per = pd.DataFrame(rows).T
    ew_on = pd.DataFrame(on_cols).mean(axis=1).dropna()
    ew_id = pd.DataFrame(id_cols).mean(axis=1).dropna()
    frac = float((per["overnight_sharpe"] > per["intraday_sharpe"]).mean()) if len(per) else np.nan
    return {
        "per_symbol": per,
        "ew_overnight": ew_on,
        "ew_intraday": ew_id,
        "frac_overnight_wins": frac,
        "n_stocks": int(len(per)),
    }
