"""Paperswithbacktest HuggingFace reader — the ONE free table, with a SURVIVORSHIP GUARD.

What is actually free (probed 2026-06-11)
-----------------------------------------
Of paperswithbacktest's ~30 HuggingFace datasets, **only `Stocks-Daily-Price` is open**.
Everything else — ETFs, Forex, Indices, Bonds, Commodities, Cryptocurrencies, the quarterly
fundamentals, bankruptcy, news — is **gated** (HTTP 401, "private or gated"), i.e. it needs an
HF token tied to a paid Papers-With-Backtest subscription. So this module deliberately wires
*one* source: individual **US single-name equities**, daily OHLCV, ~7,000 tickers, 1962→present,
as-of the dataset's monthly refresh. For ETFs / FX / futures, keep using Yahoo via
``quantlab.data.fetch`` / ``tools/fetch_altdata.py`` — do NOT reach for a gated PWB table.

Schema (verified live): ``symbol  date  open  high  low  close  volume  adj_close``.

WHY A GUARD — read before anything cross-sectional
--------------------------------------------------
The table is the **current survivor universe projected backwards**: delisted tickers
(bankruptcies, buy-outs, deletions) are absent, so any cross-sectional backtest on it is biased
**upward** — the losers were deleted before the test ever saw them. That is precisely the trap
several studies exist to expose (24 Stampede, 25 Clean-Slate, 33 Slingshot carry the same caveat
on the Yahoo panel). So ``load_panel`` **refuses** to assemble a multi-name panel unless you pass
``allow_survivorship_bias=True`` (and it still warns). **Single-name** use — where *you* picked
the ticker, not the dataset — is unbiased by construction and is free to use. The single honest
win here is *depth*: clean history back to the 1960s-80s on a name you already chose, well before
Yahoo is reliable.

Engine
------
DuckDB httpfs with predicate push-down on the (sorted) ``symbol`` column: a single name reads a
few MB of the remote shards, not the whole 520 MB table — no full download, no API key. Requires
``duckdb`` (``pip install duckdb``); everything else is stdlib + pandas.

Adjustment modes
----------------
'total_return' (default) — OHLC scaled by adj_close/close (splits + dividends removed).
'as_is'                  — close as reported. NOT 'split_only' — the table carries no split
                           events, so the genuine ex-dividend overnight gap cannot be
                           reconstructed here; use Yahoo (``quantlab.data``) when you need it.
"""

from __future__ import annotations

import json
import os
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd

HF_DATASET = "paperswithbacktest/Stocks-Daily-Price"
_PARQUET_API = f"https://huggingface.co/api/datasets/{HF_DATASET}/parquet/default/train"
_CACHE = Path(os.environ.get("OVERNIGHT_CACHE", "_cache")) / "hf"
_UA = {"User-Agent": "Mozilla/5.0 (OpenAlphaLab research)"}
_MODES = ("total_return", "as_is")


class SurvivorshipBiasError(RuntimeError):
    """Raised when a cross-sectional panel is requested without opting into the bias."""


@lru_cache(maxsize=1)
def _shard_urls() -> tuple[str, ...]:
    """The dataset's parquet shard URLs (discovered, not hard-coded — shard count drifts)."""
    req = urllib.request.Request(_PARQUET_API, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
        return tuple(json.loads(r.read()))


@lru_cache(maxsize=1)
def _con():
    """A process-wide DuckDB connection with httpfs loaded and patient retries."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "quantlab.hf_data needs duckdb for the HuggingFace push-down read; "
            "`pip install duckdb`."
        ) from exc
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET http_retries=5; SET http_timeout=60000;")
    return con


def _fetch_symbol_raw(symbol: str) -> pd.DataFrame:
    """All rows for one symbol, pushed down to the remote parquet (reads a few MB)."""
    urls = _shard_urls()
    arr = "[" + ",".join("'" + u + "'" for u in urls) + "]"
    q = (
        f"SELECT symbol,date,open,high,low,close,volume,adj_close "
        f"FROM read_parquet({arr}) WHERE symbol = ? ORDER BY date"
    )
    df = _con().execute(q, [symbol]).df()
    if df.empty:
        raise ValueError(
            f"No HF rows for {symbol!r}. Note this table is US single-name *stocks* only — "
            f"ETFs/indices/FX live in PWB's gated datasets; use Yahoo for those."
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _apply_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Raw HF columns → Open/High/Low/Close/Volume in the requested adjustment mode."""
    out = df.set_index("date")
    out.index.name = "Date"
    o, h, low, c, v = out["open"], out["high"], out["low"], out["close"], out["volume"]
    if mode == "total_return":
        factor = out["adj_close"] / out["close"]
        o, h, low, c = o * factor, h * factor, low * factor, c * factor
    res = pd.DataFrame({"Open": o, "High": h, "Low": low, "Close": c, "Volume": v})
    return res.dropna(how="all")


def load_symbol(symbol: str, mode: str = "total_return", use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV for one US stock from the free HF table, cached to ``_cache/hf/``.

    Unbiased by construction (you picked the ticker). Columns match
    ``quantlab.data.fetch`` so it is a drop-in for single-name studies — its edge is
    *depth* (history back to the 1960s-80s). ETFs/indices/FX are NOT here (gated).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE / f"{symbol}_{mode}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    out = _apply_mode(_fetch_symbol_raw(symbol), mode)
    if use_cache:
        out.to_parquet(cache_path)
    return out


def load_panel(
    symbols: list[str],
    mode: str = "total_return",
    allow_survivorship_bias: bool = False,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Multi-name panel — GUARDED. Raises unless you opt into the survivorship bias.

    The table omits delisted names, so a panel built from it over-states any cross-sectional
    edge. That is almost never what you want for a credible study; prefer a point-in-time
    membership source. If you understand the bias and want the panel anyway (a machinery
    proof, or a deliberately-biased control arm), pass ``allow_survivorship_bias=True``.
    """
    if not allow_survivorship_bias:
        raise SurvivorshipBiasError(
            f"Refusing to build a {len(symbols)}-name panel from {HF_DATASET}: it is the "
            "current-survivor universe projected backwards, so a cross-sectional backtest on it "
            "is biased upward. Pass allow_survivorship_bias=True to override (and document it in "
            "beat 4), or use a point-in-time membership source."
        )
    warnings.warn(
        f"Building a survivorship-BIASED panel from {HF_DATASET} ({len(symbols)} names). "
        "Delisted tickers are absent — treat any cross-sectional result as an upper bound.",
        stacklevel=2,
    )
    panel: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            panel[s] = load_symbol(s, mode=mode, use_cache=use_cache)
        except Exception as exc:  # a missing/typo'd name shouldn't kill the whole panel
            warnings.warn(f"HF panel: skipping {s!r} ({type(exc).__name__})", stacklevel=2)
    return panel
