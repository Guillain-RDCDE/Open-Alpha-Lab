"""Worked extension (beat 7) — run the *identical* stack across assets with different tailwinds.

The headline study shows, on the SPY, that the stack's Sharpe is vol-targeting (managed beta) and
the ARIMA forecast is a coin flip that *subtracts* value. If that is right, it makes a sharp,
falsifiable cross-asset prediction:

    * The **forecast** half should be a coin flip on **every** liquid asset — directional hit-rate
      ≈ 50%, and the ARIMA increment (Sharpe of the full stack minus the constant-long
      vol-targeting control) ≤ 0 everywhere. It is not an SPY accident.
    * The **sizing** half — the only thing that works — should track each asset's **leverage
      effect** (vol up ⇒ price down), which is strong in equity indices and weak or absent in gold,
      long bonds and FX. So the vol-targeting *lift over buy-and-hold* should be large on equities
      and small/negative off them. The "edge" travels with the sizing tailwind, not the forecast.

This module runs the same cold, parallel walk-forward as the headline on a basket of tickers
(cached or fetched via ``quantlab.data``) and returns one row per asset. It is the house-standard
*worked complement* in the study's own beat 7 — not a new study.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import log_returns
from .decompose import directional_edge, sharpe_decomposition
from .stack import generate_signals_parallel

# Strong leverage effect (equity indices) vs weak/absent (gold, long bonds, USD, oil). The split is
# the whole point: same code, different tailwind.
TAILWIND_ASSETS = ("SPY", "QQQ", "EWJ", "EWZ")
NO_TAILWIND_ASSETS = ("GLD", "TLT", "UUP", "USO")
DEFAULT_PANEL = TAILWIND_ASSETS + NO_TAILWIND_ASSETS


def _leverage_proxy(wf_frame: pd.DataFrame) -> float:
    """A simple leverage-effect read: corr(return_t, next-step GARCH σ̂_{t+1}).

    σ̂_{t+1} is the volatility forecast for the day *after* the return, made from data through day t
    (so it has seen ``return_t``). A negative correlation means down-moves are followed by *higher*
    forecast volatility — the classic equity leverage effect that makes vol-targeting pay. Near zero
    (or positive) means no tailwind for the sizing.
    """
    f = wf_frame.dropna(subset=["vol"])
    if len(f) < 31:
        return np.nan
    ret = f["ret"].to_numpy()[:-1]          # return_t
    vol_next = f["vol"].to_numpy()[1:]        # σ̂_{t+1}
    return float(np.corrcoef(ret, vol_next)[0, 1])


def walk_forward_for(prices: pd.Series, cache_path=None):
    """Cold parallel walk-forward for one price series, optionally cached to a parquet path.

    The ~thousands of daily fits per asset are the only slow part; caching the resulting frame
    keyed by the caller (typically a data fingerprint) means a re-run that only tweaks the summary
    or a secondary metric never recomputes them. Returns a :class:`WalkForward`.
    """
    from .stack import WalkForward

    if cache_path is not None and os.path.exists(cache_path):
        return WalkForward(frame=pd.read_parquet(cache_path), lookback=252, constant_long=False)
    wf = generate_signals_parallel(log_returns(prices))
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        wf.frame.to_parquet(cache_path)
    return wf


def run_asset(ticker: str, prices: pd.Series, cache_path=None) -> dict:
    """Run the full cold walk-forward stack on one price series and summarise the decomposition."""
    wf = walk_forward_for(prices, cache_path=cache_path)
    de = directional_edge(wf)
    sd = sharpe_decomposition(wf)
    return {
        "ticker": ticker,
        "n_days": de["n_graded"],
        "hit_rate_pct": de["hit_rate"] * 100.0,
        "dir_hac_t": de["dir_edge_hac_t"],
        "sharpe_stack": sd["sharpe_stack"],
        "sharpe_voltgt": sd["sharpe_voltgt"],
        "sharpe_buyhold": sd["sharpe_buyhold"],
        "arima_increment": sd["arima_increment_sharpe"],         # stack - vol-targeting; ≤0 = no edge
        "voltgt_lift": sd["sharpe_voltgt"] - sd["sharpe_buyhold"],  # the tailwind, in Sharpe
        "leverage_corr": _leverage_proxy(wf.frame),
    }


def run_panel(prices_by_ticker: dict[str, pd.Series],
              cache_paths: dict[str, str] | None = None) -> pd.DataFrame:
    """Run :func:`run_asset` across a ``{ticker: price_series}`` mapping; one row per asset.

    The caller supplies already-loaded, as-of-sliced price series (so this module stays free of any
    data-source policy). ``cache_paths`` optionally maps each ticker to a parquet path for its
    walk-forward, so re-runs that only touch the summary skip the heavy fits. Rows in input order.
    """
    cache_paths = cache_paths or {}
    rows = [run_asset(t, p, cache_path=cache_paths.get(t)) for t, p in prices_by_ticker.items()]
    return pd.DataFrame(rows).set_index("ticker")
