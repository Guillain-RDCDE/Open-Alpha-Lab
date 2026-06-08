"""The gauntlet — the tests that turn a ledger into a verdict.

The headline is :func:`signal_vs_baseline`: it pits the forward return *after a breakout*
against the forward return of a *random entry in the same stocks over the same horizon*.
That control is the whole game. A long-only momentum rule will always show a positive raw
forward return (stocks drift up; the breakout days are, by construction, days the stock was
already rising). The only question that matters is the **excess** over ambient drift — and
whether it survives autocorrelation-robust inference. Everything else (decay by year, the
cost sweep, the explosive-move audit, capacity) sizes the answer.

Leans on the shared engine: ``quantlab.analytics.mean_tstat_hac`` (HAC t on the excess),
``quantlab.stats.sharpe_ci_bootstrap`` (bootstrap CI on the trade-return Sharpe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac
from quantlab.stats import sharpe_ci_bootstrap

from . import backtest
from .signals import Params, find_signals


# --------------------------------------------------------------------------- #
# Headline falsification — breakout forward return vs same-stock random entry
# --------------------------------------------------------------------------- #

def signal_vs_baseline(
    frames: dict[str, pd.DataFrame],
    params: Params = Params(),
    horizons=(5, 10, 20),
    baseline_mult: int = 20,
    seed: int = 0,
    sigs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """For each horizon: mean forward return after a breakout vs random same-stock entry.

    For every name we collect the breakout forward returns and, as a matched control,
    ``baseline_mult`` random-entry forward returns over the same horizons. The **excess**
    (breakout minus the per-name baseline mean) is the object we test: its mean, a
    Newey-West *t*-stat, and a bootstrap CI. ``NONE`` if the excess CI straddles zero.

    Pass ``sigs`` (from :func:`backtest.signals_by_ticker`) to reuse a single detection pass.
    """
    rows = []
    for h in horizons:
        sig_rets, excess = [], []
        for tk, f in frames.items():
            sig = sigs.get(tk) if sigs is not None else find_signals(f, params)
            if sig is None or sig.empty:
                continue
            fr = backtest.forward_returns(f, sig, horizons=(h,))[f"fwd_{h}"].dropna()
            if fr.empty:
                continue
            base = backtest.baseline_forward_returns(
                f, n_draws=baseline_mult * len(fr), horizons=(h,), seed=seed
            )[f"fwd_{h}"].dropna()
            if base.empty:
                continue
            sig_rets.extend(fr.tolist())
            excess.extend((fr - base.mean()).tolist())
        if not sig_rets:
            continue
        sig_arr = np.array(sig_rets)
        exc = pd.Series(excess)
        hac = mean_tstat_hac(exc)
        boot = sharpe_ci_bootstrap(exc, periods_per_year=1)  # per-trade, not annualised
        rows.append({
            "horizon": h,
            "n_signals": len(sig_arr),
            "mean_breakout": float(sig_arr.mean()),
            "mean_excess": float(exc.mean()),
            "hac_t": hac["tstat"],
            "excess_ci_low": boot["ci_low"] * np.sqrt(1),  # CI on per-trade Sharpe of excess
            "excess_ci_high": boot["ci_high"],
            "frac_excess_neg": boot["frac_negative"],
        })
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Tradable strategy diagnostics
# --------------------------------------------------------------------------- #

def trade_sharpe_ci(ledger: pd.DataFrame, seed: int = 0) -> dict:
    """Bootstrap CI on the *per-trade* Sharpe of net returns (not annualised)."""
    if ledger.empty:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "frac_negative": float("nan")}
    return sharpe_ci_bootstrap(ledger["ret_net"], periods_per_year=1, seed=seed)


def cost_sweep(
    frames: dict[str, pd.DataFrame],
    params: Params = Params(),
    cost_grid_bps=(0, 5, 10, 20, 40, 80),
    sigs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Mean net return per trade and win rate as round-trip cost rises — where the edge dies."""
    base = backtest.run_universe(frames, params=params, cost_bps=0.0, sigs=sigs)
    rows = []
    for c in cost_grid_bps:
        net = base["ret_gross"] - c / 1e4
        rows.append({
            "cost_bps": c,
            "mean_net": float(net.mean()) if len(net) else float("nan"),
            "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
            "n_trades": int(len(net)),
        })
    return pd.DataFrame(rows).set_index("cost_bps")


def decay_by_year(
    frames: dict[str, pd.DataFrame],
    params: Params = Params(),
    cost_bps: float = 15.0,
    sigs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """The same rule, sliced by entry year — does the 'edge' cluster in one era?"""
    led = backtest.run_universe(frames, params=params, cost_bps=cost_bps, sigs=sigs)
    if led.empty:
        return pd.DataFrame(columns=["n_trades", "mean_net", "win_rate"])
    led = led.copy()
    led["year"] = pd.to_datetime(led["entry_date"]).dt.year
    g = led.groupby("year").agg(
        n_trades=("ret_net", "size"),
        mean_net=("ret_net", "mean"),
        win_rate=("ret_net", lambda s: float((s > 0).mean())),
    )
    return g


def capacity(dollar_vol: pd.Series, ledger: pd.DataFrame, participation: float = 0.01) -> dict:
    """Crude capacity: median daily $-volume of traded names x participation cap.

    A swing trade entered at the open can take only a slice of a day's volume without moving
    price. We report the median traded name's daily dollar volume and the position size at a
    ``participation`` (1%) cap — a rough ceiling on size per name, not a portfolio figure.
    """
    if ledger.empty or dollar_vol.empty:
        return {"median_dollar_vol": float("nan"), "size_at_cap": float("nan")}
    dv = float(dollar_vol.median())
    return {"median_dollar_vol": dv, "size_at_cap": dv * participation}


# --------------------------------------------------------------------------- #
# Synthetic-only: did the detector find the planted springboards?
# --------------------------------------------------------------------------- #

def detector_recall(frames, true_springs, params: Params = Params(), tol: int = 12) -> dict:
    """Share of planted breakouts the detector fires near (within ``tol`` bars), offline.

    Validates the machinery on ground truth: on the synthetic universe the rule *should*
    fire on the planted springboards. A low recall would mean the detector is broken, not
    that the market is efficient — so this guards the real-data verdict.
    """
    by_ticker: dict[str, list[int]] = {}
    for ts in true_springs:
        by_ticker.setdefault(ts.ticker, []).append(ts.breakout_index)
    hit, total = 0, 0
    for tk, planted in by_ticker.items():
        f = frames[tk].sort_index()
        pos = {d: i for i, d in enumerate(f.index)}
        sig = find_signals(f, params)
        fired = sorted(pos[d] for d in sig["breakout_date"] if d in pos)
        fired = np.array(fired, dtype=int)
        for b in planted:
            total += 1
            if fired.size and np.min(np.abs(fired - b)) <= tol:
                hit += 1
    return {"recall": hit / total if total else float("nan"), "n_planted": total, "n_hit": hit}
