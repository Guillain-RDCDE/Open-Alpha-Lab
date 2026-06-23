"""Strategy + inference for Study 397 — Hurst-Regime.

The believers' rule. Estimate the **Hurst exponent** ``H`` on a trailing window of returns.
``H`` is supposed to diagnose the *character* of the tape:

    H > 0.5  ->  PERSISTENT / trending  ->  trend-follow (go with recent direction)
    H < 0.5  ->  ANTI-PERSISTENT / mean-reverting  ->  fade (bet on a snap-back)
    H ~ 0.5  ->  random walk -> no edge.

So a **Hurst-gated switch** — trend-follow when H>0.5, mean-revert when H<0.5 — should beat
either style run alone, and certainly beat the styles run in the *wrong* regime.

We test this by building the three books (always-trend, always-revert, Hurst-gated) on a
1-day execution lag, charging one-way costs × turnover, and asking whether the *gated* book's
risk-adjusted return is reliably above the better of the two static styles. Inference is by a
**block bootstrap** of the daily P&L difference (Sharpe / mean-difference), a Welch *t* on the
gated-minus-best-static daily edge, and a **placebo** that shuffles the regime label to break
any real H→style link while preserving the marginal style mix.

The decisive question is not "does the gated book make money" (everything long-ish makes money
on an up-drifting tape) but "does conditioning the *style* on H add anything a shuffled label
wouldn't" — i.e. does the Hurst exponent forecast which style will pay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


# --------------------------------------------------------------------------- #
# Style signals (the two things the Hurst gate switches between)
# --------------------------------------------------------------------------- #
def log_returns(price: pd.Series) -> pd.Series:
    return np.log(price).diff()


def trend_signal(price: pd.Series, lookback: int = 63) -> pd.Series:
    """Trend-following position in {-1, +1}: sign of the trailing ``lookback`` return."""
    mom = np.log(price) - np.log(price.shift(lookback))
    return np.sign(mom).fillna(0.0)


def revert_signal(price: pd.Series, lookback: int = 5) -> pd.Series:
    """Mean-reversion position in {-1, +1}: *against* the trailing short return."""
    mom = np.log(price) - np.log(price.shift(lookback))
    return (-np.sign(mom)).fillna(0.0)


# --------------------------------------------------------------------------- #
# Regime gate
# --------------------------------------------------------------------------- #
def regime_position(price: pd.Series, hurst: pd.Series,
                    trend_lb: int = 63, revert_lb: int = 5,
                    h_mid: float = 0.5) -> pd.Series:
    """Hurst-gated style switch -> a desired position in {-1, 0, +1}.

    Uses the trend signal where ``hurst > h_mid`` and the reversion signal where
    ``hurst < h_mid`` (0 where Hurst is undefined). The position is the *desired* exposure at
    the close of day *t*; execution lag is applied in :func:`book_returns`.
    """
    tr = trend_signal(price, trend_lb)
    rv = revert_signal(price, revert_lb)
    pos = pd.Series(0.0, index=price.index)
    pos = pos.where(hurst.isna(), tr.where(hurst > h_mid, rv))
    pos[hurst.isna()] = 0.0
    return pos


# --------------------------------------------------------------------------- #
# Backtest (one execution lag, one-way costs x turnover)
# --------------------------------------------------------------------------- #
def book_returns(price: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> pd.Series:
    """Daily P&L of holding ``position`` in an asset with adjusted-close ``price``.

    Convention (the desk's single execution lag): the position *desired* at the close of *t*
    earns the asset return of *t+1*. Costs are one-way, charged on the change in position
    (``cost_bps`` per unit of turnover, one-way). Returns a daily P&L series net of costs.
    """
    asset_ret = price.pct_change().shift(-1)          # return earned the day AFTER the signal
    pnl = position * asset_ret
    turn = position.diff().abs().fillna(position.abs())
    cost = turn * (cost_bps / 1e4)
    net = (pnl - cost).dropna()
    return net


def sharpe(daily: pd.Series) -> float:
    """Annualised Sharpe of a daily P&L series (0 mean-vol guard -> NaN)."""
    d = daily.dropna().values
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * np.sqrt(ANN))


# --------------------------------------------------------------------------- #
# Inference — is the GATE worth anything over the best static style?
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of ``mean(a) - mean(b)`` for two paired-length daily-edge samples."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1) / len(a); vb = b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / se)


def paired_t(diff: np.ndarray) -> float:
    """One-sample t of a daily *difference* series (gated minus benchmark)."""
    d = np.asarray(diff, float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def block_bootstrap_sharpe_diff(gated: pd.Series, bench: pd.Series,
                                block: int = 21, n_boot: int = 5_000,
                                seed: int = 397) -> dict:
    """Block-bootstrap the Sharpe *difference* (gated − bench) of two aligned daily P&L books.

    Resamples contiguous blocks of length ``block`` (preserving autocorrelation/vol clustering)
    and recomputes the Sharpe difference each time. Returns the observed difference, a 95% CI,
    and the bootstrap p-value P[diff <= 0] (one-sided: is the gate's Sharpe edge above zero?).
    """
    g, b = gated.align(bench, join="inner")
    g = g.dropna(); b = b.reindex(g.index).dropna(); g = g.reindex(b.index)
    gv, bv = g.values, b.values
    n = len(gv)
    if n < 2 * block:
        return {"obs": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_le0": float("nan")}
    obs = sharpe(g) - sharpe(b)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        gi, bi = gv[idx], bv[idx]
        sg = gi.mean() / gi.std(ddof=1) * np.sqrt(ANN) if gi.std(ddof=1) > 0 else np.nan
        sb = bi.mean() / bi.std(ddof=1) * np.sqrt(ANN) if bi.std(ddof=1) > 0 else np.nan
        diffs[i] = sg - sb
    diffs = diffs[~np.isnan(diffs)]
    return {
        "obs": float(obs),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
        "p_le0": float((diffs <= 0).mean()),
    }


def placebo_shuffled_regime(price: pd.Series, hurst: pd.Series, bench: pd.Series,
                            n_draws: int = 2_000, block: int = 21,
                            cost_bps: float = 1.0, seed: int = 397,
                            trend_lb: int = 63, revert_lb: int = 5,
                            h_mid: float = 0.5) -> dict:
    """Placebo null: break the H→style link by **block-shuffling the regime label**.

    Keeps the *same* trend/revert style books and the *same* marginal mix of trending vs
    reverting days, but reassigns which days are "trend regime" by shuffling the
    ``hurst > h_mid`` mask in blocks (so the gate no longer keys off the real Hurst path). The
    p-value is P[shuffled-gate Sharpe edge >= real-gate Sharpe edge over ``bench``]. If the
    real Hurst gate carries information, the true edge should sit in the right tail.
    """
    tr = trend_signal(price, trend_lb)
    rv = revert_signal(price, revert_lb)
    valid = ~hurst.isna()
    is_trend = (hurst > h_mid) & valid

    real_pos = pd.Series(0.0, index=price.index)
    real_pos[valid] = np.where(is_trend[valid], tr[valid], rv[valid])
    real_net = book_returns(price, real_pos, cost_bps)
    real_edge = sharpe(real_net) - sharpe(bench.reindex(real_net.index))

    mask = is_trend.values.copy()
    n = len(mask)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    edges = np.empty(n_draws)
    for i in range(n_draws):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        shuf = mask[idx]
        pos = pd.Series(0.0, index=price.index)
        v = valid.values
        chosen = np.where(shuf & v, tr.values, rv.values)
        pos.iloc[:] = np.where(v, chosen, 0.0)
        net = book_returns(price, pos, cost_bps)
        edges[i] = sharpe(net) - sharpe(bench.reindex(net.index))
    edges = edges[~np.isnan(edges)]
    p = float((edges >= real_edge).mean()) if len(edges) else float("nan")
    return {"real_edge": float(real_edge), "placebo_mean": float(np.nanmean(edges)),
            "p_value": p}


# --------------------------------------------------------------------------- #
# Headline assembly
# --------------------------------------------------------------------------- #
def build_books(price: pd.Series, hurst: pd.Series, cost_bps: float = 1.0,
                trend_lb: int = 63, revert_lb: int = 5, h_mid: float = 0.5) -> dict:
    """All four daily P&L books on one tape: buy&hold, always-trend, always-revert, gated."""
    bh_pos = pd.Series(1.0, index=price.index)
    tr_pos = trend_signal(price, trend_lb)
    rv_pos = revert_signal(price, revert_lb)
    gate_pos = regime_position(price, hurst, trend_lb, revert_lb, h_mid)
    return {
        "buyhold": book_returns(price, bh_pos, 0.0),
        "trend": book_returns(price, tr_pos, cost_bps),
        "revert": book_returns(price, rv_pos, cost_bps),
        "gated": book_returns(price, gate_pos, cost_bps),
    }


def summarize_books(books: dict) -> dict:
    """Sharpe + annualised mean of each book, aligned to the gated book's dates."""
    idx = books["gated"].dropna().index
    out = {}
    for k, v in books.items():
        s = v.reindex(idx).dropna()
        out[k] = {
            "sharpe": sharpe(s),
            "ann_mean": float(s.mean() * ANN),
            "ann_vol": float(s.std(ddof=1) * np.sqrt(ANN)) if len(s) > 1 else float("nan"),
            "n": int(len(s)),
        }
    return out
