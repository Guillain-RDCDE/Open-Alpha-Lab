"""The strategy and its honest controls - Study 95 (Holiday-Cheer).

The folk claim: 'stocks reliably drift up the trading day BEFORE a market holiday - the
"pre-holiday effect" - several times the normal daily average. Buy the day before every
holiday.' We measure it the obvious way and pin it against the comparisons that decide
whether it is real, and whether it pays:

- **Pre-holiday vs the rest** - mean return on pre-holiday days against all other days,
  with a HAC (Newey-West) t-stat and Wilson win-rate intervals.
- **A pre-holiday-ONLY strategy** vs buy-and-hold - long only on pre-holiday days, in
  cash otherwise, normalised per day invested and charged realistic costs. The effect can
  be large *per day* and still capture too few days to matter as a standalone book.
- **A sub-period contrast (pre-1990 vs post-1990)** with a block-bootstrap test of the
  *difference* - the effect famously faded after Ariel (1990) published it.

Execution convention: a pre-holiday day is fixed by the *calendar* and known well in
advance (you know in June that the day before July 4th will be a pre-holiday day), so the
rule needs **no execution lag** - there is no signal to peek at. Costs are charged
one-way on each entry and each exit; cash earns 0% (a stated, conservative choice).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Pre-holiday vs rest - the core measurement
# ---------------------------------------------------------------------------
def split_returns(close: pd.Series, is_pre: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return ``(pre_holiday_returns, other_returns)`` from a close series.

    Daily simple returns are split by the boolean ``is_pre`` mask (aligned to ``close``).
    The first (NaN) return is dropped from both arms.
    """
    ret = close.pct_change()
    ip = is_pre.reindex(close.index).fillna(False).astype(bool)
    return ret[ip].dropna(), ret[~ip].dropna()


def effect_stats(close: pd.Series, is_pre: pd.Series) -> dict:
    """Mean return on pre-holiday vs other days, the gap, ratio, HAC t and win rates.

    The HAC t-stat is computed on the pre-holiday return series itself (its mean vs zero);
    we also report the t-stat of the *difference in means* via a pooled HAC on a contrast
    series. Win-rate Wilson intervals come from :func:`wilson_interval`.
    """
    r_pre, r_rest = split_returns(close, is_pre)
    mu_pre, mu_rest = float(r_pre.mean()), float(r_rest.mean())
    gap = mu_pre - mu_rest
    ratio = mu_pre / mu_rest if mu_rest != 0 else float("nan")
    win_pre = float((r_pre > 0).mean())
    win_rest = float((r_rest > 0).mean())
    return {
        "n_pre": int(len(r_pre)),
        "n_rest": int(len(r_rest)),
        "mu_pre": mu_pre,
        "mu_rest": mu_rest,
        "gap": gap,
        "ratio": ratio,
        "t_pre": hac_tstat(r_pre.to_numpy()),
        "t_gap": _diff_in_means_tstat(close, is_pre),
        "win_pre": win_pre,
        "win_rest": win_rest,
        "wilson_pre": wilson_interval((r_pre > 0).sum(), len(r_pre)),
        "wilson_rest": wilson_interval((r_rest > 0).sum(), len(r_rest)),
    }


def _diff_in_means_tstat(close: pd.Series, is_pre: pd.Series) -> float:
    """HAC t-stat for the difference (pre-holiday minus rest) in mean daily return.

    Built as a single OLS-style contrast: demean each group, then HAC the dummy-weighted
    residual. In practice we use the simple, robust route - HAC t on the series
    ``ret - mu_rest`` restricted to pre-holiday days is dominated by sample size; instead
    we report the two-sample HAC-corrected t using the pre-holiday group's long-run var.
    """
    ret = close.pct_change().dropna()
    ip = is_pre.reindex(ret.index).fillna(False).astype(bool).to_numpy()
    r = ret.to_numpy()
    rp, rr = r[ip], r[~ip]
    if rp.size <= 5 or rr.size <= 5:
        return float("nan")
    mu_d = rp.mean() - rr.mean()
    # HAC long-run variance of the pre-holiday group (autocorr-robust), plus i.i.d. var of
    # the (huge, low-autocorr) rest group; the pre group dominates the SE.
    var_pre = _hac_lrv(rp - rp.mean()) / rp.size
    var_rest = rr.var(ddof=1) / rr.size
    se = np.sqrt(var_pre + var_rest)
    return float(mu_d / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Sub-period contrast (pre-1990 vs post-1990) with a bootstrap difference test
# ---------------------------------------------------------------------------
def subperiod_contrast(
    close: pd.Series,
    is_pre: pd.Series,
    split_date: str = "1990-01-01",
    block: int = 20,
    n_boot: int = 2000,
    seed: int = 95,
) -> dict:
    """The pre-holiday gap in each sub-period and a block-bootstrap test of the decay.

    Splits the sample at ``split_date`` (justified, not snooped: Ariel's 1990 publication
    is the canonical break for this effect). Reports the pre-holiday-minus-rest gap in each
    half, and a circular-block-bootstrap distribution of the *difference of gaps*
    (early minus late) - so a claim of "faded after publication" carries a test of the
    difference, per the inference bar.
    """
    rng = np.random.default_rng(seed)
    ret = close.pct_change().dropna()
    ip = is_pre.reindex(ret.index).fillna(False).astype(bool)
    early = ret.index < split_date
    late = ret.index >= split_date

    def gap(mask: np.ndarray) -> float:
        rp = ret[mask & ip]
        rr = ret[mask & ~ip]
        return float(rp.mean() - rr.mean())

    g_early = gap(early)
    g_late = gap(late)

    def boot_gaps(sub_ret: pd.Series, sub_ip: pd.Series) -> np.ndarray:
        v = sub_ret.to_numpy()
        p = sub_ip.to_numpy()
        N = len(v)
        if N <= block or p.sum() == 0 or (~p).sum() == 0:
            return np.full(n_boot, np.nan)  # sub-period too short / one-sided to bootstrap
        nb = int(np.ceil(N / block))
        out = np.empty(n_boot)
        for b in range(n_boot):
            starts = rng.integers(0, max(N - block, 1), nb)
            sel = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
            sel = sel % N
            vv, pp = v[sel], p[sel]
            if pp.sum() == 0 or (~pp).sum() == 0:
                out[b] = np.nan
            else:
                out[b] = vv[pp].mean() - vv[~pp].mean()
        return out

    d_early = boot_gaps(ret[early], ip[early])
    d_late = boot_gaps(ret[late], ip[late])
    decay = d_early - d_late
    decay = decay[np.isfinite(decay)]
    if decay.size == 0:
        p_decay = float("nan")
        decay_ci = (float("nan"), float("nan"))
    else:
        p_decay = float((decay > 0).mean())
        decay_ci = (float(np.percentile(decay, 2.5)), float(np.percentile(decay, 97.5)))
    return {
        "split_date": split_date,
        "gap_early": g_early,
        "gap_late": g_late,
        "decay": g_early - g_late,
        "p_decay_gt0": p_decay,
        "decay_ci": decay_ci,
        "t_early": hac_tstat(ret[early & ip].to_numpy()),
        "t_late": hac_tstat(ret[late & ip].to_numpy()),
        "n_pre_early": int((early & ip).sum()),
        "n_pre_late": int((late & ip).sum()),
    }


# ---------------------------------------------------------------------------
# Backtest: a pre-holiday-only book vs buy-and-hold
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def pre_holiday_only(close: pd.Series, is_pre: pd.Series, cost_bps: float = 1.0) -> dict:
    """Long only on pre-holiday days, in cash (0%) otherwise; costs one-way per leg.

    The position is the pre-holiday mask itself (calendar-known, no lag). Each isolated
    pre-holiday day is a round-trip: one entry + one exit, each charged ``cost_bps``
    one-way. Returns net daily series + headline stats, including the per-INVESTED-day
    Sharpe (the honest "how good is the day itself" number) alongside the whole-tape CAGR
    (the honest "what does it do to your wealth" number).
    """
    ret = close.pct_change().fillna(0.0)
    pos = is_pre.reindex(close.index).fillna(False).astype(float)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * cost_bps * 1e-4
    out = _stats(net, pos, turn)
    inv = net[pos > 0]
    out["mean_invested_day"] = float(inv.mean())
    out["sharpe_invested"] = (
        float(inv.mean() / inv.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if inv.std(ddof=1) > 0 else float("nan")
    )
    out["days_invested"] = int((pos > 0).sum())
    return out


def buy_and_hold(close: pd.Series) -> dict:
    """Buy-and-hold the tape: position 1 every day, no costs after entry."""
    ret = close.pct_change().fillna(0.0)
    pos = pd.Series(1.0, index=close.index)
    return _stats(ret, pos, pd.Series(0.0, index=close.index))


def _stats(net: pd.Series, pos: pd.Series, turn: pd.Series) -> dict:
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if net.std(ddof=1) > 0 else float("nan")
    )
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "switches": int((turn > 1e-9).sum()),
        "time_in_market": float((pos > 0).mean()),
        "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Inference helpers - HAC t-stat and Wilson interval (local, no quantlab dep)
# ---------------------------------------------------------------------------
def _hac_lrv(e: np.ndarray, lags: int | None = None) -> float:
    """Newey-West long-run variance of a (demeaned) series ``e``."""
    e = np.asarray(e, dtype=float)
    n = e.size
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return lrv


def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    mu = x.mean()
    lrv = _hac_lrv(x - mu, lags)
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (win rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(centre - half), float(centre + half))
