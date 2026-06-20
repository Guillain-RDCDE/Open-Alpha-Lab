"""Strategy and honest controls -- Study 325 (Crypto-Fear-Greed).

The contrarian crypto Fear & Greed claim: forward BTC returns are higher after
the gauge sits in **Extreme Fear** and lower after **Extreme Greed**.  We test
it three ways, exactly the apparatus the desk uses for sentiment-timing claims:

1. **Regime conditional means** -- average forward daily return in each of the
   five published bands (Extreme Fear / Fear / Neutral / Greed / Extreme Greed).
   A monotone-decreasing pattern (fear best, greed worst) is the folk claim.

2. **Fear-minus-greed spread** -- a dollar-neutral long/short timing series:
   +1 (long BTC) in Extreme-Fear days, -1 (short) in Extreme-Greed days, 0
   otherwise.  This is the tradable distillation of the claim and the series
   whose HAC *t* decides the Signal axis.

3. **Linear sentiment tilt** -- regress forward return on a centred, scaled
   gauge score; a negative slope is the contrarian effect.

**No look-ahead / one execution lag**: the gauge known at the close of day *t*
is paired with the return earned over day *t+1* (read today's close gauge, trade
tomorrow).  ``forward_returns`` shifts the price return by one day so the signal
at row *t* never sees its own outcome.  The gauge itself uses only past+present
bars (see ``data.proxy_gauge``).

**Costs**: a contrarian overlay that flips long/flat/short pays one-way turnover
costs on every state change; shorts additionally pay a borrow / funding fee
(crypto perps charge funding, spot shorts pay borrow).  ``net_spread_returns``
applies both, one-way x NAV.

**Total-return caveat**: BTC has no dividends, so the price-only / total-return
distinction is moot for a single coin; the spread is in any case dollar-neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXTREME_FEAR_MAX = 25.0
EXTREME_GREED_MIN = 75.0

BAND_EDGES = [0.0, 25.0, 45.0, 55.0, 75.0, 100.01]
BAND_LABELS = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

PERIODS_PER_YEAR = 365  # crypto trades daily


# ---------------------------------------------------------------------------
# Forward returns (the one-day execution lag)
# ---------------------------------------------------------------------------
def forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach next-day simple return to each (close, fng) row.

    The signal at day *t* is ``fng`` on row *t*; the outcome is the return earned
    from close[t] to close[t+1].  The last row has no forward return and is
    dropped.  Returns a frame with columns ``fng`` and ``fwd_ret``.
    """
    df = panel.copy()
    df["fwd_ret"] = df["close"].pct_change().shift(-1)
    out = df[["fng", "fwd_ret"]].dropna()
    out.index = pd.DatetimeIndex(out.index)
    return out


# ---------------------------------------------------------------------------
# Regime binning
# ---------------------------------------------------------------------------
def regime_band(fng: pd.Series) -> pd.Series:
    """Map a 0-100 Fear & Greed series to its published band label (categorical)."""
    return pd.cut(
        fng.astype(float),
        bins=BAND_EDGES,
        labels=BAND_LABELS,
        right=False,
        include_lowest=True,
    ).rename("band")


def regime_means(fr: pd.DataFrame) -> pd.DataFrame:
    """Mean / vol / HAC-t of forward returns within each Fear & Greed band.

    Returns a DataFrame indexed by band label with columns ``n``, ``mean``,
    ``mean_ann`` (annualised, 365 days), ``vol``, ``tstat`` (HAC) and ``hit``.
    """
    band = regime_band(fr["fng"])
    rows = []
    for lbl in BAND_LABELS:
        sub = fr["fwd_ret"][band == lbl]
        s = summarize(sub)
        rows.append(
            {
                "band": lbl,
                "n": s["n"],
                "mean": s["mean"],
                "mean_ann": s["mean"] * PERIODS_PER_YEAR if np.isfinite(s["mean"]) else np.nan,
                "vol": s["vol"],
                "tstat": s["tstat"],
                "hit": s["hit_rate"],
            }
        )
    return pd.DataFrame(rows).set_index("band")


# ---------------------------------------------------------------------------
# Contrarian fear-minus-greed spread (the decisive tradable series)
# ---------------------------------------------------------------------------
def contrarian_position(
    fng: pd.Series,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> pd.Series:
    """Target position from the gauge: +1 fear, -1 greed, 0 else.

    Long BTC (+1) in Extreme Fear (fng < ``fear_max``), short (-1) in Extreme
    Greed (fng >= ``greed_min``), flat otherwise.
    """
    pos = pd.Series(0.0, index=fng.index, name="pos")
    pos[fng < fear_max] = 1.0
    pos[fng >= greed_min] = -1.0
    return pos


def spread_returns(fr: pd.DataFrame, **kw) -> pd.Series:
    """Realised daily return of the contrarian fear-minus-greed overlay.

    ``position(t) * fwd_ret(t)`` -- the position is set from the day-t gauge and
    earns day t+1's return.  Days with position 0 contribute 0.
    """
    pos = contrarian_position(fr["fng"], **kw)
    return (pos * fr["fwd_ret"]).rename("spread")


def long_only_fear_excess(fr: pd.DataFrame, fear_max: float = EXTREME_FEAR_MAX) -> pd.Series:
    """Long-only excess: (fear-day forward return) minus (all-day mean forward return).

    The no-shorting version of the claim: how much does buying *only* in
    Extreme-Fear days beat being passively long every day?
    """
    mkt_mean = fr["fwd_ret"].mean()
    fear_mask = fr["fng"] < fear_max
    return (fr["fwd_ret"][fear_mask] - mkt_mean).rename("fear_excess")


# ---------------------------------------------------------------------------
# Costs: turnover + borrow/funding on the contrarian overlay
# ---------------------------------------------------------------------------
def net_spread_returns(
    fr: pd.DataFrame,
    one_way_bps: float = 10.0,
    borrow_bps_ann: float = 1000.0,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> pd.Series:
    """Contrarian overlay return net of one-way trading costs and short funding.

    On every change in target position we pay ``one_way_bps`` x |delta position|
    x NAV (one-way x NAV).  In any day the position is short (-1) we additionally
    pay ``borrow_bps_ann / 365`` of NAV in borrow/funding -- crypto shorts are
    expensive (perp funding + spot borrow run into the high hundreds of bps/yr),
    so the default 1000 bps/yr is deliberately realistic, not punitive.
    """
    pos = contrarian_position(fr["fng"], fear_max=fear_max, greed_min=greed_min)
    gross = pos * fr["fwd_ret"]

    dpos = pos.diff().abs().fillna(pos.abs())  # first day pays to open
    trade_cost = dpos * one_way_bps * 1e-4
    borrow_cost = (pos < 0).astype(float) * (borrow_bps_ann * 1e-4) / PERIODS_PER_YEAR
    return (gross - trade_cost - borrow_cost).rename("net_spread")


def turnover_stats(
    fr: pd.DataFrame,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> dict:
    """Position-state statistics: share of days long / flat / short and trade count."""
    pos = contrarian_position(fr["fng"], fear_max=fear_max, greed_min=greed_min)
    n = len(pos)
    trades = int((pos.diff().abs() > 0).sum())
    return {
        "n_days": n,
        "pct_long": float((pos == 1).mean()) if n else float("nan"),
        "pct_flat": float((pos == 0).mean()) if n else float("nan"),
        "pct_short": float((pos == -1).mean()) if n else float("nan"),
        "n_trades": trades,
        "trades_per_year": float(trades / (n / PERIODS_PER_YEAR)) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Linear sentiment tilt (slope of forward return on gauge)
# ---------------------------------------------------------------------------
def sentiment_slope(fr: pd.DataFrame) -> dict:
    """OLS of forward return on centred/scaled gauge ((fng-50)/50), HAC t.

    A *negative* slope is the contrarian effect (more greed -> lower forward
    return).  Returns the slope (per 50 gauge points) plus its Newey-West t.
    """
    x = ((fr["fng"].astype(float) - 50.0) / 50.0).to_numpy()
    y = fr["fwd_ret"].astype(float).to_numpy()
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return {"slope": np.nan, "tstat": np.nan, "n": len(y)}
    beta = float((x @ (y - y.mean())) / denom)
    resid = (y - y.mean()) - beta * x
    n = len(y)
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    xe = x * resid
    s = float(xe @ xe)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(xe[k:] @ xe[:-k])
    var_beta = s / (denom**2)
    tstat = beta / np.sqrt(var_beta) if var_beta > 0 else np.nan
    return {"slope": beta, "tstat": float(tstat), "n": n}


# ---------------------------------------------------------------------------
# Random-timing null (shuffled gauge)
# ---------------------------------------------------------------------------
def random_timing_null(
    fr: pd.DataFrame,
    n_draws: int = 1000,
    seed: int = 325,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> pd.Series:
    """Null distribution of the spread mean under randomly *shuffled* gauge.

    For each draw, permute the gauge column (breaking any link to the forward
    returns), recompute the contrarian spread, and record its mean.  The real
    spread mean should sit in the tail if the signal is genuine.
    """
    rng = np.random.default_rng(seed)
    fng = fr["fng"].to_numpy()
    fwd = fr["fwd_ret"].to_numpy()
    means = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(fng)
        pos = np.where(perm < fear_max, 1.0, np.where(perm >= greed_min, -1.0, 0.0))
        means[i] = float((pos * fwd).mean())
    return pd.Series(means, name="null_spread_mean")


# ---------------------------------------------------------------------------
# Block-bootstrap Sharpe CI (circular, preserves vol clustering)
# ---------------------------------------------------------------------------
def block_bootstrap_sharpe_ci(
    series: pd.Series,
    n_boot: int = 2000,
    block: int = 20,
    periods_per_year: int = PERIODS_PER_YEAR,
    seed: int = 325,
) -> dict:
    """Circular block-bootstrap CI on the annualised Sharpe of a return series.

    i.i.d. resampling would destroy crypto's volatility clustering, so we resample
    contiguous blocks (circular wrap).  Returns the point Sharpe, 95% CI and the
    fraction of resamples with a negative Sharpe.
    """
    r = pd.Series(series).astype(float).dropna().to_numpy()
    n = r.size
    if n < block + 2:
        return {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan, "frac_negative": np.nan}
    ann = np.sqrt(periods_per_year)

    def _sr(a: np.ndarray) -> float:
        sd = a.std(ddof=1)
        return float(a.mean() / sd * ann) if sd > 0 else np.nan

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        boots[b] = _sr(r[idx[:n]])
    boots = boots[np.isfinite(boots)]
    return {
        "sharpe": _sr(r),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "frac_negative": float((boots < 0).mean()),
    }


# ---------------------------------------------------------------------------
# Summary statistics (Newey-West HAC t-stat)
# ---------------------------------------------------------------------------
def summarize(series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """Headline statistics for a daily return series (HAC t-stat)."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown")} | {"n": n}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


def annualise(stats: dict, periods: int = PERIODS_PER_YEAR) -> dict:
    """Convert daily stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if out.get("vol_ann", 0) > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out
