"""Strategy and honest controls -- Study 255 (Fear-Greed).

The contrarian Fear & Greed claim: forward returns are higher after the index
sits in **Extreme Fear** and lower after **Extreme Greed**.  We test it three ways:

1. **Regime conditional means** -- average forward weekly return in each of the
   five published bands (Extreme Fear / Fear / Neutral / Greed / Extreme Greed).
   A monotone-decreasing pattern (fear best, greed worst) is the folk claim.

2. **Fear-minus-greed spread** -- a dollar-neutral long/short timing series:
   +1 (long the index) in Extreme-Fear weeks, -1 (short) in Extreme-Greed weeks,
   0 otherwise.  This is the dividend-neutral, tradable distillation of the claim
   and the series whose HAC *t* decides the Signal axis.

3. **Linear sentiment tilt** -- regress forward return on a centred, scaled
   sentiment score; a negative slope is the contrarian effect.

**No look-ahead / one execution lag**: the Fear & Greed reading on Friday of
week t is paired with the return earned over week t+1 (read Friday close, trade
Monday open, hold to next Friday).  ``forward_returns`` shifts the price return
by one week so the signal at row t never sees its own outcome.

**Costs**: a contrarian timing overlay that flips long/flat/short pays one-way
turnover costs on every state change; shorts additionally pay a borrow fee.
``net_spread`` applies both.

**Price-only**: ^GSPC carries no dividends, so the long-only fear excess is
understated by the dividend yield (~1.5-2%/yr).  The fear-minus-greed spread is
dividend-neutral.  Both labels are carried in the notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXTREME_FEAR_MAX = 25.0
EXTREME_GREED_MIN = 75.0

# Published CNN band edges (upper-exclusive except the last).
BAND_EDGES = [0.0, 25.0, 45.0, 55.0, 75.0, 100.01]
BAND_LABELS = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


# ---------------------------------------------------------------------------
# Forward returns (the one-week execution lag)
# ---------------------------------------------------------------------------
def forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach next-week simple return to each (fng, close) row.

    The signal at week t is ``fng`` on row t; the outcome is the return earned
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
    cats = pd.cut(
        fng.astype(float),
        bins=BAND_EDGES,
        labels=BAND_LABELS,
        right=False,
        include_lowest=True,
    )
    return cats.rename("band")


def regime_means(fr: pd.DataFrame) -> pd.DataFrame:
    """Mean / vol / HAC-t of forward returns within each Fear & Greed band.

    Parameters
    ----------
    fr : frame with ``fng`` and ``fwd_ret`` (from :func:`forward_returns`).

    Returns a DataFrame indexed by band label with columns ``n``, ``mean``,
    ``mean_ann`` (annualised, 52 weeks), ``vol``, ``tstat`` (HAC) and ``hit``.
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
                "mean_ann": s["mean"] * 52 if np.isfinite(s["mean"]) else np.nan,
                "vol": s["vol"],
                "tstat": s["tstat"],
                "hit": s["hit_rate"],
            }
        )
    out = pd.DataFrame(rows).set_index("band")
    return out


# ---------------------------------------------------------------------------
# Contrarian fear-minus-greed spread (the decisive tradable series)
# ---------------------------------------------------------------------------
def contrarian_position(
    fng: pd.Series,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> pd.Series:
    """Target position from the Fear & Greed reading: +1 fear, -1 greed, 0 else.

    Long the index (+1) when in Extreme Fear (fng < ``fear_max``), short (-1)
    when in Extreme Greed (fng >= ``greed_min``), flat otherwise.
    """
    pos = pd.Series(0.0, index=fng.index, name="pos")
    pos[fng < fear_max] = 1.0
    pos[fng >= greed_min] = -1.0
    return pos


def spread_returns(fr: pd.DataFrame, **kw) -> pd.Series:
    """Realised weekly return of the contrarian fear-minus-greed overlay.

    ``position(t) * fwd_ret(t)`` -- the position is set from the Friday-t reading
    and earns week t+1's return.  Weeks with position 0 contribute 0.
    """
    pos = contrarian_position(fr["fng"], **kw)
    out = (pos * fr["fwd_ret"]).rename("spread")
    return out


def long_only_fear_excess(fr: pd.DataFrame, fear_max: float = EXTREME_FEAR_MAX) -> pd.Series:
    """Long-only excess: (fear-week forward return) minus (all-week mean forward return).

    The practical, no-shorting version of the claim: how much does buying *only*
    in Extreme-Fear weeks beat being passively invested every week?  Price-only,
    so this is understated by the dividend yield.
    """
    mkt_mean = fr["fwd_ret"].mean()
    fear_mask = fr["fng"] < fear_max
    excess = (fr["fwd_ret"][fear_mask] - mkt_mean).rename("fear_excess")
    return excess


# ---------------------------------------------------------------------------
# Costs: turnover + borrow on the contrarian overlay
# ---------------------------------------------------------------------------
def net_spread_returns(
    fr: pd.DataFrame,
    one_way_bps: float = 5.0,
    borrow_bps_ann: float = 50.0,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> pd.Series:
    """Contrarian overlay return net of one-way trading costs and short borrow.

    On every change in target position we pay ``one_way_bps`` x |delta position|
    x NAV.  In any week the position is short (-1) we additionally pay
    ``borrow_bps_ann / 52`` of NAV in borrow.  Both are charged on the gross
    spread series from :func:`spread_returns`.
    """
    pos = contrarian_position(fr["fng"], fear_max=fear_max, greed_min=greed_min)
    gross = pos * fr["fwd_ret"]

    dpos = pos.diff().abs().fillna(pos.abs())  # first week pays to open
    trade_cost = dpos * one_way_bps * 1e-4
    borrow_cost = (pos < 0).astype(float) * (borrow_bps_ann * 1e-4) / 52.0

    net = (gross - trade_cost - borrow_cost).rename("net_spread")
    return net


def turnover_stats(
    fr: pd.DataFrame,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> dict:
    """Position-state statistics: share of weeks long / flat / short and trade count."""
    pos = contrarian_position(fr["fng"], fear_max=fear_max, greed_min=greed_min)
    n = len(pos)
    trades = int((pos.diff().abs() > 0).sum())
    return {
        "n_weeks": n,
        "pct_long": float((pos == 1).mean()),
        "pct_flat": float((pos == 0).mean()),
        "pct_short": float((pos == -1).mean()),
        "n_trades": trades,
        "trades_per_year": float(trades / (n / 52.0)) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Linear sentiment tilt (slope of forward return on sentiment)
# ---------------------------------------------------------------------------
def sentiment_slope(fr: pd.DataFrame) -> dict:
    """OLS of forward return on centred/scaled sentiment ((fng-50)/50).

    A *negative* slope is the contrarian effect (more greed -> lower forward
    return).  Returns the slope (per unit of scaled sentiment, i.e. per 50 F&G
    points) plus its HAC-style t (Newey-West on the residual).
    """
    x = ((fr["fng"].astype(float) - 50.0) / 50.0).to_numpy()
    y = fr["fwd_ret"].astype(float).to_numpy()
    x = x - x.mean()
    denom = float(x @ x)
    if denom <= 0:
        return {"slope": np.nan, "tstat": np.nan, "n": len(y)}
    beta = float((x @ (y - y.mean())) / denom)
    resid = (y - y.mean()) - beta * x
    # Newey-West variance of beta.
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
# Random-timing null
# ---------------------------------------------------------------------------
def random_timing_null(
    fr: pd.DataFrame,
    n_draws: int = 1000,
    seed: int = 255,
    fear_max: float = EXTREME_FEAR_MAX,
    greed_min: float = EXTREME_GREED_MIN,
) -> pd.Series:
    """Null distribution of the spread mean under randomly *shuffled* sentiment.

    For each draw, permute the Fear & Greed column (breaking any link to the
    forward returns), recompute the contrarian spread, and record its mean.
    The real spread mean should sit in the tail if the signal is genuine.
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
# Summary statistics (Newey-West HAC t-stat)
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a weekly return series (HAC t-stat)."""
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


def annualise_weekly(stats: dict, periods: int = 52) -> dict:
    """Convert weekly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if out.get("vol_ann", 0) > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out
