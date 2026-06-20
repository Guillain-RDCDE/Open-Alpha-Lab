"""The bank-loan identity engine and its honest controls — Study 340.

The folk claim, sold at full strength: *"Senior bank loans (e.g. BKLN) float with
short-term rates, so they barely fall when rates rise — a juicy high-yield coupon with
almost no interest-rate risk."* We take BKLN apart against long Treasuries (TLT),
intermediate Treasuries (IEF) and equities (SPY) on two axes:

- **Rate sensitivity (the claim)** — how much of a rate move does BKLN inherit? We measure
  BKLN's beta to the rate factor (proxied by TLT, the most rate-sensitive arm) in the body
  and on the worst duration days, and BKLN's return in every >5% TLT (rate-driven)
  drawdown. The brochure's "rate protection" is real if this beta is near zero.
- **Where the risk hides (the catch)** — floating-rate loans bear *credit* risk, not
  *duration* risk. We measure BKLN's downside beta to SPY and its return in every >10%
  equity selloff. A high equity-tail beta means you swapped rate risk for credit risk.

Conventions: returns are simple daily total returns from the price frame. This is an
identity study of a single instrument — there is no trading rule, so no execution lag is
needed; where a synthetic "harvest the coupon" overlay is shown, costs are charged
one-way x NAV at ``cost_bps`` with the position entered on a one-day lag (signal known at
close *t* earns the return of *t+1*). Inference on any difference series uses a Newey-West
(HAC) *t* and a circular block bootstrap, per the desk house style.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a total-return price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def stats(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """Headline stats for a daily return series.

    If ``rf`` (a daily risk-free / cash return series) is given, the Sharpe is computed on
    the **excess-of-cash** return; CAGR/vol/drawdown stay on the raw total-return series.
    """
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float) if rf is not None else net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# The identity test — who/what does BKLN move with?
# ---------------------------------------------------------------------------
def ols_beta(y: pd.Series, x: pd.Series) -> float:
    """Univariate OLS slope of ``y`` on ``x`` (the beta of y to x), aligned & dropna."""
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    yy = df.iloc[:, 0].to_numpy(dtype=float)
    xx = df.iloc[:, 1].to_numpy(dtype=float)
    xc = xx - xx.mean()
    denom = float(xc @ xc)
    return float((xc @ (yy - yy.mean())) / denom) if denom > 0 else float("nan")


def hac_beta_t(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """(beta, HAC t) of ``y`` on ``x`` via the influence series whose mean is the slope."""
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    p = df.iloc[:, 0].to_numpy(dtype=float)
    xx = df.iloc[:, 1].to_numpy(dtype=float)
    xc = xx - xx.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return float("nan"), float("nan")
    infl = (p - p.mean()) * xc / denom * len(xc)
    return float(infl.mean()), hac_tstat(infl)


def downside_beta(y: pd.Series, x: pd.Series, q: float = 0.10) -> float:
    """Beta of ``y`` to ``x`` restricted to the worst ``q`` days of ``x``.

    For the credit-risk test this is the number that matters: when equities are falling
    hard, how much of that fall does BKLN inherit? A high downside beta to SPY means the
    "low rate risk" instrument is really a credit-risk instrument in disguise.
    """
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    thresh = df.iloc[:, 1].quantile(q)
    tail = df[df.iloc[:, 1] <= thresh]
    if len(tail) < 5:
        return float("nan")
    return ols_beta(tail.iloc[:, 0], tail.iloc[:, 1])


def asset_drawdowns(
    returns: pd.DataFrame, driver: str, thresh: float = -0.10
) -> list[dict]:
    """Peak-to-trough ``driver`` drawdowns deeper than ``thresh`` (e.g. -10%).

    For each episode (peak -> trough) returns the driver loss and the *contemporaneous*
    total return of every other column over the same window. Use ``driver='SPY'`` for the
    credit test (equity crashes) and ``driver='TLT'`` for the rate test (rate-driven bond
    selloffs) — the same machinery, two questions.
    """
    px = (1.0 + returns).cumprod()
    s = px[driver]
    peak = s.cummax()
    dd = s / peak - 1.0
    episodes: list[dict] = []
    in_dd = False
    peak_date = s.index[0]
    for i in range(len(s)):
        if not in_dd and dd.iloc[i] < 0:
            in_dd = True
            peak_date = s.index[i - 1] if i > 0 else s.index[0]
        elif in_dd and dd.iloc[i] >= 0:
            in_dd = False
            episodes += _maybe_episode(px, returns, driver, peak_date, s.index[i - 1], thresh)
    if in_dd:  # an open drawdown at the end of the tape
        episodes += _maybe_episode(px, returns, driver, peak_date, s.index[-1], thresh)
    return episodes


def _maybe_episode(px, returns, driver, peak_date, end_date, thresh):
    seg = px.loc[peak_date:end_date]
    trough_date = (seg[driver] / seg[driver].iloc[0] - 1.0).idxmin()
    loss = float(seg[driver].loc[trough_date] / seg[driver].iloc[0] - 1.0)
    if loss > thresh:
        return []
    window = px.loc[peak_date:trough_date]
    others = {c: float(window[c].iloc[-1] / window[c].iloc[0] - 1.0)
              for c in returns.columns if c != driver}
    return [{"peak": peak_date, "trough": trough_date, "driver_loss": loss, "others": others}]


def rolling_correlation(
    returns: pd.DataFrame, a: str, b: str, window: int = 63
) -> pd.Series:
    """Rolling correlation between two daily return series (default ~quarter window)."""
    return returns[a].rolling(window, min_periods=window).corr(returns[b])


def window_return(returns: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Total return of every column over a calendar window [start, end] (inclusive)."""
    seg = returns.loc[start:end]
    px = (1.0 + seg).cumprod()
    return {c: float(px[c].iloc[-1] - 1.0) for c in seg.columns}


# ---------------------------------------------------------------------------
# Inference: HAC t-stat and circular block bootstrap
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def bootstrap_beta_diff(
    loan: pd.Series,
    rate_ref: pd.Series,
    equity_ref: pd.Series,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 340,
) -> dict:
    """Circular block bootstrap CI for (full beta to equity − full beta to the rate ref).

    Resamples the three aligned daily return series **jointly** in circular blocks (so the
    cross-asset co-movement and the volatility clustering both survive), recomputes both
    betas per resample, and returns the point difference with a 95% CI and the fraction of
    resamples in which the equity beta exceeds the rate beta. A CI strictly above zero is
    the quantitative version of "BKLN is more a credit/equity bet than a duration bet" —
    i.e. the rate-protection claim is real but the risk just moved to credit.
    """
    idx = loan.index.intersection(rate_ref.index).intersection(equity_ref.index)
    L = loan.reindex(idx).to_numpy(dtype=float)
    R = rate_ref.reindex(idx).to_numpy(dtype=float)
    E = equity_ref.reindex(idx).to_numpy(dtype=float)
    n = len(idx)

    def _beta(p, ref):
        xc = ref - ref.mean()
        denom = float(xc @ xc)
        return float((xc @ (p - p.mean())) / denom) if denom > 0 else float("nan")

    point = _beta(L, E) - _beta(L, R)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        be, br = _beta(L[sel], E[sel]), _beta(L[sel], R[sel])
        diffs[i] = be - br
        if be > br:
            wins += 1
    finite = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": (float(lo), float(hi)),
        "frac_equity_wins": wins / n_boot,
        "n": n, "block": block, "n_boot": n_boot,
    }
