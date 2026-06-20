"""The BDC-yield identity engine and its honest controls — Study 342.

The folk claim, sold at full strength: *"Business development companies (and the BIZD
ETF that bundles them) pay a ~10% yield — high income with bond-like steadiness, because
they hold senior secured loans to private companies."* We take BIZD (VanEck BDC Income
ETF) apart against a bond proxy (IEF, 7-10y Treasuries) and an equity proxy (SPY) on
three axes:

- **Yield vs total return** — does the ~10% headline distribution actually arrive as
  *total return*, or is a chunk of it the fund paying out its own NAV (return of capital)?
- **Volatility & drawdown** — is the ride bond-like, or levered-equity-like?
- **Crash co-movement (the decisive test)** — *into an equity/credit crash*, does BIZD
  follow bonds up (the cushion) or stocks down, amplified (the trap)? We measure downside
  beta to SPY vs to IEF, and BIZD's drawdown alongside each in every >10% equity selloff.

Conventions: returns are simple daily total returns from the price frame. There is no
trading rule here (it is an identity study of a single instrument), so no execution lag is
needed; where a synthetic "harvest the distribution" overlay is shown, costs are charged
one-way × NAV at ``cost_bps`` with the signal-to-fill convention of one ``shift``.
Inference on any difference series uses a Newey-West (HAC) *t* and a circular block
bootstrap, per the desk house style.
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
# The identity test — who does BIZD move with?
# ---------------------------------------------------------------------------
def ols_beta(y: pd.Series, x: pd.Series) -> float:
    """Univariate OLS slope of ``y`` on ``x`` (the beta of y to x), aligned & dropna."""
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    yy = df.iloc[:, 0].to_numpy(dtype=float)
    xx = df.iloc[:, 1].to_numpy(dtype=float)
    xc = xx - xx.mean()
    denom = float(xc @ xc)
    return float((xc @ (yy - yy.mean())) / denom) if denom > 0 else float("nan")


def downside_beta(y: pd.Series, x: pd.Series, q: float = 0.10) -> float:
    """Beta of ``y`` to ``x`` restricted to the worst ``q`` days of ``x``.

    This is the number that matters for the "steady income" claim: when the equity market
    is falling hard, how much of that fall does BIZD inherit? A downside beta to SPY above
    1 (and a low/negative one to IEF) means BIZD is *levered* equity exactly when you'd
    want it to be a calm income sleeve.
    """
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    thresh = df.iloc[:, 1].quantile(q)
    tail = df[df.iloc[:, 1] <= thresh]
    if len(tail) < 5:
        return float("nan")
    return ols_beta(tail.iloc[:, 0], tail.iloc[:, 1])


def equity_drawdowns(
    returns: pd.DataFrame, stock: str, thresh: float = -0.10
) -> list[dict]:
    """Peak-to-trough ``stock`` drawdowns deeper than ``thresh`` (e.g. -10%).

    For each episode (peak → trough) returns the stock loss and the *contemporaneous*
    total return of every other column over the same window — the direct test of whether
    the BDC leg cushioned (rose, like a bond) or piled on (fell, like levered equity).
    """
    px = (1.0 + returns).cumprod()
    s = px[stock]
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
            episodes += _maybe_episode(px, returns, stock, peak_date, s.index[i - 1], thresh)
    if in_dd:  # an open drawdown at the end of the tape
        episodes += _maybe_episode(px, returns, stock, peak_date, s.index[-1], thresh)
    return episodes


def _maybe_episode(px, returns, stock, peak_date, end_date, thresh):
    seg = px.loc[peak_date:end_date]
    trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
    loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
    if loss > thresh:
        return []
    window = px.loc[peak_date:trough_date]
    others = {c: float(window[c].iloc[-1] / window[c].iloc[0] - 1.0)
              for c in returns.columns if c != stock}
    return [{"peak": peak_date, "trough": trough_date, "stock_loss": loss, "others": others}]


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
# The yield-vs-total-return gap — the "is the 10% real?" check
# ---------------------------------------------------------------------------
def headline_vs_realised(total_return_cagr: float, quoted_yield: float = 0.10) -> dict:
    """Compare a quoted distribution yield to the realised total-return CAGR.

    The gap is the part of the headline ~10% that did *not* show up as wealth — fees,
    NAV erosion, return of capital, and crash losses eating the carry. A large positive
    gap (quoted ≫ realised) is the quantitative version of "the yield is partly your own
    money handed back."
    """
    gap = float(quoted_yield - total_return_cagr)
    return {
        "quoted_yield": float(quoted_yield),
        "realised_cagr": float(total_return_cagr),
        "gap": gap,
        "fraction_phantom": float(gap / quoted_yield) if quoted_yield != 0 else float("nan"),
    }


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


def bootstrap_downside_beta_diff(
    bizd: pd.Series,
    spy: pd.Series,
    bnd: pd.Series,
    q: float = 0.10,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 342,
) -> dict:
    """Circular block bootstrap CI for (downside-beta-to-SPY − downside-beta-to-IEF).

    Resamples the three aligned daily return series **jointly** in circular blocks (so the
    cross-asset co-movement and the volatility clustering both survive), recomputes both
    downside betas per resample on that resample's own worst-q SPY days, and returns the
    point difference with a 95% CI and the fraction of resamples in which the SPY downside
    beta exceeds the IEF one. A CI strictly above zero is the quantitative version of "in
    the crash, BIZD is more (levered) equity than bond."
    """
    idx = bizd.index.intersection(spy.index).intersection(bnd.index)
    P = bizd.reindex(idx).to_numpy(dtype=float)
    S = spy.reindex(idx).to_numpy(dtype=float)
    B = bnd.reindex(idx).to_numpy(dtype=float)
    n = len(idx)

    def _dbeta(p, ref):
        thresh = np.quantile(ref, q)
        m = ref <= thresh
        if m.sum() < 5:
            return float("nan")
        xc = ref[m] - ref[m].mean()
        denom = float(xc @ xc)
        return float((xc @ (p[m] - p[m].mean())) / denom) if denom > 0 else float("nan")

    point = _dbeta(P, S) - _dbeta(P, B)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        ds, db = _dbeta(P[sel], S[sel]), _dbeta(P[sel], B[sel])
        diffs[i] = ds - db
        if ds > db:
            wins += 1
    finite = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": (float(lo), float(hi)),
        "frac_spy_wins": wins / n_boot,
        "n": n, "block": block, "n_boot": n_boot,
    }
