"""Strategy and honest controls — Study 207 (REITs-Diversifier).

We test three interrelated claims about REITs (proxied by VNQ):

1. **Diversification claim.** Adding a REIT sleeve (e.g. 60/20/20 SPY/VNQ/TLT)
   improves the Sharpe and reduces drawdown vs a vanilla 60/40 (SPY/TLT).
2. **Inflation-hedge claim.** REITs outperform in high-CPI regimes (they hold real
   assets, rents adjust upward) — but this claim collides with interest-rate
   sensitivity (rising rates hurt REIT valuations).
3. **Crisis resilience claim.** Unlike stocks, REITs hold up in equity crashes
   because they have stable rental income.

We run two honest controls against all three:

- **Buy-and-hold SPY** — the return ceiling, no diversification benefit needed.
- **60/40 (SPY/TLT) annually rebalanced** — the mainstream diversified baseline;
  the REIT sleeve must clear this bar to matter.

All portfolios are rebalanced annually on the first trading day of each year.
Costs: one-way round-trip turnover charged at ``cost_bps`` on each rebalance leg.
No look-ahead: rebalance weights are fixed ex-ante; no hindsight optimisation.

Conventions
-----------
- Inputs are **price frames** (total-return, from ``data.load_real``).
- Returns are simple daily: ``pct_change().dropna()``.
- Sharpe is annualised on **excess-of-SHY** returns (like-for-like basis).
- HAC t-stat uses Newey-West on the annual return *differences*
  (REIT-sleeve portfolio minus benchmark), matching the rebalance cadence.
- Correlation is computed on rolling 252-day windows to track regime variation.
- CPI regimes split the Shiller monthly CPI series into high/low inflation terciles.
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
    """Maximum peak-to-trough drawdown from an equity curve."""
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _worst_year(daily_ret: pd.Series) -> float:
    """Worst calendar-year total return from a daily return series."""
    annual = (1.0 + daily_ret).groupby(daily_ret.index.year).prod() - 1.0
    return float(annual.min())


def annual_returns(daily: pd.Series) -> pd.Series:
    """Calendar-year total returns from a daily-return series."""
    return (1.0 + daily).groupby(daily.index.year).prod() - 1.0


# ---------------------------------------------------------------------------
# Portfolio engine — fixed weights, annual rebalance
# ---------------------------------------------------------------------------
def blended_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "annual",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of a fixed-weight blend, rebalanced on a calendar schedule.

    Weights drift between rebalances; on each rebalance date the book is reset to
    ``weights`` and the one-way turnover (sum of absolute weight changes) is charged
    at ``cost_bps``. ``rebalance='annual'`` resets on the first trading day of each
    calendar year; ``'none'`` lets the mix drift forever (buy-and-hold).

    Returns a daily net-return Series aligned to ``returns.index``.
    """
    cols = list(weights.keys())
    R = returns[cols].to_numpy()
    w_target = np.array([weights[c] for c in cols], dtype=float)
    n = R.shape[0]
    idx = returns.index

    if rebalance == "annual":
        marks = idx.to_series().groupby(idx.year).head(1).index
    elif rebalance == "none":
        marks = idx[:1]
    else:
        raise ValueError(f"unknown rebalance schedule: {rebalance!r}")
    rebal_set = set(marks)

    w = w_target.copy()
    out = np.empty(n)
    cost = cost_bps * 1e-4
    for t in range(n):
        turn = 0.0
        if idx[t] in rebal_set:
            turn = float(np.abs(w_target - w).sum())
            w = w_target.copy()
        port_ret = float(w @ R[t]) - turn * cost
        out[t] = port_ret
        w = w * (1.0 + R[t])
        s = w.sum()
        if s > 0:
            w = w / s
    return pd.Series(out, index=idx, name="blend")


def reit_sleeve(
    returns: pd.DataFrame,
    reit_wt: float = 0.20,
    eq_wt: float = 0.60,
    bond_wt: float | None = None,
    cost_bps: float = 0.0,
    eq: str = "SPY",
    reit: str = "VNQ",
    bond: str = "TLT",
) -> pd.Series:
    """60/20/20 REIT-sleeved portfolio (default: eq_wt/reit_wt/(1-eq-reit) bonds),
    annually rebalanced.

    The bond weight defaults to ``1 - eq_wt - reit_wt`` so the portfolio sums to 1.
    """
    if bond_wt is None:
        bond_wt = 1.0 - eq_wt - reit_wt
    weights = {eq: eq_wt, reit: reit_wt, bond: bond_wt}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


def sixty_forty(
    returns: pd.DataFrame,
    cost_bps: float = 0.0,
    eq: str = "SPY",
    bond: str = "TLT",
) -> pd.Series:
    """Classic 60/40 (SPY/TLT), annually rebalanced — the primary benchmark."""
    weights = {eq: 0.60, bond: 0.40}
    return blended_portfolio(returns, weights, rebalance="annual", cost_bps=cost_bps)


def spy_only(returns: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
    """100% SPY — no rebalancing, no costs — the return ceiling."""
    return returns[ticker].rename("SPY")


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------
def portfolio_stats(
    net: pd.Series,
    rf: pd.Series | None = None,
) -> dict:
    """Headline annual stats for a daily-return series.

    ``rf`` is the daily return of a cash proxy (SHY); Sharpe is computed on
    excess-of-cash so all arms are compared like-for-like.
    """
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "worst_year": _worst_year(net),
        "years": years,
    }


# ---------------------------------------------------------------------------
# Inference: Newey-West HAC t-stat on annual return differences
# ---------------------------------------------------------------------------
def hac_tstat_annual(a_daily: pd.Series, b_daily: pd.Series) -> float:
    """Newey-West HAC t-stat for mean(annual return of A − annual return of B).

    Annual granularity matches the rebalance cadence and avoids within-year
    autocorrelation inflating the power. The HAC correction still applies because
    annual excess returns can be serially correlated across macro cycles.
    """
    ar_a = annual_returns(a_daily)
    ar_b = annual_returns(b_daily)
    common = ar_a.index.intersection(ar_b.index)
    diff = ar_a.loc[common] - ar_b.loc[common]
    x = diff.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return float("nan")
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Rolling correlation — does VNQ's equity correlation rise in crashes?
# ---------------------------------------------------------------------------
def rolling_correlation(
    returns: pd.DataFrame,
    asset_a: str = "VNQ",
    asset_b: str = "SPY",
    window: int = 63,
) -> pd.Series:
    """Rolling Pearson correlation between two return series.

    ``window=63`` ≈ one calendar quarter; ``window=252`` ≈ one year.
    Returns a Series with the same index as ``returns``, NaN in the
    initial ``window-1`` rows.
    """
    return returns[asset_a].rolling(window).corr(returns[asset_b])


# ---------------------------------------------------------------------------
# Crisis drawdown analysis — does VNQ cushion or pile on during crashes?
# ---------------------------------------------------------------------------
def equity_drawdown_episodes(
    returns: pd.DataFrame,
    stock: str = "SPY",
    thresh: float = -0.15,
) -> list[dict]:
    """SPY drawdown episodes deeper than ``thresh``, with contemporaneous returns
    of every other column in ``returns``.

    For each episode (peak-to-trough in the SPY leg) we record the SPY loss and
    the total return of VNQ, TLT, GLD, and SHY over the *same* window. This is the
    direct test of the crisis-diversification claim: if VNQ's return over the crisis
    window is substantially better than SPY's, it provided cushion; if it roughly
    matched or exceeded the SPY loss, it piled on.

    The threshold ``-0.15`` (−15%) targets true crashes, not routine 5–10%
    corrections.
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
            peak_date = s.index[max(0, i - 1)]
        elif in_dd and dd.iloc[i] >= 0:
            in_dd = False
            seg = px.loc[peak_date : s.index[i - 1]]
            trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
            loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
            if loss <= thresh:
                w = px.loc[peak_date:trough_date]
                others = {
                    c: float(w[c].iloc[-1] / w[c].iloc[0] - 1.0)
                    for c in returns.columns
                    if c != stock
                }
                episodes.append(
                    {"peak": peak_date, "trough": trough_date,
                     "stock_loss": loss, "others": others}
                )
    # Handle open drawdown at tape end
    if in_dd:
        seg = px.loc[peak_date:]
        trough_date = (seg[stock] / seg[stock].iloc[0] - 1.0).idxmin()
        loss = float(seg[stock].loc[trough_date] / seg[stock].iloc[0] - 1.0)
        if loss <= thresh:
            w = px.loc[peak_date:trough_date]
            others = {
                c: float(w[c].iloc[-1] / w[c].iloc[0] - 1.0)
                for c in returns.columns
                if c != stock
            }
            episodes.append(
                {"peak": peak_date, "trough": trough_date,
                 "stock_loss": loss, "others": others}
            )
    return episodes


# ---------------------------------------------------------------------------
# CPI regime analysis — annual CPI from Shiller data
# ---------------------------------------------------------------------------
def cpi_regimes(
    shiller_path: str,
    low_pct: float = 33.3,
    high_pct: float = 66.7,
) -> pd.Series:
    """Year-over-year CPI inflation from the Shiller dataset, classified into
    tercile regimes: 'low', 'medium', 'high'.

    Returns a Series indexed by year (int) with string labels.
    The Shiller monthly CPI column is 'Consumer Price Index'.
    """
    df = pd.read_parquet(shiller_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    cpi = df["Consumer Price Index"].dropna()
    cpi = cpi[cpi > 0]
    # Year-over-year inflation rate
    yoy = cpi.pct_change(12).dropna()
    # Annual average of monthly YoY
    ann = yoy.groupby(yoy.index.year).mean()
    lo_cut = float(ann.quantile(low_pct / 100.0))
    hi_cut = float(ann.quantile(high_pct / 100.0))

    def _label(x: float) -> str:
        if x <= lo_cut:
            return "low"
        if x >= hi_cut:
            return "high"
        return "medium"

    return ann.map(_label)


def returns_by_cpi_regime(
    daily_ret: pd.Series,
    regimes: pd.Series,
) -> dict[str, dict]:
    """Annual total returns of ``daily_ret`` grouped by CPI regime.

    Returns a dict of regime label → {'mean', 'n', 'years', 'cagr'}.
    """
    ann = annual_returns(daily_ret)
    out: dict[str, dict] = {}
    for label in ("low", "medium", "high"):
        yrs = regimes[regimes == label].index
        subset = ann.loc[ann.index.isin(yrs)]
        n = len(subset)
        if n == 0:
            out[label] = {"n": 0, "mean": float("nan"), "cagr": float("nan")}
            continue
        mu = float(subset.mean())
        cagr = float((1.0 + subset).prod() ** (1.0 / n) - 1.0) if n > 0 else float("nan")
        out[label] = {"n": n, "mean": mu, "cagr": cagr}
    return out


# ---------------------------------------------------------------------------
# Bootstrap Sharpe-difference CI
# ---------------------------------------------------------------------------
def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    rf: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 207,
) -> dict:
    """Circular block bootstrap 95% CI for the Sharpe difference (a − b).

    Resamples both series jointly in circular blocks of length ``block``
    (default 21 days ≈ one calendar month) so volatility clustering survives.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f

    def _sharpe(r: np.ndarray) -> float:
        sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    n = len(ra)
    point = _sharpe(ra) - _sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs, wins = [], 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _sharpe(ra[sel]), _sharpe(rb[sel])
        d = sa - sb
        diffs.append(d)
        if sa > sb:
            wins += 1
    arr = np.array(diffs)
    lo = float(np.nanpercentile(arr, 2.5))
    hi = float(np.nanpercentile(arr, 97.5))
    return {
        "point": float(point),
        "ci95": (lo, hi),
        "frac_a_wins": wins / n_boot,
        "n": n,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# Synthetic summary helper — used in tests and run_synthetic_demo.py
# ---------------------------------------------------------------------------
def summarize_portfolio(
    prices: pd.DataFrame,
    weights_sleeve: dict[str, float],
    weights_bench: dict[str, float],
    cost_bps: float = 1.0,
) -> dict:
    """Compare a REIT sleeve vs a 60/40 benchmark on synthetic price data.

    Returns headline stats for both portfolios and the HAC t-stat of the annual
    return difference. Used in run_synthetic_demo.py and in tests (with synthetic
    data so no cache is needed).
    """
    ret = to_returns(prices)
    # Map the synthetic column names to those in ``weights``
    sleeve_net = blended_portfolio(ret, weights_sleeve, cost_bps=cost_bps)
    bench_net = blended_portfolio(ret, weights_bench, cost_bps=cost_bps)

    s_sleeve = portfolio_stats(sleeve_net)
    s_bench = portfolio_stats(bench_net)
    t = hac_tstat_annual(sleeve_net, bench_net)

    return {
        "sleeve_cagr": s_sleeve["cagr"],
        "sleeve_sharpe": s_sleeve["sharpe"],
        "sleeve_maxdd": s_sleeve["max_dd"],
        "bench_cagr": s_bench["cagr"],
        "bench_sharpe": s_bench["sharpe"],
        "bench_maxdd": s_bench["max_dd"],
        "sharpe_diff": s_sleeve["sharpe"] - s_bench["sharpe"],
        "dd_diff": s_sleeve["max_dd"] - s_bench["max_dd"],
        "hac_t": t,
    }
