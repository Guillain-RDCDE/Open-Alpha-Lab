"""Pumpkin-spice-season: month-window seasonality tests and a PSL-season rotation.

The folklore: Starbucks launches the Pumpkin Spice Latte in late August and "pumpkin-spice season"
(Aug–Nov) is when SBUX *beats the market*. We test the tradable version on SBUX total-return **excess
over SPY**: (1) per-month one-sample HAC t-stats on the excess series; (2) is the season group
(Aug–Nov) significantly stronger than the off-season? (3) a block-bootstrap CI on the season-minus-off
spread; (4) a 12-window placebo — is Aug–Nov special among *all* four-month windows? (5) a seasonal
rotation (long SBUX in season, SPY otherwise) raced against buy-and-hold, gross and net.

Conventions, stated up front:

  * **The headline series is EXCESS (SBUX − SPY).** The claim is "beats the market", so the object of
    study is market-*relative* return, not SBUX's raw return (SBUX has trounced SPY over its life — a
    single-name survivor pick — so a raw-return test would just re-discover that selection).
  * **HAC (Newey-West) t-stats**, not naive ones — a single high-beta name's excess clusters; the
    naive t overstates significance. We expose both for honesty.
  * **Total-return, both legs.** SBUX and SPY are dividend-reinvested (yfinance ``auto_adjust``), so
    the excess is a clean total-return spread, labelled as such.
  * **Calendar-known rule, no execution lag.** The season months are known in advance, so positions
    are set at the start of each month with no ``shift`` — a PSL-season rule needs no signal-to-trade lag.
  * **Costs one-way × NAV; the rotation trades twice a year** (into SBUX for the season, back to SPY).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
SEASON_MONTHS = [8, 9, 10, 11]   # Aug–Nov: PSL launch through Thanksgiving → the bullish window
OFF_MONTHS = [m for m in range(1, 13) if m not in SEASON_MONTHS]


def _hac_se(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (Bartlett-kernel) standard error of the sample mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan
    e = x - x.mean()
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        if k >= n:
            break
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (float(e[k:] @ e[:-k]) / n)
    return np.sqrt(max(lrv, 0.0) / n)


def month_stats(series: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, count, naive t-stat and HAC t-stat for a monthly return series.

    Applied to the SBUX-minus-SPY excess series. Returns a DataFrame indexed 1..12 with columns
    ``mean``, ``std``, ``n``, ``tstat`` (naive), ``tstat_hac`` (Newey-West). A robust seasonality
    claim needs |t_HAC| ≥ 2 *after* multiple-testing adjustment (Bonferroni for 12 months:
    0.05/12 ≈ 0.004, so effectively |t| ≈ 3).
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    rows = {}
    for m in range(1, 13):
        vals = s[s.index.month == m].dropna()
        n = len(vals)
        if n < 2:
            rows[m] = {"mean": np.nan, "std": np.nan, "n": n, "tstat": np.nan, "tstat_hac": np.nan}
            continue
        v = vals.to_numpy()
        mu, sigma = v.mean(), v.std(ddof=1)
        se_hac = _hac_se(v)
        rows[m] = {
            "mean": float(mu),
            "std": float(sigma),
            "n": int(n),
            "tstat": float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan,
            "tstat_hac": float(mu / se_hac) if se_hac and se_hac > 0 else np.nan,
        }
    return pd.DataFrame(rows).T.rename_axis("month")


def season_tstat(series: pd.Series, season: list[int] = SEASON_MONTHS) -> dict:
    """Welch two-sample t-stat comparing season (Aug–Nov) vs off-season monthly excess returns.

    Returns ``season_mean``, ``off_mean``, ``spread``, ``tstat``, ``n_season``, ``n_off``.
    Hypothesis: pumpkin-spice-season months earn more excess-over-market than the rest of the year.
    Robust result needs |t| ≥ 2.
    """
    s = pd.Series(series).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    on = s[s.index.month.isin(season)].dropna().to_numpy()
    off = s[~s.index.month.isin(season)].dropna().to_numpy()
    if len(on) < 2 or len(off) < 2:
        return {k: np.nan for k in ("season_mean", "off_mean", "spread", "tstat", "n_season", "n_off")}
    mu_on, mu_off = on.mean(), off.mean()
    var_on, var_off = on.var(ddof=1), off.var(ddof=1)
    n_on, n_off = len(on), len(off)
    se = np.sqrt(var_on / n_on + var_off / n_off)
    return {
        "season_mean": float(mu_on),
        "off_mean": float(mu_off),
        "spread": float(mu_on - mu_off),
        "tstat": float((mu_on - mu_off) / se) if se > 0 else np.nan,
        "n_season": int(n_on),
        "n_off": int(n_off),
    }


def spread_bootstrap_ci(
    series: pd.Series, season: list[int] = SEASON_MONTHS,
    n_boot: int = 2000, block: int = 12, seed: int = 724, alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the season-minus-off-season monthly-mean excess spread.

    Resamples 12-month blocks (one calendar year) to respect the annual seasonal structure, recomputes
    the season-minus-off spread on each resample, and returns the percentile CI. ``[lo, hi]``
    straddling 0 means the spread is indistinguishable from noise.
    """
    s = pd.Series(series).astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    months = s.index.month.to_numpy()
    vals = s.to_numpy()
    n = len(vals)
    if n < block * 2:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "n_boot": 0}
    rng = np.random.default_rng(seed)
    season_arr = np.asarray(season)

    def spread_of(idx):
        mv, vv = months[idx], vals[idx]
        on = vv[np.isin(mv, season_arr)]
        off = vv[~np.isin(mv, season_arr)]
        if len(on) < 1 or len(off) < 1:
            return np.nan
        return on.mean() - off.mean()

    point = spread_of(np.arange(n))
    n_blocks = int(np.ceil(n / block))
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(st, st + block) % n) for st in starts])[:n]
        d = spread_of(idx)
        if np.isfinite(d):
            draws.append(d)
    draws = np.array(draws)
    return {
        "point": float(point),
        "lo": float(np.quantile(draws, alpha / 2)),
        "hi": float(np.quantile(draws, 1 - alpha / 2)),
        "n_boot": int(len(draws)),
    }


def window_placebo(series: pd.Series, width: int = 4) -> pd.DataFrame:
    """Every ``width``-month rolling window's excess mean vs its complement — the placebo.

    Builds all 12 wrap-around windows of ``width`` consecutive calendar months, and for each computes
    (window-months mean − rest-of-year mean) and its naive Welch t. The pumpkin-spice window (Aug–Nov)
    is just *one* of twelve. If its spread is unremarkable in the ranking, "Aug–Nov" is a story we
    drew around an ordinary slice of the year. Returned DataFrame is sorted by spread, descending,
    with a ``months`` label and an ``is_psl`` flag on the Aug–Nov row.
    """
    s = pd.Series(series).astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    months = s.index.month.to_numpy()
    vals = s.to_numpy()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = []
    for start in range(1, 13):
        win = [((start - 1 + k) % 12) + 1 for k in range(width)]
        on = vals[np.isin(months, win)]
        off = vals[~np.isin(months, win)]
        if len(on) < 2 or len(off) < 2:
            continue
        se = np.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off))
        rows.append({
            "start_month": start,
            "months": "-".join(names[m - 1] for m in win),
            "spread": float(on.mean() - off.mean()),
            "tstat": float((on.mean() - off.mean()) / se) if se > 0 else np.nan,
            "is_psl": win == SEASON_MONTHS,
        })
    out = pd.DataFrame(rows).sort_values("spread", ascending=False).reset_index(drop=True)
    out.index.name = "rank"
    return out


def seasonal_rotation(
    sbux: pd.Series, spy: pd.Series, season: list[int] = SEASON_MONTHS,
) -> pd.Series:
    """Long SBUX in the pumpkin-spice season (Aug–Nov), hold SPY the rest of the year.

    A long-only, always-invested rotation: the believer's tradable thesis (be in Starbucks *for the
    season*, in the market otherwise). Calendar-known → no execution lag. Returns a monthly total-return
    series aligned to ``sbux``.
    """
    sb = pd.Series(sbux).astype(float)
    sb.index = pd.DatetimeIndex(sb.index)
    sp = pd.Series(spy).astype(float).reindex(sb.index)
    in_season = pd.Series(sb.index.month, index=sb.index).isin(season).to_numpy()
    return pd.Series(np.where(in_season, sb.to_numpy(), sp.to_numpy()), index=sb.index).rename("rotation")


def spread_timer(
    excess: pd.Series, tbill: pd.Series | None = None, season: list[int] = SEASON_MONTHS,
) -> pd.Series:
    """Long-SBUX / short-SPY market-neutral pair, on only in the pumpkin-spice season (Aug–Nov).

    Captures the *excess* (SBUX − SPY) in season months and earns the cash leg (``tbill``, or 0)
    when flat. Calendar-known → no execution lag. Returns a monthly return series aligned to ``excess``.
    """
    ex = pd.Series(excess).astype(float)
    ex.index = pd.DatetimeIndex(ex.index)
    cash = pd.Series(0.0, index=ex.index) if tbill is None else pd.Series(tbill).astype(float).reindex(ex.index).fillna(0.0)
    in_season = pd.Series(ex.index.month, index=ex.index).isin(season).to_numpy()
    return pd.Series(np.where(in_season, ex.to_numpy(), cash.to_numpy()), index=ex.index).rename("spread_timer")


def buy_hold(series: pd.Series) -> pd.Series:
    return pd.Series(series).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, periods_per_year: int = MONTHS, rf: pd.Series | None = None) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown for a monthly return series.

    **Sharpe convention**: raw (``mean/std``) when ``rf`` is None; excess-of-cash when ``rf`` is given
    (``mean(r−rf)/std(r−rf)``). Pass the *same* ``rf`` to both legs of a race so it is like-for-like.
    CAGR / vol / max-drawdown always describe the raw series.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    ex = r if rf is None else (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)).dropna()
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    std = r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }


def apply_costs(returns: pd.Series, n_trades_per_year: float, cost_bps_one_way: float) -> pd.Series:
    """Subtract transaction cost spread evenly across the year.

    ``cost_bps_one_way`` is one-way cost in basis points × NAV. The seasonal rotation makes
    ``n_trades_per_year`` one-way trades (into SBUX for the season, back to SPY = 2/yr for the
    long-only rotation; the market-neutral pair has two legs, so 4/yr). We deduct the annual cost
    budget spread across the 12 months.
    """
    r = pd.Series(returns).astype(float).copy()
    monthly_cost = (n_trades_per_year * cost_bps_one_way / 1e4) / MONTHS
    return (r - monthly_cost).rename("net")
