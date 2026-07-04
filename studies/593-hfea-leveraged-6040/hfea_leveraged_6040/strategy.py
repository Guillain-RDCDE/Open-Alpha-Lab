"""Strategy + stats for Study 593 — HFEA (UPRO/TMF 55/45).

The tested rule (the Bogleheads "Hedgefundie's Excellent Adventure" recipe,
stated exactly):

    hold 55% UPRO (3x S&P 500) / 45% TMF (3x 20+yr Treasuries);
    at the close of the LAST trading day of each calendar quarter, reset the
    drifted weights back to 55/45. The new weights earn from the NEXT session
    on — the single, documented execution lag. There is no forecast signal:
    the rebalance calendar is known in advance.

The claim under test: this levered, diversified pair COMPOUNDS FASTER than SPY
(and than a plain 60/40) thanks to leveraged diversification — negative
stock–bond correlation lets each leg insure the other, so the pair can carry
3x exposure without a 3x-single-fund's decay-and-drawdown profile.

Inference, the desk's shared spirit:

  * CAGR / vol / max-drawdown / Sharpe races, always **excess-vs-excess**
    (monthly returns minus the compounded ^IRX bill leg);
  * a **Newey-West (HAC) t** on the mean monthly log-return difference
    (HFEA − benchmark) — the honest test of "compounds faster";
  * costs one-way × NAV traded at each quarterly reset (both legs pay);
  * the 2022 autopsy: the stock–bond correlation flip that hit both legs at
    once, per-leg calendar-year damage, and where the pair stands vs its 2021
    peak today;
  * a synthetic control in which the diversification engine (rho < 0 + bond
    carry) is PLANTED or REMOVED, averaged over >= 20 seeds (single-seed
    baselines are banned on this desk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12

HFEA_W = (0.55, 0.45)          # UPRO / TMF
SIXTY_FORTY_W = (0.60, 0.40)   # SPY / TLT


# --------------------------------------------------------------------------- #
# Quarterly-rebalanced two-asset portfolio (daily, with drift + costs)
# --------------------------------------------------------------------------- #
def quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar quarter present in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("Q")).max().values)


def rebalanced(r_a: pd.Series, r_b: pd.Series, w: tuple[float, float],
               cost_bps: float = 0.0) -> dict:
    """Daily returns of a two-asset portfolio reset to ``w`` at quarter ends.

    Weights drift with returns between resets; the reset trade executes at the
    quarter-end close and the new weights earn from the next session (one lag).
    Cost: ``cost_bps`` one-way x the NAV fraction traded (sum of |delta w| over
    both legs), charged on the reset day.
    """
    df = pd.concat({"a": r_a, "b": r_b}, axis=1).dropna()
    qe = set(quarter_ends(df.index))
    wa, wb = float(w[0]), float(w[1])
    out = np.empty(len(df))
    turn = 0.0
    n_reb = 0
    ra_v, rb_v = df["a"].to_numpy(), df["b"].to_numpy()
    dates = df.index
    for i in range(len(df)):
        rp = wa * ra_v[i] + wb * rb_v[i]
        # drift
        wa = wa * (1.0 + ra_v[i]) / (1.0 + rp)
        wb = wb * (1.0 + rb_v[i]) / (1.0 + rp)
        cost = 0.0
        if dates[i] in qe:
            traded = abs(w[0] - wa) + abs(w[1] - wb)
            turn += traded
            n_reb += 1
            cost = cost_bps / 1e4 * traded
            wa, wb = float(w[0]), float(w[1])
        out[i] = rp - cost
    r = pd.Series(out, index=df.index, name="port")
    return {"ret": r, "avg_turnover": turn / max(n_reb, 1), "n_rebalances": n_reb,
            "ann_cost_bps": cost_bps * turn / (len(df) / TRADING_DAYS)}


# --------------------------------------------------------------------------- #
# Performance + inference
# --------------------------------------------------------------------------- #
def monthly(r_daily: pd.Series) -> pd.Series:
    """Compounded calendar-month returns."""
    return (1.0 + r_daily).groupby(r_daily.index.to_period("M")).prod() - 1.0


def max_drawdown(r_daily: pd.Series) -> float:
    nav = (1.0 + r_daily).cumprod()
    return float((nav / nav.cummax() - 1.0).min())


def nw_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_mean_t(x: np.ndarray, lags: int | None = None) -> dict:
    """Newey-West (Bartlett) t of the mean of ``x``."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    mu = x.mean()
    if lags is None:
        lags = nw_lags(n)
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 1e-30) / n)
    return {"mean": float(mu), "t": float(mu / se), "n": n, "lags": lags}


def perf(r_daily: pd.Series, rf_daily: pd.Series) -> dict:
    """CAGR / vol / maxDD / excess-vs-excess Sharpe (+ HAC t of the monthly
    excess mean), all on the aligned daily tape."""
    df = pd.concat({"r": r_daily, "rf": rf_daily}, axis=1).dropna()
    n = len(df)
    yrs = n / TRADING_DAYS
    cagr = float((1.0 + df["r"]).prod() ** (1.0 / yrs) - 1.0)
    m = monthly(df["r"])
    m_rf = monthly(df["rf"])
    exc = (m - m_rf).to_numpy()
    sd = exc.std(ddof=1)
    h = hac_mean_t(exc)
    return {
        "cagr_pct": cagr * 100.0,
        "vol_ann_pct": float(df["r"].std(ddof=1) * np.sqrt(TRADING_DAYS)) * 100.0,
        "max_dd_pct": max_drawdown(df["r"]) * 100.0,
        "sharpe": float(exc.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else np.nan,
        "t_exc_hac": h["t"], "years": yrs, "n_months": len(exc),
    }


def race(r_p: pd.Series, r_bench: pd.Series, rf_daily: pd.Series) -> dict:
    """HFEA-vs-benchmark: HAC t of the mean monthly LOG-return difference (the
    'compounds faster' claim is a geometric claim, so the race runs in logs),
    plus the simple monthly excess difference for reference."""
    df = pd.concat({"p": r_p, "b": r_bench, "rf": rf_daily}, axis=1).dropna()
    mp, mb = monthly(df["p"]), monthly(df["b"])
    dlog = (np.log1p(mp) - np.log1p(mb)).to_numpy()
    h = hac_mean_t(dlog)
    return {
        "dlog_ann_pct": h["mean"] * MONTHS * 100.0,   # log-CAGR gap, %/yr
        "t_dlog_hac": h["t"], "n_months": h["n"], "lags": h["lags"],
    }


# --------------------------------------------------------------------------- #
# The 2022 autopsy + correlation regime
# --------------------------------------------------------------------------- #
def year_return(r_daily: pd.Series, year: int) -> float:
    sel = r_daily[r_daily.index.year == year]
    return float((1.0 + sel).prod() - 1.0) * 100.0


def stock_bond_corr(tape: pd.DataFrame, start: str | None = None,
                    end: str | None = None) -> float:
    """Correlation of MONTHLY SPY/TLT (1x) returns over [start, end]."""
    ms, mb = monthly(tape["spy"]), monthly(tape["tlt"])
    df = pd.concat({"s": ms, "b": mb}, axis=1).dropna()
    if start:
        df = df[df.index >= pd.Period(start, "M")]
    if end:
        df = df[df.index <= pd.Period(end, "M")]
    return float(df["s"].corr(df["b"]))


def rolling_corr(tape: pd.DataFrame, window: int = 24) -> pd.Series:
    """Rolling ``window``-month SPY/TLT correlation (monthly returns)."""
    ms, mb = monthly(tape["spy"]), monthly(tape["tlt"])
    df = pd.concat({"s": ms, "b": mb}, axis=1).dropna()
    return df["s"].rolling(window).corr(df["b"])


def drawdown_state(r_daily: pd.Series) -> dict:
    """Peak date, trough date/depth, and where the NAV stands now vs the peak."""
    nav = (1.0 + r_daily).cumprod()
    dd = nav / nav.cummax() - 1.0
    trough = dd.idxmin()
    peak = nav[:trough].idxmax()
    return {
        "peak": str(peak.date()), "trough": str(trough.date()),
        "depth_pct": float(dd.min()) * 100.0,
        "now_vs_peak_pct": float(nav.iloc[-1] / nav[:trough].max() - 1.0) * 100.0,
    }


# --------------------------------------------------------------------------- #
# Synthetic control — plant / remove the diversification engine (>= 20 seeds)
# --------------------------------------------------------------------------- #
def synthetic_check(rho: float, bond_mu_exc: float, n_seeds: int = 20,
                    n_years: int = 20, base_seed: int = 593) -> dict:
    """Run the full HFEA machinery on ``n_seeds`` independent synthetic worlds.

    Each world: correlated stock/bond daily returns; both legs levered 3x by the
    daily identity (1%/yr all-in fee); 55/45 quarterly rebalance; race vs the 1x
    stock. Reports the across-seed mean HAC t of the monthly log gap, the mean
    log-CAGR gap, and the share of seeds with t >= 2.
    """
    from . import data as d
    ts, gaps = [], []
    for s in range(n_seeds):
        w = d.synthetic_world(n_years=n_years, rho=rho, bond_mu_exc=bond_mu_exc,
                              seed=base_seed + 1000 * s)
        s3 = d.synth_letf(w["spy"], w["rf"], fee_ann=0.01)
        b3 = d.synth_letf(w["tlt"], w["rf"], fee_ann=0.01)
        port = rebalanced(s3, b3, HFEA_W)["ret"]
        r = race(port, w["spy"], w["rf"])
        ts.append(r["t_dlog_hac"])
        gaps.append(r["dlog_ann_pct"])
    ts, gaps = np.array(ts), np.array(gaps)
    return {"mean_t": float(ts.mean()), "sd_t": float(ts.std(ddof=1)),
            "mean_gap_ann_pct": float(gaps.mean()),
            "share_t_ge_2": float((ts >= 2.0).mean()), "n_seeds": n_seeds}
