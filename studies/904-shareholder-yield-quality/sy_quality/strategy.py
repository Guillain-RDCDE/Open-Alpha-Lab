"""Strategy + inference for Study 904 — Shareholder-Yield + Quality.

We build long-only, equal-weight, **monthly-rebalanced** sleeves from live ETFs and
race them (and SPY) on **excess-of-cash** terms:

* **QSY** (quality-screened shareholder yield) = equal weight PKW + QUAL — the "real
  buybacks, not dilution theatre" blend.
* **RAW** (raw buyback) = PKW alone — the unscreened shareholder-yield vehicle.

The two questions:

* *Does quality-screened shareholder yield beat the market?* — QSY vs **SPY**.
* *Does the quality overlay add value over raw buybacks?* — QSY vs **RAW**.

Answered on:

* **Excess-of-cash Sharpe race.** Each sleeve's monthly return minus BIL (the realized
  cash return); annualised Sharpe, plus the **Sharpe gap** for each pairing. Because
  cash cancels in a difference, the sleeve-minus-sleeve monthly spread is cash-independent
  — we put a Newey-West (HAC) *t* on its mean and a **paired moving-block bootstrap** CI
  on the Sharpe gap itself.
* **Drawdown & calendar table.** Max drawdown per sleeve/benchmark and a calendar-year
  table — is the quality-screened ride genuinely shallower?
* **Era cut.** Split the race in two (2020-01) — does either half certify?
* **Costed net version.** Each sleeve is rebalanced monthly to equal weight; we charge a
  one-way spread on the realized turnover (drift back to 50/50) — long-only, no borrow.

Everything is deterministic (fixed seeds); pure numpy + pandas + statsmodels-free HAC.
Mirrors the [601](../../601-factor-etf-live-test/) / [900](../../900-quality-income/)
live-ETF templates and reuses ``quantlab`` helpers where they fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 6) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0 — monthly series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 6:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Sleeves — equal-weight, monthly-rebalanced
# --------------------------------------------------------------------------- #
def sleeve_returns(monthly: pd.DataFrame, members: list[str]) -> pd.Series:
    """Equal-weight, monthly-rebalanced sleeve total return.

    With a rebalance to equal weight at every month-end, the sleeve's month-``t`` return
    is the equal-weight mean of its members' month-``t`` returns. Rows where any member
    is missing (a not-yet-listed ETF) are dropped, so the sleeve starts only once ALL
    members trade — no partial-basket look-ahead. A single-member sleeve (RAW = PKW) is
    just that member's return.
    """
    sub = monthly[members].dropna(how="any")
    return sub.mean(axis=1).rename("+".join(members))


def sleeve_turnover(monthly: pd.DataFrame, members: list[str]) -> pd.Series:
    """Realized one-way turnover of the equal-weight monthly rebalance, per month.

    Over a month the members drift away from equal weight; at month-end we trade back.
    Turnover_t = 0.5 * sum_i |w_drift_i - w_target_i| (one-way fraction of NAV traded).
    The FIRST in-sample month is the initial build (turnover = 0.5 * sum of target
    weights = 0.5). Uses only within-month drift (returns known at ``t``) — no future
    information. A single-member sleeve never drifts, so turnover is 0 after the build.
    """
    sub = monthly[members].dropna(how="any")
    n = len(members)
    w_t = 1.0 / n
    out = np.empty(len(sub))
    prev_w = np.full(n, w_t)
    first = True
    R = sub.to_numpy(dtype=float)
    for i in range(len(sub)):
        if first:
            out[i] = 0.5 * n * w_t  # initial build from cash = 0.5 (one-way)
            first = False
        else:
            grown = prev_w * (1.0 + R[i])
            drift_w = grown / grown.sum()
            out[i] = 0.5 * float(np.abs(drift_w - w_t).sum())
        prev_w = np.full(n, w_t)  # rebalanced back to equal weight
    return pd.Series(out, index=sub.index, name="turnover")


def cash_returns(monthly: pd.DataFrame, cash: str = "BIL") -> pd.Series:
    """The realized monthly cash return = BIL's monthly total return."""
    return monthly[cash].dropna().rename("cash")


def excess(series: pd.Series, cash: pd.Series) -> pd.Series:
    """Excess-of-cash monthly return, aligned on the common index."""
    df = pd.concat([series, cash], axis=1).dropna()
    return (df.iloc[:, 0] - df.iloc[:, 1]).rename(series.name)


# --------------------------------------------------------------------------- #
# Performance / risk
# --------------------------------------------------------------------------- #
def max_drawdown(total_ret: pd.Series) -> float:
    """Max drawdown of a monthly total-return series (compounded wealth curve)."""
    r = total_ret.dropna()
    if len(r) == 0:
        return float("nan")
    curve = (1.0 + r).cumprod()
    return float((curve / curve.cummax() - 1.0).min())


def ann_stats(total_ret: pd.Series, cash: pd.Series) -> dict:
    """CAGR, annualised vol, excess-of-cash Sharpe, max drawdown, growth of $1."""
    df = pd.concat([total_ret, cash], axis=1).dropna()
    r, c = df.iloc[:, 0], df.iloc[:, 1]
    n = len(r)
    if n == 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan,
                "wealth": np.nan, "n_months": 0}
    wealth = float((1.0 + r).prod())
    cagr = wealth ** (MONTHS / n) - 1.0
    vol = float(r.std() * np.sqrt(MONTHS))
    ex = r - c
    sharpe = float(ex.mean() / ex.std() * np.sqrt(MONTHS)) if ex.std() > 0 else np.nan
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "maxdd": max_drawdown(r),
            "wealth": wealth, "n_months": n}


def calendar_year_table(returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Compounded calendar-year total returns for each named sleeve/benchmark."""
    cols = {}
    for name, s in returns.items():
        s = s.dropna()
        cols[name] = s.groupby(s.index.year).apply(lambda x: float((1.0 + x).prod() - 1.0))
    return pd.DataFrame(cols).sort_index()


# --------------------------------------------------------------------------- #
# The Sharpe race + the sleeve-minus-sleeve test
# --------------------------------------------------------------------------- #
def excess_sharpe(sleeve: pd.Series, cash: pd.Series) -> float:
    ex = excess(sleeve, cash)
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(MONTHS)) if sd and sd > 0 else float("nan")


def sharpe_gap_test(a: pd.Series, b: pd.Series, cash: pd.Series,
                    lags: int = 6) -> dict:
    """Excess-of-cash Sharpe race ``a`` vs ``b`` + HAC *t* on the monthly difference.

    The a-minus-b monthly spread ``d = a - b`` is cash-independent (cash cancels), so its
    mean/HAC *t* is the clean "does a out-earn b?" statistic. Reports both legs' excess
    Sharpe, the Sharpe gap, and the spread's mean / one-sample *t* / Newey-West *t*.
    """
    df = pd.concat([a, b, cash], axis=1).dropna()
    av, bv, c = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    d = (av - bv).to_numpy(dtype=float)
    return {
        "n_months": len(df),
        "sharpe_a": excess_sharpe(av, c),
        "sharpe_b": excess_sharpe(bv, c),
        "sharpe_gap": excess_sharpe(av, c) - excess_sharpe(bv, c),
        "diff_mean_bps": float(np.mean(d) * 1e4),
        "diff_ann_pct": float(np.mean(d) * MONTHS * 100),
        "t_1s": one_sample_t(d),
        "t_nw": newey_west_t(d, lags),
    }


def sharpe_gap_bootstrap(a: pd.Series, b: pd.Series, cash: pd.Series,
                         n_draws: int = 4000, block: int = 6, seed: int = 904) -> dict:
    """Paired moving-block bootstrap CI for the excess-of-cash Sharpe GAP (a - b).

    Joint monthly rows (a, b, cash) are resampled in blocks so serial and cross
    correlation survive; the Sharpe gap is recomputed per draw. Reports the 95% CI and
    the share of draws with a NEGATIVE gap (a blunt "could a be no better than b?").
    """
    df = pd.concat([a, b, cash], axis=1).dropna()
    av = df.iloc[:, 0].to_numpy(dtype=float)
    bv = df.iloc[:, 1].to_numpy(dtype=float)
    c = df.iloc[:, 2].to_numpy(dtype=float)
    n = len(av)
    obs = excess_sharpe(df.iloc[:, 0], df.iloc[:, 2]) - excess_sharpe(df.iloc[:, 1], df.iloc[:, 2])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offs = np.arange(block)
    draws = np.empty(n_draws)
    ann = np.sqrt(MONTHS)
    for i in range(n_draws):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        ae, be = av[idx] - c[idx], bv[idx] - c[idx]
        sda, sdb = ae.std(ddof=1), be.std(ddof=1)
        ga = ae.mean() / sda * ann if sda > 0 else np.nan
        gb = be.mean() / sdb * ann if sdb > 0 else np.nan
        draws[i] = ga - gb
    valid = draws[np.isfinite(draws)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return {"obs": float(obs), "lo": float(lo), "hi": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_draws": int(valid.size),
            "block": block, "n_months": n}


def era_cut(a: pd.Series, b: pd.Series, cash: pd.Series,
            split: str = "2020-01-01", lags: int = 6) -> dict:
    """Split the a-vs-b race into two eras at ``split``; report each half."""
    out = {}
    df = pd.concat([a, b, cash], axis=1).dropna()
    for lbl, lo, hi in [("early", df.index.min(), pd.Timestamp(split)),
                        ("late", pd.Timestamp(split), df.index.max() + pd.Timedelta(days=1))]:
        sub = df[(df.index >= lo) & (df.index < hi)]
        if len(sub) < 6:
            out[lbl] = {"n_months": len(sub)}
            continue
        out[lbl] = sharpe_gap_test(sub.iloc[:, 0], sub.iloc[:, 1], sub.iloc[:, 2], lags)
    return out


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def costed_sleeve(monthly: pd.DataFrame, members: list[str], cash: pd.Series,
                  one_way_bps: float = 3.0) -> dict:
    """Gross vs net excess-of-cash Sharpe after charging the monthly-rebalance turnover.

    Charge = one_way_bps x realized one-way turnover, each month. Long-only, no borrow.
    Returns gross/net excess Sharpe, the annual cost drag, and net CAGR.
    """
    gross = sleeve_returns(monthly, members)
    turn = sleeve_turnover(monthly, members)
    cost = turn * (one_way_bps / 1e4)
    net = (gross - cost).rename(gross.name + "(net)")
    g = ann_stats(gross, cash)
    n = ann_stats(net, cash)
    return {
        "n_months": g["n_months"],
        "gross_sharpe": g["sharpe"], "net_sharpe": n["sharpe"],
        "gross_cagr": g["cagr"], "net_cagr": n["cagr"],
        "cost_drag_bps_yr": float(cost.mean() * MONTHS * 1e4),
        "avg_turnover_pct": float(turn.mean() * 100),
        "net": net, "gross": gross,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = 6) -> dict:
    """Run the QSY-vs-raw Sharpe-gap test on a synthetic world (excess-of-cash).

    ``world`` columns ``qsy`` / ``raw`` are already excess of cash, so a zero cash
    series recovers the same statistics the real race computes.
    """
    zero_cash = pd.Series(0.0, index=world.index, name="cash")
    q = world["qsy"].rename("qsy")
    r = world["raw"].rename("raw")
    out = sharpe_gap_test(q, r, zero_cash, lags)
    return {"sharpe_gap": out["sharpe_gap"], "t_nw": out["t_nw"],
            "diff_ann_pct": out["diff_ann_pct"], "n_months": out["n_months"]}
