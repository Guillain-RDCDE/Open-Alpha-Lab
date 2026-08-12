"""The seniority-premium engine and its honest controls — Study 907.

The folk claim, at full strength: *"Senior secured loans sit above high-yield bonds in the
capital stack at a similar yield — better recovery, floating coupon — so a loan sleeve pays
you the same carry with less risk: a **seniority premium** collected for free."* We take the
loan sleeve (BKLN, SRLN) apart against the high-yield sleeve (HYG, JNK) on the axes that
decide it:

- **The risk-adjusted race (the claim)** — both legs measured **excess of cash** (BIL); does
  the loan leg earn a genuinely higher **excess-Sharpe**, and does a bootstrap keep that
  advantage clear of zero *and* hold across sub-eras? A real seniority premium shows up here.
- **The return premium (the carry claim)** — is the loans-minus-HY mean return positive and
  HAC-significant, or do loans quietly earn *less* (lower vol paid for with lower carry)?
- **Where seniority helps and where it bites (the catch)** — loans really do cushion
  rate/spread selloffs (floating rate, first lien), but they are the *less liquid* leg and
  can gap **worse** in a pure liquidity crisis. The per-episode stress table shows which.
- **Tradability** — the natural trade (long loans / short HY) charged one-way cost × NAV per
  rebalance + borrow on the short HY leg. If the spread is negative gross, no cost schedule
  saves it.

Conventions: returns are simple daily total returns. Sharpe is always **excess of the cash
leg**. Inference on any difference series uses a Newey-West (HAC) *t* and a circular block
bootstrap, per the desk house style. The costed long-short enters on a one-day lag (a
constant-weight pair rebalanced monthly; no per-date look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return / composite helpers
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily total returns from a price frame (first row dropped)."""
    return prices.pct_change().iloc[1:]


def composite(ret: pd.DataFrame, cols) -> pd.Series:
    """Equal-weight composite of the columns present, skipping NaN (a late lister just
    joins the average when it starts trading)."""
    present = [c for c in cols if c in ret.columns]
    return ret[present].mean(axis=1)


def excess(r: pd.Series, cash: pd.Series) -> pd.Series:
    """Return series minus the aligned cash (BIL) return — the excess-of-cash leg."""
    return (r - cash.reindex(r.index)).dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def annualized_sharpe(excess_ret: pd.Series | np.ndarray) -> float:
    """Annualised Sharpe of an already-excess-of-cash return series."""
    x = np.asarray(excess_ret, dtype=float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def arm_stats(r: pd.Series, cash: pd.Series) -> dict:
    """Headline stats for one arm: CAGR / vol / excess-Sharpe / max drawdown."""
    r = r.dropna().astype(float)
    equity = (1.0 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = annualized_sharpe(excess(r, cash))
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()), "n": len(r),
    }


# --------------------------------------------------------------------------- #
# Inference primitives (local, no quantlab dep — mirrors the house helpers)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
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
# The spread: loans minus HY
# --------------------------------------------------------------------------- #
def spread_stats(loans: pd.Series, hy: pd.Series, nw_lags: int = 10) -> dict:
    """The loans-minus-HY daily return spread — is there a return *premium* to seniority?

    (Excess-of-cash cancels in the difference, so the raw return spread already is the
    excess-vs-excess return difference.) Positive & HAC-significant => loans out-earn HY.
    """
    df = pd.concat([loans.rename("L"), hy.rename("H")], axis=1).dropna()
    sp = (df["L"] - df["H"]).to_numpy(dtype=float)
    return {
        "n_days": int(len(sp)),
        "mean_bps": float(np.mean(sp) * 1e4),
        "ann_pct": float(np.mean(sp) * TRADING_DAYS * 100.0),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
    }


def sharpe_advantage(loans_ex: pd.Series, hy_ex: pd.Series) -> dict:
    """Point excess-Sharpe of each leg and the loans-minus-HY Sharpe advantage."""
    df = pd.concat([loans_ex.rename("L"), hy_ex.rename("H")], axis=1).dropna()
    sL = annualized_sharpe(df["L"]); sH = annualized_sharpe(df["H"])
    return {"sharpe_loans": sL, "sharpe_hy": sH, "advantage": sL - sH, "n_days": int(len(df))}


def bootstrap_sharpe_adv(
    loans_ex: pd.Series,
    hy_ex: pd.Series,
    block: int = 21,
    n_boot: int = 5000,
    seed: int = 907,
) -> dict:
    """Circular block bootstrap CI for the excess-Sharpe advantage (loans − HY).

    The two excess-of-cash series are resampled **jointly** in circular blocks (co-movement
    and vol clustering survive); each resample recomputes both annualised Sharpes and their
    difference. Returns the point advantage, a 95% CI and the fraction of resamples in which
    loans' Sharpe exceeds HY's. A CI that straddles zero => the advantage is within noise.
    """
    df = pd.concat([loans_ex.rename("L"), hy_ex.rename("H")], axis=1).dropna()
    L = df["L"].to_numpy(dtype=float)
    H = df["H"].to_numpy(dtype=float)
    n = len(L)
    point = annualized_sharpe(L) - annualized_sharpe(H)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    advs = np.empty(n_boot)
    wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        off = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = off.reshape(-1)[:n]
        a = annualized_sharpe(L[sel]) - annualized_sharpe(H[sel])
        advs[i] = a
        if a > 0:
            wins += 1
    finite = advs[np.isfinite(advs)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "advantage": float(point), "ci95": (float(lo), float(hi)),
        "frac_loans_wins": wins / n_boot, "n": n, "block": block, "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# Calendar-year and era tables
# --------------------------------------------------------------------------- #
def calendar_year_table(loans: pd.Series, hy: pd.Series) -> pd.DataFrame:
    """Per-calendar-year total return (%) of loans and HY, plus their difference."""
    df = pd.concat([loans.rename("loans"), hy.rename("hy")], axis=1).dropna()
    yr = (1.0 + df).groupby(df.index.year).prod() - 1.0
    yr["loans_minus_hy"] = yr["loans"] - yr["hy"]
    return (yr * 100.0).round(2)


def era_table(loans: pd.Series, hy: pd.Series, cash: pd.Series, cuts) -> list[dict]:
    """Excess-Sharpe of each leg and the loans-minus-HY spread over each ``(lo, hi, label)``.

    Tests whether the sign of the risk-adjusted advantage is *era-robust* — the bar a real
    premium must clear beyond a lucky full-sample number.
    """
    out = []
    for lo, hi, label in cuts:
        lsub = loans[(loans.index >= lo) & (loans.index < hi)]
        hsub = hy[(hy.index >= lo) & (hy.index < hi)]
        ex_l = excess(lsub, cash); ex_h = excess(hsub, cash)
        sp = spread_stats(lsub, hsub)
        out.append({
            "era": label, "n_days": sp["n_days"],
            "sharpe_loans": annualized_sharpe(ex_l),
            "sharpe_hy": annualized_sharpe(ex_h),
            "advantage": annualized_sharpe(ex_l) - annualized_sharpe(ex_h),
            "spread_bps": sp["mean_bps"], "spread_t": sp["t_nw"],
        })
    return out


# --------------------------------------------------------------------------- #
# Stress table — per-episode total return of every leg
# --------------------------------------------------------------------------- #
def stress_table(prices: pd.DataFrame, windows) -> list[dict]:
    """Total return (%) of each price column over each ``(start, end, label)`` window.

    Use it for the credit-stress episodes (energy 2015-16, COVID 2020, the 2022 rate shock):
    where does seniority + floating rate cushion, and where does the loan sleeve's own
    illiquidity make it gap *worse* than HY?
    """
    out = []
    for start, end, label in windows:
        seg = prices.loc[start:end].dropna(how="all")
        rec = {"episode": label, "start": start, "end": end}
        for c in prices.columns:
            s = seg[c].dropna()
            rec[c] = float(s.iloc[-1] / s.iloc[0] - 1.0) * 100.0 if len(s) > 1 else float("nan")
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# The costed long-short (tradability)
# --------------------------------------------------------------------------- #
def costed_long_short(
    loans: pd.Series,
    hy: pd.Series,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 60.0,
    rebals_per_year: int = 12,
    lag: int = 1,
) -> dict:
    """Cost the natural trade: **long loans, short HY**, dollar-neutral.

    The gross daily spread is ``loans − HY`` (both legs long/short one NAV; excess-of-cash
    cancels). We charge, per day: a monthly-rebalanced round trip amortised as
    ``2 × cost_bps × (rebals_per_year / 252)`` on the two-sided book, plus daily borrow
    ``borrow_bps_yr / 252`` on the short HY leg. The book is entered on a ``lag``-day delay
    (a constant-weight pair known at yesterday's close), so there is no look-ahead. If the
    gross spread is already negative, no cost schedule rescues it.
    """
    df = pd.concat([loans.rename("L"), hy.rename("H")], axis=1).dropna()
    sp = (df["L"] - df["H"]).shift(lag).dropna().to_numpy(dtype=float)
    round_trip = 2.0 * cost_bps / 1e4 * (rebals_per_year / TRADING_DAYS)
    borrow_daily = (borrow_bps_yr / 1e4) / TRADING_DAYS
    net = sp - round_trip - borrow_daily
    return {
        "n_days": int(len(sp)),
        "gross_ann_pct": float(np.mean(sp) * TRADING_DAYS * 100.0),
        "net_ann_pct": float(np.mean(net) * TRADING_DAYS * 100.0),
        "cost_bps_per_day": float((round_trip + borrow_daily) * 1e4),
        "net_bps_per_day": float(np.mean(net) * 1e4),
        "t_net_nw": newey_west_t(net),
        "net_sharpe": annualized_sharpe(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(frame: pd.DataFrame, n_boot: int = 1500, seed: int = 907) -> dict:
    """Run the headline detector (excess-Sharpe advantage + bootstrap) on a synthetic frame.

    ``frame`` has columns ``['LOANS','HY','CASH']`` (from :func:`data.synthetic_pair`).
    Returns the point advantage, the spread HAC *t*, and the bootstrap win-fraction — so a
    planted edge must light up and the null must not.
    """
    ret = to_returns(frame)
    cash = ret["CASH"]
    ex_l = excess(ret["LOANS"], cash)
    ex_h = excess(ret["HY"], cash)
    adv = sharpe_advantage(ex_l, ex_h)
    boot = bootstrap_sharpe_adv(ex_l, ex_h, n_boot=n_boot, seed=seed)
    sp = spread_stats(ret["LOANS"], ret["HY"])
    return {
        "advantage": adv["advantage"],
        "sharpe_loans": adv["sharpe_loans"],
        "sharpe_hy": adv["sharpe_hy"],
        "frac_loans_wins": boot["frac_loans_wins"],
        "ci95": boot["ci95"],
        "spread_bps": sp["mean_bps"],
        "spread_t": sp["t_nw"],
        "n_days": adv["n_days"],
    }
