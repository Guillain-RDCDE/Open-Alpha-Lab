"""Strategy + inference for Study 887 — High-Yield Muni Premium.

The claim: high-yield munis (HYD) pay a fat, tax-advantaged credit spread over
investment-grade munis (MUB). Three questions, three honest answers:

  1. **Is the credit premium real?** Mean monthly ``HYD - MUB`` with a Newey-West
     (HAC) *t*, a block-bootstrap mean CI, an era cut, and the crisis windows (2020,
     2022) where illiquidity bites — plus the excess-vs-excess Sharpe race (minus BIL).
  2. **Does the tax wrapper matter?** Muni coupons are federally tax-exempt. Back out
     each fund's monthly income (distribution) return as ``total_return - price_return``,
     gross the muni yield up to its **tax-equivalent yield** ``y/(1-t)`` and compare with
     taxable HY (HYG); then race after-tax total returns for a top-bracket investor.
  3. **Is it bankable?** A long-only MUB→HYD substitution: one-way cost × NAV on the
     single switch, ETF fees already inside the net-of-fee tape; plus the drawdown bill
     (HYD's crisis illiquidity vs MUB) — the price of admission.

Inference is HAC throughout: monthly credit-spread returns are serially correlated
through the muni-market cycle, so plain *t*'s are banned. Reuses
``quantlab.analytics.mean_tstat_hac`` (HAC mean *t*) and ``quantlab.stats`` Sharpe
helpers where they fit; the block-bootstrap mean CI is local (the desk's mean-CI, not
the Sharpe-CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame, asof: str | None = None) -> pd.DataFrame:
    """Wide monthly simple total returns (month-end to month-end).

    Drops the final calendar month if the price tape ends before that month is over,
    so a stamped run never contains a partial month.
    """
    px = prices if asof is None else prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    ret = m.pct_change()
    last_px = px.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret


def monthly_income(prices_tr: pd.DataFrame, prices_pr: pd.DataFrame,
                   asof: str | None = None) -> pd.DataFrame:
    """Monthly income (distribution) return = total-return minus price-only return.

    The coupon leg — the only part of a muni fund's return that is federally tax-exempt.
    Aligned month-ends; a small negative residual can appear when the two Yahoo series
    are adjusted on slightly different dates, so treat this as an approximation good to
    the annual scale, not a per-month exactness.
    """
    tr = monthly_returns(prices_tr, asof)
    pr = monthly_returns(prices_pr, asof)
    return (tr - pr).dropna(how="all")


def align_common(monthly: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows where every column in ``cols`` has a return (the common sample)."""
    return monthly[cols].dropna()


# --------------------------------------------------------------------------- #
# HAC inference (the single significance engine)
# --------------------------------------------------------------------------- #
def hac_mean(x, lags: int | None = None) -> dict:
    """HAC (Newey-West, Bartlett) mean and *t* of a monthly series vs 0.

    ``lags=None`` uses the rule-of-thumb ``floor(4*(n/100)^(2/9))``. Returns mean in
    bps/month, the HAC t, the lag count and n. This is the excess-spread test.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return {"mean_bps": float("nan"), "tstat": float("nan"), "lags": 0, "n": n}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (float(e[k:] @ e[:-k]) / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean_bps": mu * 1e4, "tstat": mu / se if se > 0 else float("nan"),
            "lags": lags, "n": n}


def bootstrap_mean_ci(x, n_boot: int = 5000, alpha: float = 0.05,
                      seed: int = 887, block_size: int | None = None) -> dict:
    """Circular block-bootstrap CI for the MEAN of a monthly series (bps/month).

    Monthly credit-spread returns cluster, so an i.i.d. bootstrap understates the
    interval. Block length defaults to ``round(n**(1/3))`` (Politis-Romano rate).
    Returns the point mean, the (1-alpha) percentile interval and the share of
    resampled means below zero — a blunt "could it be zero?" read.
    """
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    rng = np.random.default_rng(seed)
    blk = int(block_size) if block_size is not None else max(1, round(n ** (1.0 / 3.0)))
    blk = max(1, min(blk, n))
    n_blocks = n // blk + 1
    offsets = np.arange(blk)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_bps": float(r.mean() * 1e4), "ci_low_bps": float(lo * 1e4),
            "ci_high_bps": float(hi * 1e4), "frac_negative": float((boots < 0).mean()),
            "n": n, "block_size": blk}


# --------------------------------------------------------------------------- #
# Performance & risk
# --------------------------------------------------------------------------- #
def ann_return(monthly_ret: pd.Series) -> float:
    """Geometric annualised return (%) from monthly simple returns."""
    r = pd.Series(monthly_ret).dropna()
    return ((1.0 + r).prod() ** (TRADING_MONTHS / len(r)) - 1.0) * 100.0


def excess(monthly: pd.DataFrame, col: str, rf: str = "BIL") -> pd.Series:
    """Monthly excess return of ``col`` over the T-bill ETF proxy."""
    return (monthly[col] - monthly[rf]).dropna()


def sharpe_excess(monthly: pd.DataFrame, asset: str, rf: str = "BIL") -> float:
    """Annualised Sharpe of monthly EXCESS returns (excess-vs-excess race only)."""
    e = excess(monthly, asset, rf)
    sd = e.std(ddof=1)
    return float(e.mean() / sd * np.sqrt(TRADING_MONTHS)) if sd > 0 else float("nan")


def max_drawdown(prices: pd.Series) -> dict:
    """Max drawdown of a daily total-return price series: depth (%), peak/trough dates."""
    px = pd.Series(prices).dropna()
    peak = px.cummax()
    dd = px / peak - 1.0
    trough = dd.idxmin()
    peak_date = px.loc[:trough].idxmax()
    return {"depth_pct": float(dd.min() * 100.0),
            "peak": str(peak_date.date()), "trough": str(trough.date())}


# --------------------------------------------------------------------------- #
# The credit premium — spread, eras, crises
# --------------------------------------------------------------------------- #
def premium_series(monthly: pd.DataFrame, hy: str = "HYD", ig: str = "MUB") -> pd.Series:
    """Monthly HY-muni credit spread = HY-muni total return minus IG-muni."""
    return (monthly[hy] - monthly[ig]).dropna()


def era_table(spread: pd.Series, cuts: list[tuple[str, str, str]]) -> list[dict]:
    """HAC stats for the credit spread over a list of ``(lo, hi, label)`` windows."""
    out = []
    for lo, hi, lbl in cuts:
        s = spread.loc[lo:hi]
        h = hac_mean(s.values)
        out.append({"label": lbl, "mean_bps": h["mean_bps"], "tstat": h["tstat"],
                    "n": h["n"]})
    return out


def calendar_year_table(monthly: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Calendar-year total returns (%) for ``cols`` plus the HYD-MUB spread (pp)."""
    yr = (1.0 + monthly[cols]).groupby(monthly.index.year).prod() - 1.0
    out = (yr * 100).round(2)
    if "HYD" in cols and "MUB" in cols:
        out["HYD-MUB"] = (out["HYD"] - out["MUB"]).round(2)
    return out


# --------------------------------------------------------------------------- #
# The tax wrapper — tax-equivalent yield & after-tax race
# --------------------------------------------------------------------------- #
def income_yields(income: pd.DataFrame, cols: list[str]) -> dict:
    """Average annualised income (distribution) yield (%) per fund."""
    return {c: float(income[c].mean() * TRADING_MONTHS * 100) for c in cols}


def tax_equivalent_yield(muni_income_yield_pct: float, rate: float) -> float:
    """Tax-equivalent yield of a tax-exempt muni: ``y / (1 - rate)`` (%).

    The pre-tax yield a *taxable* bond would have to pay to leave a top-bracket investor
    equally well off after federal tax.
    """
    return muni_income_yield_pct / (1.0 - rate)


def after_tax_returns(monthly_tr: pd.DataFrame, income: pd.DataFrame, col: str,
                      rate: float, tax_exempt: bool) -> pd.Series:
    """Monthly after-tax total return for a top-bracket investor.

    Price appreciation is left pre-tax (a held position defers/roughly matches across
    funds); the *income* leg is taxed at ``rate`` unless ``tax_exempt`` (munis). So
    ``after_tax = (total - income) + income * (1 - tax)``. A deliberately simple, honest
    convention — it isolates the coupon-tax difference that is the whole tax story.
    """
    tr = monthly_tr[col]
    inc = income[col]
    keep = 1.0 if tax_exempt else (1.0 - rate)
    df = pd.concat([tr.rename("tr"), inc.rename("inc")], axis=1).dropna()
    return (df["tr"] - df["inc"]) + df["inc"] * keep


def after_tax_sharpe(after_tax: pd.Series, monthly_tr: pd.DataFrame, rate: float,
                     rf: str = "BIL") -> float:
    """Annualised Sharpe of an after-tax series over the after-tax T-bill leg.

    The cash leg is taxable ordinary income, so its after-tax return is ``BIL*(1-rate)``
    — the honest risk-free for a taxable investor.
    """
    rf_at = monthly_tr[rf] * (1.0 - rate)
    e = (after_tax - rf_at).dropna()
    sd = e.std(ddof=1)
    return float(e.mean() / sd * np.sqrt(TRADING_MONTHS)) if sd > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Tradability — the one-switch cost
# --------------------------------------------------------------------------- #
def switch_cost_drag(spread_bps_oneway: float, years: float) -> float:
    """Annualised drag (bps/yr) of ONE round-trip switch (sell MUB, buy HYD).

    One-way × NAV on each of the 2 legs, amortised over the holding period. Long-only,
    no shorts, no borrow. ETF expense ratios are already inside the net-of-fee tape.
    """
    return 2.0 * spread_bps_oneway / years


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame) -> dict:
    """Run the headline premium stats on a synthetic world (HYD, MUB)."""
    sp = premium_series(world)
    h = hac_mean(sp.values)
    boot = bootstrap_mean_ci(sp.values)
    return {"mean_bps": h["mean_bps"], "tstat": h["tstat"], "n": h["n"],
            "ci_low_bps": boot["ci_low_bps"], "ci_high_bps": boot["ci_high_bps"],
            "frac_negative": boot["frac_negative"]}
