"""Strategy + inference for Study 908 — Optimized-Roll Commodities.

The race: does an **optimized-roll** commodity wrapper (USCI — it holds the 14 most
backwardated of 27 commodities and rolls each into a cheapest-carry contract) beat a
**front-month** roller (GSG, DJP) and the semi-optimized DB "Optimum Yield" index (DBC)
on a **higher excess-of-cash return / Sharpe**, over the full sample and net of costs?

Everything is measured **excess of cash (BIL)**. A commodity index is fully collateralised,
so its total return bundles a big **T-bill yield** leg (≈5 %/yr in 2023-26) on top of the
spot + roll return. That collateral yield is identical across every wrapper and is NOT a
roll edge — so we subtract BIL from *both* sides and race excess-vs-excess. The only thing
left in ``optimized_excess − front_excess`` is the difference in **spot exposure + roll
yield** — i.e. the thing the claim is about.

Distinct from:

* [35-contango](../../35-contango/) — *times* the futures curve (a signal that goes to
  cash / flips when the term structure is in contango); here nothing is timed, we compare
  two **always-invested** index wrappers.
* [794-commodity-carry](../../794-commodity-carry/) — a **cross-sectional** long-short of
  individual commodities sorted on carry; here we buy whole packaged **indices** and ask
  whether the optimized wrapper's structural roll rule pays.
* [661-uso-roll-decay](../../661-uso-roll-decay/) — the roll *decay* of a single-commodity
  front-month vehicle (USO, crude); here it is a broad multi-commodity index race.
* [226-crude-seasonality](../../226-crude-seasonality/) — a calendar effect in one
  commodity; unrelated claim.

Inference: Newey-West (HAC) *t* on the monthly return **difference** (optimized − benchmark);
an annualised Sharpe with a delta-method SE and a block-bootstrap CI on each leg and on the
**Sharpe advantage** (paired circular block bootstrap); max drawdown; a calendar-year table;
an era cut (deep-contango 2010-2015 vs the 2016+ recovery / 2021+ backwardation); and a
costed version (total returns already net each fund's expense ratio — the honest race — plus
an incremental bid-ask charge on reconstitution turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~1.5 * T^(1/3) at T ~ 190)


# --------------------------------------------------------------------------- #
# Excess-of-cash frame
# --------------------------------------------------------------------------- #
def excess_frame(rets: pd.DataFrame, cash: str = "cash") -> pd.DataFrame:
    """Subtract the cash column from every other column: excess-of-cash returns.

    Every leg becomes ``r_leg − r_cash`` so a Sharpe computed on it is an
    excess-of-cash Sharpe and the difference of two legs has the collateral-yield
    component (identical across wrappers) already netted out.
    """
    if cash not in rets.columns:
        raise KeyError(f"cash column {cash!r} not in {list(rets.columns)}")
    others = [c for c in rets.columns if c != cash]
    ex = rets[others].sub(rets[cash], axis=0)
    return ex


def common_sample(ex: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows where every column in ``cols`` is present (the shared window)."""
    sub = ex[cols].dropna()
    return sub


# --------------------------------------------------------------------------- #
# HAC inference primitives
# --------------------------------------------------------------------------- #
def nw_mean_t(x: np.ndarray, lags: int = NW_LAGS) -> tuple[float, float]:
    """Mean of ``x`` and its Newey-West (Bartlett) HAC t-statistic vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    s = g0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    se = np.sqrt(max(s, 1e-18) / n)
    return float(x.mean()), float(x.mean() / se)


def annualized_sharpe(x: np.ndarray, ppy: int = MONTHS_PER_YEAR) -> float:
    """Annualised Sharpe of an already-excess-of-cash return series."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")


def max_drawdown(monthly: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the cumulative TOTAL (not excess) return, %."""
    r = np.asarray(monthly, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(curve)
    dd = curve / peak - 1.0
    return float(dd.min() * 100.0)


# --------------------------------------------------------------------------- #
# The Sharpe race — optimized vs one benchmark, excess-vs-excess
# --------------------------------------------------------------------------- #
def sharpe_race(ex: pd.DataFrame, opt: str, bench: str, lags: int = NW_LAGS,
                n_boot: int = 2000, block: int = 6, seed: int = 908) -> dict:
    """Excess-vs-excess race between two wrappers on their common window.

    Both ``opt`` and ``bench`` are already excess-of-cash. Reports each leg's annualised
    excess return / vol / Sharpe, the **Sharpe advantage** (opt − bench) with a paired
    circular-block-bootstrap CI, and the HAC *t* on the monthly return **difference**
    (opt − bench). The difference series strips the shared commodity beta partially — it
    is the head-to-head roll pickup.
    """
    df = common_sample(ex, [opt, bench])
    a = df[opt].to_numpy(dtype=float)
    b = df[bench].to_numpy(dtype=float)
    d = a - b
    n = len(df)

    sr_a = annualized_sharpe(a)
    sr_b = annualized_sharpe(b)
    adv = sr_a - sr_b
    mean_d, t_d = nw_mean_t(d, lags)

    # paired circular block bootstrap on the Sharpe advantage
    rng = np.random.default_rng(seed)
    blk = max(1, min(block, n))
    n_blocks = int(np.ceil(n / blk))
    offs = np.arange(blk)
    advs = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        aa, bb = a[idx], b[idx]
        sda, sdb = aa.std(ddof=1), bb.std(ddof=1)
        if sda > 0 and sdb > 0:
            advs[i] = (aa.mean() / sda - bb.mean() / sdb) * np.sqrt(MONTHS_PER_YEAR)
    valid = advs[np.isfinite(advs)]
    ci_lo, ci_hi = (np.percentile(valid, [2.5, 97.5]) if valid.size else (np.nan, np.nan))
    frac_le0 = float((valid <= 0).mean()) if valid.size else float("nan")

    return {
        "opt": opt, "bench": bench, "n": n,
        "start": str(df.index.min()), "end": str(df.index.max()),
        "ann_ex_opt": float(a.mean() * MONTHS_PER_YEAR * 100),
        "ann_ex_bench": float(b.mean() * MONTHS_PER_YEAR * 100),
        "vol_opt": float(a.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) * 100),
        "vol_bench": float(b.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) * 100),
        "sharpe_opt": sr_a, "sharpe_bench": sr_b, "sharpe_adv": adv,
        "adv_ci_lo": float(ci_lo), "adv_ci_hi": float(ci_hi),
        "adv_frac_le0": frac_le0,
        "diff_bps_mo": mean_d * 1e4, "diff_ann_pct": mean_d * MONTHS_PER_YEAR * 100,
        "t_diff": t_d,
    }


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_race(ex: pd.DataFrame, opt: str, bench: str, splits: list[tuple[str, str, str]],
             lags: int = NW_LAGS) -> list[dict]:
    """Run the return-difference stats on each (label, start, end) era window."""
    out = []
    df = common_sample(ex, [opt, bench])
    is_period = isinstance(df.index, pd.PeriodIndex)

    def _bound(s):
        return pd.Period(s, freq="M") if is_period else pd.Timestamp(s)

    for label, start, end in splits:
        w = df.copy()
        if start:
            w = w[w.index >= _bound(start)]
        if end:
            # inclusive of the whole end month
            hi = _bound(end) if is_period else pd.Timestamp(end) + pd.offsets.MonthEnd(0)
            w = w[w.index <= hi]
        if len(w) < 8:
            out.append({"era": label, "n": len(w), "diff_ann_pct": float("nan"),
                        "t_diff": float("nan"), "sharpe_adv": float("nan")})
            continue
        a = w[opt].to_numpy(dtype=float)
        b = w[bench].to_numpy(dtype=float)
        mean_d, t_d = nw_mean_t(a - b, lags)
        out.append({
            "era": label, "n": len(w),
            "diff_ann_pct": mean_d * MONTHS_PER_YEAR * 100, "t_diff": t_d,
            "sharpe_adv": annualized_sharpe(a) - annualized_sharpe(b),
        })
    return out


# --------------------------------------------------------------------------- #
# Calendar-year table (excess-of-cash annual returns)
# --------------------------------------------------------------------------- #
def calendar_year_table(ex: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-calendar-year compounded excess-of-cash return (%) for each column."""
    df = ex[cols].dropna(how="all")
    years = df.index.year if isinstance(df.index, pd.PeriodIndex) else df.index.year
    out = {}
    for c in cols:
        s = df[c]
        out[c] = s.groupby(years).apply(lambda v: (np.prod(1.0 + v.dropna().to_numpy()) - 1.0) * 100.0)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Costed version
# --------------------------------------------------------------------------- #
def costed_race(ex: pd.DataFrame, opt: str, bench: str,
                spread_bps: dict[str, float] | None = None,
                turnover_per_year: float = 2.0, lags: int = NW_LAGS) -> dict:
    """Charge an incremental bid-ask cost on top of the (already net-of-ER) total returns.

    Total-return NAVs already embed each fund's expense ratio, so the raw ``sharpe_race``
    is *already* net of fees (USCI's 1.03 % vs GSG's 0.48 %). This adds the round-trip
    **bid-ask** cost a holder pays on the fund's reconstitution/rebalance turnover:
    ``spread_bps`` one-way × ``turnover_per_year`` × 2 sides, spread pro-rated monthly. The
    less-liquid optimized wrapper (USCI) carries the wider spread, so this can only *shrink*
    its advantage — the conservative direction.
    """
    if spread_bps is None:
        spread_bps = {"USCI": 8.0, "PDBC": 4.0, "DBC": 3.0, "GSG": 3.0, "DJP": 6.0}
    df = common_sample(ex, [opt, bench])
    ca = df[opt] - spread_bps.get(opt, 5.0) / 1e4 * turnover_per_year * 2 / MONTHS_PER_YEAR
    cb = df[bench] - spread_bps.get(bench, 5.0) / 1e4 * turnover_per_year * 2 / MONTHS_PER_YEAR
    a, b = ca.to_numpy(dtype=float), cb.to_numpy(dtype=float)
    mean_d, t_d = nw_mean_t(a - b, lags)
    return {
        "opt": opt, "bench": bench, "n": len(df),
        "sharpe_opt_net": annualized_sharpe(a), "sharpe_bench_net": annualized_sharpe(b),
        "sharpe_adv_net": annualized_sharpe(a) - annualized_sharpe(b),
        "diff_ann_pct_net": mean_d * MONTHS_PER_YEAR * 100, "t_diff_net": t_d,
        "charge_opt_ann_pct": spread_bps.get(opt, 5.0) / 1e4 * turnover_per_year * 2 * 100,
        "charge_bench_ann_pct": spread_bps.get(bench, 5.0) / 1e4 * turnover_per_year * 2 * 100,
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Run the excess-vs-excess race on a synthetic world (optimized vs front)."""
    ex = excess_frame(world, cash="cash")   # -> columns optimized, front
    return sharpe_race(ex, "optimized", "front", lags=lags)
