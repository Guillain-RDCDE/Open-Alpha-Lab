"""Race + inference for Study 915 — K-1 vs 1099.

Two ETPs sit on the same commodity beta and differ almost entirely in **legal wrapper**:

- **DBC** is a commodity pool. It holds the futures directly, is a partnership for tax
  purposes, and sends investors a **Schedule K-1** every spring. Its futures gains are
  **Section 1256** contracts: marked to market at every year end whether or not you sell,
  and taxed **60% long-term / 40% short-term** regardless of holding period.
- **PDBC** wraps the *same* strategy family inside a 1940-Act fund (the futures live in a
  Cayman controlled subsidiary), so it sends a **Form 1099**. Its income arrives as
  ordinary **distributions** taxed in the year received; the rest of the return is price
  appreciation, **deferred** until you sell and then taxed at long-term rates.

The marketing claim is convenience: same exposure, no K-1, IRA-friendly. The question the
desk asks is whether that convenience is **paid for in performance**.

The study has three layers, in strictly increasing order of assumption:

1. **The pre-tax race** — pure tape. Excess-of-cash Sharpe for each wrapper, the HAC
   *t* on the daily return difference, a paired block-bootstrap CI on both the annualised
   return difference and the Sharpe advantage, tracking error, an era cut and a cost sweep.
   Nothing here is assumed.
2. **The fee-gap reconciliation** — one PROXY (the two prospectus expense ratios) used
   only to compare what the fee gap *predicts* against what the tape *delivered*.
3. **The after-tax layer** — an explicit, stated **model**, not a measurement. It applies
   the two tax regimes to the realised annual returns under stated marginal rates and a
   stated distribution-payout share, and sweeps both. Every input is an ASSUMPTION.

**Execution lag.** There is exactly one, and it is trivial because nothing is timed: the
wrapper is chosen once, bought at the close of the first common session, and the first
return counted is the following session's (``pct_change`` on aligned closes, first bar
dropped). No signal is formed from data it could not have seen. **No shorting**, so no
borrow cost arises anywhere in this study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0.

    ``lags=None`` uses the usual automatic rule ``floor(4 (n/100)^(2/9))``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Returns & summary
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Aligned simple daily total returns.

    Columns are inner-joined on their common sessions first, so every leg is measured on
    exactly the same days. The first bar disappears with ``pct_change`` — that is the
    single execution lag: the position is entered at the first common close and the first
    return counted is the next session's.
    """
    df = prices if cols is None else prices[list(cols)]
    df = df.dropna().sort_index()
    return df.pct_change().dropna()


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an *excess* series for
    the excess-of-cash Sharpe)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy()),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# Tracking decomposition (the "same exposure?" question)
# --------------------------------------------------------------------------- #
def tracking_stats(r_a: pd.Series, r_b: pd.Series,
                   periods_per_year: int = TRADING_DAYS) -> dict:
    """How closely does wrapper ``a`` track wrapper ``b``, and at what drift?

    ``diff_ann`` is the annualised arithmetic mean of the daily difference (a − b);
    ``te_ann`` is its annualised standard deviation — the tracking error. ``t_diff`` is
    the HAC *t* on the daily difference: it answers "is the drift distinguishable from
    zero at all?", which is the whole pre-tax question.
    """
    d = (pd.Series(r_a) - pd.Series(r_b)).dropna()
    a = pd.Series(r_a).reindex(d.index)
    b = pd.Series(r_b).reindex(d.index)
    var_b = float(b.var(ddof=1))
    return {
        "n_days": int(len(d)),
        "diff_ann": float(d.mean() * periods_per_year),
        "diff_bps_day": float(d.mean() * 1e4),
        "te_ann": float(d.std(ddof=1) * np.sqrt(periods_per_year)),
        "t_diff": newey_west_t(d.to_numpy()),
        "corr": float(a.corr(b)),
        "beta": float(a.cov(b) / var_b) if var_b > 0 else float("nan"),
        "r2": float(a.corr(b) ** 2),
    }


def horizon_tracking(prices: pd.DataFrame, leg_a: str, leg_b: str,
                     freq: str = "ME") -> dict:
    """Tracking difference and error measured on *resampled* (default monthly) returns.

    Two ETFs on the same beta disagree at the daily close for reasons that are not
    economic — the two prints do not clear at the same instant, and each carries its own
    premium/discount noise that reverses the next session. That shows up as a large
    *negative* lag-1 autocorrelation in the daily difference, which inflates the naive
    ``sqrt(252)``-scaled tracking error far above the divergence an actual holder lives
    through. Measuring at a lower frequency lets the bounce cancel.
    """
    r = daily_returns(prices, [leg_a, leg_b])
    per_year = {"ME": 12, "QE": 4, "YE": 1, "W": 52}.get(freq, 12)
    res = (1.0 + r).resample(freq).prod() - 1.0
    d = (res[leg_a] - res[leg_b]).dropna()
    daily_d = (r[leg_a] - r[leg_b]).dropna()
    return {
        "freq": freq, "n_obs": int(len(d)),
        "diff_ann": float(d.mean() * per_year),
        "te_ann": float(d.std(ddof=1) * np.sqrt(per_year)),
        "t_diff": newey_west_t(d.to_numpy()),
        "te_ann_daily_scaled": float(daily_d.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "daily_autocorr_lag1": float(daily_d.autocorr(1)),
        "resampled_autocorr_lag1": float(d.autocorr(1)) if len(d) > 3 else float("nan"),
    }


def power_stats(r_a: pd.Series, r_b: pd.Series, boot_sd_ann: float | None = None,
                periods_per_year: int = TRADING_DAYS) -> dict:
    """How large a wrapper cost could this sample actually have detected?

    A null is only informative if the test had the power to reject. Reports the annualised
    standard error of the return difference three ways — naive i.i.d., HAC-implied, and
    (optionally) the block-bootstrap sd — and the corresponding **minimum detectable
    difference** at |*t*| = 2. Anything smaller than that was never in reach, and the null
    must be read as "no cost larger than this", not "no cost".
    """
    d = (pd.Series(r_a) - pd.Series(r_b)).dropna()
    n = len(d)
    mean_ann = float(d.mean() * periods_per_year)
    se_iid = float(d.std(ddof=1) / np.sqrt(n) * periods_per_year)
    t_hac = newey_west_t(d.to_numpy())
    se_hac = abs(mean_ann / t_hac) if np.isfinite(t_hac) and t_hac != 0 else float("nan")
    out = {
        "n_days": int(n), "mean_ann": mean_ann,
        "se_iid_ann": se_iid, "mde_iid_ann": 2.0 * se_iid,
        "se_hac_ann": se_hac, "mde_hac_ann": 2.0 * se_hac,
        "t_hac": t_hac,
        "daily_autocorr_lag1": float(d.autocorr(1)),
    }
    if boot_sd_ann is not None:
        out["se_boot_ann"] = float(boot_sd_ann)
        out["mde_boot_ann"] = 2.0 * float(boot_sd_ann)
    return out


# --------------------------------------------------------------------------- #
# The pre-tax race (the headline; pure tape)
# --------------------------------------------------------------------------- #
def race(
    prices: pd.DataFrame,
    leg_a: str,
    leg_b: str,
    cash: str,
    extra_cost_bps_a: float = 0.0,
    extra_cost_bps_b: float = 0.0,
    holding_years: float | None = None,
) -> dict:
    """Race two buy-and-hold wrappers, both measured **excess of cash**.

    ``leg_a`` is the leg under test (the 1099 wrapper), ``leg_b`` the benchmark (the K-1
    twin). Total-return closes already net each fund's expense ratio, so the only cost
    layer that can be added is **trading friction**: a one-way spread paid on entry and
    on exit, amortised over the holding period (``2 * cost / holding_years`` per year).
    Buy-and-hold means one round trip, so friction here is genuinely second-order — the
    sweep exists to show exactly how second-order.
    """
    r = daily_returns(prices, [leg_a, leg_b, cash])
    years = holding_years if holding_years else len(r) / TRADING_DAYS
    drag_a = 2.0 * extra_cost_bps_a * 1e-4 / max(years, 1e-9) / TRADING_DAYS
    drag_b = 2.0 * extra_cost_bps_b * 1e-4 / max(years, 1e-9) / TRADING_DAYS

    ra = r[leg_a] - drag_a
    rb = r[leg_b] - drag_b
    rc = r[cash]
    ea = (ra - rc).rename("e_a")
    eb = (rb - rc).rename("e_b")

    s_a, s_b = summary(ea), summary(eb)
    trk = tracking_stats(ra, rb)
    return {
        "leg_a": leg_a, "leg_b": leg_b, "cash": cash,
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "n_days": int(len(r)), "years": float(years),
        "a": s_a, "b": s_b,
        "cagr_a": summary(ra)["cagr"], "cagr_b": summary(rb)["cagr"],
        "dd_a": max_drawdown(ra), "dd_b": max_drawdown(rb),
        "sharpe_adv": s_a["sharpe"] - s_b["sharpe"],
        "tracking": trk,
        "t_diff": trk["t_diff"],
        "diff_ann": trk["diff_ann"],
        "te_ann": trk["te_ann"],
        "e_a": ea, "e_b": eb, "r_a": ra, "r_b": rb, "r_cash": rc,
    }


# --------------------------------------------------------------------------- #
# Block bootstrap (paired — the same resampled dates hit both legs)
# --------------------------------------------------------------------------- #
def _block_indices(rng, n: int, block: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def bootstrap_diff_ci(
    r_a: pd.Series, r_b: pd.Series,
    n_boot: int = 4000, block: int = 21, seed: int = 915, alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Circular block-bootstrap CI for the **annualised return difference** (a − b).

    Blocks of ``block`` consecutive days preserve the day-to-day dependence in the
    tracking difference (creation/redemption timing, roll dates, NAV-price noise).
    """
    d = (pd.Series(r_a) - pd.Series(r_b)).dropna().to_numpy(dtype=float)
    n = d.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = d[_block_indices(rng, n, block)].mean() * periods_per_year
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(d.mean() * periods_per_year),
        "ci_low": float(lo), "ci_high": float(hi),
        "boot_sd": float(boots.std(ddof=1)),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n), "n_boot": int(n_boot), "block": int(block),
    }


def bootstrap_sharpe_adv_ci(
    e_a: pd.Series, e_b: pd.Series,
    n_boot: int = 4000, block: int = 21, seed: int = 915, alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Paired circular block-bootstrap CI for the **excess-of-cash Sharpe advantage**.

    Both legs are resampled on the *same* dates, so the shared commodity beta cancels and
    the CI reflects only the wrapper difference — a far tighter (and fairer) interval than
    bootstrapping each Sharpe separately.
    """
    df = pd.concat([pd.Series(e_a).rename("a"), pd.Series(e_b).rename("b")], axis=1).dropna()
    a = df["a"].to_numpy(dtype=float); b = df["b"].to_numpy(dtype=float)
    n = a.size
    ann = np.sqrt(periods_per_year)
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = _block_indices(rng, n, block)
        x, y = a[idx], b[idx]
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        if sx > 0 and sy > 0:
            boots[i] = (x.mean() / sx - y.mean() / sy) * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * ann
    return {
        "point": float(point), "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((valid < 0).mean()),
        "n_obs": int(n), "n_boot": int(valid.size), "block": int(block),
    }


# --------------------------------------------------------------------------- #
# Era cut, calendar years, cost sweep
# --------------------------------------------------------------------------- #
def era_cut(prices: pd.DataFrame, leg_a: str, leg_b: str, cash: str,
            split: str = "2021-01-01") -> dict:
    """Split at ``split`` and re-run the pre-tax race on each half."""
    out: dict[str, dict | None] = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = prices.loc[sl]
        if len(sub.dropna()) < 120:
            out[tag] = None
            continue
        res = race(sub, leg_a, leg_b, cash)
        out[tag] = {
            "start": res["start"], "end": res["end"], "n_days": res["n_days"],
            "diff_ann": res["diff_ann"], "t_diff": res["t_diff"],
            "te_ann": res["te_ann"],
            "sharpe_a": res["a"]["sharpe"], "sharpe_b": res["b"]["sharpe"],
            "sharpe_adv": res["sharpe_adv"],
        }
    return out


def calendar_year_table(prices: pd.DataFrame, leg_a: str, leg_b: str,
                        cash: str, complete_years_only: bool = True) -> pd.DataFrame:
    """Per-calendar-year total return (%) for both wrappers plus the cash leg.

    ``complete_years_only`` drops the first and last calendar years when they are partial
    (they always are: PDBC's inception is mid-November and the as-of is 30 June), so the
    annual after-tax model never mixes a stub year with a full one.
    """
    r = daily_returns(prices, [leg_a, leg_b, cash])
    yr = (1.0 + r).groupby(r.index.year).prod() - 1.0
    counts = r.groupby(r.index.year).size()
    if complete_years_only:
        keep = counts[counts >= 200].index
        yr = yr.loc[yr.index.isin(keep)]
    yr = yr.rename(columns={leg_a: "a_ret", leg_b: "b_ret", cash: "cash_ret"})
    yr["diff_pp"] = (yr["a_ret"] - yr["b_ret"]) * 100.0
    yr.index.name = "year"
    return yr


def sign_test(diffs) -> dict:
    """How many years did the 1099 wrapper win, and is that count distinguishable from
    a coin flip? (Wilson interval on the win rate.)"""
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    k = int((d > 0).sum())
    n = int(len(d))
    lo, hi = wilson_interval(k, n)
    return {"wins": k, "n": n, "win_rate": (k / n if n else float("nan")),
            "ci_low": lo, "ci_high": hi}


def cost_sweep(prices: pd.DataFrame, leg_a: str, leg_b: str, cash: str,
               spread_bps_grid=(0.0, 3.0, 10.0, 25.0, 50.0),
               holding_years: float | None = None) -> list[dict]:
    """Charge the 1099 leg an *extra* one-way spread (round trip, amortised) and re-race.

    This is deliberately asymmetric and deliberately punitive: it asks how large an
    execution disadvantage the 1099 wrapper would need before the pre-tax verdict changed.
    Buy-and-hold means one round trip over the whole holding period, so the annual drag is
    ``2 x spread / years``.
    """
    out = []
    for c in spread_bps_grid:
        res = race(prices, leg_a, leg_b, cash, extra_cost_bps_a=c,
                   holding_years=holding_years)
        out.append({
            "extra_spread_bps": c,
            "annual_drag_bps": 2.0 * c / max(res["years"], 1e-9),
            "diff_ann": res["diff_ann"], "t_diff": res["t_diff"],
            "sharpe_a": res["a"]["sharpe"], "sharpe_b": res["b"]["sharpe"],
            "sharpe_adv": res["sharpe_adv"],
        })
    return out


def dispersion_table(prices: pd.DataFrame, legs, cash: str) -> pd.DataFrame:
    """CAGR / vol / excess-of-cash Sharpe / max drawdown for several vehicles.

    Context, not a race: it sets the size of the *wrapper* difference against the size of
    the *index and vehicle* differences an investor is choosing between at the same time.
    """
    r = daily_returns(prices, list(legs) + [cash])
    rows = []
    for leg in legs:
        s = summary(r[leg] - r[cash])
        rows.append({
            "leg": leg,
            "cagr": summary(r[leg])["cagr"],
            "vol_ann": s["vol_ann"],
            "excess_sharpe": s["sharpe"],
            "max_drawdown": max_drawdown(r[leg]),
        })
    return pd.DataFrame(rows).set_index("leg")


# --------------------------------------------------------------------------- #
# The after-tax layer — a MODEL built from stated ASSUMPTIONS, not a measurement
# --------------------------------------------------------------------------- #
def blended_1256_rate(ordinary_rate: float, ltcg_rate: float) -> float:
    """The Section 1256 blended rate: 60% long-term + 40% short-term, by statute."""
    return 0.6 * ltcg_rate + 0.4 * ordinary_rate


def after_tax_terminal_k1(
    annual_returns, annual_interest, ordinary_rate: float, ltcg_rate: float,
) -> float:
    """Terminal after-tax wealth of 1 unit held in the **K-1 commodity pool**.

    ASSUMPTIONS (all stated, none measured):

    - the pool's futures results are Section 1256 contracts, **marked to market each 31
      December** whether or not the investor sells, and taxed at the 60/40 blend;
    - the collateral interest component is ordinary income, proxied by the **BIL** total
      return of that year applied to the year's opening balance;
    - net Section 1256 losses are carried forward within the model (no carry-back
      election, no other 1256 income to offset);
    - tax is paid **out of the account** at each year end;
    - annual mark-to-market steps the basis up, so **nothing is owed on liquidation** —
      that is the structural point of this wrapper.
    """
    blend = blended_1256_rate(ordinary_rate, ltcg_rate)
    g = pd.Series(annual_returns).astype(float)
    i = pd.Series(annual_interest).astype(float).reindex(g.index)
    nav, carry = 1.0, 0.0
    for y in g.index:
        opening = nav
        nav *= (1.0 + g[y])
        interest = float(i[y]) * opening
        sec1256 = g[y] * opening - interest
        tax = max(interest, 0.0) * ordinary_rate
        taxable = sec1256 + carry
        if taxable > 0:
            tax += taxable * blend
            carry = 0.0
        else:
            carry = taxable
        nav -= tax
    return float(nav)


def after_tax_terminal_ric(
    annual_returns, annual_interest, ordinary_rate: float, ltcg_rate: float,
    payout_share: float = 1.0,
) -> float:
    """Terminal after-tax wealth of 1 unit held in the **1099 fund**, sold at the end.

    ASSUMPTIONS (all stated, none measured):

    - the fund distributes ordinary income each year equal to ``payout_share`` x the
      collateral-interest proxy (BIL's total return) on the year's opening balance;
      distributions are reinvested and add to basis, and the tax on them is paid out of
      the account (the withdrawal removes basis pro rata);
    - everything else is **deferred** price appreciation, taxed once at the long-term rate
      when the position is liquidated on the last day of the sample;
    - a terminal loss is not monetised (no offsetting gains assumed elsewhere).

    ``payout_share`` is the load-bearing assumption and is swept: 0 = the fund distributes
    nothing (maximum deferral), 1 = it distributes exactly the collateral interest,
    2 = it distributes twice that (realised subsidiary gains on top). No distribution
    filing was parsed for this study, so the true payout is UNKNOWN here — that is exactly
    why it is swept rather than set. The sign of the after-tax verdict turns on it.
    """
    g = pd.Series(annual_returns).astype(float)
    i = pd.Series(annual_interest).astype(float).reindex(g.index)
    nav, basis = 1.0, 1.0
    for y in g.index:
        opening = nav
        nav *= (1.0 + g[y])
        dist = max(float(i[y]) * payout_share, 0.0) * opening
        basis += dist                       # reinvested distribution lifts basis
        tax = dist * ordinary_rate
        if nav > 0:
            basis -= tax * (basis / nav)    # the withdrawal removes basis pro rata
        nav -= tax
    gain = max(nav - basis, 0.0)
    return float(nav - gain * ltcg_rate)


def _cagr(terminal: float, n_years: int) -> float:
    return float(terminal ** (1.0 / n_years) - 1.0) if terminal > 0 and n_years > 0 else float("nan")


def after_tax_race(
    year_table: pd.DataFrame, ordinary_rate: float, ltcg_rate: float,
    payout_share: float = 1.0,
) -> dict:
    """After-tax CAGR of both wrappers on the realised annual tape, under stated rates.

    ``year_table`` is the output of ``calendar_year_table`` (complete years only), with
    ``a_ret`` = the 1099 wrapper, ``b_ret`` = the K-1 wrapper, ``cash_ret`` = the
    collateral-interest proxy.
    """
    n = len(year_table)
    k1 = after_tax_terminal_k1(year_table["b_ret"], year_table["cash_ret"],
                               ordinary_rate, ltcg_rate)
    ric = after_tax_terminal_ric(year_table["a_ret"], year_table["cash_ret"],
                                 ordinary_rate, ltcg_rate, payout_share=payout_share)
    c_k1, c_ric = _cagr(k1, n), _cagr(ric, n)
    return {
        "n_years": int(n),
        "ordinary_rate": ordinary_rate, "ltcg_rate": ltcg_rate,
        "blended_1256": blended_1256_rate(ordinary_rate, ltcg_rate),
        "payout_share": payout_share,
        "cagr_k1": c_k1, "cagr_ric": c_ric,
        "gap_pp": (c_ric - c_k1) * 100.0,
        "pretax_cagr_k1": float((1.0 + year_table["b_ret"]).prod() ** (1.0 / n) - 1.0),
        "pretax_cagr_ric": float((1.0 + year_table["a_ret"]).prod() ** (1.0 / n) - 1.0),
    }


def same_tape_regime_gap(
    year_table: pd.DataFrame, ordinary_rate: float, ltcg_rate: float,
    payout_share: float = 1.0,
) -> dict:
    """Isolate the **tax regime** from the **index**: apply both regimes to one tape.

    Running the K-1 and 1099 tax rules over the *same* (K-1 fund's) annual returns removes
    the tracking difference entirely, so what is left is purely the value of 60/40 taxation
    without deferral against ordinary distributions with deferral.
    """
    n = len(year_table)
    k1 = after_tax_terminal_k1(year_table["b_ret"], year_table["cash_ret"],
                               ordinary_rate, ltcg_rate)
    ric = after_tax_terminal_ric(year_table["b_ret"], year_table["cash_ret"],
                                 ordinary_rate, ltcg_rate, payout_share=payout_share)
    return {
        "n_years": int(n),
        "cagr_k1": _cagr(k1, n), "cagr_ric": _cagr(ric, n),
        "regime_gap_pp": (_cagr(ric, n) - _cagr(k1, n)) * 100.0,
        "ordinary_rate": ordinary_rate, "ltcg_rate": ltcg_rate,
        "payout_share": payout_share,
    }


# Bracket grid — every entry is an ASSUMPTION about the investor, not a fact about the
# tape. (ordinary, ltcg) are all-in marginal rates; NIIT is already folded in where noted.
BRACKETS = (
    ("22% / 15%, no NIIT", 0.22, 0.15),
    ("24% / 15%, no NIIT", 0.24, 0.15),
    ("35.8% / 18.8% (32% + NIIT)", 0.358, 0.188),
    ("40.8% / 23.8% (top + NIIT)", 0.408, 0.238),
)

PAYOUT_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)


def bracket_sweep(year_table: pd.DataFrame, brackets=BRACKETS,
                  payout_grid=PAYOUT_GRID) -> pd.DataFrame:
    """The full (bracket x payout-share) after-tax grid — the honesty exhibit.

    One row per assumption pair. If the sign of ``gap_pp`` flips inside this grid, the
    after-tax answer is a function of the assumption, not of the tape, and must be
    reported that way.
    """
    rows = []
    for tag, o, l in brackets:
        for p in payout_grid:
            res = after_tax_race(year_table, o, l, payout_share=p)
            rows.append({
                "bracket": tag, "ordinary": o, "ltcg": l,
                "blended_1256": res["blended_1256"], "payout_share": p,
                "cagr_k1_pct": res["cagr_k1"] * 100.0,
                "cagr_ric_pct": res["cagr_ric"] * 100.0,
                "gap_pp": res["gap_pp"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, n_boot: int = 500, seed: int = 915) -> dict:
    """Run the pre-tax race on a synthetic wrapper pair with a known planted drag.

    On the planted world the estimated annual drag must land near the truth with |*t*| >= 2;
    on the null it must be centred on zero and stay quiet. Proves the estimator neither
    misses a genuine wrapper cost nor manufactures one out of tracking noise.
    """
    res = race(prices, "ric", "k1", "cash")
    ci = bootstrap_diff_ci(res["r_a"], res["r_b"], n_boot=n_boot, seed=seed)
    adv = bootstrap_sharpe_adv_ci(res["e_a"], res["e_b"], n_boot=n_boot, seed=seed)
    return {
        "diff_ann": res["diff_ann"],
        "estimated_drag_ann": -res["diff_ann"],   # a drag on the 1099 leg is a negative diff
        "t_diff": res["t_diff"],
        "te_ann": res["te_ann"],
        "sharpe_adv": res["sharpe_adv"],
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "sharpe_adv_ci": (adv["ci_low"], adv["ci_high"]),
        "n_days": res["n_days"],
    }
