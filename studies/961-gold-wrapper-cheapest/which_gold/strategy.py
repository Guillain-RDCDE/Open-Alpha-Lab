"""Estimators, tests and rules for Study 961 — Which Gold.

Five wrappers hold the same allocated London bullion and strike at the same 16:00 New
York close. Whatever separates their returns is not the metal; the only recurring,
contractual leak is the sponsor fee. So a clean tape should hand the fee back, in basis
points per year, wrapper by wrapper — and the *ranking* of realised tracking differences
should be the reverse of the ranking of published fees.

Three views of the same quantity, because they differ in what noise they carry:

1. ``td_daily`` — the mean daily log difference, annualised, with a HAC *t*. The
   difference between two wrappers is dominated by a **mean-reverting** premium/discount
   (first-order autocorrelation about −0.5 on the real tape), so the HAC long-run variance
   is *smaller* than the iid one and the HAC *t* is materially larger than the naive one.
   Quoted because the brief asks for the daily spread, and read with that caveat.
2. ``td_monthly`` — the mean of **non-overlapping complete-calendar-month** differences,
   annualised. Aggregation kills most of the transient premium noise, and the observations
   are genuinely non-overlapping. Every headline number is quoted from it.
3. ``td_endpoint`` — the cumulative log difference over the window, annualised. Two
   anchors only, so it inherits the premium/discount at both; quoted for completeness.

**The HAC *t* here is anti-conservative, at BOTH frequencies — read the naive one.**
The usual reason to put a Newey-West *t* on a return series is that positive serial
dependence inflates the iid *t*. This series does the opposite: the wrapper-versus-wrapper
residual mean-reverts at the daily frequency (acf1 ≈ −0.46) *and still* at the monthly one
(acf1 ≈ −0.42), so the HAC long-run variance sits **below** the iid variance and the HAC
*t* comes out roughly **twice** the naive one — and it keeps climbing with the bandwidth
(lag 1 → 4.2, lag 3 → 6.3, lag 12 → 9.4 on the headline pair). A number that grows with the
bandwidth is not evidence. So every estimator in this module returns the **naive iid *t*
alongside the HAC one**, the study leads with the naive one, and the same discipline
applies to the interval: the block bootstrap inherits the mean reversion and prints a
narrow CI, so an **iid** bootstrap CI is reported next to it and the wider one is the one
the verdict has to survive.

The cross-sectional measure is each wrapper minus the **equal-weight cohort mean**: the
bullion price and any common strike effect cancel exactly, leaving the fee.

**One execution lag, once.** The measurement above involves no decision and therefore no
lag. The only rule that *trades* — ``chase_the_winner`` — ranks the cohort on data through
the last session of a calendar year and switches at the **next session's close**, paying
``cost_bps`` one-way x NAV on each change of wrapper. No short leg in the ownership race;
the long/short version is handled separately by ``long_short_net``, where the short pays
borrow.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12


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
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0."""
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


def rankdata(a) -> np.ndarray:
    """Average-tie ranks (a short stand-in for scipy, which the desk does not import)."""
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    """Spearman rank correlation, ties averaged."""
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def block_bootstrap_ci(
    x, n_boot: int = 5000, block: int = 3, seed: int = 961,
    scale: float = MONTHS * 1e4, alpha: float = 0.05,
) -> dict:
    """Moving-block bootstrap CI for the mean of a (monthly) series, in bp/yr.

    Blocks of ``block`` consecutive months preserve whatever serial dependence survives
    monthly aggregation. ``scale`` converts the mean into bp/yr.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = x.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    offs = np.arange(block)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + offs[None, :]) % n).ravel()[:n]
        boots[b] = x[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(x.mean() * scale), "ci_low": float(lo * scale),
        "ci_high": float(hi * scale), "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n), "block": int(block), "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# The tracking-difference estimators
# --------------------------------------------------------------------------- #
def monthly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping complete-calendar-month log returns (the first partial month drops)."""
    m = np.log(prices).resample("ME").last()
    return m.diff().dropna()


def td_endpoint(fund: pd.Series, bench: pd.Series) -> float:
    """Cumulative log tracking difference, annualised, in bp/yr — two anchors only."""
    common = fund.index.intersection(bench.index)
    r = (np.log(fund.reindex(common)) - np.log(bench.reindex(common))).dropna()
    if len(r) < 3:
        return float("nan")
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float((r.iloc[-1] - r.iloc[0]) / years * 1e4) if years > 0 else float("nan")


def td_daily(fund: pd.Series, bench: pd.Series, lags: int | None = None) -> dict:
    """Mean **daily** log difference, annualised (bp/yr), with HAC and naive *t*.

    Reported with the difference's first-order autocorrelation, because on this tape it is
    strongly *negative*: the wrapper-versus-wrapper residual is a mean-reverting
    premium/discount, so the HAC long-run variance sits below the iid one and the HAC *t*
    overstates the naive one rather than the usual other way round. Both are printed.
    """
    common = fund.index.intersection(bench.index)
    d = (np.log(fund.reindex(common)) - np.log(bench.reindex(common))).diff().dropna()
    n = len(d)
    if n < 30:
        return {"td_bpy": float("nan"), "tstat": float("nan"), "t_naive": float("nan"),
                "n_days": int(n), "sd_daily_bp": float("nan"), "acf1": float("nan")}
    x = d.to_numpy()
    return {
        "td_bpy": float(x.mean() * TRADING_DAYS * 1e4),
        "tstat": newey_west_t(x, lags=lags),
        "t_naive": one_sample_t(x),
        "n_days": int(n),
        "sd_daily_bp": float(x.std(ddof=1) * 1e4),
        "acf1": float(np.corrcoef(x[1:], x[:-1])[0, 1]),
    }


def td_monthly(fund_m: pd.Series, bench_m: pd.Series, lags: int = 3) -> dict:
    """Mean non-overlapping monthly tracking difference, annualised (bp/yr), naive + HAC *t*.

    The headline estimator: immune to any single anchor, computed on genuinely
    non-overlapping observations. **``t_naive`` is the one to read** — the monthly
    difference still mean-reverts (``acf1`` is returned so the reader can see it), which
    makes ``tstat`` (HAC) the *anti*-conservative of the two. Both are returned, always.
    """
    x = (fund_m - bench_m).dropna().to_numpy()
    if len(x) < 4:
        return {"td_bpy": float("nan"), "tstat": float("nan"), "t_naive": float("nan"),
                "n_months": int(len(x)), "sd_month_bp": float("nan"), "pos_months": 0,
                "acf1": float("nan")}
    return {
        "td_bpy": float(x.mean() * MONTHS * 1e4),
        "tstat": newey_west_t(x, lags=lags),
        "t_naive": one_sample_t(x),
        "n_months": int(len(x)),
        "sd_month_bp": float(x.std(ddof=1) * 1e4),
        "pos_months": int((x > 0).sum()),
        "acf1": float(np.corrcoef(x[1:], x[:-1])[0, 1]) if len(x) > 2 else float("nan"),
    }


def hac_lag_sweep(x, lags_grid=(1, 2, 3, 4, 6, 8, 12)) -> list[dict]:
    """The HAC *t* as a function of bandwidth, plus the naive *t* — a fragility check.

    With a mean-reverting series the Bartlett-kernel long-run variance keeps shrinking as
    the bandwidth grows, so the HAC *t* keeps growing. Printing the sweep is the honest way
    to say that the HAC number is a choice and the naive one is not.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    rows = [{"lags": 0, "tstat": one_sample_t(x), "label": "naive (iid)"}]
    rows += [{"lags": int(L), "tstat": newey_west_t(x, lags=int(L)), "label": "HAC"}
             for L in lags_grid]
    return rows


def tracking_table(prices: pd.DataFrame, cohort, fees: dict | None = None,
                   lags: int = 3) -> pd.DataFrame:
    """Every wrapper's realised tracking difference vs the equal-weight cohort mean.

    The cohort mean is the right benchmark here: the bullion price and any common strike
    effect cancel exactly out of a wrapper-minus-cohort difference, so what is left is the
    fee plus each wrapper's own premium/discount noise. Columns: the published fee (if
    given, an ASSUMPTION), the endpoint / daily / monthly estimators, and **two** monthly
    *t*s — ``t_monthly`` is the naive iid one (the conservative one on this tape, and the
    one every write-up leads with) and ``t_monthly_hac`` the Newey-West one, which the
    mean-reverting residual inflates.
    """
    px = prices[list(cohort)].dropna()
    mm = monthly_log_returns(px)
    cohort_m = mm[list(cohort)].mean(axis=1)
    cohort_px = np.exp(np.log(px).mean(axis=1))
    rows = []
    for tk in cohort:
        dl = td_daily(px[tk], cohort_px)
        mo = td_monthly(mm[tk], cohort_m, lags=lags)
        rows.append({
            "fund": tk,
            "fee_bps": float(fees[tk]) if fees else np.nan,
            "td_endpoint_bpy": td_endpoint(px[tk], cohort_px),
            "td_daily_bpy": dl["td_bpy"], "t_daily": dl["tstat"],
            "td_monthly_bpy": mo["td_bpy"], "t_monthly": mo["t_naive"],
            "t_monthly_hac": mo["tstat"],
            "pos_months": mo["pos_months"], "n_months": mo["n_months"],
            "sd_month_bp": mo["sd_month_bp"], "sd_daily_bp": dl["sd_daily_bp"],
        })
    return pd.DataFrame(rows).set_index("fund")


# --------------------------------------------------------------------------- #
# The headline — a wrapper-versus-wrapper spread (pure tape, no fee assumption)
# --------------------------------------------------------------------------- #
def pair_spread(prices: pd.DataFrame, cheap: str, dear: str, lags: int = 3,
                seed: int = 961) -> dict:
    """Tracking-difference spread ``cheap - dear``, annualised, daily and monthly.

    Both legs hold the same metal and strike at the same close, so the bullion price
    cancels: this is the one comparison the tape can resolve at basis-point scale. The
    *measurement* uses no fee schedule at all — but note that **which** two wrappers get
    compared is read off the fee sheet (an ASSUMPTION, and today's sheet at that), so the
    pair choice is a selection even though the number is pure tape.

    Returns the naive iid *t* (``t_naive``, the one to lead with here), the HAC *t*
    (``tstat``, inflated by the mean-reverting residual), a moving-block bootstrap CI and
    an **iid** bootstrap CI (the wider, conservative interval), plus the count of calendar
    years in which the spread was positive.
    """
    px = prices[[cheap, dear]].dropna()
    mm = monthly_log_returns(px)
    x = (mm[cheap] - mm[dear]).dropna()
    mo = td_monthly(mm[cheap], mm[dear], lags=lags)
    dl = td_daily(px[cheap], px[dear])
    ci = block_bootstrap_ci(x, seed=seed, block=3)
    ci_iid = block_bootstrap_ci(x, seed=seed, block=1)
    by_year = x.groupby(x.index.year).sum()
    return {
        "cheap": cheap, "dear": dear,
        "spread_bpy": mo["td_bpy"], "tstat": mo["t_naive"], "t_hac": mo["tstat"],
        "monthly_acf1": mo["acf1"],
        "n_months": mo["n_months"], "pos_months": mo["pos_months"],
        "sd_month_bp": mo["sd_month_bp"],
        "pos_years": int((by_year > 0).sum()), "n_years": int(len(by_year)),
        "daily_bpy": dl["td_bpy"], "t_daily": dl["tstat"], "t_daily_naive": dl["t_naive"],
        "sd_daily_bp": dl["sd_daily_bp"], "daily_acf1": dl["acf1"], "n_days": dl["n_days"],
        "endpoint_bpy": td_endpoint(px[cheap], px[dear]),
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "frac_negative": ci["frac_negative"],
        "ci_iid_low": ci_iid["ci_low"], "ci_iid_high": ci_iid["ci_high"],
        "frac_negative_iid": ci_iid["frac_negative"],
        "monthly": x,
    }


def all_pairs(prices: pd.DataFrame, cohort, fees: dict | None = None,
              lags: int = 3) -> pd.DataFrame:
    """Every unordered pair's monthly spread, naive and HAC *t*, and (if given) its fee gap.

    Ten rows for five wrappers. The point of the table is the *joint* pattern: pairs with a
    wide fee gap should print a wide spread and pairs with none should print nothing.
    """
    rows = []
    for a, b in itertools.combinations(list(cohort), 2):
        px = prices[[a, b]].dropna()
        mm = monthly_log_returns(px)
        mo = td_monthly(mm[a], mm[b], lags=lags)
        rows.append({
            "pair": f"{a}-{b}", "a": a, "b": b,
            "fee_gap_bps": (float(fees[b]) - float(fees[a])) if fees else np.nan,
            "spread_bpy": mo["td_bpy"], "tstat": mo["t_naive"], "t_hac": mo["tstat"],
            "pos_months": mo["pos_months"], "n_months": mo["n_months"],
        })
    return pd.DataFrame(rows).set_index("pair")


def measurement_floor(prices: pd.DataFrame, cohort) -> dict:
    """How small a fee gap this tape could resolve at ``|t| = 2``.

    Uses the average pairwise dispersion of monthly wrapper-minus-wrapper differences: the
    smallest annualised gap that ``n`` non-overlapping months could separate from zero.
    A study whose claimed effect sits below this floor is measuring nothing.
    """
    px = prices[list(cohort)].dropna()
    mm = monthly_log_returns(px)
    d = np.log(px).diff().dropna()
    pairs = list(itertools.combinations(list(cohort), 2))
    sd_m = float(np.mean([(mm[a] - mm[b]).std(ddof=1) for a, b in pairs]) * 1e4)
    sd_d = float(np.mean([(d[a] - d[b]).std(ddof=1) for a, b in pairs]) * 1e4)
    n_months = len(mm)
    return {
        "sd_month_pair_bp": sd_m,
        "sd_daily_pair_bp": sd_d,
        "n_months": int(n_months),
        "n_days": int(len(d)),
        "detectable_bpy": float(2.0 * sd_m / np.sqrt(n_months) * MONTHS),
    }


# --------------------------------------------------------------------------- #
# The fee-rank test (this one DOES use the fee ASSUMPTION)
# --------------------------------------------------------------------------- #
def rank_test(fees, tds, n_perm: int = 20000, seed: int = 961) -> dict:
    """Spearman(fee, tracking difference) against a permutation null.

    A working fee ladder implies a strongly **negative** rank correlation (higher fee,
    worse tracking). The null shuffles the tracking differences across wrappers, which is
    the right null for "the fee ranking carries no information about the outcome ranking".
    With five funds the 120 permutations are enumerated exactly, so ``p_perm`` cannot fall
    below 2/120 = 0.0167 however perfect the match — a hard floor worth stating.
    """
    fees = np.asarray(fees, dtype=float)
    tds = np.asarray(tds, dtype=float)
    rho = spearman(fees, tds)
    n = len(fees)
    ra = rankdata(fees); ra = ra - ra.mean()
    rb = rankdata(tds); rb = rb - rb.mean()
    den = np.sqrt(float(ra @ ra) * float(rb @ rb))
    if den <= 0:
        return {"spearman": float("nan"), "p_perm": float("nan"), "n_funds": int(n),
                "exact": False, "n_draws": 0, "crit_5pct": float("nan"),
                "p_floor": float("nan")}
    if n <= 8:
        perms = np.array(list(itertools.permutations(rb)))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.tile(rb, (n_perm, 1)), axis=1)
        exact = False
    draws = perms @ ra / den
    p = float((np.abs(draws) >= abs(rho) - 1e-12).mean())
    return {
        "spearman": float(rho), "p_perm": p, "n_funds": int(n),
        "exact": bool(exact), "n_draws": int(len(draws)),
        "crit_5pct": float(np.percentile(np.abs(draws), 95.0)),
        "p_floor": float((np.abs(draws) >= np.abs(draws).max() - 1e-12).mean()),
    }


def pass_through(fees, tds) -> dict:
    """Cross-sectional OLS of tracking difference on fee: ``TD = a + b * fee``.

    A wrapper that leaks exactly its fee and nothing else gives ``b = -1``. The R-squared
    says how much of the five-point cross-section the fee sheet explains.
    """
    fees = np.asarray(fees, dtype=float)
    tds = np.asarray(tds, dtype=float)
    if len(fees) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"),
                "n": int(len(fees))}
    b, a = np.polyfit(fees, tds, 1)
    pred = a + b * fees
    ss_res = float(((tds - pred) ** 2).sum())
    ss_tot = float(((tds - tds.mean()) ** 2).sum())
    return {"slope": float(b), "intercept": float(a),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(fees))}


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_cut(prices: pd.DataFrame, cheap: str, dear: str, split: str = "2022-01-01",
            lags: int = 3) -> dict:
    """The cheapest-minus-priciest spread, re-split at ``split`` on the monthly series.

    Split on the already-computed monthly spreads rather than re-slicing the price frame,
    so the month straddling the cut is not silently lost to the second window's first diff.
    """
    px = prices[[cheap, dear]].dropna()
    mm = monthly_log_returns(px)
    x = (mm[cheap] - mm[dear]).dropna()
    out = {}
    for tag, mask in [("early", x.index < pd.Timestamp(split)),
                      ("late", x.index >= pd.Timestamp(split))]:
        seg = x[mask]
        if len(seg) < 4:
            out[tag] = None
            continue
        out[tag] = {
            "spread_bpy": float(seg.mean() * MONTHS * 1e4),
            "tstat": one_sample_t(seg.to_numpy()),          # naive: the conservative one here
            "t_hac": newey_west_t(seg.to_numpy(), lags=lags),
            "n_months": int(len(seg)),
            "pos_months": int((seg > 0).sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# What the fee gap is worth to a holder
# --------------------------------------------------------------------------- #
def holding_cost(spread_bpy: float, years=(1, 5, 10, 20)) -> list[dict]:
    """Compounded shortfall from owning the dear wrapper instead of the cheap one.

    ``(1 - spread)^n`` of terminal wealth, expressed as a percentage. On a
    buy-and-hold bullion sleeve there is no turnover and no timing: the gap simply
    compounds, which is the entire economic content of the result.
    """
    s = spread_bpy * 1e-4
    return [{"years": int(y),
             "shortfall_pct": float((1.0 - (1.0 - s) ** y) * 100.0),
             "per_100k_usd": float((1.0 - (1.0 - s) ** y) * 100_000.0)}
            for y in years]


def spread_break_even(spread_bpy: float,
                      round_trip_bps_grid=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)) -> list[dict]:
    """Days for the fee gap to repay a **wider** round-trip execution cost.

    The honest counterweight: a cheap wrapper that trades on a wider spread costs you the
    difference up front. The round-trip differential is an ASSUMPTION — no free tape
    carries quoted spreads — so it is swept from a penny-wide 0 bp to a punitive 30 bp
    rather than picked.
    """
    out = []
    for c in round_trip_bps_grid:
        days = (c / spread_bpy) * 365.0 if spread_bpy > 0 else float("inf")
        out.append({"round_trip_bps": float(c), "break_even_days": float(days)})
    return out


def tax_break_even(spread_bpy: float, gains=(0.20, 0.50, 1.00, 3.00),
                   tax_rates=(0.15, 0.20, 0.238)) -> list[dict]:
    """Years for the fee gap to repay the capital-gains bill on *switching* wrapper.

    The tax rate and the embedded gain are ASSUMPTIONS — a taxable holder's own numbers —
    so both are swept. Selling a position with embedded gain ``g`` (as a fraction of cost)
    realises ``g / (1 + g)`` of current NAV as gain; taxing it at ``rate`` costs that much
    of NAV today, against ``spread_bpy`` per year forever after. US bullion trusts are
    taxed as collectibles (a 28% top rate plus the 3.8% surtax), so the high end of the
    sweep is the realistic one for a long-held gold sleeve.
    """
    out = []
    for rate in tax_rates:
        for g in gains:
            cost = rate * g / (1.0 + g)
            years = (float(np.log(1.0 / (1.0 - cost)) / (spread_bpy * 1e-4))
                     if spread_bpy > 0 else float("inf"))
            out.append({"tax_rate": float(rate), "embedded_gain": float(g),
                        "cost_of_nav": float(cost), "break_even_years": years})
    return out


def long_short_net(gross_bpy: float,
                   borrow_grid=(0.0, 10.0, 25.0, 50.0, 100.0),
                   cost_bps: float = 2.0, turns_per_year: float = 1.0) -> list[dict]:
    """The market-neutral version: long the cheap wrapper, short the dear one.

    Self-financing, so the spread return is already an excess return — but the **short
    pays borrow**, and the borrow rate is an ASSUMPTION (no free tape carries it), so it
    is swept rather than picked. One-way cost x NAV is charged on both legs,
    ``turns_per_year`` times.
    """
    return [
        {"borrow_bpy": float(b),
         "net_bpy": float(gross_bpy - b - 4.0 * cost_bps * turns_per_year),
         "alive": bool(gross_bpy - b - 4.0 * cost_bps * turns_per_year > 0)}
        for b in borrow_grid
    ]


# --------------------------------------------------------------------------- #
# Liquidity — the counterweight, on real tape
# --------------------------------------------------------------------------- #
def liquidity_table(dvol: pd.DataFrame, cohort, trade_usd: float = 100e6,
                    recent_from: str | None = None) -> pd.DataFrame:
    """Median daily dollar volume per wrapper, and what a ``trade_usd`` ticket is worth in it.

    ``days_of_adv`` = trade size / median daily dollar volume. Pure tape (close x volume);
    it does not observe the quoted spread, which is why the spread itself is swept
    separately in :func:`spread_break_even`.
    """
    d = dvol[list(cohort)].dropna()
    if recent_from is not None:
        d = d[d.index >= pd.Timestamp(recent_from)]
    rows = []
    for tk in cohort:
        med = float(d[tk].median())
        rows.append({"fund": tk, "median_dvol_usd_m": med / 1e6,
                     "days_of_adv": float(trade_usd / med) if med > 0 else float("inf"),
                     "n_days": int(len(d))})
    return pd.DataFrame(rows).set_index("fund")


# --------------------------------------------------------------------------- #
# The rules — ownership race and the tracking-winner chase
# --------------------------------------------------------------------------- #
def ownership_race(prices: pd.DataFrame, cohort, cash: str, fees: dict,
                   cost_bps: float = 2.0, lookback_months: int = 12) -> dict:
    """Race three ways of choosing a wrapper, all measured **excess-of-cash**.

    - ``own_cheapest`` — hold the lowest-fee wrapper for the whole window. A purchase
      decision: no forecast, no turnover, no cost after the first trade.
    - ``own_priciest`` — hold the highest-fee wrapper. The legacy holder who never looked.
    - ``chase_winner`` — every calendar year-end, rank the cohort by its trailing
      ``lookback_months``-month tracking difference against the cohort mean and hold last
      year's winner. Ranked on data through the last session of the year (day *T*),
      **executed at the next session's close** (*T+1*), so the new wrapper's return only
      starts accruing on *T+2* — a genuine one-session lag, not a same-close fill. Costs:
      ``cost_bps`` one-way x NAV on each leg of a change of wrapper. No short leg, so no
      borrow.

      Before the first year-end there is no signal, so the arm defaults to the fee-cheapest
      wrapper — knowable ex ante from the published sheet, and a default that *flatters*
      the chase arm (it starts life holding the eventual winner). It loses anyway.

    Excess-of-cash means each arm's daily simple return minus ``cash``'s, so the Sharpe
    race is apples-to-apples. At ~17% annualised vol a 30 bp/yr fee gap is worth about two
    hundredths of a Sharpe point; the race is reported precisely to show that. The *t*
    returned alongside is a HAC *t* on the mean daily **return difference** — it is not an
    inference on the Sharpe gap, and on this mean-reverting series it is the
    anti-conservative one, so the naive *t* is returned next to it.
    """
    px = prices[list(cohort) + [cash]].dropna()
    r = px.pct_change().dropna()
    r_cash = r[cash]
    cheapest = min(cohort, key=lambda t: fees[t])
    priciest = max(cohort, key=lambda t: fees[t])

    mm = monthly_log_returns(px[list(cohort)])
    cohort_m = mm[list(cohort)].mean(axis=1)
    rel = mm[list(cohort)].sub(cohort_m, axis=0)

    y_ends = pd.Series(px.index, index=px.index).resample("YE").last().dropna()
    holdings = pd.Series(index=r.index, dtype=object)
    for i, ye in enumerate(y_ends.to_numpy()):
        ye = pd.Timestamp(ye)
        window = rel[rel.index <= ye].tail(lookback_months)
        pick = str(window.mean().idxmax()) if len(window) == lookback_months else cheapest
        nxt = r.index[r.index > ye]
        # Ranked at T's close, filled at T+1's close -> the new holding first earns on T+2.
        # Taking nxt[0] here would have credited the T->T+1 move to the new wrapper, i.e.
        # a same-close fill on the very session the signal was formed.
        if len(nxt) < 2:
            continue
        start = nxt[1]
        end = pd.Timestamp(y_ends.to_numpy()[i + 1]) if i + 1 < len(y_ends) else r.index[-1]
        holdings.loc[(holdings.index >= start) & (holdings.index <= end)] = pick
    holdings = holdings.ffill().fillna(cheapest)

    r_chase = pd.Series([r.at[d, holdings.at[d]] for d in r.index], index=r.index, dtype=float)
    switch = (holdings != holdings.shift(1)) & holdings.shift(1).notna()
    r_chase = r_chase - switch.astype(float) * 2.0 * cost_bps * 1e-4  # round trip: sell + buy

    arms = {"own_cheapest": r[cheapest], "own_priciest": r[priciest], "chase_winner": r_chase}
    out = {"cheapest": cheapest, "priciest": priciest, "n_switches": int(switch.sum()),
           "cost_bps": float(cost_bps), "n_days": int(len(r)), "arms": {}}
    for tag, series in arms.items():
        e = (series - r_cash).dropna()
        sd = e.std(ddof=1)
        wealth = (1.0 + series.dropna()).cumprod()
        years = len(series.dropna()) / TRADING_DAYS
        out["arms"][tag] = {
            "excess_sharpe": float(e.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan"),
            "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan"),
            "vol_ann": float(series.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "total_return": float(wealth.iloc[-1] - 1.0),
        }
    out["sharpe_gap_cheap_minus_dear"] = (out["arms"]["own_cheapest"]["excess_sharpe"]
                                          - out["arms"]["own_priciest"]["excess_sharpe"])
    cmd = (r[cheapest] - r[priciest]).dropna().to_numpy()
    out["t_cheap_minus_dear"] = newey_west_t(cmd)          # HAC, anti-conservative here
    out["t_cheap_minus_dear_naive"] = one_sample_t(cmd)    # the conservative daily t
    chase_diff = (r_chase - r[cheapest]).dropna().to_numpy()
    out["t_chase_minus_cheapest"] = newey_west_t(chase_diff)
    out["t_chase_minus_cheapest_naive"] = one_sample_t(chase_diff)
    out["holdings"] = holdings
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict) -> dict:
    """Run the whole measurement stack on a planted panel.

    The tests read the **published** fee sheet (``truth['fees_bps']``) — the thing an
    investor can look up — while the world charges ``truth['fees_eff_bps']``. At
    ``signal_strength=1`` the two coincide, so the estimators must recover the planted gap
    and the rank test must fire. At ``signal_strength=0`` every wrapper charges the same
    fee while the sheet still shows dispersion, so the rank test faces a genuine,
    informative-looking input with nothing behind it and must stay quiet. Proves the
    harness is unbiased — it never supports a real-tape stamp.
    """
    funds = truth["fund_cols"]
    fees = np.asarray(truth["fees_bps"], dtype=float)          # the published sheet
    fees_eff = np.asarray(truth["fees_eff_bps"], dtype=float)  # what the world charges
    px = prices[funds]
    mm = monthly_log_returns(px)
    cohort_m = mm[funds].mean(axis=1)

    tds = np.asarray([td_monthly(mm[f], cohort_m)["td_bpy"] for f in funds], dtype=float)

    cheap, dear = funds[int(np.argmin(fees))], funds[int(np.argmax(fees))]
    if cheap == dear:  # degenerate sheet: pick two arbitrary distinct wrappers
        cheap, dear = funds[0], funds[-1]
    ps = pair_spread(px, cheap, dear)
    rt = rank_test(fees, tds)
    pt = pass_through(fees, tds)

    return {
        "pair_spread_bpy": ps["spread_bpy"], "pair_t": ps["tstat"],  # naive iid t
        "pair_t_hac": ps["t_hac"],
        "pair_daily_bpy": ps["daily_bpy"], "pair_daily_t": ps["t_daily"],
        "planted_gap_bpy": float(fees_eff.max() - fees_eff.min()),
        "spearman": rt["spearman"], "p_perm": rt["p_perm"],
        "pass_through_slope": pt["slope"], "pass_through_r2": pt["r2"],
        "td_rel_bpy": tds.tolist(),
        "n_months": ps["n_months"],
    }
