"""Estimators, tests and rules for Study 959 — Crypto Fee War.

A wrapper's **tracking difference** is what it actually delivered minus what the asset
delivered. For a spot-bitcoin ETF the only recurring, contractual leak is the sponsor
fee, so a clean tape should hand the fee back, in basis points per year, fund by fund.

Three estimators of the same quantity, because on this tape they disagree and the
disagreement is the story:

1. ``td_endpoint`` — cumulative log difference over the window, annualised. One number
   from two anchors, and it therefore inherits the premium/discount at *both* anchors.
   On the real tape the first anchor is GBTC's 2024-01-11 conversion day, which is worth
   ~250 bp on its own; the estimator is quoted for completeness and never trusted.
2. ``td_trend_slope`` — OLS slope of the log ratio on time, with a HAC standard error.
   Uses all 618 anchors instead of two, so the anchor noise averages out.
3. ``td_monthly`` — the mean of **non-overlapping complete-calendar-month** tracking
   differences, annualised, with a HAC *t*. The one every headline number is quoted from.
   **It is not anchor-free, and this study does not pretend otherwise**: the mean of
   non-overlapping log increments telescopes, so its point estimate is *exactly* the
   endpoint estimate taken between the first and last complete month-end. What it buys over
   ``td_endpoint`` is (a) a pair of anchors that excludes the contaminated first session
   (GBTC's 2024-01-11 conversion close) and (b) a genuine dispersion — the month-by-month
   signs and the HAC *t* — which the endpoint estimator cannot give at all. The anchors
   themselves are checked separately, by trimming months off both ends and by
   ``td_trend_slope``, which uses every session as an anchor.

The measurement floor decides what any of them can see. Against ``BTC-USD`` the daily
difference has a standard deviation of ~135 bp — bitcoin trades 24/7 and the funds are
struck at 16:00 New York, so every comparison carries a clock stub. Against *another
wrapper* the same difference is ~8-9 bp, because the stub is common and cancels. A 20
bp/yr fee is 0.08 bp/day. That is why the fund-versus-fund spread is the instrument and
the fund-versus-spot number is an indication.

**One execution lag, once.** The measurement above involves no decision and therefore no
lag. The only rule that *trades* — ``rotation_race``'s tracking-winner rotation — ranks on
data through the last day of a quarter and switches at the next day's close; a switch is
sell-plus-buy, so it is charged a **round trip** of 2 x ``cost_bps`` x NAV (4 bp at the 2 bp
one-way default). No short leg in the ownership race; the long/short
version is handled separately by ``long_short_net``, where the short pays borrow.
"""

from __future__ import annotations

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
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def rankdata(a) -> np.ndarray:
    """Average-tie ranks (a two-line stand-in for scipy, which the desk does not import)."""
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
    x, n_boot: int = 5000, block: int = 3, seed: int = 959,
    scale: float = MONTHS * 1e4, alpha: float = 0.05,
) -> dict:
    """Circular moving-block bootstrap CI for the mean of a (monthly) series, in bp/yr.

    Blocks of ``block`` consecutive months preserve whatever serial dependence the
    premium/discount leaves behind; blocks wrap around the end of the sample (circular), so
    every observation gets equal weight. ``scale`` converts the mean into bp/yr.
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
# The three tracking-difference estimators
# --------------------------------------------------------------------------- #
def _log_ratio(fund: pd.Series, bench: pd.Series) -> pd.Series:
    common = fund.index.intersection(bench.index)
    return (np.log(fund.reindex(common)) - np.log(bench.reindex(common))).dropna()


def td_endpoint(fund: pd.Series, bench: pd.Series) -> float:
    """Cumulative log tracking difference, annualised, in bp/yr — two anchors only."""
    r = _log_ratio(fund, bench)
    if len(r) < 3:
        return float("nan")
    years = (r.index[-1] - r.index[0]).days / 365.25
    return float((r.iloc[-1] - r.iloc[0]) / years * 1e4) if years > 0 else float("nan")


def td_trend_slope(fund: pd.Series, bench: pd.Series, lags: int | None = None) -> dict:
    """OLS slope of the log ratio on time (bp/yr) with a HAC standard error.

    Uses every session as an anchor, so a single dislocated endpoint cannot move it.
    Reported with the residual's first-order autocorrelation: where that residual is
    persistent (fund-versus-fund, an AR(1) premium) the HAC *t* is **optimistic** and the
    monthly estimator is the one to quote.
    """
    r = _log_ratio(fund, bench)
    n = len(r)
    if n < 30:
        return {"slope_bpy": float("nan"), "tstat": float("nan"), "n": int(n),
                "resid_acf1": float("nan")}
    years = np.asarray((r.index - r.index[0]).days, dtype=float) / 365.25
    X = np.column_stack([np.ones(n), years])
    beta, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    resid = r.to_numpy() - X @ beta
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.inv(X.T @ X)
    u = X * resid[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S = S + w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = float(np.sqrt(V[1, 1]))
    acf1 = float(np.corrcoef(resid[1:], resid[:-1])[0, 1]) if n > 3 else float("nan")
    return {"slope_bpy": float(beta[1] * 1e4), "tstat": float(beta[1] / se) if se > 0 else float("nan"),
            "n": int(n), "resid_acf1": acf1}


def monthly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping complete-calendar-month log returns (the first partial month drops)."""
    m = np.log(prices).resample("ME").last()
    return m.diff().dropna()


def td_monthly(fund_m: pd.Series, bench_m: pd.Series, lags: int = 3) -> dict:
    """Mean non-overlapping monthly tracking difference, annualised (bp/yr), with a HAC *t*.

    The headline estimator. Its *t* is computed on genuinely non-overlapping observations,
    and it reports how many of them are individually positive.

    **Honest caveat.** Log increments telescope, so ``td_bpy`` is algebraically identical to
    an endpoint estimate between the first and last complete month-end. It is *not* immune
    to its own two anchors — it merely uses better ones than :func:`td_endpoint` (the
    contaminated first session is dropped) and, unlike the endpoint estimator, it exposes
    the dispersion that says whether the gap is a steady drip or one jump. Use
    :func:`td_trend_slope` (all sessions as anchors) and an end-trim sweep to check the
    anchors; ``examples/verify.py`` prints both.
    """
    x = (fund_m - bench_m).dropna().to_numpy()
    if len(x) < 4:
        return {"td_bpy": float("nan"), "tstat": float("nan"), "n_months": int(len(x)),
                "sd_month_bp": float("nan"), "pos_months": 0}
    return {
        "td_bpy": float(x.mean() * MONTHS * 1e4),
        "tstat": newey_west_t(x, lags=lags),
        "n_months": int(len(x)),
        "sd_month_bp": float(x.std(ddof=1) * 1e4),
        "pos_months": int((x > 0).sum()),
    }


def tracking_table(
    prices: pd.DataFrame, cohort, bench: str, fees: dict | None = None, lags: int = 3,
) -> pd.DataFrame:
    """Every estimator, every fund, on one common window.

    Columns: the published fee (if given), the three benchmark-relative estimators, and
    the **cohort-relative** monthly tracking difference (each fund minus the equal-weight
    cohort mean) — the sharp cross-sectional measure, because the clock stub in the
    benchmark cancels out of it entirely.
    """
    px = prices[list(cohort) + [bench]].dropna()
    mm = monthly_log_returns(px)
    cohort_m = mm[list(cohort)].mean(axis=1)
    rows = []
    for tk in cohort:
        slope = td_trend_slope(px[tk], px[bench])
        mo = td_monthly(mm[tk], mm[bench], lags=lags)
        rel = td_monthly(mm[tk], cohort_m, lags=lags)
        row = {
            "fund": tk,
            "fee_bps": float(fees[tk]) if fees else np.nan,
            "td_endpoint_bpy": td_endpoint(px[tk], px[bench]),
            "td_slope_bpy": slope["slope_bpy"], "t_slope": slope["tstat"],
            "td_monthly_bpy": mo["td_bpy"], "t_monthly": mo["tstat"],
            "td_rel_bpy": rel["td_bpy"], "t_rel": rel["tstat"],
            "sd_month_bp": mo["sd_month_bp"], "sd_month_rel_bp": rel["sd_month_bp"],
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("fund")


# --------------------------------------------------------------------------- #
# The headline — a fund-versus-fund spread (pure tape, no fee assumption)
# --------------------------------------------------------------------------- #
def pair_spread(prices: pd.DataFrame, cheap: str, dear: str, lags: int = 3) -> dict:
    """Monthly tracking-difference spread ``cheap - dear``, annualised, with HAC *t* and CI.

    Both legs hold the same asset and are struck at the same close, so the benchmark's
    clock stub cancels: this is the one comparison the tape can resolve at basis-point
    scale. Nothing here uses the fee schedule — it is a pure tape measurement.
    """
    px = prices[[cheap, dear]].dropna()
    mm = monthly_log_returns(px)
    x = (mm[cheap] - mm[dear]).dropna()
    st = td_monthly(mm[cheap], mm[dear], lags=lags)
    ci = block_bootstrap_ci(x)
    daily = np.log(px).diff().dropna()
    return {
        "cheap": cheap, "dear": dear,
        "spread_bpy": st["td_bpy"], "tstat": st["tstat"],
        "n_months": st["n_months"], "pos_months": st["pos_months"],
        "sd_month_bp": st["sd_month_bp"],
        "sd_daily_bp": float((daily[cheap] - daily[dear]).std(ddof=1) * 1e4),
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "frac_negative": ci["frac_negative"],
        "monthly": x,
    }


def measurement_floor(prices: pd.DataFrame, cohort, bench: str) -> dict:
    """The daily noise of a fund-vs-benchmark difference and a fund-vs-fund difference.

    The ratio between them is the reason a fee is measurable against a sibling wrapper and
    not against the 24/7 spot tape.
    """
    px = prices[list(cohort) + [bench]].dropna()
    d = np.log(px).diff().dropna()
    mm = monthly_log_returns(px)
    ref = cohort[0]
    peers = [t for t in cohort if t != ref]
    vs_bench_d = np.mean([float((d[t] - d[bench]).std(ddof=1)) for t in cohort]) * 1e4
    vs_peer_d = np.mean([float((d[t] - d[ref]).std(ddof=1)) for t in peers]) * 1e4
    vs_bench_m = np.mean([float((mm[t] - mm[bench]).std(ddof=1)) for t in cohort]) * 1e4
    vs_peer_m = np.mean([float((mm[t] - mm[ref]).std(ddof=1)) for t in peers]) * 1e4
    n_months = len(mm)
    # The smallest annualised gap a two-sided t = 2 could resolve on this many months.
    det = lambda sd_m: float(2.0 * sd_m / np.sqrt(n_months) * MONTHS)  # noqa: E731
    return {
        "sd_daily_vs_bench_bp": float(vs_bench_d),
        "sd_daily_vs_peer_bp": float(vs_peer_d),
        "sd_month_vs_bench_bp": float(vs_bench_m),
        "sd_month_vs_peer_bp": float(vs_peer_m),
        "ratio": float(vs_bench_d / vs_peer_d) if vs_peer_d > 0 else float("nan"),
        "detectable_vs_bench_bpy": det(vs_bench_m),
        "detectable_vs_peer_bpy": det(vs_peer_m),
        "n_months": int(n_months),
    }


# --------------------------------------------------------------------------- #
# The fee-rank test (this one DOES use the fee ASSUMPTION)
# --------------------------------------------------------------------------- #
def rank_test(fees, tds, n_perm: int = 20000, seed: int = 959) -> dict:
    """Spearman(fee, tracking difference) against a permutation null.

    A working fee ladder implies a strongly **negative** rank correlation (higher fee,
    worse tracking). The null shuffles the tracking differences across funds, which is the
    right null for "the fee ranking carries no information about the outcome ranking".
    With <= 8 funds the permutation is enumerated exactly; above that it is sampled.
    """
    import itertools

    fees = np.asarray(fees, dtype=float)
    tds = np.asarray(tds, dtype=float)
    rho = spearman(fees, tds)
    n = len(fees)
    # Permuting the tracking differences permutes their ranks, so the whole null can be
    # built once on centred rank vectors — exactly equivalent, and fast enough to enumerate.
    ra = rankdata(fees); ra = ra - ra.mean()
    rb = rankdata(tds); rb = rb - rb.mean()
    den = np.sqrt(float(ra @ ra) * float(rb @ rb))
    if den <= 0:
        return {"spearman": float("nan"), "p_perm": float("nan"), "n_funds": int(n),
                "exact": False, "n_draws": 0, "crit_5pct": float("nan")}
    if n <= 8:
        perms = np.array(list(itertools.permutations(rb)))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.tile(rb, (n_perm, 1)), axis=1)
        exact = False
    draws = perms @ ra / den
    p = float((np.abs(draws) >= abs(rho) - 1e-12).mean())
    return {"spearman": float(rho), "p_perm": p, "n_funds": int(n),
            "exact": bool(exact), "n_draws": int(len(draws)),
            # the |rho| this fee vector would need to clear 5% — a fee sheet with five
            # funds tied at one number cannot generate a fine-grained rank statistic.
            "crit_5pct": float(np.percentile(np.abs(draws), 95.0))}


def pass_through(fees, tds) -> dict:
    """Cross-sectional OLS of tracking difference on fee: TD = a + b * fee.

    A wrapper that leaks exactly its fee and nothing else gives ``b = -1``. The R-squared
    says how much of the cross-section the fee sheet explains — and dropping the priciest
    fund says how much of that was one anchor doing all the work.
    """
    fees = np.asarray(fees, dtype=float)
    tds = np.asarray(tds, dtype=float)
    if len(fees) < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": len(fees)}
    b, a = np.polyfit(fees, tds, 1)
    pred = a + b * fees
    ss_res = float(((tds - pred) ** 2).sum())
    ss_tot = float(((tds - tds.mean()) ** 2).sum())
    return {"slope": float(b), "intercept": float(a),
            "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "n": int(len(fees))}


# --------------------------------------------------------------------------- #
# Waiver expiries — do they show up in the tape?
# --------------------------------------------------------------------------- #
def waiver_event_test(prices: pd.DataFrame, cohort, waiver: dict, bench: str | None = None) -> pd.DataFrame:
    """For each fund, the step in its cohort-relative tracking difference at its waiver end.

    Measured on non-overlapping months, fund minus the equal-weight cohort mean, so the
    benchmark's clock stub is gone. A waiver that expires should print a **negative** step
    of roughly the fund's headline fee. The waiver dates are an ASSUMPTION (see
    ``data.WAIVER``); several were AUM-capped and may have ended earlier than scheduled.
    """
    px = prices[list(cohort) + ([bench] if bench else [])].dropna()
    mm = monthly_log_returns(px)
    cohort_m = mm[list(cohort)].mean(axis=1)
    rows = []
    for tk in cohort:
        if tk not in waiver:
            continue
        wfee, wend = waiver[tk]
        x = (mm[tk] - cohort_m).dropna()
        cut = pd.Timestamp(wend)
        pre, post = x[x.index <= cut].to_numpy(), x[x.index > cut].to_numpy()
        if len(pre) < 3 or len(post) < 3:
            rows.append({"fund": tk, "waiver_end": wend, "n_pre": len(pre), "n_post": len(post),
                         "pre_bpy": np.nan, "post_bpy": np.nan, "step_bpy": np.nan, "welch_t": np.nan})
            continue
        rows.append({
            "fund": tk, "waiver_end": wend, "n_pre": int(len(pre)), "n_post": int(len(post)),
            "pre_bpy": float(pre.mean() * MONTHS * 1e4),
            "post_bpy": float(post.mean() * MONTHS * 1e4),
            "step_bpy": float((post.mean() - pre.mean()) * MONTHS * 1e4),
            "welch_t": welch_t(post, pre),
        })
    return pd.DataFrame(rows).set_index("fund")


# --------------------------------------------------------------------------- #
# Era cut
# --------------------------------------------------------------------------- #
def era_cut(prices: pd.DataFrame, cheap: str, dear: str, split: str = "2025-01-01") -> dict:
    """The cheapest-minus-priciest spread, re-split at ``split`` on the monthly series.

    Split on the already-computed monthly spreads rather than re-slicing the price frame,
    so the month straddling the cut is not silently lost to the second window's first diff.
    """
    px = prices[[cheap, dear]].dropna()
    mm = monthly_log_returns(px)
    x = (mm[cheap] - mm[dear]).dropna()
    out = {}
    for tag, mask in [("early", x.index < pd.Timestamp(split)), ("late", x.index >= pd.Timestamp(split))]:
        seg = x[mask]
        if len(seg) < 4:
            out[tag] = None
            continue
        out[tag] = {
            "spread_bpy": float(seg.mean() * MONTHS * 1e4),
            "tstat": newey_west_t(seg.to_numpy(), lags=3),
            "n_months": int(len(seg)),
            "pos_months": int((seg > 0).sum()),
        }
    return out


# --------------------------------------------------------------------------- #
# The rules — ownership race and the long/short
# --------------------------------------------------------------------------- #
def rotation_race(
    prices: pd.DataFrame, cohort, cash: str, fees: dict,
    cost_bps: float = 2.0, lookback_months: int = 3,
) -> dict:
    """Race three ways of choosing a wrapper, all measured **excess-of-cash**.

    - ``own_cheapest`` — hold the fund with the lowest published fee for the whole window.
      A purchase decision: no forecast, no turnover, no cost after the first trade.
    - ``own_priciest`` — hold the fund with the highest published fee. The do-nothing
      legacy holder.
    - ``rotate_winner`` — every quarter, rank the cohort by its trailing
      ``lookback_months``-month tracking difference against the cohort mean and hold last
      quarter's winner. Ranked on data through the last session of the quarter, **switched
      at the next session's close** (one execution lag), paying a **round trip** —
      2 x ``cost_bps`` x NAV, i.e. sell plus buy — on each change of fund. No short leg, so
      no borrow. The trailing window is taken from month-end labels at or before the ranking
      session, so in quarters whose last session falls before the calendar month end the
      just-closed month is excluded: the rule is never fed anything it could not have had,
      and in those quarters it is fed strictly less.

    Excess-of-cash means each arm's daily simple return minus ``cash``'s, so the Sharpe
    race is apples-to-apples. At ~50% annualised vol a basis point a month is invisible in
    Sharpe; the race is reported precisely to show that.
    """
    px = prices[list(cohort) + [cash]].dropna()
    r = px.pct_change().dropna()
    r_cash = r[cash]
    cheapest = min(cohort, key=lambda t: fees[t])
    priciest = max(cohort, key=lambda t: fees[t])

    mm = monthly_log_returns(px[list(cohort)])
    cohort_m = mm[list(cohort)].mean(axis=1)
    rel = mm[list(cohort)].sub(cohort_m, axis=0)

    # Quarter-end ranking dates; hold from the NEXT session (one execution lag).
    q_ends = pd.Series(px.index, index=px.index).resample("QE").last().dropna()
    holdings = pd.Series(index=r.index, dtype=object)
    current = cheapest
    for i, qe in enumerate(q_ends.to_numpy()):
        qe = pd.Timestamp(qe)
        window = rel[rel.index <= qe].tail(lookback_months)
        if len(window) == lookback_months:
            pick = str(window.mean().idxmax())
        else:
            pick = cheapest
        nxt = r.index[r.index > qe]
        if len(nxt) == 0:
            continue
        start = nxt[0]
        end = pd.Timestamp(q_ends.to_numpy()[i + 1]) if i + 1 < len(q_ends) else r.index[-1]
        holdings.loc[(holdings.index >= start) & (holdings.index <= end)] = pick
    holdings = holdings.ffill().fillna(cheapest)

    r_rot = pd.Series(
        [r.at[d, holdings.at[d]] for d in r.index], index=r.index, dtype=float
    )
    switch = (holdings != holdings.shift(1)) & holdings.shift(1).notna()
    r_rot = r_rot - switch.astype(float) * 2.0 * cost_bps * 1e-4  # round trip: sell + buy

    arms = {
        "own_cheapest": r[cheapest], "own_priciest": r[priciest], "rotate_winner": r_rot,
    }
    out = {"cheapest": cheapest, "priciest": priciest,
           "n_switches": int(switch.sum()), "cost_bps": cost_bps,
           "n_days": int(len(r)), "arms": {}}
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
    out["sharpe_gap_cheap_minus_dear"] = (
        out["arms"]["own_cheapest"]["excess_sharpe"] - out["arms"]["own_priciest"]["excess_sharpe"]
    )
    out["holdings"] = holdings
    return out


def long_short_net(gross_bpy: float, borrow_grid=(0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0),
                   cost_bps: float = 2.0, turns_per_year: float = 1.0) -> list[dict]:
    """The market-neutral version: long the cheap wrapper, short the dear one.

    Self-financing, so the spread return is already an excess return — but the **short
    pays borrow**, and a hard-to-borrow legacy trust is exactly where borrow is dear. The
    borrow rate is an ASSUMPTION (no free tape carries it), so it is swept rather than
    picked. One-way cost x NAV is charged on both legs, ``turns_per_year`` times.
    """
    return [
        {"borrow_bpy": float(b),
         "net_bpy": float(gross_bpy - b - 4.0 * cost_bps * turns_per_year),
         "alive": bool(gross_bpy - b - 4.0 * cost_bps * turns_per_year > 0)}
        for b in borrow_grid
    ]


def switch_break_even(spread_bpy: float, one_way_bps_grid=(0.0, 2.0, 5.0, 10.0, 25.0)) -> list[dict]:
    """Days for the spread to repay a round-trip switch out of the dear wrapper."""
    out = []
    for c in one_way_bps_grid:
        days = (2.0 * c) / spread_bpy * 365.0 if spread_bpy > 0 else float("inf")
        out.append({"one_way_bps": float(c), "break_even_days": float(days)})
    return out


def tax_break_even(spread_bpy: float, gains=(0.20, 0.50, 1.00, 3.00), tax_rates=(0.15, 0.20, 0.238)) -> list[dict]:
    """Years for the spread to repay the capital-gains bill on switching wrappers.

    The tax rate and the embedded gain are ASSUMPTIONS — a taxable holder's own numbers —
    so both are swept. Selling a position with embedded gain ``g`` (as a fraction of cost)
    realises ``g / (1 + g)`` of current NAV as gain; taxing it at ``rate`` costs that much
    of NAV today, against ``spread_bpy`` per year forever after.
    """
    out = []
    for rate in tax_rates:
        for g in gains:
            cost = rate * g / (1.0 + g)
            years = float(np.log(1.0 / (1.0 - cost)) / (spread_bpy * 1e-4)) if spread_bpy > 0 else float("inf")
            out.append({"tax_rate": float(rate), "embedded_gain": float(g),
                        "cost_of_nav": float(cost), "break_even_years": years})
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, n_perm: int = 4000) -> dict:
    """Run the whole measurement stack on a planted panel.

    The tests read the **published** fee sheet (``truth['fees_bps']``) — the thing an
    investor can look up — while the world charges ``truth['fees_eff_bps']``. At
    ``signal_strength=1`` the two coincide, so the estimators must recover the gap and the
    rank test must fire. At ``signal_strength=0`` every fund charges the same fee while the
    sheet still shows dispersion, so the rank test faces a genuine, informative-looking
    input with nothing behind it and must stay quiet. Proves the harness is unbiased — it
    never supports a real-tape stamp.
    """
    funds = truth["fund_cols"]
    fees = np.asarray(truth["fees_bps"], dtype=float)          # the published sheet
    fees_eff = np.asarray(truth["fees_eff_bps"], dtype=float)  # what the world charges
    mm = monthly_log_returns(prices[funds + ["bench"]])
    cohort_m = mm[funds].mean(axis=1)

    tds, ts = [], []
    for f in funds:
        st = td_monthly(mm[f], cohort_m)
        tds.append(st["td_bpy"]); ts.append(st["tstat"])
    tds = np.asarray(tds)

    cheap, dear = funds[int(np.argmin(fees))], funds[int(np.argmax(fees))]
    if cheap == dear:  # the null: every fee identical, so pick two arbitrary distinct funds
        cheap, dear = funds[0], funds[-1]
    ps = pair_spread(prices[funds], cheap, dear)
    rt = rank_test(fees, tds, n_perm=n_perm)
    pt = pass_through(fees, tds)
    bench_st = td_monthly(mm[dear], mm["bench"])

    return {
        "pair_spread_bpy": ps["spread_bpy"], "pair_t": ps["tstat"],
        "planted_gap_bpy": float(fees_eff.max() - fees_eff.min()),
        "spearman": rt["spearman"], "p_perm": rt["p_perm"],
        "pass_through_slope": pt["slope"], "pass_through_r2": pt["r2"],
        "td_rel_bpy": tds.tolist(), "t_rel": ts,
        "dear_vs_bench_bpy": bench_st["td_bpy"], "dear_vs_bench_t": bench_st["tstat"],
        "n_months": ps["n_months"],
    }
