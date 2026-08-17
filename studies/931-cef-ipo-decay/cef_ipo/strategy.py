"""Event study + inference for Study 931 — the closed-end-fund IPO hole.

The claim. A closed-end fund IPOs at a round offering price ($20, or $25 for preferred
funds); the underwriting load (3-6%) comes out of the proceeds, so the fund's net asset
value on day one is already several percent below what the subscriber paid. The syndicate
supports the price for a few weeks, then the support ends and the fund drifts to the
discount that seasoned CEFs trade at. The subscriber's money goes into a hole.

What we measure. For each fund on the hardcoded list we take the **total return from the
offering price** over 1 / 3 / 6 / 12 months and subtract the **total return of an
asset-class benchmark ETF** over the identical calendar window. That difference is the
*abnormal* return: it strips out the fact that a 2021-vintage credit fund was going to have
a bad 2022 whatever it did. A one-sample *t* across funds asks whether the average abnormal
return is reliably negative.

Conventions, all deliberate:

- **Entry.** ``entry="offer"`` (the headline) prices a genuine subscriber: they pay the
  offering price, so the day-one gap ``raw_close_day1 / offer - 1`` is part of their return.
  ``entry="close"`` prices a buyer at the first traded close instead — a sweep, not the
  headline.
- **One execution lag, and where.** The event study itself has no rebalancing decision to
  lag — the IPO date *is* the entry. The lag lives in the tradable mirror
  (:func:`short_trade`): the "this fund just IPO'd" information is known at the day-one
  close, and the short is opened at the **next** close (t+1).
- **Costs.** One-way bps x NAV per leg per side (entry and exit, two legs = four charges),
  and the short leg pays a borrow rate that is swept — newly issued CEFs are the textbook
  hard-to-borrow name.
- **Total return everywhere.** Both legs are ``auto_adjust=True`` closes, so distributions
  (which are enormous for CEFs) are reinvested on both sides. A price-only comparison would
  invent a fake decay out of thin air.
- **The load is never subtracted.** It is the *mechanism*, and the price tape already
  contains its consequences. Subtracting it too would double-count.
- **Beta = 1, and then not.** Subtracting one unit of benchmark is an ASSUMPTION: closed-end
  funds are usually 25-40% leveraged, and a levered fund launched at the top of its sleeve
  would print a negative "abnormal" return from leverage alone. :func:`beta_adjusted_table`
  re-runs the event window against each fund's own beta, fitted on its *seasoned* life — an
  ex-post diagnostic (it uses data from after the event window), never a tradable weight.
- **No excess-of-cash adjustment anywhere, on purpose.** Every number here is either a
  difference of two fully invested legs or a self-financing long/short spread, so the cash
  rate cancels identically on both sides. BIL is cached as the desk's cash reference and is
  deliberately *not* used: there is no Sharpe race between a funded and an unfunded leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZONS = (21, 63, 126, 252)          # 1, 3, 6, 12 months
HORIZON_LABEL = {21: "1m", 63: "3m", 126: "6m", 252: "12m"}


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
# --------------------------------------------------------------------------- #
def one_sample_t(x) -> float:
    """Plain one-sample *t* of mean(x) vs 0 (cross-sectional, one obs per fund)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0, for time-series legs."""
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
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def cluster_bootstrap_ci(
    values,
    groups,
    n_boot: int = 5000,
    seed: int = 931,
    alpha: float = 0.05,
) -> dict:
    """Block/cluster bootstrap of the mean, resampling whole ``groups`` with replacement.

    The funds are *not* independent draws: three 2021-vintage credit funds all met the
    2022 rate shock together. Resampling whole **vintage years** (the cluster) instead of
    individual funds keeps that calendar dependence intact, so the interval is honest
    about how few genuinely independent market episodes the sample contains.
    """
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    keep = np.isfinite(v)
    v, g = v[keep], g[keep]
    if v.size < 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_positive": float("nan"), "n": int(v.size), "n_clusters": 0}
    uniq = np.unique(g)
    idx_by_group = [np.flatnonzero(g == u) for u in uniq]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[p] for p in pick])
        boots[b] = v[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "frac_positive": float((boots > 0).mean()), "n": int(v.size),
        "n_clusters": int(len(uniq)), "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# The event study
# --------------------------------------------------------------------------- #
def _tr(series: pd.Series, i0: int, i1: int) -> float:
    """Total return of an adjusted-close series between positional indices i0 and i1."""
    if i1 >= len(series) or i0 >= len(series):
        return float("nan")
    a, b = float(series.iloc[i0]), float(series.iloc[i1])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return float("nan")
    return b / a - 1.0


def fund_event_table(
    fund_tr: pd.Series,
    bench_tr: pd.Series,
    offer: float | None = None,
    raw_fund: pd.Series | None = None,
    horizons=HORIZONS,
    entry: str = "offer",
) -> dict:
    """Abnormal total return of one fund vs its benchmark over each horizon.

    ``fund_tr`` / ``bench_tr`` are daily **total-return** closes; ``raw_fund`` the fund's
    unadjusted closes (only its first value is read). With ``entry="offer"`` the
    subscriber's return is ``(raw_day1 / offer) * (adj_h / adj_day1) - 1`` — the day-one
    gap from the offering price compounded with the total return from the day-one close.
    With ``entry="close"`` the gap factor is dropped.
    """
    f = fund_tr.dropna()
    if len(f) < 30:
        return {}
    t0 = f.index[0]
    b = bench_tr.dropna()
    b = b[b.index >= t0]
    if len(b) < 30:
        return {}
    # Align the benchmark onto the fund's trading calendar.
    b = b.reindex(f.index).ffill()

    gap = 1.0
    if entry == "offer":
        if raw_fund is None or offer is None:
            raise ValueError("entry='offer' needs raw_fund and offer")
        r = raw_fund.dropna()
        r = r[r.index >= t0]
        if len(r) == 0:
            return {}
        gap = float(r.iloc[0]) / float(offer)

    out = {"ipo_date": t0, "vintage": int(t0.year), "n_days": int(len(f)),
           "day1_gap_pct": (gap - 1.0) * 100.0}
    for h in horizons:
        lab = HORIZON_LABEL.get(h, f"{h}d")
        rf = _tr(f, 0, h)
        rb = _tr(b, 0, h)
        if np.isfinite(rf):
            rf = gap * (1.0 + rf) - 1.0
        out[f"fund_{lab}"] = rf
        out[f"bench_{lab}"] = rb
        out[f"abn_{lab}"] = rf - rb if np.isfinite(rf) and np.isfinite(rb) else float("nan")
    return out


def build_table(
    prices: pd.DataFrame,
    ipos,
    raw: pd.DataFrame | None = None,
    mapping: dict | None = None,
    horizons=HORIZONS,
    entry: str = "offer",
) -> pd.DataFrame:
    """Run :func:`fund_event_table` across the whole hardcoded IPO list.

    ``ipos`` is the ``(ticker, offer, sleeve, benchmark)`` list; ``mapping`` optionally
    overrides the fund->benchmark assignment (used by the mapping sweep).
    """
    rows = []
    for tk, offer, sleeve, bench in ipos:
        bm = (mapping or {}).get(tk, bench)
        if tk not in prices.columns or bm not in prices.columns:
            continue
        rf = raw[tk] if (raw is not None and tk in raw.columns) else None
        rec = fund_event_table(prices[tk], prices[bm], offer=offer, raw_fund=rf,
                               horizons=horizons, entry=entry)
        if not rec:
            continue
        rec.update({"ticker": tk, "sleeve": sleeve, "bench": bm, "offer": offer})
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker").sort_values("ipo_date")
    return df


def horizon_summary(table: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Mean / median abnormal return, hit rate and one-sample *t* at each horizon."""
    rows = []
    for h in horizons:
        lab = HORIZON_LABEL.get(h, f"{h}d")
        col = f"abn_{lab}"
        if col not in table.columns:
            continue
        v = table[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        n = len(v)
        neg = int((v < 0).sum())
        lo, hi = wilson_interval(neg, n)
        rows.append({
            "horizon": lab, "n_funds": n,
            "mean_abn_pct": float(v.mean() * 100) if n else float("nan"),
            "median_abn_pct": float(np.median(v) * 100) if n else float("nan"),
            "share_negative": neg / n if n else float("nan"),
            "share_neg_lo": lo, "share_neg_hi": hi,
            "tstat": one_sample_t(v),
        })
    return pd.DataFrame(rows).set_index("horizon")


# --------------------------------------------------------------------------- #
# Time-to-discount distribution
# --------------------------------------------------------------------------- #
def time_to_threshold(
    fund_tr: pd.Series,
    bench_tr: pd.Series,
    threshold_pct: float = 5.0,
    offer: float | None = None,
    raw_fund: pd.Series | None = None,
    max_days: int = 504,
    entry: str = "offer",
) -> float:
    """Trading days until the fund's cumulative abnormal return first falls below −threshold.

    A **PROXY for time-to-discount**: true daily NAV is not on the free tape, so we cannot
    watch the discount itself. What we can watch is the fund losing ground to the assets it
    claims to hold. ``nan`` means the threshold was never crossed inside ``max_days``.
    """
    f = fund_tr.dropna()
    if len(f) < 30:
        return float("nan")
    b = bench_tr.dropna()
    b = b[b.index >= f.index[0]].reindex(f.index).ffill()
    n = min(len(f), max_days)
    gap = 1.0
    if entry == "offer":
        r = raw_fund.dropna()
        r = r[r.index >= f.index[0]]
        if len(r) == 0 or offer is None:
            return float("nan")
        gap = float(r.iloc[0]) / float(offer)
    fv = gap * (f.iloc[:n].to_numpy(dtype=float) / float(f.iloc[0]))
    bv = b.iloc[:n].to_numpy(dtype=float) / float(b.iloc[0])
    abn = fv - bv
    hit = np.flatnonzero(abn <= -threshold_pct / 100.0)
    return float(hit[0]) if hit.size else float("nan")


def time_to_discount_table(prices, ipos, raw, threshold_pct: float = 5.0,
                           max_days: int = 504, entry: str = "offer") -> pd.DataFrame:
    """Time-to-threshold for every fund, plus the sleeve label."""
    rows = []
    for tk, offer, sleeve, bench in ipos:
        if tk not in prices.columns or bench not in prices.columns:
            continue
        d = time_to_threshold(prices[tk], prices[bench], threshold_pct=threshold_pct,
                              offer=offer, raw_fund=raw[tk] if raw is not None else None,
                              max_days=max_days, entry=entry)
        rows.append({"ticker": tk, "sleeve": sleeve, "days_to_threshold": d})
    return pd.DataFrame(rows).set_index("ticker")


def time_to_discount_stats(tbl: pd.DataFrame, max_days: int = 504) -> dict:
    """Median / quartiles of the time-to-threshold, and the share that never cross."""
    v = tbl["days_to_threshold"].to_numpy(dtype=float)
    crossed = v[np.isfinite(v)]
    n = len(v)
    return {
        "n_funds": int(n),
        "n_crossed": int(len(crossed)),
        "share_crossed": float(len(crossed) / n) if n else float("nan"),
        "median_days": float(np.median(crossed)) if len(crossed) else float("nan"),
        "q1_days": float(np.percentile(crossed, 25)) if len(crossed) else float("nan"),
        "q3_days": float(np.percentile(crossed, 75)) if len(crossed) else float("nan"),
        "max_days": int(max_days),
    }


# --------------------------------------------------------------------------- #
# Calendar-time portfolio — the cross-correlation-proof version of the test
# --------------------------------------------------------------------------- #
def calendar_time_spread(
    prices: pd.DataFrame,
    ipos,
    window: int = 252,
    start_day: int = 1,
) -> pd.Series:
    """Daily equal-weight spread return of every fund inside its post-IPO ``window``.

    On each date, average ``r_fund - r_benchmark`` across the funds whose age (in trading
    days since their first close) lies in ``(start_day, window]``; dates with no eligible
    fund are dropped. This is the Fama-style **calendar-time portfolio**: it turns 28
    overlapping, cross-correlated event windows into ONE daily time series whose
    Newey-West *t* needs no independence assumption across funds — the fix for the fact
    that the whole 2021 vintage met the 2022 rate shock together.

    **The lag, spelled out.** The daily return at age *k* is earned by somebody who was
    already positioned at the close of age *k−1*. ``start_day = 1`` therefore keeps ages
    ``2 … window``: the book is established at the **t+1 close** and earns from the next
    day on — byte-for-byte the convention of :func:`short_trade` (``i0 = 1``,
    ``i1 = 1 + horizon``) and of :func:`seasoned_window`. The earlier version of this
    function kept ages ``1 … window−1``, i.e. it credited the book with the IPO-close →
    next-close move it could not have held; that move averages *positive* here, so the
    old convention flattered the null rather than the finding.

    The spread is self-financing (long fund, short benchmark, equal notional), so the
    cash leg cancels exactly and no excess-of-cash adjustment applies on either side.
    """
    legs = []
    for tk, offer, sleeve, bench in ipos:
        if tk not in prices.columns or bench not in prices.columns:
            continue
        f = prices[tk].dropna()
        b = prices[bench].dropna()
        b = b[b.index >= f.index[0]].reindex(f.index).ffill()
        age = np.arange(len(f))
        m = (age > start_day) & (age <= window)
        d = (f.pct_change(fill_method=None) - b.pct_change(fill_method=None))[m]
        legs.append(d.rename(tk))
    if not legs:
        return pd.Series(dtype=float)
    wide = pd.concat(legs, axis=1, sort=True).sort_index()
    return wide.mean(axis=1, skipna=True).dropna().rename("spread")


def calendar_time_stats(spread: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats and the HAC *t* of a daily calendar-time spread series."""
    r = pd.Series(spread).astype(float).dropna()
    n = len(r)
    if n < 30:
        return {"n_days": int(n), "tstat_hac": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return {
        "n_days": int(n),
        "mean_daily_bps": float(mu * 1e4),
        "ann_pct": float(mu * periods_per_year * 100),
        "vol_ann_pct": float(sd * np.sqrt(periods_per_year) * 100),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "tstat_hac": newey_west_t(r.to_numpy(), lags=lags),
        "hac_lags": lags,
    }


def block_bootstrap_mean_ci(
    spread: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 931,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
) -> dict:
    """Circular block bootstrap of the annualised mean of a daily spread series."""
    r = np.asarray(pd.Series(spread).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"ann_pct": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[b] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "ann_pct": float(r.mean() * periods_per_year * 100),
        "ci_low": float(lo * periods_per_year * 100),
        "ci_high": float(hi * periods_per_year * 100),
        "frac_positive": float((boots > 0).mean()),
        "n_obs": int(n), "block": int(block), "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# Market-model (beta-adjusted) CAR — the "is it just leverage?" control
# --------------------------------------------------------------------------- #
def seasoned_beta(
    fund_tr: pd.Series,
    bench_tr: pd.Series,
    start_day: int = 253,
    end_day: int = 757,
    min_obs: int = 60,
) -> float:
    """OLS beta of daily fund returns on its benchmark, fitted on the **seasoned** window.

    Why this exists. The headline abnormal return subtracts *one* unit of benchmark —
    an implicit **beta = 1 ASSUMPTION**. Closed-end funds are usually 25-40% leveraged,
    so a levered fund launched into the top of its sleeve would print a negative
    "abnormal" return from leverage alone, with no IPO mechanism involved. This fits each
    fund's own beta to remove that.

    Why it is a diagnostic and not a rule. The estimation window (ages
    ``start_day … end_day``, i.e. months 12-36) is **disjoint from the event window it
    adjusts, but it lies AFTER it**. Nobody standing at the IPO knows this number. It is
    an ex-post decomposition — labelled as such everywhere it is reported — never a
    tradable weight, and it is never used in :func:`short_trade`.

    Returns ``nan`` when the fund has fewer than ``min_obs`` seasoned observations.
    """
    f = fund_tr.dropna()
    b = bench_tr.dropna()
    b = b[b.index >= f.index[0]].reindex(f.index).ffill()
    rf = f.pct_change(fill_method=None).to_numpy(dtype=float)
    rb = b.pct_change(fill_method=None).to_numpy(dtype=float)
    hi = min(end_day, len(rf))
    if hi - start_day < min_obs:
        return float("nan")
    x, y = rb[start_day:hi], rf[start_day:hi]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < min_obs or x.var(ddof=1) <= 0:
        return float("nan")
    return float(np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1))


def market_model_car(
    fund_tr: pd.Series,
    bench_tr: pd.Series,
    horizon: int = 252,
    lag_days: int = 1,
    beta: float = 1.0,
) -> float:
    """Cumulative abnormal return: sum of daily ``r_fund − beta * r_bench`` over the window.

    Entered at the **t+1** close like everything else on this desk, so the daily returns
    used are ages ``lag_days + 1 … lag_days + horizon``. Summing daily differences (rather
    than differencing two compounded totals) is what makes the ``beta`` weighting
    well-defined, so the ``beta = 1`` column of this table is the like-for-like control
    for the ``beta``-fitted one — it is *not* the headline number, which compounds.
    """
    f = fund_tr.dropna()
    b = bench_tr.dropna()
    b = b[b.index >= f.index[0]].reindex(f.index).ffill()
    rf = f.pct_change(fill_method=None).to_numpy(dtype=float)
    rb = b.pct_change(fill_method=None).to_numpy(dtype=float)
    lo, hi = lag_days + 1, lag_days + 1 + horizon
    if hi > len(rf) or not np.isfinite(beta):
        return float("nan")
    d = rf[lo:hi] - beta * rb[lo:hi]
    d = d[np.isfinite(d)]
    if len(d) < horizon // 2:
        return float("nan")
    return float(d.sum())


def beta_adjusted_table(
    prices: pd.DataFrame,
    ipos,
    horizon: int = 252,
    lag_days: int = 1,
    fit_start: int = 253,
    fit_end: int = 757,
) -> pd.DataFrame:
    """Per-fund seasoned beta and the event-window CAR at beta = 1 and at the fitted beta."""
    rows = []
    for tk, offer, sleeve, bench in ipos:
        if tk not in prices.columns or bench not in prices.columns:
            continue
        f, b = prices[tk], prices[bench]
        bt = seasoned_beta(f, b, start_day=fit_start, end_day=fit_end)
        rows.append({
            "ticker": tk, "sleeve": sleeve, "beta": bt,
            "car_beta1_pct": market_model_car(f, b, horizon, lag_days, 1.0) * 100,
            "car_betafit_pct": market_model_car(f, b, horizon, lag_days, bt) * 100,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ticker")


def beta_adjusted_summary(tbl: pd.DataFrame) -> dict:
    """Mean / share-negative / one-sample *t* of both CAR columns, plus the beta spread."""
    out = {"n_funds": int(len(tbl)),
           "mean_beta": float(np.nanmean(tbl["beta"].to_numpy(dtype=float))),
           "median_beta": float(np.nanmedian(tbl["beta"].to_numpy(dtype=float)))}
    for col, tag in (("car_beta1_pct", "beta1"), ("car_betafit_pct", "betafit")):
        v = tbl[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        out[f"{tag}_mean_pct"] = float(v.mean()) if v.size else float("nan")
        out[f"{tag}_median_pct"] = float(np.median(v)) if v.size else float("nan")
        out[f"{tag}_share_neg"] = float((v < 0).mean()) if v.size else float("nan")
        out[f"{tag}_t"] = one_sample_t(v)
    return out


# --------------------------------------------------------------------------- #
# The seasoned mirror + the pseudo-IPO placebo
# --------------------------------------------------------------------------- #
def seasoned_window(
    fund_tr: pd.Series,
    bench_tr: pd.Series,
    start_day: int = 252,
    end_day: int = 756,
) -> float:
    """Abnormal total return between two post-IPO trading-day offsets (the seasoned leg).

    The tradable read of this study is a *mirror*: if the hole is an IPO-window artefact,
    the same fund measured from its first birthday onward should stop bleeding relative to
    its benchmark. This measures exactly that window (default months 12 → 36).
    """
    f = fund_tr.dropna()
    b = bench_tr.dropna()
    b = b[b.index >= f.index[0]].reindex(f.index).ffill()
    if len(f) <= start_day + 21:
        return float("nan")
    e = min(end_day, len(f) - 1)
    return _tr(f, start_day, e) - _tr(b, start_day, e)


def seasoned_table(prices, ipos, start_day: int = 252, end_day: int = 756) -> pd.DataFrame:
    rows = []
    for tk, offer, sleeve, bench in ipos:
        if tk not in prices.columns or bench not in prices.columns:
            continue
        rows.append({
            "ticker": tk, "sleeve": sleeve,
            "abn_seasoned": seasoned_window(prices[tk], prices[bench],
                                            start_day=start_day, end_day=end_day),
        })
    return pd.DataFrame(rows).set_index("ticker")


def pseudo_ipo_placebo(
    prices: pd.DataFrame,
    ipos,
    horizon: int = 252,
    min_offset: int = 252,
    n_draws: int = 500,
    seed: int = 931,
) -> dict:
    """Placebo: re-run the 12-month abnormal return from *random seasoned* start dates.

    Same funds, same benchmarks, same estimator — only the event date is fake (drawn at
    least ``min_offset`` trading days after the real IPO). If these CEFs are simply bad
    assets forever, the placebo mean is as negative as the real one and the "IPO hole"
    is a mislabelled fund-quality effect. If the placebo is centred near zero, the damage
    is genuinely concentrated in the IPO window.
    """
    rng = np.random.default_rng(seed)
    draws = np.full(n_draws, np.nan)
    per_fund = {}
    for tk, offer, sleeve, bench in ipos:
        if tk not in prices.columns or bench not in prices.columns:
            continue
        f = prices[tk].dropna()
        b = prices[bench].dropna()
        b = b[b.index >= f.index[0]].reindex(f.index).ffill()
        lo, hi = min_offset, len(f) - horizon - 1
        if hi <= lo:
            continue
        per_fund[tk] = (f, b, lo, hi)
    if not per_fund:
        return {"mean_pct": float("nan"), "sd_pct": float("nan"), "n_draws": 0}
    for d in range(n_draws):
        vals = []
        for tk, (f, b, lo, hi) in per_fund.items():
            s = int(rng.integers(lo, hi))
            v = _tr(f, s, s + horizon) - _tr(b, s, s + horizon)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            draws[d] = float(np.mean(vals))
    ok = draws[np.isfinite(draws)]
    return {
        "mean_pct": float(ok.mean() * 100), "sd_pct": float(ok.std(ddof=1) * 100),
        "p05_pct": float(np.percentile(ok, 5) * 100),
        "p95_pct": float(np.percentile(ok, 95) * 100),
        "n_draws": int(ok.size), "n_funds": int(len(per_fund)),
    }


# --------------------------------------------------------------------------- #
# Era cut, mapping sweep, entry sweep
# --------------------------------------------------------------------------- #
def era_cut(table: pd.DataFrame, split_year: int = 2019, horizon_label: str = "12m") -> dict:
    """Split the fund list by IPO **vintage** and re-run the one-sample test on each half."""
    col = f"abn_{horizon_label}"
    out = {}
    for tag, mask in [("early", table["vintage"] < split_year),
                      ("late", table["vintage"] >= split_year)]:
        v = table.loc[mask, col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        out[tag] = {
            "n_funds": int(len(v)),
            "mean_abn_pct": float(v.mean() * 100) if len(v) else float("nan"),
            "share_negative": float((v < 0).mean()) if len(v) else float("nan"),
            "tstat": one_sample_t(v),
            "vintages": sorted(set(table.loc[mask, "vintage"].tolist())),
        }
    return out


def mapping_sweep(prices, ipos, raw, mappings: dict, horizon_label: str = "12m") -> list[dict]:
    """Re-run the headline under alternative fund->benchmark mappings (an ASSUMPTION sweep)."""
    rows = []
    for name, mp in mappings.items():
        tbl = build_table(prices, ipos, raw=raw, mapping=mp, entry="offer")
        v = tbl[f"abn_{horizon_label}"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append({
            "mapping": name, "n_funds": int(len(v)),
            "mean_abn_pct": float(v.mean() * 100) if len(v) else float("nan"),
            "share_negative": float((v < 0).mean()) if len(v) else float("nan"),
            "tstat": one_sample_t(v),
        })
    return rows


def entry_sweep(prices, ipos, raw, horizon_label: str = "12m") -> list[dict]:
    """Subscriber (pays the offering price) vs day-one-close buyer."""
    rows = []
    for entry in ("offer", "close"):
        tbl = build_table(prices, ipos, raw=raw, entry=entry)
        v = tbl[f"abn_{horizon_label}"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append({
            "entry": entry, "n_funds": int(len(v)),
            "mean_abn_pct": float(v.mean() * 100) if len(v) else float("nan"),
            "tstat": one_sample_t(v),
        })
    return rows


# --------------------------------------------------------------------------- #
# The tradable mirror — short the new issue, long the benchmark
# --------------------------------------------------------------------------- #
def short_trade(
    prices: pd.DataFrame,
    ipos,
    horizon: int = 252,
    cost_bps: float = 20.0,
    borrow_bps_year: float = 500.0,
    lag_days: int = 1,
) -> dict:
    """The only *position* this study can express: short the new CEF, long its benchmark.

    Equal notional, held ``horizon`` trading days. The IPO is known at the day-one close,
    so the trade is opened at the **next** close (``lag_days = 1``) — the study's single
    execution lag. Costs: ``cost_bps`` one-way x NAV per leg, charged on entry and exit
    (four charges). Borrow: ``borrow_bps_year`` x NAV x holding period on the short leg —
    a fresh CEF with a tiny float is the archetype of a hard-to-borrow name, so this rate
    is swept rather than assumed away.
    """
    vals, tickers = [], []
    for tk, offer, sleeve, bench in ipos:
        if tk not in prices.columns or bench not in prices.columns:
            continue
        f = prices[tk].dropna()
        b = prices[bench].dropna()
        b = b[b.index >= f.index[0]].reindex(f.index).ffill()
        i0 = lag_days
        i1 = i0 + horizon
        if i1 >= len(f):
            continue
        rf = _tr(f, i0, i1)
        rb = _tr(b, i0, i1)
        if not (np.isfinite(rf) and np.isfinite(rb)):
            continue
        gross = -rf + rb
        cost = 4.0 * cost_bps * 1e-4
        borrow = borrow_bps_year * 1e-4 * (horizon / TRADING_DAYS)
        vals.append(gross - cost - borrow)
        tickers.append(tk)
    v = np.asarray(vals, dtype=float)
    return {
        "n_funds": int(v.size),
        "mean_net_pct": float(v.mean() * 100) if v.size else float("nan"),
        "median_net_pct": float(np.median(v) * 100) if v.size else float("nan"),
        "share_positive": float((v > 0).mean()) if v.size else float("nan"),
        "tstat": one_sample_t(v),
        "cost_bps": cost_bps, "borrow_bps_year": borrow_bps_year,
        "per_fund": dict(zip(tickers, v.tolist())),
    }


def cost_borrow_sweep(prices, ipos, grid=((20.0, 0.0), (20.0, 300.0), (20.0, 500.0),
                                          (20.0, 1000.0), (50.0, 1500.0))) -> list[dict]:
    """Sweep the mirror trade over (one-way cost, annual borrow) pairs."""
    out = []
    for c, bo in grid:
        r = short_trade(prices, ipos, cost_bps=c, borrow_bps_year=bo)
        r.pop("per_fund", None)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Synthetic control (machinery proof — never supports the real-tape stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict, horizon: int = 252) -> dict:
    """Run the abnormal-return estimator over a synthetic panel of staggered IPOs.

    On a panel with a planted first-year slide the mean abnormal return must come back
    near the planted magnitude with a decisively negative *t*; on the null panel it must
    be centred at zero. Proves the estimator is unbiased — it never supports a real stamp.
    """
    vals = []
    for key, px in panel.items():
        f, b = px["fund"], px["bench"]
        if len(f) <= horizon:
            continue
        vals.append(_tr(f, 0, horizon) - _tr(b, 0, horizon))
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n_funds": int(v.size),
        "mean_abn_pct": float(v.mean() * 100) if v.size else float("nan"),
        "share_negative": float((v < 0).mean()) if v.size else float("nan"),
        "tstat": one_sample_t(v),
    }
