"""The ex-dividend drop, and whether it can be measured — Study 984.

The textbook claim is an accounting identity waiting to happen: a share that is about to pay a
dollar must be worth a dollar more than the same share after it pays, so on the ex-date the
price should fall by the dividend. Elton and Gruber (1970) measured the fall at about 78% of the
dividend and read the shortfall as a tax effect. Fifty years of papers followed.

This module measures it on modern data, and treats the *measurement* as the harder half of the
problem for three reasons that compound:

1. **The denominator is small.** A quarterly dividend is roughly 0.5-0.8% of the price. A single
   day's move is roughly 1.2%. The per-event ratio ``(P_cum - P_ex) / D`` therefore divides
   something noisy by something small — the classic recipe for a statistic with fat tails and,
   in the limit of a small enough dividend, no finite variance at all. ``drop_ratios`` returns
   the per-event ratios; ``estimator_table`` shows how far four defensible summaries of them can
   sit from each other on the same data.

2. **The market moves on the ex-day too.** An uncorrected ex-day return is the drop plus that
   day's market move times beta. Boyd and Jagannathan (1994) among others insist on removing it;
   ``drop_ratios`` does so with a strictly backward-looking beta and reports both versions.

3. **Averaging ratios is not the same as the ratio of averages.** Both are used in the
   literature. They answer different questions and, on this data, give different numbers. The
   study reports both plus the regression slope, which is the estimator with the cleanest
   interpretation: *how much of a dollar of dividend shows up in the price*, weighted the way
   a portfolio would experience it.

The tradability half is the dividend-capture trade: buy the day before the ex-date, sell on the
ex-date, collect the dividend and keep the difference if the price falls by less than the
dividend. ``capture_trade`` prices it with costs and a tax knob.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
ELTON_GRUBER_1970 = 0.778     # the number the literature has argued about ever since


# --------------------------------------------------------------------------- #
# Building the event set
# --------------------------------------------------------------------------- #
def ex_dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    """Every date on which a cash dividend went ex."""
    d = bars["dividend"]
    return pd.DatetimeIndex(d[d > 0].index)


def rolling_beta(stock_rets: pd.Series, market_rets: pd.Series, window: int = 252) -> pd.Series:
    """Backward-looking beta: the value on day *t* uses returns through *t-1* only."""
    cov = stock_rets.rolling(window).cov(market_rets).shift(1)
    var = market_rets.rolling(window).var().shift(1)
    return (cov / var).rename("beta")


def drop_ratios(bars: pd.DataFrame, market: pd.DataFrame, beta_window: int = 252,
                min_yield: float = 1e-4) -> pd.DataFrame:
    """One row per ex-date: the raw drop, the market-adjusted drop, and both as ratios.

    ``raw_ratio``      = (P_cum - P_ex) / D
    ``adjusted_ratio`` = (P_cum * (1 + beta * r_market) - P_ex) / D

    The adjusted version asks what the price *would* have closed at had the stock simply
    tracked the market that day, and measures the shortfall from there. It is the version the
    modern literature uses, and it is materially less noisy.
    """
    px = bars["close"].dropna()
    div = bars["dividend"].reindex(px.index).fillna(0.0)
    mkt = market["close"].reindex(px.index).ffill()
    r_stock = px.pct_change()
    r_mkt = mkt.pct_change()
    beta = rolling_beta(r_stock, r_mkt, beta_window)
    rows = []
    pos = {d: i for i, d in enumerate(px.index)}
    for d in ex_dates(bars):
        if d not in pos or pos[d] == 0:
            continue
        i = pos[d]
        p_ex = float(px.iloc[i])
        p_cum = float(px.iloc[i - 1])
        D = float(div.loc[d])
        if D <= 0 or p_cum <= 0 or D / p_cum < min_yield:
            continue
        b = beta.iloc[i]
        rm = r_mkt.iloc[i]
        b = 1.0 if not np.isfinite(b) else float(b)
        rm = 0.0 if not np.isfinite(rm) else float(rm)
        counterfactual = p_cum * (1.0 + b * rm)
        rows.append({
            "ex_date": d, "price_cum": p_cum, "price_ex": p_ex, "dividend": D,
            "yield": D / p_cum, "beta": b, "market_return": rm,
            "raw_drop": p_cum - p_ex, "adjusted_drop": counterfactual - p_ex,
            "raw_ratio": (p_cum - p_ex) / D, "adjusted_ratio": (counterfactual - p_ex) / D,
        })
    if not rows:
        return pd.DataFrame(columns=["ex_date", "price_cum", "price_ex", "dividend", "yield",
                                     "beta", "market_return", "raw_drop", "adjusted_drop",
                                     "raw_ratio", "adjusted_ratio"]).set_index("ex_date")
    return pd.DataFrame(rows).set_index("ex_date").sort_index()


def build_events(bars_by_ticker: dict, market_key: str, beta_window: int = 252) -> pd.DataFrame:
    """The pooled event set across every payer."""
    market = bars_by_ticker[market_key]
    frames = []
    for tk, b in bars_by_ticker.items():
        if tk == market_key:
            continue
        ev = drop_ratios(b, market, beta_window)
        if len(ev):
            ev = ev.assign(ticker=tk)
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


# --------------------------------------------------------------------------- #
# Four ways to summarise the same events
# --------------------------------------------------------------------------- #
def mean_of_ratios(ev: pd.DataFrame, col: str = "adjusted_ratio") -> float:
    """The naive average. Dominated by events with tiny dividends and fat tails."""
    return float(ev[col].mean())


def median_of_ratios(ev: pd.DataFrame, col: str = "adjusted_ratio") -> float:
    """Robust to the tails, but estimates the median ratio, not the average dollar recovered."""
    return float(ev[col].median())


def ratio_of_sums(ev: pd.DataFrame, drop_col: str = "adjusted_drop") -> float:
    """Total dollars dropped over total dollars paid — the portfolio-weighted answer."""
    tot_d = float(ev["dividend"].sum())
    return float(ev[drop_col].sum() / tot_d) if tot_d > 0 else np.nan


def regression_slope(ev: pd.DataFrame, drop_col: str = "adjusted_drop") -> dict:
    """Regress the dollar drop on the dollar dividend, no intercept forced.

    The slope answers "per dollar of dividend, how many cents came out of the price?" and its
    standard error is the only one of these four estimators that is honestly computable.
    """
    y = ev[drop_col].to_numpy(float)
    x = ev["dividend"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 30:
        return {"n": int(n), "slope": np.nan, "t": np.nan, "intercept": np.nan}
    A = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    XtX_inv = np.linalg.pinv(A.T @ A)
    meat = A.T @ np.diag(resid ** 2) @ A * n / max(n - 2, 1)
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(max(V[1, 1], 0.0))
    return {"n": int(n), "intercept": float(coef[0]), "slope": float(coef[1]),
            "se": float(se), "t_vs_zero": float(coef[1] / se) if se > 0 else np.nan,
            "t_vs_one": float((coef[1] - 1.0) / se) if se > 0 else np.nan}


def estimator_table(ev: pd.DataFrame, col: str = "adjusted_ratio",
                    drop_col: str = "adjusted_drop") -> pd.DataFrame:
    """All four estimators side by side. They will not agree, and that is the finding."""
    reg = regression_slope(ev, drop_col)
    rows = [
        {"estimator": "mean of per-event ratios", "value": mean_of_ratios(ev, col),
         "note": "fat-tailed; dominated by small dividends"},
        {"estimator": "median of per-event ratios", "value": median_of_ratios(ev, col),
         "note": "robust, but answers a different question"},
        {"estimator": "total drop / total dividend", "value": ratio_of_sums(ev, drop_col),
         "note": "what a portfolio actually experienced"},
        {"estimator": "regression slope", "value": reg["slope"],
         "note": f"t vs 1.0 = {reg.get('t_vs_one', np.nan):+.2f}"},
    ]
    return pd.DataFrame(rows).set_index("estimator")


def ratio_dispersion(ev: pd.DataFrame, col: str = "adjusted_ratio") -> dict:
    """Evidence that the per-event ratio is a badly behaved statistic.

    If the standard deviation of the per-event ratios barely shrinks as events are added, and
    the trimmed mean sits far from the raw mean, the distribution has tails heavy enough that
    "the average drop ratio" is not a quantity a mean can estimate.
    """
    r = ev[col].dropna().to_numpy()
    n = len(r)
    if n < 30:
        return {"n": int(n)}
    lo, hi = np.percentile(r, [5, 95])
    trimmed = r[(r >= lo) & (r <= hi)]
    halves = [np.std(r[:n // 2], ddof=1), np.std(r, ddof=1)]
    return {"n": int(n), "sd": float(np.std(r, ddof=1)),
            "iqr": float(np.subtract(*np.percentile(r, [75, 25]))),
            "mean": float(r.mean()), "trimmed_mean": float(trimmed.mean()),
            "median": float(np.median(r)),
            "min": float(r.min()), "max": float(r.max()),
            "sd_ratio_half_to_full": float(halves[0] / halves[1]) if halves[1] > 0 else np.nan,
            "share_outside_0_2": float(np.mean((r < 0) | (r > 2)))}


def bootstrap_ci(ev: pd.DataFrame, estimator=None, n_boot: int = 2000,
                 seed: int = 984) -> dict:
    """Percentile bootstrap over events, for whichever estimator is passed."""
    estimator = estimator or ratio_of_sums
    rng = np.random.default_rng(seed)
    n = len(ev)
    if n < 30:
        return {"n": int(n)}
    vals = np.empty(n_boot)
    arr = ev.reset_index(drop=True)
    for b in range(n_boot):
        vals[b] = estimator(arr.iloc[rng.integers(0, n, n)])
    return {"n": int(n), "point": float(estimator(arr)),
            "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "boot_sd": float(np.std(vals, ddof=1))}


def by_group(ev: pd.DataFrame, key: str, drop_col: str = "adjusted_drop",
             min_n: int = 30) -> pd.DataFrame:
    """The portfolio-weighted ratio within each group of events."""
    rows = []
    for g, sl in ev.groupby(key):
        if len(sl) < min_n:
            continue
        reg = regression_slope(sl, drop_col)
        rows.append({key: g, "n": len(sl), "ratio": ratio_of_sums(sl, drop_col),
                     "slope": reg["slope"], "t_vs_one": reg.get("t_vs_one", np.nan),
                     "median_yield": float(sl["yield"].median())})
    return pd.DataFrame(rows).set_index(key) if rows else pd.DataFrame()


def yield_buckets(ev: pd.DataFrame, n_buckets: int = 4,
                  drop_col: str = "adjusted_drop") -> pd.DataFrame:
    """By dividend size. If the ratio is a measurement artefact it will vary here systematically.

    A small dividend is a small denominator, so the *noise* in the per-event ratio falls as the
    yield rises. A tax-clientele story predicts a genuine relationship with yield too, so this
    table cannot separate the two by itself — but a ratio that is wild for tiny dividends and
    orderly for large ones is telling you about arithmetic, not about tax.
    """
    e = ev.copy()
    e["yield_bucket"] = pd.qcut(e["yield"], n_buckets,
                                labels=[f"Q{i + 1}" for i in range(n_buckets)])
    return by_group(e, "yield_bucket", drop_col, min_n=20)


# --------------------------------------------------------------------------- #
# The trade
# --------------------------------------------------------------------------- #
def capture_trade(ev: pd.DataFrame, cost_bps: float = 2.0, div_tax: float = 0.0) -> dict:
    """Buy at the cum close, sell at the ex close, collect the dividend.

    Profit per dollar invested = (P_ex - P_cum + D*(1 - tax)) / P_cum - 2 * cost.
    With no tax and no costs this is positive exactly when the drop ratio is below one, so the
    trade *is* the study's finding restated as money. The costs matter enormously: the whole
    gross edge is a fraction of a typical dividend, which is itself under 1% of the price.
    """
    p_cum = ev["price_cum"].to_numpy(float)
    p_ex = ev["price_ex"].to_numpy(float)
    D = ev["dividend"].to_numpy(float)
    gross = (p_ex - p_cum + D * (1.0 - div_tax)) / p_cum
    net = gross - 2.0 * cost_bps / 1e4
    n = len(net)
    se = net.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"n": int(n), "mean_gross_bps": float(gross.mean() * 1e4),
            "mean_net_bps": float(net.mean() * 1e4),
            "median_net_bps": float(np.median(net) * 1e4),
            "sd_bps": float(net.std(ddof=1) * 1e4),
            "t": float(net.mean() / se) if se and se > 0 else np.nan,
            "hit_rate": float((net > 0).mean()),
            "breakeven_cost_bps": float(gross.mean() * 1e4 / 2.0),
            "cost_bps": cost_bps, "div_tax": div_tax, "net": net}


def verdict(h: dict) -> dict:
    """Stamps by a pre-registered rule.

    - **Signal**: **Real** if the regression slope is statistically distinguishable from 1.0
      (|*t*| >= 2) — i.e. the price provably does *not* fall by the full dividend; **Weak** if
      the point estimate is below 0.95 but not significantly so; **None** if the evidence is
      consistent with a full drop.
    - **Tradability**: **Investable** if the dividend-capture trade nets a positive mean with
      |*t*| >= 2 at realistic costs; **Fragile** if positive without significance; **Mirage**
      if negative.
    """
    distinguishable = abs(h["t_vs_one"]) >= 2.0
    below = h["slope"] < 0.95
    signal = "Real" if (distinguishable and below) else ("Weak" if below else "None")
    trad = ("Investable" if h["net_bps"] > 0 and abs(h["t_trade"]) >= 2.0
            else ("Fragile" if h["net_bps"] > 0 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Across **{h['n_events']} ex-dates** on {h['n_tickers']} mega-cap payers "
            f"({h['window'][0]} → {h['window'][1]}), a dollar of dividend took "
            f"**{h['slope']:.3f}** of a dollar out of the price once that day's market move is "
            f"removed (HC1 *t* against 1.0 = **{h['t_vs_one']:+.2f}**; against zero "
            f"{h['t_vs_zero']:+.2f}). The four defensible summaries of the same events disagree "
            f"by more than the effect anyone is arguing about: the mean per-event ratio is "
            f"{h['mean_ratio']:+.2f}, the median {h['median_ratio']:.2f}, total-drop-over-total-"
            f"dividend {h['ratio_of_sums']:.3f}, the regression slope {h['slope']:.3f}. That "
            f"spread is not sloppiness — the per-event ratio divides a {h['typical_move']:.1%} "
            f"daily move by a {h['typical_yield']:.2%} dividend, so "
            f"{h['share_wild']:.0%} of individual events land outside the range [0, 2] "
            f"entirely. Elton and Gruber's {h['elton_gruber']:.3f} sits "
            f"{'inside' if h['eg_inside_ci'] else 'outside'} the bootstrap interval "
            f"[{h['ci_lo']:.3f}, {h['ci_hi']:.3f}]."),
        "trad": trad,
        "trad_why": (
            f"Buying at the cum close and selling at the ex close, keeping the dividend, earned "
            f"**{h['gross_bps']:+.1f} bps** gross per event and **{h['net_bps']:+.1f} bps** "
            f"after {h['cost_bps']:.0f} bps a side (*t* = {h['t_trade']:+.2f}, "
            f"{h['hit_rate']:.0%} of events profitable, dispersion {h['sd_bps']:.0f} bps). The "
            f"trade breaks even at {h['breakeven_bps']:.1f} bps of round-trip cost, and at the "
            f"{h['tax_rate']:.0%} qualified-dividend rate a taxable holder nets "
            f"{h['net_after_tax_bps']:+.1f} bps. The edge, if it is one, is smaller than the "
            f"noise on a single event by a factor of {h['sd_bps'] / max(abs(h['net_bps']), 1):.0f}."),
        "one_sentence": (
            f"A dollar of dividend removes {h['slope']:.2f} of a dollar from the price — but the "
            f"four standard ways of computing that number span "
            f"{h['estimator_spread']:.2f}, which is larger than the effect the literature has "
            f"spent fifty years arguing about."),
    }
