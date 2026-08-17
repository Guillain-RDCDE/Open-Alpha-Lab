"""Strategy + inference for Study 947 — The Buffer Ladder.

The question: **BUFR** is a single ticker that holds the whole Innovator Power Buffer
ladder. A private investor can build his own ladder by buying the vintages directly and
equal-weighting them. Does the wrapper earn the extra fee layer it charges — is there a
*laddering* premium beyond (a) averaging away entry-point luck and (b) simply carrying more
equity beta?

Five arms, all measured **excess-of-cash** (minus BIL's total return) so the Sharpe race
is apples-to-apples:

1. **ladder** — BUFR, the laddered wrapper (buy-and-hold).
2. **single vintages** — PJAN / PAPR / PJUL / POCT held individually (buy-and-hold): what
   you get if you pick one entry point and live with it.
3. **DIY basket** — an equal-weight basket of the four vintages, rebalanced quarterly with
   an explicit one-way cost. The cheap home-made ladder.
4. **beta-matched DIY ladder** — the DIY basket topped up with the market until its
   SPY-beta equals the wrapper's. This is the arm that matters: the wrapper carries *more*
   equity beta than a four-vintage basket, so an un-matched comparison flatters it.
5. **beta-matched SPY/BIL mix** — the dumb mix each arm supposedly justifies its fee
   against (the Study 624 benchmark, reused here for continuity).

**Exactly one execution lag, everywhere.** Every weight — the basket's rebalance target and
every beta used to build a matched arm — is estimated on data through the close of day
``t`` and applied to day ``t+1`` returns. Betas are expanding-window (out-of-sample) by
default; the full-sample beta is reported alongside and clearly labelled *in-sample*.

Costs are one-way x NAV, charged on realised turnover. No arm shorts, so no borrow applies
to the headline; the beta-matched arms can call for a short market leg when the target beta
falls below the basket's, and ``borrow_bps`` charges that leg honestly.

Returns are **simple** (arithmetic) so basket weights aggregate exactly and wealth paths
compound correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return plumbing
# --------------------------------------------------------------------------- #
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple total returns from a frame of adjusted close levels."""
    return prices.sort_index().pct_change().dropna(how="all")


def excess_returns(rets: pd.DataFrame, cash_col: str) -> pd.DataFrame:
    """Subtract the cash leg's *actual* total return from every column.

    The cash leg is BIL's realised total return, so the excess-of-cash race credits the
    real path of short rates (~0% in 2020-2021, ~5% in 2023-2026). That matters a great
    deal here: a buffer fund parks most of its notional in the option collateral, so a
    naive nominal comparison would hand the buffers the 2023-2026 rate rise for free.
    """
    return rets.sub(rets[cash_col], axis=0).drop(columns=[cash_col])


# --------------------------------------------------------------------------- #
# The DIY basket (equal weight, periodic rebalance, explicit cost)
# --------------------------------------------------------------------------- #
def equal_weight_basket(
    rets: pd.DataFrame,
    cols,
    rebalance: str = "Q",
    cost_bps: float = 5.0,
) -> pd.Series:
    """Equal-weight basket of ``cols`` with drifting weights and a periodic reset.

    Weights drift with realised returns between rebalances; on the first trading day of
    each ``rebalance`` period they are reset to 1/N and the realised one-way turnover is
    charged at ``cost_bps`` x NAV. The reset decision uses only the weights implied by
    returns through day ``t-1``, so no future information enters day ``t``'s return.

    ``rebalance`` accepts ``"Q"`` (quarterly), ``"M"`` (monthly), ``"A"`` (annual) or
    ``"N"`` (never — pure buy-and-hold drift).
    """
    sub = rets[list(cols)].dropna()
    if sub.empty:
        return pd.Series(dtype=float, name="basket")
    n = sub.shape[1]
    idx = sub.index

    if rebalance == "N":
        marks = np.zeros(len(idx), dtype=bool)
    else:
        key = {"Q": idx.to_period("Q"), "M": idx.to_period("M"), "A": idx.to_period("Y")}[rebalance]
        marks = np.empty(len(idx), dtype=bool)
        marks[0] = False
        marks[1:] = key[1:] != key[:-1]

    w = np.full(n, 1.0 / n)
    cost = cost_bps * 1e-4
    vals = sub.to_numpy(dtype=float)
    out = np.empty(len(idx))
    for i in range(len(idx)):
        drag = 0.0
        if marks[i]:
            target = np.full(n, 1.0 / n)
            drag = cost * np.abs(target - w).sum()
            w = target
        r = float(w @ vals[i]) - drag
        out[i] = r
        grown = w * (1.0 + vals[i])
        tot = grown.sum()
        w = grown / tot if tot > 0 else np.full(n, 1.0 / n)
    return pd.Series(out, index=idx, name="basket")


# --------------------------------------------------------------------------- #
# Beta machinery — expanding-window (OOS) and full-sample (in-sample, labelled)
# --------------------------------------------------------------------------- #
def full_sample_beta(y: pd.Series, x: pd.Series) -> float:
    """OLS slope of ``y`` on ``x`` over the whole sample. IN-SAMPLE — label it as such."""
    a, b = y.align(x, join="inner")
    a, b = a.dropna(), b.dropna()
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    if len(a) < 3 or b.var() == 0:
        return float("nan")
    return float(np.cov(a, b, ddof=1)[0, 1] / b.var(ddof=1))


def expanding_beta(y: pd.Series, x: pd.Series, min_obs: int = 252) -> pd.Series:
    """Expanding-window OLS beta of ``y`` on ``x``, **lagged one day**.

    The value at day ``t`` is estimated from returns up to and including day ``t-1``, so it
    is a weight you could actually have set before day ``t`` traded. NaN until ``min_obs``
    observations exist. This is the study's single execution lag.
    """
    a, b = y.align(x, join="inner")
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    ya, xa = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    n = len(ya)
    cx = np.cumsum(xa)
    cy = np.cumsum(ya)
    cxx = np.cumsum(xa * xa)
    cxy = np.cumsum(xa * ya)
    k = np.arange(1, n + 1, dtype=float)
    var = cxx - cx * cx / k
    cov = cxy - cx * cy / k
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(var > 0, cov / var, np.nan)
    beta[: max(min_obs - 1, 0)] = np.nan
    return pd.Series(beta, index=a.index, name="beta").shift(1)


def beta_matched_mix(
    market_excess: pd.Series,
    beta: pd.Series | float,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
) -> pd.Series:
    """The dumb ``beta * SPY + (1 - beta) * BIL`` mix, in **excess-of-cash** space.

    In excess space the cash leg is identically zero, so the mix is simply
    ``beta_t * r_market_excess_t`` with ``beta_t`` already lagged one day by
    :func:`expanding_beta`. Turnover in the market leg is charged at ``cost_bps`` one-way x
    NAV; any day the required weight is negative pays ``borrow_bps``/yr on the short
    notional (it never is on the real tape, but the machinery is honest about it).
    """
    b = beta if isinstance(beta, pd.Series) else pd.Series(
        float(beta), index=market_excess.index, name="beta")
    b = b.reindex(market_excess.index)
    r = b * market_excess
    if cost_bps:
        r = r - b.diff().abs().fillna(0.0) * cost_bps * 1e-4
    if borrow_bps:
        r = r - b.clip(upper=0.0).abs() * (borrow_bps * 1e-4) / TRADING_DAYS
    return r.rename("beta_mix")


def beta_matched_ladder(
    basket_excess: pd.Series,
    market_excess: pd.Series,
    beta_target: pd.Series | float,
    beta_basket: pd.Series | float,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
) -> pd.Series:
    """The DIY basket topped up with the market until it carries ``beta_target``.

    ``basket + (beta_target - beta_basket) * market``, all in excess space, with both betas
    already lagged one day. This is the arm that makes the wrapper-vs-DIY race fair: BUFR
    carries materially more equity beta than a four-vintage basket, and beta is not a
    laddering premium — it is something anyone can buy for the price of an S&P 500 ETF.
    """
    bt = beta_target if isinstance(beta_target, pd.Series) else pd.Series(
        float(beta_target), index=basket_excess.index)
    bb = beta_basket if isinstance(beta_basket, pd.Series) else pd.Series(
        float(beta_basket), index=basket_excess.index)
    gap = (bt - bb).reindex(basket_excess.index)
    r = basket_excess + gap * market_excess.reindex(basket_excess.index)
    if cost_bps:
        r = r - gap.diff().abs().fillna(0.0) * cost_bps * 1e-4
    if borrow_bps:
        r = r - gap.clip(upper=0.0).abs() * (borrow_bps * 1e-4) / TRADING_DAYS
    return r.rename("diy_beta_matched")


# --------------------------------------------------------------------------- #
# Inference primitives (mirror of Study 912)
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
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0. Default lag = 4(n/100)^(2/9)."""
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


def gap_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """HAC *t* on the daily return **difference** r1 - r2 (Jobson-Korkie, NW form).

    Both arms are excess-of-cash, so the cash leg cancels and the difference is exactly
    the wrapper-minus-DIY gap. |*t*| >= 2 is the desk's bar for a real edge.
    """
    diff = (r1 - r2).dropna()
    if len(diff) < 3:
        return float("nan")
    return newey_west_t(diff.to_numpy())


# --------------------------------------------------------------------------- #
# Performance summary
# --------------------------------------------------------------------------- #
def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass excess for excess stats)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {k: float("nan") for k in
                ("n_days", "ann_return_pct", "sharpe", "vol_ann", "max_drawdown", "tstat")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "ann_return_pct": float(mu * periods_per_year * 100.0),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": max_drawdown(r),
        "tstat": newey_west_t(r.to_numpy()),
    }


# --------------------------------------------------------------------------- #
# Block bootstrap CIs
# --------------------------------------------------------------------------- #
def _block_indices(n: int, block: int, rng) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def bootstrap_gap_ci(
    r1: pd.Series,
    r2: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 947,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the **annualised gap** (r1 - r2), in pp/yr.

    Blocks of ``block`` consecutive days preserve the vol clustering and the residual
    autocorrelation that a daily t-test on 1,478 overlapping observations would otherwise
    understate. Also returns the share of resamples on the wrong side of zero.
    """
    d = (pd.Series(r1) - pd.Series(r2)).dropna().to_numpy(dtype=float)
    n = d.size
    if n < block + 2:
        return {"gap_ann_pp": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "frac_negative": float("nan"), "n_obs": n}
    point = float(d.mean() * TRADING_DAYS * 100.0)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = d[_block_indices(n, block, rng)].mean() * TRADING_DAYS * 100.0
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"gap_ann_pp": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((boots < 0).mean()), "n_obs": n,
            "n_boot": n_boot, "block": block}


def bootstrap_sharpe_gap_ci(
    r1: pd.Series,
    r2: pd.Series,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 947,
    alpha: float = 0.05,
) -> dict:
    """Circular block-bootstrap CI for the **Sharpe difference** (r1 - r2), paired draws.

    The two arms are resampled with the *same* block indices, so the pairing (and hence the
    correlation between the arms) is preserved — the right way to bootstrap a difference of
    Sharpe ratios between two highly correlated funds.
    """
    a, b = pd.Series(r1).align(pd.Series(r2), join="inner")
    m = a.notna() & b.notna()
    x, y = a[m].to_numpy(dtype=float), b[m].to_numpy(dtype=float)
    n = x.size
    ann = np.sqrt(TRADING_DAYS)
    if n < block + 2:
        return {"sharpe_gap": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_obs": n}
    point = float(x.mean() / x.std(ddof=1) * ann - y.mean() / y.std(ddof=1) * ann)
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = _block_indices(n, block, rng)
        xs, ys = x[idx], y[idx]
        sx, sy = xs.std(ddof=1), ys.std(ddof=1)
        if sx > 0 and sy > 0:
            boots[i] = xs.mean() / sx * ann - ys.mean() / sy * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe_gap": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": n, "block": block}


def bootstrap_block_sensitivity(
    r1: pd.Series,
    r2: pd.Series,
    blocks=(5, 10, 21, 42, 63),
    seeds=(947, 1, 2),
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> list[dict]:
    """Does a block-bootstrap CI's verdict survive the **choice of block length**?

    The block length is a free parameter of the bootstrap, not a property of the data. A CI
    that excludes zero at one block length and straddles it at another has not established
    anything — it has revealed that the resampling scheme, not the tape, is doing the work.
    This sweeps the block length (and a few RNG seeds per block, so the answer is not one
    draw's luck) and reports whether the CI excludes zero at each setting. Use it on any CI
    that is about to be quoted as "excludes zero"; if ``excludes_zero`` is not True across
    the sweep, quote the HAC *t* instead and say why.
    """
    rows = []
    for block in blocks:
        for seed in seeds:
            b = bootstrap_gap_ci(r1, r2, n_boot=n_boot, block=block, seed=seed, alpha=alpha)
            lo, hi = b["ci_low"], b["ci_high"]
            rows.append({
                "block": block, "seed": seed, "gap_ann_pp": b["gap_ann_pp"],
                "ci_low": lo, "ci_high": hi, "frac_negative": b["frac_negative"],
                "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0),
            })
    return rows


# --------------------------------------------------------------------------- #
# Entry-point luck — how big is the thing laddering is supposed to average away?
# --------------------------------------------------------------------------- #
def dispersion_stats(rets: pd.DataFrame, cols, window: int = TRADING_DAYS) -> dict:
    """How much entry-point luck is there across vintages, and does averaging remove it?

    Rolling ``window``-day compounded returns per vintage; we report the mean/median/max
    best-minus-worst spread (the size of the luck), the standard deviation of those rolling
    returns per vintage and for the equal-weight basket, and the mean pairwise correlation
    of daily returns. If the vintages are near-perfectly correlated, averaging them removes
    almost no variance and laddering solves a problem that was never large.
    """
    sub = rets[list(cols)].dropna()
    if len(sub) <= window:
        return {}
    roll = (1.0 + sub).rolling(window).apply(np.prod, raw=True) - 1.0
    roll = roll.dropna()
    spread = (roll.max(axis=1) - roll.min(axis=1))
    ew = sub.mean(axis=1)
    roll_ew = ((1.0 + ew).rolling(window).apply(np.prod, raw=True) - 1.0).dropna()
    sd_single = {c: float(roll[c].std(ddof=1) * 100.0) for c in roll.columns}
    corr = sub.corr().to_numpy()
    iu = np.triu_indices_from(corr, k=1)
    mean_sd_single = float(np.mean(list(sd_single.values())))
    sd_ew = float(roll_ew.std(ddof=1) * 100.0)
    # Daily-return basis too: this is the quantity the equally-correlated-legs closed form
    # sd(basket)/sd(leg) = sqrt((1 + (N-1)*rho) / N) actually predicts. The holding-period
    # figure above is smaller because the vintages' path effects do not average as cleanly.
    sd_daily_single = float(sub.std(ddof=1).mean())
    sd_daily_ew = float(ew.std(ddof=1))
    n_legs = sub.shape[1]
    rho = float(corr[iu].mean())
    return {
        "window": window,
        "daily_variance_reduction_pct": float((1.0 - sd_daily_ew / sd_daily_single) * 100.0)
        if sd_daily_single > 0 else float("nan"),
        "daily_variance_reduction_closed_form_pct":
            float((1.0 - np.sqrt((1.0 + (n_legs - 1) * rho) / n_legs)) * 100.0),
        "spread_mean_pp": float(spread.mean() * 100.0),
        "spread_median_pp": float(spread.median() * 100.0),
        "spread_max_pp": float(spread.max() * 100.0),
        "sd_single_pct": sd_single,
        "sd_single_mean_pct": mean_sd_single,
        "sd_basket_pct": sd_ew,
        "variance_reduction_pct": float((1.0 - sd_ew / mean_sd_single) * 100.0)
        if mean_sd_single > 0 else float("nan"),
        "mean_pairwise_corr": float(corr[iu].mean()),
        "n_windows": int(len(roll)),
    }


# --------------------------------------------------------------------------- #
# The race (the headline)
# --------------------------------------------------------------------------- #
def race(
    prices: pd.DataFrame,
    ladder: str,
    vintages,
    market: str,
    cash: str,
    rebalance: str = "Q",
    cost_bps: float = 5.0,
    borrow_bps: float = 50.0,
    min_obs: int = 252,
    extra_fee_pct: float = 0.0,
) -> dict:
    """Race the laddered wrapper against every DIY alternative, excess-of-cash.

    Arms: the wrapper, each single vintage, the equal-weight DIY basket, the
    **beta-matched** DIY ladder, and the beta-matched SPY/BIL mix.

    Two windows, deliberately. The arms that need **no estimated weight** (the wrapper,
    the vintages, the DIY basket) are raced over the whole common sample. The
    **beta-matched** arms need an expanding-window beta, so they burn the first
    ``min_obs`` days and are raced over the shorter window — the price of estimating a
    weight out-of-sample rather than peeking at the full sample. Both ``n`` values are
    reported so nothing is quietly compared on different samples.

    ``extra_fee_pct`` builds the *pre-extra-fee* counterfactual: the (assumed, quoted)
    incremental wrapper fee in pp/yr is added back to the wrapper's return, answering
    "would the ladder have won if the extra layer were waived?". It is a PROXY, not a tape
    measurement, and :func:`fee_sweep` sweeps it.

    Returns a dict of per-arm summaries, the gaps with HAC *t*, the betas, and the raw
    excess series for downstream bootstrap / era work.
    """
    rets = to_returns(prices)
    ex = excess_returns(rets, cash)

    e_ladder = ex[ladder].dropna()
    if extra_fee_pct:
        e_ladder = e_ladder + extra_fee_pct / 100.0 / TRADING_DAYS
    e_mkt = ex[market].dropna()
    basket = equal_weight_basket(ex, vintages, rebalance=rebalance, cost_bps=cost_bps)

    b_ladder = expanding_beta(e_ladder, e_mkt, min_obs=min_obs)
    b_basket = expanding_beta(basket, e_mkt, min_obs=min_obs)
    idx_b = b_ladder.index[b_ladder.notna() & b_basket.notna()]
    bl, bb = b_ladder.reindex(idx_b), b_basket.reindex(idx_b)

    diy_matched = beta_matched_ladder(basket.reindex(idx_b), e_mkt.reindex(idx_b), bl, bb,
                                      cost_bps=cost_bps, borrow_bps=borrow_bps)
    mix_ladder = beta_matched_mix(e_mkt.reindex(idx_b), bl,
                                  cost_bps=cost_bps, borrow_bps=borrow_bps)
    mix_basket = beta_matched_mix(e_mkt.reindex(idx_b), bb,
                                  cost_bps=cost_bps, borrow_bps=borrow_bps)

    arms = {"ladder": e_ladder, "diy_basket": basket, "market": e_mkt,
            "diy_beta_matched": diy_matched, "beta_mix_ladder": mix_ladder,
            "beta_mix_basket": mix_basket}
    for v in vintages:
        arms[v] = ex[v].dropna()

    summaries = {k: summary(v) for k, v in arms.items()}
    # Absolute (lived) drawdowns on the nominal, not excess, series.
    abs_dd = {k: max_drawdown(rets[k]) for k in [ladder, market] + list(vintages)}
    abs_dd["diy_basket"] = max_drawdown(
        equal_weight_basket(rets, vintages, rebalance=rebalance, cost_bps=cost_bps))

    def _gap(a: pd.Series, b: pd.Series) -> dict:
        d = (a - b).dropna()
        return {
            "gap_ann_pp": float(d.mean() * TRADING_DAYS * 100.0),
            "t_hac": newey_west_t(d.to_numpy()),
            "tracking_error_pct": float(d.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0),
            "sharpe_gap": summary(a.reindex(d.index))["sharpe"] - summary(b.reindex(d.index))["sharpe"],
            "n_days": int(len(d)),
        }

    gaps = {"vs_diy_basket": _gap(e_ladder, basket)}
    for v in vintages:
        gaps[f"vs_{v}"] = _gap(e_ladder, arms[v])
    gaps["vs_diy_beta_matched"] = _gap(e_ladder.reindex(idx_b), diy_matched)
    gaps["vs_beta_mix"] = _gap(e_ladder.reindex(idx_b), mix_ladder)
    gaps["basket_vs_beta_mix"] = _gap(basket.reindex(idx_b), mix_basket)

    return {
        "arms": arms,
        "summary": summaries,
        "abs_drawdown": abs_dd,
        "gaps": gaps,
        "beta_ladder_oos_last": float(bl.iloc[-1]) if len(bl) else float("nan"),
        "beta_basket_oos_last": float(bb.iloc[-1]) if len(bb) else float("nan"),
        "beta_ladder_full_sample": full_sample_beta(e_ladder, e_mkt),
        "beta_basket_full_sample": full_sample_beta(basket, e_mkt),
        "corr_ladder_basket": float(e_ladder.corr(basket)),
        "window": (e_ladder.index[0], e_ladder.index[-1]),
        "window_matched": (idx_b[0], idx_b[-1]) if len(idx_b) else (None, None),
        "n_days": int(len(e_ladder)),
        "n_days_matched": int(len(idx_b)),
        "extra_fee_pct": extra_fee_pct,
    }


# --------------------------------------------------------------------------- #
# Robustness — eras, costs, the assumed fee layer
# --------------------------------------------------------------------------- #
def era_cut(prices: pd.DataFrame, ladder: str, vintages, market: str, cash: str,
            split: str, **kw) -> dict:
    """Re-run the race on each side of ``split``. A real premium shows in both halves.

    Both halves reuse the *same* expanding beta path (estimated on the full history up to
    each day, lagged one day) rather than re-estimating within the half — re-fitting inside
    a two-year window would be noisier than the effect under test.
    """
    full = race(prices, ladder, vintages, market, cash, **kw)
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        arms = {k: v.loc[sl] for k, v in full["arms"].items()}
        if len(arms["ladder"]) < 60:
            out[tag] = None
            continue
        row = {"n_days": int(len(arms["ladder"])),
               "sharpe_ladder": summary(arms["ladder"])["sharpe"],
               "sharpe_basket": summary(arms["diy_basket"])["sharpe"],
               "sharpe_matched": summary(arms["diy_beta_matched"])["sharpe"]}
        for name, other in [("vs_diy_basket", arms["diy_basket"]),
                            ("vs_diy_beta_matched", arms["diy_beta_matched"]),
                            ("vs_beta_mix", arms["beta_mix_ladder"])]:
            d = (arms["ladder"] - other).dropna()
            row[name + "_gap_pp"] = float(d.mean() * TRADING_DAYS * 100.0)
            row[name + "_t"] = newey_west_t(d.to_numpy())
        out[tag] = row
    return out


def cost_sweep(prices: pd.DataFrame, ladder: str, vintages, market: str, cash: str,
               cost_grid=(0.0, 2.0, 5.0, 10.0, 25.0), **kw) -> list[dict]:
    """Sweep the one-way cost charged on the DIY basket's rebalance and the beta legs.

    The DIY arm is the one paying friction (the wrapper's own trading is already inside its
    NAV), so raising the cost can only *help* the wrapper. If the wrapper still fails to win
    at 25 bps one-way, cost is not what is hiding the premium.
    """
    rows = []
    for c in cost_grid:
        res = race(prices, ladder, vintages, market, cash, cost_bps=c, **kw)
        rows.append({
            "cost_bps": c,
            "gap_vs_basket_pp": res["gaps"]["vs_diy_basket"]["gap_ann_pp"],
            "t_vs_basket": res["gaps"]["vs_diy_basket"]["t_hac"],
            "gap_vs_matched_pp": res["gaps"]["vs_diy_beta_matched"]["gap_ann_pp"],
            "t_vs_matched": res["gaps"]["vs_diy_beta_matched"]["t_hac"],
        })
    return rows


def fee_sweep(prices: pd.DataFrame, ladder: str, vintages, market: str, cash: str,
              fee_grid=(0.0, 0.10, 0.20, 0.26, 0.40), **kw) -> list[dict]:
    """Sweep the ASSUMED incremental wrapper fee added back to the wrapper's return.

    Published NAV returns are already net of whatever was charged, so this is a
    counterfactual on a **quoted, non-tape** number: had the extra layer been ``fee`` pp/yr
    and had it been waived, the wrapper's gap would improve by exactly that much. The sweep
    exists so no conclusion rests on one guess at the fee.
    """
    rows = []
    for f in fee_grid:
        res = race(prices, ladder, vintages, market, cash, extra_fee_pct=f, **kw)
        rows.append({
            "extra_fee_pct": f,
            "gap_vs_basket_pp": res["gaps"]["vs_diy_basket"]["gap_ann_pp"],
            "t_vs_basket": res["gaps"]["vs_diy_basket"]["t_hac"],
            "gap_vs_matched_pp": res["gaps"]["vs_diy_beta_matched"]["gap_ann_pp"],
            "t_vs_matched": res["gaps"]["vs_diy_beta_matched"]["t_hac"],
        })
    return rows


def rebalance_sweep(prices: pd.DataFrame, ladder: str, vintages, market: str, cash: str,
                    grid=("N", "A", "Q", "M"), **kw) -> list[dict]:
    """Does the DIY basket's rebalance frequency matter? (It should barely.)"""
    rows = []
    for g in grid:
        res = race(prices, ladder, vintages, market, cash, rebalance=g, **kw)
        rows.append({
            "rebalance": g,
            "sharpe_basket": res["summary"]["diy_basket"]["sharpe"],
            "gap_vs_basket_pp": res["gaps"]["vs_diy_basket"]["gap_ann_pp"],
            "t_vs_basket": res["gaps"]["vs_diy_basket"]["t_hac"],
        })
    return rows


def calendar_year_table(prices: pd.DataFrame, ladder: str, vintages, market: str,
                        cash: str, rebalance: str = "Q", cost_bps: float = 5.0) -> pd.DataFrame:
    """Per-calendar-year nominal total return (%) for the wrapper, each vintage, the DIY
    basket and the market — the lived experience, and where entry-point luck is visible.

    Partial calendar years (the wrapper's 2020 stub and the as-of 2026 stub) are dropped so
    no year is compared on an unequal number of trading days.
    """
    rets = to_returns(prices)
    basket = equal_weight_basket(rets, vintages, rebalance=rebalance, cost_bps=cost_bps)
    rows = []
    years = sorted(set(rets.index.year))
    for y in years:
        m = rets.index.year == y
        if m.sum() < 200:          # drop partial years
            continue
        row = {"year": int(y)}
        row[ladder] = float((1.0 + rets[ladder][m]).prod() - 1.0) * 100.0
        for v in vintages:
            row[v] = float((1.0 + rets[v][m]).prod() - 1.0) * 100.0
        row["diy_basket"] = float((1.0 + basket[basket.index.year == y]).prod() - 1.0) * 100.0
        row["vintage_spread_pp"] = max(row[v] for v in vintages) - min(row[v] for v in vintages)
        row[market] = float((1.0 + rets[market][m]).prod() - 1.0) * 100.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, cost_bps: float = 0.0) -> dict:
    """Run the race on a synthetic panel and report what it recovered vs what was planted.

    With a planted laddering premium the measured wrapper-minus-basket gap must land near
    ``truth['expected_gap_ann']`` (the premium net of the planted fee) with |*t*| >= 2; on
    the null (``signal_strength = 0``) it must land near ``-extra_fee_ann`` and never invent
    a positive premium. Proves the detector is unbiased — it never supports a real stamp.
    """
    res = race(prices, "ladder", truth["vintages"], "market", "cash",
               cost_bps=cost_bps, borrow_bps=0.0)
    g = res["gaps"]["vs_diy_basket"]
    return {
        "gap_ann_pp": g["gap_ann_pp"],
        "t_hac": g["t_hac"],
        "expected_gap_pp": float(truth["expected_gap_ann"] * 100.0),
        "error_pp": g["gap_ann_pp"] - float(truth["expected_gap_ann"] * 100.0),
        "gap_matched_pp": res["gaps"]["vs_diy_beta_matched"]["gap_ann_pp"],
        "t_matched": res["gaps"]["vs_diy_beta_matched"]["t_hac"],
        "n_days": res["n_days"],
    }
