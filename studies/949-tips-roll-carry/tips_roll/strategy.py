"""Strategy + inference for Study 949 — Riding the TIPS Curve.

Two questions, two engines.

**1. Does extending along the real curve pay?** ``ladder_race`` races the four
inflation-linked maturity buckets (VTIP → SCHP → TIP → LTPZ) **excess-of-cash** (minus
BIL's actual total return), reports each leg's excess Sharpe, drawdown and HAC *t*, and
puts a Newey-West *t* on every long-minus-short difference. If riding the real curve
pays a roll-down carry, the long buckets should out-earn the short one per unit of risk.

**2. Is what is left after duration a real carry?** ``hedged_sleeve`` strips the
duration factor out of each linker by shorting a **duration-matched nominal Treasury
fund**, and asks whether the residual is reliably positive:

    residual_t = ex_tips_t − beta_{t-1} · ex_nominal_t

``beta_{t-1}`` is estimated by rolling OLS on the trailing ``window`` days ending at
``t-1`` and applied to day ``t``. **That is the study's one and only execution lag**,
and it is the honest one: the legs themselves are buy-and-hold funds with no timing
signal, so the only thing that could peek is the hedge ratio. ``static_alpha`` also
reports the full-sample OLS alpha for reference — clearly labelled *in-sample*, since
it uses the whole history to fit the hedge.

Costs are charged the desk way: the one-way cost is applied to the **change in the
short leg's weight** (|Δbeta| × NAV, no cost on the buy-and-hold long leg), and the
short nominal leg pays a **borrow fee** of ``borrow_bps_yr`` per year on |beta| × NAV.
The borrow rate is a PROXY (Treasury-ETF borrow is cheap but not free) and is swept in
``cost_borrow_sweep`` — at some rate the residual dies, and it matters where.

**Identification, stated plainly.** A long-linker / short-nominal residual is a
*long-breakeven* position. Its expected return is roll-down in real yields **plus**
(realised inflation − the breakeven priced at entry). Fund total returns cannot tell
those apart, so a positive residual is evidence of *something* the duration hedge does
not explain — not proof it is roll-down. The era cut around the 2021-2023 inflation
shock is what decides which story the tape actually supports.

Returns are **simple** (arithmetic) throughout, so a hedged sleeve's return is exactly
the weighted sum of its legs and wealth paths compound correctly.
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


def _auto_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_t(x, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) *t* of mean(x) vs 0.

    Daily bond-fund returns are mildly autocorrelated (stale NAV marks, curve
    momentum), so a naive *t* over-states significance. ``lags`` defaults to the
    standard 4(n/100)^(2/9) rule.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = _auto_lags(n)
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * float(u[l:] @ u[:-l]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Returns & summaries
# --------------------------------------------------------------------------- #
def excess_returns(prices: pd.DataFrame, cash_col: str = "BIL") -> pd.DataFrame:
    """Daily simple total returns, every column net of the cash leg's actual return.

    The cash column is kept (as itself minus itself it would be zero, so it is left as
    the raw cash return) under the name ``cash_col`` so callers can still see what the
    bill leg earned; every other column is excess-of-cash.
    """
    r = prices.sort_index().pct_change().dropna(how="all")
    ex = r.sub(r[cash_col], axis=0)
    ex[cash_col] = r[cash_col]
    return ex.dropna()


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised stats for a daily simple-return series (pass an *excess* series
    for excess Sharpe): mean, vol, Sharpe, CAGR, max drawdown and the HAC *t*."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": int(n), "ann_return": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "cagr": float("nan"),
                "max_drawdown": float("nan"), "tstat": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan")
    return {
        "n_days": int(n),
        "ann_return": float(mu * periods_per_year),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "cagr": cagr,
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "tstat": newey_west_t(r.to_numpy()),
    }


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


# --------------------------------------------------------------------------- #
# Question 1 — does extending along the real curve pay?
# --------------------------------------------------------------------------- #
def ladder_race(ex: pd.DataFrame, legs, base: str | None = None) -> dict:
    """Race the maturity buckets excess-of-cash; HAC *t* on each leg-minus-``base``.

    ``ex`` is an excess-return frame (see :func:`excess_returns`); ``legs`` is the
    ladder ordered shortest → longest. ``base`` defaults to the shortest leg, so every
    difference answers "did extending duration pay?". A roll-down carry story predicts
    positive, significant differences; a pure-duration-risk story predicts nothing.
    """
    legs = list(legs)
    base = base or legs[0]
    per_leg = {lg: summary(ex[lg]) for lg in legs}
    diffs = {}
    for lg in legs:
        if lg == base:
            continue
        d = (ex[lg] - ex[base]).dropna()
        diffs[lg] = {
            "ann_diff": float(d.mean() * TRADING_DAYS),
            "t_diff": newey_west_t(d.to_numpy()),
            "sharpe_gap": per_leg[lg]["sharpe"] - per_leg[base]["sharpe"],
        }
    return {"legs": per_leg, "base": base, "diff_vs_base": diffs}


def absolute_stats(prices: pd.DataFrame, legs) -> dict:
    """Absolute (not excess) CAGR and lived max drawdown per leg — what a holder felt."""
    r = prices.sort_index().pct_change().dropna(how="all")
    out = {}
    for lg in legs:
        s = summary(r[lg])
        out[lg] = {"cagr": s["cagr"], "max_drawdown": s["max_drawdown"],
                   "vol_ann": s["vol_ann"]}
    return out


# --------------------------------------------------------------------------- #
# Question 2 — the duration-hedged residual
# --------------------------------------------------------------------------- #
def static_alpha(y_ex: pd.Series, x_ex: pd.Series) -> dict:
    """Full-sample OLS of the linker's excess return on the nominal's excess return.

    Reported for reference and **labelled in-sample**: the hedge ratio is fitted on the
    whole history, so this alpha is not tradable. The tradable version is
    :func:`hedged_sleeve`, whose beta is estimated strictly out-of-sample.
    """
    df = pd.concat([y_ex.rename("y"), x_ex.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return {"alpha_ann": float("nan"), "beta": float("nan"),
                "t_alpha": float("nan"), "r2": float("nan"), "n_days": len(df)}
    y = df["y"].to_numpy(); x = df["x"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")
    return {
        "alpha_ann": float(coef[0] * TRADING_DAYS),
        "beta": float(coef[1]),
        # HAC t on the residual re-centred at alpha == the HAC t of the intercept.
        "t_alpha": newey_west_t(resid + coef[0]),
        "r2": float(r2),
        "n_days": int(len(df)),
    }


def rolling_beta(y_ex: pd.Series, x_ex: pd.Series, window: int = 252) -> pd.Series:
    """Trailing-``window`` OLS beta of ``y_ex`` on ``x_ex``, **lagged one day**.

    The beta stamped on day ``t`` is fitted on data through ``t-1`` and applied to day
    ``t``'s returns. One lag, no look-ahead, and the only lag in the study.
    """
    cov = y_ex.rolling(window).cov(x_ex)
    var = x_ex.rolling(window).var()
    return (cov / var).shift(1).rename("beta")


def hedged_sleeve(
    y_ex: pd.Series,
    x_ex: pd.Series,
    window: int = 252,
    cost_bps: float = 2.0,
    borrow_bps_yr: float = 30.0,
) -> pd.DataFrame:
    """Long the linker, short ``beta_{t-1}`` of the duration-matched nominal fund.

    ``cost_bps`` is charged one-way on the *change* in the short weight (|Δbeta| × NAV);
    the long leg is buy-and-hold and pays nothing. ``borrow_bps_yr`` is the annual stock
    borrow on the short nominal leg, accrued daily on |beta| × NAV — a PROXY, swept in
    :func:`cost_borrow_sweep`.

    Returns a frame with ``beta``, ``gross`` and ``net`` daily residual returns.
    """
    beta = rolling_beta(y_ex, x_ex, window=window)
    gross = (y_ex - beta * x_ex).rename("gross")
    turnover = beta.diff().abs().fillna(0.0)
    fee = turnover * cost_bps * 1e-4 + beta.abs() * (borrow_bps_yr * 1e-4) / TRADING_DAYS
    net = (gross - fee).rename("net")
    return pd.concat([beta, gross, net, turnover.rename("turnover")], axis=1).dropna()


def sleeve_stats(sleeve: pd.DataFrame) -> dict:
    """Headline stats for a hedged sleeve: gross and net annualised carry + HAC *t*."""
    g, n = sleeve["gross"], sleeve["net"]
    sg, sn = summary(g), summary(n)
    return {
        "n_days": int(len(sleeve)),
        "mean_beta": float(sleeve["beta"].mean()),
        "turnover_ann": float(sleeve["turnover"].sum() / (len(sleeve) / TRADING_DAYS)),
        "gross_ann": sg["ann_return"], "t_gross": sg["tstat"], "sharpe_gross": sg["sharpe"],
        "net_ann": sn["ann_return"], "t_net": sn["tstat"], "sharpe_net": sn["sharpe"],
        "vol_ann": sn["vol_ann"], "max_drawdown": sn["max_drawdown"],
    }


def sleeve_table(ex: pd.DataFrame, pairs, **kw) -> dict:
    """Run :func:`hedged_sleeve` for every ``(linker, nominal)`` pair in ``pairs``."""
    out = {}
    for a, b in pairs:
        out[(a, b)] = sleeve_stats(hedged_sleeve(ex[a], ex[b], **kw))
    return out


# --------------------------------------------------------------------------- #
# Robustness — eras, costs, pairings, bootstrap
# --------------------------------------------------------------------------- #
def era_cut(ex: pd.DataFrame, a: str, b: str, eras, **kw) -> dict:
    """Re-run the hedged sleeve inside each ``(label, start, end)`` era.

    A carry that is a *structural* feature of the real curve should show up in every
    era; one that is an inflation surprise shows up only around the shock.
    """
    sleeve = hedged_sleeve(ex[a], ex[b], **kw)
    out = {}
    for label, start, end in eras:
        sl = sleeve.loc[str(start):str(end)]
        if len(sl) < 60:
            out[label] = None
            continue
        st = sleeve_stats(sl)
        st["start"], st["end"] = str(sl.index[0].date()), str(sl.index[-1].date())
        out[label] = st
    return out


def drop_year(ex: pd.DataFrame, a: str, b: str, year: int, **kw) -> dict:
    """The hedged sleeve with one calendar year excised — the jackknife that matters
    when a single year is suspected of carrying the whole result."""
    sleeve = hedged_sleeve(ex[a], ex[b], **kw)
    kept = sleeve[sleeve.index.year != year]
    st = sleeve_stats(kept)
    st["dropped_year"] = int(year)
    return st


def cost_borrow_sweep(ex: pd.DataFrame, a: str, b: str,
                      cost_grid=(0.0, 2.0, 5.0, 10.0),
                      borrow_grid=(0.0, 25.0, 30.0, 50.0, 100.0),
                      window: int = 252) -> dict:
    """Sweep the two frictions independently around the base case (2 bps / 30 bps/yr).

    Borrow is the one that bites: the sleeve is short a nominal Treasury fund every day,
    so the fee is a *level* charge, not a turnover charge.
    """
    base_borrow, base_cost = 30.0, 2.0
    costs = []
    for c in cost_grid:
        st = sleeve_stats(hedged_sleeve(ex[a], ex[b], window=window,
                                        cost_bps=c, borrow_bps_yr=base_borrow))
        costs.append({"cost_bps": c, "net_ann": st["net_ann"], "t_net": st["t_net"]})
    borrows = []
    for bo in borrow_grid:
        st = sleeve_stats(hedged_sleeve(ex[a], ex[b], window=window,
                                        cost_bps=base_cost, borrow_bps_yr=bo))
        borrows.append({"borrow_bps_yr": bo, "net_ann": st["net_ann"],
                        "t_net": st["t_net"], "sharpe_net": st["sharpe_net"]})
    return {"cost_sweep": costs, "borrow_sweep": borrows}


def pairing_sweep(ex: pd.DataFrame, a: str, alternatives, **kw) -> list:
    """Re-hedge the same linker against alternative nominal funds.

    The duration match is an assumption from sponsor fact sheets; if the residual only
    survives one pairing it is a pairing artefact, not a carry.
    """
    out = []
    for b in alternatives:
        st = sleeve_stats(hedged_sleeve(ex[a], ex[b], **kw))
        out.append({"nominal": b, "mean_beta": st["mean_beta"],
                    "gross_ann": st["gross_ann"], "t_gross": st["t_gross"],
                    "net_ann": st["net_ann"], "t_net": st["t_net"]})
    return out


def window_sweep(ex: pd.DataFrame, a: str, b: str, windows=(126, 252, 504), **kw) -> list:
    """Re-estimate the hedge beta over different trailing windows."""
    out = []
    for w in windows:
        st = sleeve_stats(hedged_sleeve(ex[a], ex[b], window=w, **kw))
        out.append({"window": w, "mean_beta": st["mean_beta"],
                    "net_ann": st["net_ann"], "t_net": st["t_net"]})
    return out


def bootstrap_mean_ci(x, n_boot: int = 2000, block: int = 21, seed: int = 949,
                      alpha: float = 0.05, periods_per_year: int = TRADING_DAYS) -> dict:
    """Circular block-bootstrap CI for the *annualised mean* of a daily series."""
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        boots[i] = r[idx].mean() * periods_per_year
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(r.mean() * periods_per_year), "ci_low": float(lo),
            "ci_high": float(hi), "frac_negative": float((boots < 0).mean()),
            "n_obs": int(n), "block": int(block)}


def bootstrap_sharpe_ci(x, n_boot: int = 2000, block: int = 21, seed: int = 949,
                        alpha: float = 0.05, periods_per_year: int = TRADING_DAYS) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of a daily series."""
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": int(n)}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sdi = s.std(ddof=1)
        if sdi > 0:
            boots[i] = s.mean() / sdi * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": int(n),
            "n_boot": int(valid.size), "block": int(block)}


def calendar_year_table(sleeve: pd.DataFrame) -> pd.DataFrame:
    """Per-calendar-year compounded gross and net residual carry (%) of a sleeve."""
    rows = []
    for y in sorted(set(sleeve.index.year)):
        m = sleeve.index.year == y
        rows.append({
            "year": int(y),
            "gross_pct": float((1.0 + sleeve["gross"][m]).prod() - 1.0) * 100,
            "net_pct": float((1.0 + sleeve["net"][m]).prod() - 1.0) * 100,
            "n_days": int(m.sum()),
        })
    return pd.DataFrame(rows).set_index("year")


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, window: int = 252,
                     cost_bps: float = 0.0, borrow_bps_yr: float = 0.0) -> dict:
    """Run the duration-hedge detector on a synthetic ``tips``/``nominal``/``cash`` tape.

    On a tape with a planted residual carry the hedged alpha should recover it with a
    clearly significant HAC *t*; on the null (no planted carry) it must sit at zero.
    Frictions default to zero here — this measures the *detector*, not a trade.
    """
    ex = excess_returns(prices, cash_col="cash")
    sleeve = hedged_sleeve(ex["tips"], ex["nominal"], window=window,
                           cost_bps=cost_bps, borrow_bps_yr=borrow_bps_yr)
    st = sleeve_stats(sleeve)
    stat = static_alpha(ex["tips"], ex["nominal"])
    st["static_alpha_ann"] = stat["alpha_ann"]
    st["static_beta"] = stat["beta"]
    st["static_t_alpha"] = stat["t_alpha"]
    return st


def synthetic_ladder_detect(prices: pd.DataFrame, truth: dict) -> dict:
    """Ladder race + per-bucket hedged carry on a synthetic panel.

    The panel plants the *same* carry in every bucket, so a healthy detector recovers a
    flat carry term structure — and the excess-of-cash race, which cannot separate carry
    from duration, must NOT be mistaken for one.
    """
    ex = excess_returns(prices, cash_col="cash")
    k = int(truth["n_buckets"])
    tips = [f"tips_{i}" for i in range(k)]
    race = ladder_race(ex, tips)
    carries = {}
    for i in range(k):
        st = sleeve_stats(hedged_sleeve(ex[f"tips_{i}"], ex[f"nom_{i}"],
                                        cost_bps=0.0, borrow_bps_yr=0.0))
        carries[f"bucket_{i}"] = {"gross_ann": st["gross_ann"], "t_gross": st["t_gross"],
                                  "mean_beta": st["mean_beta"]}
    return {"race": race, "carries": carries,
            "planted_carry_ann": truth["carry_ann_planted"]}
