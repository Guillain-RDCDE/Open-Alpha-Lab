"""Measurement + inference for Study 914 — Securities-Lending Offset.

The claim: lending-heavy funds (small caps, emerging markets) earn securities-lending
revenue that offsets part of the expense ratio, so their realised tracking difference is
*better* than ``-ER``.

The measurement. Fund-level lending revenue is not public tape, and a licensed
net-of-withholding index total-return series is not free, so a *level* tracking difference
is out of reach. What is in reach is the **relative** version, between two funds in the
same asset class:

    drift(a, b)     = annualised mean of ( log r_a - log r_b )       [the tape]
    delta_er        = ER_a - ER_b                                    [a PROXY, see data.py]
    residual(a, b)  = drift(a, b) + delta_er                         [bp per year]

Read it as: *the part of the realised return gap that the fee gap does not explain.* If
neither fund passed back any lending revenue and both tracked their benchmarks perfectly,
the residual is zero by construction. A **positive** residual means ``a`` delivered more
than its fee disadvantage implied — the signature the lending-offset claim predicts for
the lending-heavier leg. A negative residual means it delivered less.

Three disciplines are baked in.

1. **No signal, therefore no execution lag — and none is faked.** The residual is an
   accounting identity on contemporaneous returns. The *tradable* leg (``pair_trade``) is
   a **static** long/short on two named funds: there is no forecast, no rule, and nothing
   to lag. Its return series is the month-on-month spread of two total-return closes,
   which is arithmetically a book **rebalanced to equal notional every month**. The
   friction is charged once a year (in January, one period after the December book date)
   as a deliberately conservative round-trip stand-in — *not* as an execution lag, and the
   study does not claim one. See ``pair_trade`` for the size of that conservatism.

2. **Costs and borrow.** The pair trade is long one fund and **short** the other, so the
   short leg pays a borrow fee — swept from 0 to 100 bp/yr, because ETF borrow is the
   single input that decides whether a few basis points of fee gap survive. Rebalance
   friction is one-way bp x NAV on the notional actually reset.

3. **Frequency.** The headline runs on **monthly** log returns. Daily total-return closes
   from Yahoo carry large, transient dividend-timing noise between two funds that pay on
   different dates (SPY/IVV: 2.2% annualised daily tracking error against 0.5% monthly) —
   noise that cancels within the month and would otherwise inflate every variance in the
   study. The daily version is reported as a robustness check and agrees.

Everything is **total return** (``auto_adjust=True``); nothing here is price-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MONTHS = 12
BP = 1e4


# --------------------------------------------------------------------------- #
# Inference primitives (shared desk mirror)
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
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
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


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def block_bootstrap_mean_ci(x, n_boot: int = 4000, block: int = 6,
                            seed: int = 914, alpha: float = 0.05):
    """Circular block-bootstrap percentile CI for the mean of a return series.

    Blocks of ``block`` consecutive observations preserve any persistence in the
    tracking wedge (index reconstitution effects cluster).
    """
    r = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    off = np.arange(block)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + off[None, :]) % n).ravel()[:n]
        boots[i] = r[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(r.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(n), "n_boot": int(n_boot), "block": int(block)}


# --------------------------------------------------------------------------- #
# Return construction
# --------------------------------------------------------------------------- #
def monthly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end log total returns. Partial trailing months are dropped by the as-of."""
    m = np.log(prices.dropna()).resample("ME").last()
    return m.diff().dropna()


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.dropna()).diff().dropna()


# --------------------------------------------------------------------------- #
# The headline measurement — the fee-adjusted residual
# --------------------------------------------------------------------------- #
def pair_residual(
    prices: pd.DataFrame,
    a: str,
    b: str,
    er_a: float,
    er_b: float,
    freq: str = "M",
    n_boot: int = 4000,
    seed: int = 914,
) -> dict:
    """Fee-adjusted relative tracking residual of fund ``a`` against peer ``b``.

    ``er_a`` / ``er_b`` are expense ratios in **percent per year** (a PROXY — see
    ``data.EXPENSE_RATIOS``). All rates are reported in **bp per year**.

    Returns the realised relative drift, the fee gap, the residual, its HAC *t*, a block
    bootstrap CI, the tracking error, the noise floor (the smallest effect this sample
    could resolve at |t| = 2), and the OLS beta of ``a`` on ``b`` — a blunt read on how
    different the two baskets actually are.
    """
    px = prices[[a, b]].dropna()
    lr = monthly_log_returns(px) if freq == "M" else daily_log_returns(px)
    per_year = MONTHS if freq == "M" else TRADING_DAYS
    block = 6 if freq == "M" else 21

    d = (lr[a] - lr[b]).astype(float)
    delta_er = (er_a - er_b) / 100.0
    resid = d + delta_er / per_year

    te = float(d.std(ddof=1) * np.sqrt(per_year))
    years = len(d) / per_year
    se_ann = te / np.sqrt(years) if years > 0 else float("nan")
    ci = block_bootstrap_mean_ci(resid.to_numpy(), n_boot=n_boot, block=block, seed=seed)
    beta = float(np.polyfit(lr[b].to_numpy(), lr[a].to_numpy(), 1)[0])

    return {
        "a": a, "b": b, "freq": freq,
        "start": str(lr.index[0].date()), "end": str(lr.index[-1].date()),
        "n_obs": int(len(d)), "years": float(years),
        "drift_bp": float(d.mean() * per_year * BP),
        "delta_er_bp": float(delta_er * BP),
        "residual_bp": float(resid.mean() * per_year * BP),
        "t_hac": newey_west_t(resid.to_numpy(), lags=6 if freq == "M" else None),
        "ci_low_bp": float(ci["ci_low"] * per_year * BP),
        "ci_high_bp": float(ci["ci_high"] * per_year * BP),
        "te_pct": te * 100.0,
        "se_bp": float(se_ann * BP),
        "noise_floor_bp": float(2.0 * se_ann * BP),
        "beta": beta,
        "resid_series": resid,
    }


def residual_table(prices: pd.DataFrame, pairs, expense_ratios,
                   freq: str = "M", n_boot: int = 4000, seed: int = 914) -> pd.DataFrame:
    """Run ``pair_residual`` over every pair and return a tidy frame (no series column)."""
    rows = []
    for a, b, klass, _note in pairs:
        r = pair_residual(prices, a, b, expense_ratios[a], expense_ratios[b],
                          freq=freq, n_boot=n_boot, seed=seed)
        r.pop("resid_series")
        r["asset_class"] = klass
        rows.append(r)
    cols = ["a", "b", "asset_class", "start", "end", "n_obs", "years", "drift_bp",
            "delta_er_bp", "residual_bp", "t_hac", "ci_low_bp", "ci_high_bp",
            "te_pct", "se_bp", "noise_floor_bp", "beta"]
    return pd.DataFrame(rows)[cols]


# --------------------------------------------------------------------------- #
# Robustness — eras, the expense-ratio assumption sweep, the frequency check
# --------------------------------------------------------------------------- #
def era_cut(prices: pd.DataFrame, a: str, b: str, er_a: float, er_b: float,
            split: str = "2020-01-01", freq: str = "M") -> dict:
    """Split at ``split`` and re-measure the residual on each half.

    A genuine, structural passthrough should be present in both halves; a wandering
    composition wedge will not be.
    """
    out = {}
    for tag, sl in [("early", slice(None, split)), ("late", slice(split, None))]:
        sub = prices.loc[sl]
        px = sub[[a, b]].dropna()
        if len(px) < (60 if freq == "M" else 500):
            out[tag] = None
            continue
        r = pair_residual(px, a, b, er_a, er_b, freq=freq, n_boot=1000)
        r.pop("resid_series")
        out[tag] = r
    return out


def er_sweep(prices: pd.DataFrame, a: str, b: str, er_a_grid, er_b: float,
             freq: str = "M") -> list:
    """Sweep the (assumed, non-tape) expense ratio of leg ``a`` and re-report the residual.

    The residual moves one-for-one with the assumed fee gap, so this sweep states exactly
    how much of the answer is tape and how much is the assumption.
    """
    out = []
    for er_a in er_a_grid:
        r = pair_residual(prices, a, b, er_a, er_b, freq=freq, n_boot=800)
        out.append({"er_a": er_a, "delta_er_bp": r["delta_er_bp"],
                    "residual_bp": r["residual_bp"], "t_hac": r["t_hac"]})
    return out


def er_sweep_b(prices: pd.DataFrame, a: str, b: str, er_a: float, er_b_grid,
               freq: str = "M") -> list:
    """Sweep the assumed expense ratio of the CHEAP leg ``b`` — the decisive sweep.

    Sweeping only the dear leg is a **one-sided** sweep. Every cheap leg on this desk had
    its fee cut hard during the sample (IVV 0.0945 -> 0.03, IEMG 0.18 -> 0.09), so pinning
    it at today's value overstates the fee gap and pushes the residual up. Raising ``er_b``
    lowers the residual one-for-one; this is the sweep that decides the residual's *sign*.
    """
    out = []
    for er_b in er_b_grid:
        r = pair_residual(prices, a, b, er_a, er_b, freq=freq, n_boot=800)
        out.append({"er_b": er_b, "delta_er_bp": r["delta_er_bp"],
                    "residual_bp": r["residual_bp"], "t_hac": r["t_hac"]})
    return out


# --------------------------------------------------------------------------- #
# The tradable leg — can you harvest the gap? (long cheap / short dear, with borrow)
# --------------------------------------------------------------------------- #
def pair_trade(
    prices: pd.DataFrame,
    long: str,
    short: str,
    borrow_bps: float = 50.0,
    cost_bps: float = 3.0,
    reset: str = "A",
) -> pd.Series:
    """Dollar-neutral long/short pair on two same-class funds, monthly return series.

    Long ``long``, short ``short``, equal notional. **There is no signal here** — the two
    tickers are fixed for the whole sample — so there is no decision to lag and none is
    claimed. What the arithmetic below computes (``r[long] - r[short]`` month by month) is
    a book **restored to equal notional at every month end**.

    Friction. ``cost_bps`` is charged one-way on the **full** notional of both legs, once a
    calendar year (``reset='A'``, booked in January, one period after the December book
    date) or every month (``reset='M'``). Charging full notional annually is a deliberately
    **conservative stand-in**, not a model of the actual turnover: restoring equal notional
    only ever trades the legs' drift, ``|r_long - r_short| / 2`` per leg, which on this
    tape is 1.0–9.5%/yr of notional, i.e. **0.03–0.29 bp/yr** at 3 bp one-way against the
    **6.0 bp/yr** actually charged. The trade is charged 20–200x its true rebalancing cost;
    the answer is therefore not flattered by the cost model.

    The short leg pays ``borrow_bps`` per year of borrow, accrued monthly — the input that
    actually decides this trade, and the reason it is swept rather than assumed.

    Because the book is dollar-neutral and self-financing, its return is already an
    **excess-of-cash** quantity — the short proceeds earn the cash rate that the long
    leg's funding costs, and what remains is the fee/lending/composition wedge minus
    borrow. That is what makes the Sharpe here directly comparable to any other
    excess-of-cash Sharpe on the desk. (The standing caveat: a real short rebate lands
    *below* the cash rate, which the borrow sweep is there to cover.)
    """
    px = prices[[long, short]].dropna()
    m = px.resample("ME").last()
    r = m.pct_change().dropna()

    if reset == "A":
        is_reset = pd.Series(r.index.month == 12, index=r.index)
    elif reset == "M":
        is_reset = pd.Series(True, index=r.index)
    else:
        raise ValueError("reset must be 'A' or 'M'")
    # The December book date is charged in January (a cost convention, not a lag —
    # there is no signal in this trade to lag).
    applied = is_reset.shift(1).fillna(False).astype(float)

    gross = r[long] - r[short]
    borrow = borrow_bps * 1e-4 / MONTHS
    friction = applied * 2.0 * cost_bps * 1e-4      # both legs reset, one-way each
    return (gross - borrow - friction).rename(f"{long}_minus_{short}")


def pair_trade_stats(ret: pd.Series) -> dict:
    r = pd.Series(ret).dropna().astype(float)
    sd = r.std(ddof=1)
    return {
        "n_months": int(len(r)),
        "ann_bp": float(r.mean() * MONTHS * BP),
        "vol_pct": float(sd * np.sqrt(MONTHS) * 100.0),
        "sharpe": float(r.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan"),
        "t_hac": newey_west_t(r.to_numpy(), lags=6),
        "max_drawdown": float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()),
    }


def rebalance_turnover(prices: pd.DataFrame, long: str, short: str,
                       cost_bps: float = 3.0) -> dict:
    """What the monthly equal-notional restore *actually* costs, vs what is charged.

    Restoring equal notional trades only the legs' drift: ``|r_long - r_short| / 2`` per
    leg, so total traded notional per month is ``|r_long - r_short|``. This makes the
    conservatism of ``pair_trade``'s annual full-notional charge auditable instead of
    asserted.
    """
    px = prices[[long, short]].dropna()
    r = px.resample("ME").last().pct_change().dropna()
    drift = (r[long] - r[short]).abs()
    turnover_ann = float(drift.mean() * MONTHS)
    return {
        "pair": f"{long}/{short}",
        "mean_monthly_drift_pct": float(drift.mean() * 100.0),
        "turnover_ann_pct": turnover_ann * 100.0,
        "true_cost_bp_yr": turnover_ann * cost_bps,
        "charged_bp_yr": 2.0 * cost_bps,
    }


def borrow_sweep(prices: pd.DataFrame, long: str, short: str,
                 borrow_grid=(0.0, 25.0, 50.0, 100.0), cost_bps: float = 3.0) -> list:
    """The decisive sweep: how much borrow does the fee gap survive?"""
    out = []
    for bb in borrow_grid:
        s = pair_trade_stats(pair_trade(prices, long, short, borrow_bps=bb, cost_bps=cost_bps))
        s["borrow_bps"] = bb
        out.append(s)
    return out


def excess_of_cash_sharpe(prices: pd.DataFrame, ticker: str, cash: str = "BIL") -> dict:
    """Each leg's own excess-of-cash Sharpe, for context (monthly, total return)."""
    px = prices[[ticker, cash]].dropna()
    r = px.resample("ME").last().pct_change().dropna()
    e = (r[ticker] - r[cash]).astype(float)
    sd = e.std(ddof=1)
    return {"ticker": ticker, "n_months": int(len(e)),
            "excess_sharpe": float(e.mean() / sd * np.sqrt(MONTHS)) if sd > 0 else float("nan"),
            "ann_excess_pct": float(e.mean() * MONTHS * 100.0)}


# --------------------------------------------------------------------------- #
# Pooling across pairs and the intensity check
# --------------------------------------------------------------------------- #
def pooled_residual(prices: pd.DataFrame, pairs, expense_ratios,
                    classes=None, freq: str = "M", seed: int = 914) -> dict:
    """Equal-weight pool of the per-pair residual series (optionally one asset class set).

    Pooling is the only way this tape gains power: each pair's tracking error is far
    larger than any plausible lending revenue, so a single pair cannot resolve the claim.
    """
    series = []
    used = []
    for a, b, klass, _note in pairs:
        if classes is not None and klass not in classes:
            continue
        r = pair_residual(prices, a, b, expense_ratios[a], expense_ratios[b],
                          freq=freq, n_boot=1)
        series.append(r["resid_series"].rename(f"{a}-{b}"))
        used.append(f"{a}-{b}")
    if not series:
        return {"n_pairs": 0}
    # Pool only over the window where EVERY pair is live, so the pooled series is not
    # secretly one pair in its early years.
    pooled = pd.concat(series, axis=1).dropna().mean(axis=1)
    per_year = MONTHS if freq == "M" else TRADING_DAYS
    ci = block_bootstrap_mean_ci(pooled.to_numpy(), block=6 if freq == "M" else 21, seed=seed)
    te = float(pooled.std(ddof=1) * np.sqrt(per_year))
    years = len(pooled) / per_year
    return {
        "pairs": used, "n_pairs": len(used), "n_obs": int(len(pooled)),
        "start": str(pooled.index[0].date()), "end": str(pooled.index[-1].date()),
        "residual_bp": float(pooled.mean() * per_year * BP),
        "t_hac": newey_west_t(pooled.to_numpy(), lags=6 if freq == "M" else None),
        "ci_low_bp": float(ci["ci_low"] * per_year * BP),
        "ci_high_bp": float(ci["ci_high"] * per_year * BP),
        "te_pct": te * 100.0,
        "noise_floor_bp": float(2.0 * te / np.sqrt(years) * BP) if years > 0 else float("nan"),
    }


def upper_bound_bp(res: dict, side: str = "low") -> float:
    """The bound the tape actually delivers: the CI edge on the unexplained residual.

    With a residual indistinguishable from zero, the honest output is not an estimate but
    a **bound** — "any passthrough visible in returns is no larger than this".
    """
    return abs(res["ci_low_bp"] if side == "low" else res["ci_high_bp"])


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, truth: dict, freq: str = "M") -> dict:
    """Recover the planted lending differential from a synthetic pair.

    On the planted world the residual must land near ``truth['planted_residual_bp']`` with
    |t| >= 2; on the null (``signal_strength=0``) it must be centred on zero. Proves the
    estimator is unbiased — it never supports a real-tape stamp.
    """
    r = pair_residual(prices.rename(columns={"fund_a": "A", "fund_b": "B"}),
                      "A", "B", truth["er_a"], truth["er_b"], freq=freq, n_boot=800)
    r.pop("resid_series")
    r["planted_bp"] = truth["planted_residual_bp"]
    r["error_bp"] = r["residual_bp"] - truth["planted_residual_bp"]
    return r


def synthetic_panel_detect(frames: dict, truth: dict, freq: str = "M") -> dict:
    """Pool the synthetic panel's residuals — the power gain from pooling, quantified."""
    per_year = MONTHS if freq == "M" else TRADING_DAYS
    series = []
    ers = dict(zip(range(len(truth["expense_ratios"])), truth["expense_ratios"]))
    for i, (name, px) in enumerate(sorted(frames.items())):
        er_a, er_b = ers[i]
        r = pair_residual(px.rename(columns={"fund_a": "A", "fund_b": "B"}),
                          "A", "B", er_a, er_b, freq=freq, n_boot=1)
        series.append(r["resid_series"].rename(name))
    pooled = pd.concat(series, axis=1).mean(axis=1).dropna()
    return {
        "n_pairs": len(series), "n_obs": int(len(pooled)),
        "residual_bp": float(pooled.mean() * per_year * BP),
        "planted_bp": truth["planted_residual_bp"],
        "t_hac": newey_west_t(pooled.to_numpy(), lags=6 if freq == "M" else None),
    }
