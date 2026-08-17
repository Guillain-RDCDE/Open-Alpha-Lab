"""Strategy + inference for Study 932 — the pre-deal SPAC trust yield.

The claim: a pre-deal SPAC is a T-bill portfolio in a box. Every public share carries a
right to redeem for its pro-rata share of the trust (~$10 plus accrued interest) at the
deal vote or at liquidation. So a SPAC quoted **below** trust is a T-bill bought at a
discount, and the discount, annualised over the time left to the redemption date, is a
yield **on top of** the bill yield — with the deal upside thrown in for free.

The rule tested here is the flat, unglamorous version of that trade:

1. At each month-end, look at every SPAC still in its pre-deal window with between 30
   and 730 days left to the (buffered) redemption date.
2. Buy the ones quoted below their accrued trust value, at the **next** day's close
   (one execution lag — the signal is formed on the month-end close, acted on at t+1).
3. Hold to the redemption date and take the trust value. Never sell into the market;
   never hold through the de-SPAC. No shorting anywhere, so no borrow leg.

Two books are measured, because they answer different questions:

- **accrual** — the hold-to-redemption book. Each position pulls from its entry price to
  trust value along its own implied yield. This is what the arb actually earns if the
  deadline holds.
- **mark-to-market** — the same positions marked at the daily quote, with the final day
  snapped to the trust value. This is what the account statement shows on the way, and
  it is the one that carries gap risk.

Both are raced **excess-of-cash** against BIL's total return over the identical windows,
gross and net of a one-way entry cost charged on NAV. Redemption itself is at trust and
costs nothing extra.

Inference: a one-sample HAC (Newey-West) *t* on the position-level excess returns, a
**cluster bootstrap over SPACs** (the honest CI, because monthly entries in the same name
are the same bet), a one-position-per-name variant for near-independent observations, an
era cut on entry year, and sweeps over every hardcoded assumption in ``data.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    BUFFER_DAYS,
    DAYS_PER_YEAR,
    DEAL_CLOSE,
    MAX_DISCOUNT,
    SPACS,
    TRADING_DAYS_PER_YEAR,
    TRUST0,
)

TRADING_DAYS = TRADING_DAYS_PER_YEAR


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


def newey_west_t(x, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    var = float(u @ u) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        var += 2.0 * w * float(u[lag:] @ u[:-lag]) / n
    if var <= 0:
        return float("nan")
    return float(mu / np.sqrt(var / n))


def auto_lags(n: int) -> int:
    return max(1, int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0))))


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Headline annualised stats for a daily simple-return series (pass excess-of-cash)."""
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    if n < 3:
        return {"n_days": n, "cagr": float("nan"), "sharpe": float("nan"),
                "vol_ann": float("nan"), "max_drawdown": float("nan"),
                "mean_daily_bps": float("nan"), "tstat": float("nan")}
    mu, sd = r.mean(), r.std(ddof=1)
    wealth = (1.0 + r).cumprod()
    years = n / periods_per_year
    return {
        "n_days": int(n),
        "cagr": float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if wealth.iloc[-1] > 0 else float("nan"),
        "sharpe": float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": newey_west_t(r.to_numpy(), lags=auto_lags(n)),
    }


# --------------------------------------------------------------------------- #
# The trust path (an accrual PROXY, swept)
# --------------------------------------------------------------------------- #
def trust_path(cash: pd.Series, start, end, trust0: float = TRUST0,
               fee_bps: float = 0.0) -> pd.Series:
    """Per-share trust value from ``start`` to ``end``, accrued at the cash leg.

    PROXY: the trust is assumed to earn the cash ETF's total return from ``start``, less
    a constant ``fee_bps`` a year of fees / franchise tax. Real trusts hold T-bills or a
    government money-market fund, so this tracks closely, but it is not a filed number —
    which is why ``trust0`` and ``fee_bps`` are both swept.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    c = pd.Series(cash).astype(float).dropna()
    c = c[(c.index >= start) & (c.index <= end)]
    if len(c) == 0:
        return pd.Series(dtype=float)
    yrs = (c.index - c.index[0]).days.to_numpy() / DAYS_PER_YEAR
    drag = (1.0 - fee_bps * 1e-4) ** yrs
    return pd.Series(trust0 * (c.to_numpy() / c.iloc[0]) * drag, index=c.index, name="trust")


def implied_ytr(price: float, trust_terminal: float, days: float) -> float:
    """Annualised yield-to-redemption implied by a price, a trust value and a horizon."""
    if price <= 0 or days <= 0:
        return float("nan")
    return float((trust_terminal / price) ** (DAYS_PER_YEAR / days) - 1.0)


# --------------------------------------------------------------------------- #
# Position construction
# --------------------------------------------------------------------------- #
def spac_windows(closes: pd.DataFrame, spacs=SPACS, buffer_days: int = BUFFER_DAYS) -> pd.DataFrame:
    """Per-SPAC pre-deal window table: first tape date, assumed redemption deadline, n obs.

    The deadline is the hardcoded deal-close date minus ``buffer_days``: the redemption
    *election* is due before the vote, and the vote before the close. Beyond it the trust
    put is gone and the quote is a bet on the target, not a bill.
    """
    rows = []
    for tk, sym, close in spacs:
        if tk not in closes.columns:
            continue
        px = closes[tk].dropna()
        deadline = pd.Timestamp(close) - pd.Timedelta(days=buffer_days)
        pre = px[px.index <= deadline]
        if len(pre) == 0:
            continue
        rows.append({
            "ticker": tk, "spac": sym, "deal_close": pd.Timestamp(close),
            "deadline": deadline, "first": pre.index[0], "last": pre.index[-1],
            "n_obs": int(len(pre)), "px_first": float(pre.iloc[0]),
            "px_last": float(pre.iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("ticker")


def build_positions(
    closes: pd.DataFrame,
    cash: pd.Series,
    spacs=SPACS,
    trust0: float = TRUST0,
    buffer_days: int = BUFFER_DAYS,
    fee_bps: float = 0.0,
    max_discount: float = MAX_DISCOUNT,
    min_days: int = 30,
    max_days: int = 730,
    cost_bps: float = 15.0,
    min_obs: int = 60,
    one_per_spac: bool = False,
    market_floor: bool = False,
    payoff: dict | None = None,
) -> pd.DataFrame:
    """Every buy-below-trust position the rule would have taken, with its outcome.

    Signals are formed on **month-end closes** and executed at the **next** trading close
    (one lag). ``cost_bps`` is charged one-way on NAV at entry; the redemption is at trust
    and free. Quotes implying a discount deeper than ``max_discount`` are treated as
    untradable (stale print / expired put) and skipped — that guard is swept.

    ``payoff`` overrides what a redeemed share actually pays at the deadline (default: the
    accrued trust). It exists for the **synthetic control only**, where the null world is
    one in which the trust put does not bind — never for the real tape.

    ``market_floor`` is the adversarial payoff: pay the **worse** of the assumed accrued
    trust and the deadline-day market quote. The assumed trust ($10.00 accreted at BIL) is
    the load-bearing guess in this study, and on 41% of positions the market's own closing
    quote on the deadline sat *below* it — so this setting lets the tape veto the
    assumption name by name instead of trusting it. It is deliberately too harsh (the
    redemption right pays trust whatever the quote says), which is the point of a floor.

    Returns one row per position with the entry price, the accrued trust at entry, the
    terminal trust, the raw and excess-of-cash return over the identical window, and both
    the implied and the realised annualised yield.
    """
    cash = pd.Series(cash).astype(float).dropna()
    rows = []
    for tk, sym, close in spacs:
        if tk not in closes.columns:
            continue
        px = closes[tk].dropna()
        deadline = pd.Timestamp(close) - pd.Timedelta(days=buffer_days)
        pre = px[px.index <= deadline]
        if len(pre) < min_obs:
            continue
        trust = trust_path(cash, pre.index[0], deadline, trust0=trust0, fee_bps=fee_bps)
        trust = trust.reindex(pre.index).ffill().dropna()
        pre = pre.reindex(trust.index)
        if len(pre) < min_obs:
            continue
        trust_T = float(trust.iloc[-1]) if payoff is None else float(payoff[tk])
        exit_date = trust.index[-1]
        px_exit_mkt = float(pre.iloc[-1])
        if market_floor:
            trust_T = min(trust_T, px_exit_mkt)

        # Month-end signal dates, executed at the next available close.
        month_ends = pre.groupby([pre.index.year, pre.index.month]).tail(1).index
        taken_one = False
        for sig_date in month_ends:
            loc = pre.index.get_loc(sig_date)
            if loc + 1 >= len(pre):
                continue
            exec_date = pre.index[loc + 1]
            days = float((exit_date - exec_date).days)
            if days < min_days or days > max_days:
                continue
            p_sig = float(pre.iloc[loc])
            t_sig = float(trust.iloc[loc])
            disc_sig = t_sig / p_sig - 1.0            # the signal, formed at t
            if not (0.0 < disc_sig <= max_discount):
                continue
            p_exec = float(pre.iloc[loc + 1])
            if p_exec <= 0:
                continue
            entry = p_exec * (1.0 + cost_bps * 1e-4)   # one-way cost x NAV, on entry
            ret = trust_T / entry - 1.0
            # Two alternatives to redeeming, both paying the cost on the way out too:
            #   ret_sell — sell into the deadline-day quote. This is the ONLY payoff read
            #     on this table that assumes nothing about the trust: it is a tape price.
            #   ret_best — take the better of the trust and that quote. An upper bound,
            #     since redeeming always forfeits any deal premium on offer.
            px_sell = px_exit_mkt * (1.0 - cost_bps * 1e-4)
            ret_sell = px_sell / entry - 1.0
            ret_best = max(trust_T, px_sell) / entry - 1.0
            cash_ret = float(cash.asof(exit_date) / cash.asof(exec_date) - 1.0)
            rows.append({
                "ticker": tk, "spac": sym, "signal_date": sig_date, "entry_date": exec_date,
                "exit_date": exit_date, "days": days, "years": days / DAYS_PER_YEAR,
                "px_signal": p_sig, "px_entry": p_exec, "px_entry_net": entry,
                "trust_signal": t_sig,
                "trust_exit": trust_T, "discount_signal": disc_sig,
                "discount_entry": t_sig / p_exec - 1.0,
                "implied_ytr": implied_ytr(p_exec, trust_T, days),
                "ret": ret,
                "px_exit_mkt": px_exit_mkt,
                "ret_sell": ret_sell,
                "excess_sell": ret_sell - cash_ret,
                "ret_best": ret_best,
                "excess_best": ret_best - cash_ret,
                "cash_ret": cash_ret,
                "excess": ret - cash_ret,
                "excess_ann": (1.0 + ret) ** (DAYS_PER_YEAR / days)
                              - (1.0 + cash_ret) ** (DAYS_PER_YEAR / days),
                "year": int(exec_date.year),
            })
            taken_one = True
            if one_per_spac and taken_one:
                break
    cols = ["ticker", "spac", "signal_date", "entry_date", "exit_date", "days", "years",
            "px_signal", "px_entry", "px_entry_net", "trust_signal", "trust_exit",
            "discount_signal", "discount_entry", "implied_ytr", "ret", "px_exit_mkt",
            "ret_sell", "excess_sell", "ret_best", "excess_best", "cash_ret", "excess",
            "excess_ann", "year"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("entry_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Daily books (accrual and mark-to-market), both excess-of-cash
# --------------------------------------------------------------------------- #
def portfolio_daily(
    positions: pd.DataFrame,
    closes: pd.DataFrame,
    cash: pd.Series,
    mode: str = "accrual",
) -> pd.Series:
    """Equal-weighted daily return of the open book.

    ``accrual`` values each position by geometric pull-to-trust from entry price to trust
    value across its own holding window (what a hold-to-redemption arb earns if the
    deadline holds). ``mtm`` values it at the daily quote, snapping the final day to the
    trust value (what the statement shows, gap risk included). Both are equal-weighted
    across whatever is open that day; days with no position earn the cash leg.

    Both books are **net of the entry cost**: the cost basis is ``px_entry_net`` (the fill
    grossed up by ``cost_bps``), not the raw fill, so a book run at 200 bps really is worse
    than the same book run gross.
    """
    if len(positions) == 0:
        return pd.Series(dtype=float)
    cash = pd.Series(cash).astype(float).dropna()
    all_days = cash.index
    lo = positions["entry_date"].min()
    hi = positions["exit_date"].max()
    days = all_days[(all_days >= lo) & (all_days <= hi)]
    acc = pd.DataFrame(index=days, columns=range(len(positions)), dtype=float)

    for i, row in positions.reset_index(drop=True).iterrows():
        win = days[(days > row["entry_date"]) & (days <= row["exit_date"])]
        if len(win) == 0:
            continue
        # The cost basis, not the raw fill — otherwise the books would be gross of cost
        # while the position table is net of it.
        basis = float(row.get("px_entry_net", row["px_entry"]))
        if mode == "accrual":
            total = row["trust_exit"] / basis - 1.0
            per = (1.0 + total) ** (1.0 / len(win)) - 1.0
            acc.loc[win, i] = per
        else:
            px = closes[row["ticker"]].dropna()
            marks = px.reindex(days).ffill().loc[win].to_numpy(dtype=float)
            marks = np.where(np.isfinite(marks), marks, float(row["px_entry"]))
            # The last mark is the redemption itself: the trust, not the quote. The rule
            # redeems unconditionally, so any premium on offer that day is forfeited.
            marks[-1] = float(row["trust_exit"])
            prev = np.concatenate(([basis], marks[:-1]))
            acc.loc[win, i] = marks / prev - 1.0

    port = acc.mean(axis=1, skipna=True)
    r_cash = cash.reindex(days).pct_change()
    port = port.fillna(r_cash)
    return (port - r_cash).dropna().rename(f"excess_{mode}")


# --------------------------------------------------------------------------- #
# Cluster bootstrap over SPACs (the honest CI given overlapping monthly entries)
# --------------------------------------------------------------------------- #
def cluster_bootstrap_mean(
    positions: pd.DataFrame,
    column: str = "excess",
    n_boot: int = 4000,
    seed: int = 932,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI for the mean of ``column``, resampling **whole SPACs**.

    Monthly entries in the same shell are the same bet on the same deadline, so the
    independent unit is the name, not the position. Resampling names is the conservative
    read and is the CI the verdict leans on.
    """
    if len(positions) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_clusters": 0, "n_obs": 0}
    g = positions.groupby("ticker")[column]
    sums = g.sum().to_numpy(dtype=float)
    counts = g.size().to_numpy(dtype=float)
    k = len(sums)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(n_boot, k))
    draws = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(positions[column].mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "frac_negative": float((draws < 0).mean()),
        "n_clusters": int(k), "n_obs": int(len(positions)), "n_boot": int(n_boot),
    }


def bootstrap_sharpe_ci(excess: pd.Series, n_boot: int = 2000, block: int = 21,
                        seed: int = 932, periods_per_year: int = TRADING_DAYS,
                        alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI for the annualised Sharpe of a daily excess series."""
    r = np.asarray(pd.Series(excess).dropna(), dtype=float)
    n = r.size
    if n < block + 2:
        return {"sharpe": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "frac_negative": float("nan"), "n_obs": n}
    ann = np.sqrt(periods_per_year)
    sd = r.std(ddof=1)
    point = float(r.mean() / sd * ann) if sd > 0 else float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        s = r[idx]
        sdb = s.std(ddof=1)
        if sdb > 0:
            boots[b] = s.mean() / sdb * ann
    valid = boots[np.isfinite(boots)]
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": point, "ci_low": float(lo), "ci_high": float(hi),
            "frac_negative": float((valid < 0).mean()), "n_obs": n,
            "n_boot": int(valid.size), "block": block}


# --------------------------------------------------------------------------- #
# What the headline actually is — and the one thing the tape can say about it
# --------------------------------------------------------------------------- #
def identity_residual(positions: pd.DataFrame) -> dict:
    """How close the headline excess is to being **pure arithmetic**.

    Because the modelled payoff *is* the accrued trust and the benchmark *is* the same
    cash index the trust accrues at, the position excess collapses algebraically to

        excess  =  (1 + cash_ret) x (trust_at_signal / cost-loaded entry price - 1)

    i.e. to the **entry discount**, grossed up by the cash return. Nothing about the
    redemption actually happening is measured — it is imposed by the ``payoff`` model. So
    the position-level *t*, the hit rate and the cluster CI are statistics about how
    reliably a selected discount was still there one day later, **not** evidence that the
    trust paid. This function reports the residual so the claim is checkable, not rhetorical.
    """
    if len(positions) == 0:
        return {"max_abs_residual": float("nan"), "corr_with_discount": float("nan"), "n": 0}
    implied = (1.0 + positions["cash_ret"]) * (
        positions["trust_signal"] / positions["px_entry_net"] - 1.0)
    resid = (positions["excess"] - implied).abs()
    return {
        "max_abs_residual": float(resid.max()),
        "corr_with_discount": float(np.corrcoef(positions["excess"],
                                                positions["discount_entry"])[0, 1]),
        "n": int(len(positions)),
    }


def trust_anchor_check(positions: pd.DataFrame) -> pd.DataFrame:
    """Per-SPAC: where the **market** was on the deadline, against the assumed trust line.

    This is the study's only real-tape test of its own load-bearing assumption. The payoff
    model says a share was worth the accrued trust at the deadline; the tape says what the
    share was quoted at on the same day. If the quotes had sat well *below* the assumed
    line, the $10.00-accreted-at-BIL trust would be a fiction and the headline with it.
    """
    if len(positions) == 0:
        return pd.DataFrame(columns=["spac", "trust_exit", "px_exit_mkt", "gap"])
    one = positions.groupby("ticker").first()
    out = pd.DataFrame({
        "spac": one["spac"],
        "trust_exit": one["trust_exit"],
        "px_exit_mkt": one["px_exit_mkt"],
        "gap": one["px_exit_mkt"] / one["trust_exit"] - 1.0,
    })
    return out.sort_values("gap")


# --------------------------------------------------------------------------- #
# The headline race
# --------------------------------------------------------------------------- #
def evaluate(positions: pd.DataFrame, closes: pd.DataFrame, cash: pd.Series,
             seed: int = 932) -> dict:
    """Position-level and daily-book statistics for one parameter set."""
    if len(positions) == 0:
        return {"n_pos": 0, "n_spacs": 0}
    exc = positions["excess"].to_numpy(dtype=float)
    e_acc = portfolio_daily(positions, closes, cash, mode="accrual")
    e_mtm = portfolio_daily(positions, closes, cash, mode="mtm")
    return {
        "n_pos": int(len(positions)),
        "n_spacs": int(positions["ticker"].nunique()),
        "mean_excess": float(np.mean(exc)),
        "median_excess": float(np.median(exc)),
        "hit_rate": float((exc > 0).mean()),
        "mean_excess_ann": float(positions["excess_ann"].mean()),
        "mean_hold_days": float(positions["days"].mean()),
        "mean_discount": float(positions["discount_entry"].mean()),
        "mean_implied_ytr": float(positions["implied_ytr"].mean()),
        "t_pos": one_sample_t(exc),
        "t_pos_hac": newey_west_t(exc, lags=auto_lags(len(exc))),
        "mean_excess_best": float(positions["excess_best"].mean()),
        "t_pos_best": one_sample_t(positions["excess_best"].to_numpy(dtype=float)),
        "mean_excess_sell": float(positions["excess_sell"].mean()),
        "t_pos_sell": one_sample_t(positions["excess_sell"].to_numpy(dtype=float)),
        "boot": cluster_bootstrap_mean(positions, seed=seed),
        "boot_best": cluster_bootstrap_mean(positions, column="excess_best", seed=seed),
        "boot_sell": cluster_bootstrap_mean(positions, column="excess_sell", seed=seed),
        "identity": identity_residual(positions),
        "anchor": trust_anchor_check(positions),
        "acc": summary(e_acc),
        "mtm": summary(e_mtm),
        "e_acc": e_acc,
        "e_mtm": e_mtm,
    }


def race(closes: pd.DataFrame, cash: pd.Series, **kw) -> dict:
    """Build positions with the given assumptions and evaluate them."""
    seed = kw.pop("seed", 932)
    pos = build_positions(closes, cash, **kw)
    out = evaluate(pos, closes, cash, seed=seed)
    out["positions"] = pos
    return out


# --------------------------------------------------------------------------- #
# Era cut, sweeps, and the cross-sectional yield path
# --------------------------------------------------------------------------- #
def era_cut(positions: pd.DataFrame, split_year: int = 2022) -> dict:
    """Split positions by *entry* year: the zero-rate era vs the hiking era."""
    out = {}
    for tag, mask in [("pre", positions["year"] < split_year),
                      ("post", positions["year"] >= split_year)]:
        sub = positions[mask]
        if len(sub) < 3:
            out[tag] = None
            continue
        exc = sub["excess"].to_numpy(dtype=float)
        out[tag] = {
            "n_pos": int(len(sub)), "n_spacs": int(sub["ticker"].nunique()),
            "mean_excess": float(exc.mean()),
            "mean_excess_ann": float(sub["excess_ann"].mean()),
            "hit_rate": float((exc > 0).mean()),
            "t_pos": one_sample_t(exc),
            "boot": cluster_bootstrap_mean(sub),
            "mean_discount": float(sub["discount_entry"].mean()),
            "mean_hold_days": float(sub["days"].mean()),
        }
    return out


def sweep(closes: pd.DataFrame, cash: pd.Series, param: str, grid, **base) -> list[dict]:
    """Re-run the headline across a grid of one hardcoded assumption."""
    rows = []
    for v in grid:
        kw = dict(base)
        kw[param] = v
        pos = build_positions(closes, cash, **kw)
        if len(pos) == 0:
            rows.append({param: v, "n_pos": 0})
            continue
        exc = pos["excess"].to_numpy(dtype=float)
        boot = cluster_bootstrap_mean(pos)
        rows.append({
            param: v, "n_pos": int(len(pos)), "n_spacs": int(pos["ticker"].nunique()),
            "mean_excess": float(exc.mean()),
            "mean_excess_ann": float(pos["excess_ann"].mean()),
            "hit_rate": float((exc > 0).mean()),
            "t_pos": one_sample_t(exc),
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        })
    return rows


def yield_path(closes: pd.DataFrame, cash: pd.Series, spacs=SPACS,
               trust0: float = TRUST0, buffer_days: int = BUFFER_DAYS,
               fee_bps: float = 0.0, min_days: int = 30, max_days: int = 730,
               max_discount: float = MAX_DISCOUNT) -> pd.DataFrame:
    """Month-by-month cross-section of the implied yield-to-redemption.

    For each month-end, across every SPAC still in its pre-deal window with a usable
    horizon: the count, the median discount to trust, and the median implied annualised
    yield-to-redemption. This is the 'trust yield' curve the README's chart shows.
    """
    cash = pd.Series(cash).astype(float).dropna()
    recs = []
    for tk, _sym, close in spacs:
        if tk not in closes.columns:
            continue
        px = closes[tk].dropna()
        deadline = pd.Timestamp(close) - pd.Timedelta(days=buffer_days)
        pre = px[px.index <= deadline]
        if len(pre) < 60:
            continue
        trust = trust_path(cash, pre.index[0], deadline, trust0=trust0, fee_bps=fee_bps)
        trust = trust.reindex(pre.index).ffill().dropna()
        pre = pre.reindex(trust.index)
        trust_T = float(trust.iloc[-1])
        exit_date = trust.index[-1]
        month_ends = pre.groupby([pre.index.year, pre.index.month]).tail(1).index
        for d in month_ends:
            days = float((exit_date - d).days)
            if days < min_days or days > max_days:
                continue
            p = float(pre.loc[d])
            disc = float(trust.loc[d]) / p - 1.0
            if abs(disc) > max(max_discount, 0.5):
                continue
            recs.append({"month": d.to_period("M"), "ticker": tk, "discount": disc,
                         "ytr": implied_ytr(p, trust_T, days), "days": days,
                         "below": disc > 0})
    if not recs:
        return pd.DataFrame(columns=["n_live", "n_below", "median_discount", "median_ytr"])
    df = pd.DataFrame(recs)
    g = df.groupby("month")
    out = pd.DataFrame({
        "n_live": g.size(),
        "n_below": g["below"].sum(),
        "median_discount": g["discount"].median(),
        "median_ytr": g["ytr"].median(),
    })
    below = df[df["below"]].groupby("month")["ytr"].median().rename("median_ytr_below")
    return out.join(below)


# --------------------------------------------------------------------------- #
# Synthetic control (the machinery proof — never supports the stamp)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, cash: pd.Series, meta: pd.DataFrame,
                     cost_bps: float = 15.0, max_discount: float = 0.25,
                     seed: int = 932, n_boot: int = 2000) -> dict:
    """Run the buy-below-trust rule on a synthetic panel.

    In the world where the trust put binds the rule must earn a clearly positive excess
    over cash; in the world where the put is a fiction and the terminal payoff is just the
    last quote, it must earn nothing. Proves the harness is unbiased — it never supports a
    real-tape stamp.
    """
    spacs = tuple((tk, tk, str((meta.loc[tk, "deadline"] + pd.Timedelta(days=1)).date()))
                  for tk in meta.index)
    pay = meta["payoff"].to_dict() if "payoff" in meta.columns else None
    pos = build_positions(prices, cash, spacs=spacs, buffer_days=1, payoff=pay,
                          cost_bps=cost_bps, max_discount=max_discount, min_obs=40)
    if len(pos) == 0:
        return {"n_pos": 0, "mean_excess": float("nan"), "t_pos": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    exc = pos["excess"].to_numpy(dtype=float)
    boot = cluster_bootstrap_mean(pos, seed=seed, n_boot=n_boot)
    return {
        "n_pos": int(len(pos)), "n_spacs": int(pos["ticker"].nunique()),
        "mean_excess": float(exc.mean()),
        "mean_excess_ann": float(pos["excess_ann"].mean()),
        "hit_rate": float((exc > 0).mean()),
        "t_pos": one_sample_t(exc),
        "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
        "mean_discount": float(pos["discount_entry"].mean()),
    }
