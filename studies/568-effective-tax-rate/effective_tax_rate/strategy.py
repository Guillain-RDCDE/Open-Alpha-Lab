"""Strategy for Study 568 (Effective-Tax-Rate) — the tax-rate return anomaly.

**The effective-tax-rate anomaly.** A firm's ETR (income-tax expense / pretax income) is read two
ways. The *quality* story: low-ETR firms are efficiently managed cash machines the market
underprices, so they earn *higher* future returns. The *risk* story: a suspiciously low ETR is a
fragile loophole that can reverse, so low-ETR firms earn *lower* returns. The **sign is the whole
question**, so this module reports it both directions and lets a placebo null and a robustness
sweep decide.

**Signal**::

    ETR   = income_tax_expense / pretax_income        (only when pretax_income > 0)
    dETR  = ETR_y - ETR_{y-1}                          (the change, a second sort key)

We sort the survivor basket into quintiles each fiscal year. The *headline* long-short is
**long low-ETR, short high-ETR** (Q1 - Q5 on the ETR level) — the quality/tax-avoidance framing;
a positive hedge means low-ETR firms out-earn. The sign, its t-stat, its placebo p-value and its
stability across windows are what carry the verdict.

**Honest controls.**

1. Equal-weight market: all names with a valid ETR that year.
2. Random-portfolio null: same-size random subsets, to ask whether the long leg's edge exceeds
   what concentration alone would give.
3. Label-shuffle placebo: permute the ETR labels *within each year*, breaking the signal->return
   link while preserving each year's marginal return distribution. A real edge should die.
4. Information coefficient: the cross-sectional Spearman rank correlation between ETR and the
   next-year return, averaged over years (a rank-IC), with a t-stat on its year series.

**Costs.** One-way bps x NAV x turnover, plus borrow on the short leg. Annual rebalance, so
turnover is modest. Gross and net are both reported.

**Reporting lag.** Fundamentals from fiscal year y predict calendar-year y+1 returns — a
conservative lag (the 10-K is not actionable until well into the following year). Exactly one
execution lag, no same-bar fills, no look-ahead.

**Survivorship bias.** The basket is a fixed set of names still trading in 2026. Every real-tape
number is an upper bound; named on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def etr_signal(
    fund: pd.DataFrame,
    min_pretax: float = 1e7,
    lo: float = -0.5,
    hi: float = 1.2,
) -> pd.DataFrame:
    """Build the effective-tax-rate ratio from a long fundamentals frame.

    ``fund`` has columns ``ticker, year, tax_expense, pretax_income``. Computes::

        ETR = tax_expense / pretax_income

    only where ``pretax_income >= min_pretax`` (the ratio is meaningless for near-zero or negative
    pretax income — a loss-making firm has no economic tax *rate*). ETRs outside ``[lo, hi]`` are
    winsorised to the bounds (one-off items can push a reported ratio absurdly negative or above
    100%). Returns a (year x ticker) DataFrame of ETRs (higher = pays more tax).
    """
    df = fund.copy()
    tax = df["tax_expense"].to_numpy(dtype=float)
    pre = df["pretax_income"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        etr = np.where(pre >= min_pretax, tax / pre, np.nan)
    etr = np.clip(etr, lo, hi)
    df["etr"] = etr
    panel = df.pivot_table(index="year", columns="ticker", values="etr")
    panel.index.name = "year"
    panel.columns.name = None
    return panel.sort_index()


def detr_signal(etr: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year change in ETR (dETR); NaN in the first available year per name."""
    return etr.diff()


# --------------------------------------------------------------------------- #
# Portfolio construction and returns
# --------------------------------------------------------------------------- #
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 15,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on a signal (default ETR level) vs next-year returns.

    Q1 = lowest signal (lowest ETR) — the long leg for the quality framing.
    Q5 = highest signal (highest ETR) — the short leg.
    ``hedge`` = Q1 - Q5 (long low-ETR, short high-ETR); positive = low-ETR firms out-earn.

    Returns a DataFrame with columns ``q1..q5``, ``market`` (equal-weight all), ``n``, and
    ``hedge``, indexed by signal year.
    """
    rows: dict[int, dict] = {}
    n_q = int(round(1 / q))
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue

        ranks = s.rank(method="first")
        bins = np.ceil(ranks / len(s) * n_q).astype(int).clip(1, n_q)
        row: dict = {"market": float(nxt.mean()), "n": int(len(s))}
        qrets = []
        for qi in range(1, n_q + 1):
            mask = (bins == qi).values
            qr = float(nxt[mask].mean()) if mask.any() else np.nan
            qrets.append(qr)
            row[f"q{qi}"] = qr
        row["hedge"] = qrets[0] - qrets[-1]  # long low-ETR, short high-ETR
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def turnover_series(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 15,
) -> pd.Series:
    """One-sided name turnover of the Q1 (long) basket from year to year."""
    n_q = int(round(1 / q))
    prev_long: set | None = None
    out: dict[int, float] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, _ = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue
        ranks = s.rank(method="first")
        bins = np.ceil(ranks / len(s) * n_q).astype(int).clip(1, n_q)
        long_names = set(s.index[(bins == 1).values])
        if prev_long is not None and len(prev_long) > 0:
            replaced = len(long_names - prev_long) / len(prev_long)
            out[int(y)] = float(replaced)
        prev_long = long_names
    return pd.Series(out, name="turnover").sort_index()


def apply_costs(
    hedge: pd.Series,
    turnover: pd.Series,
    one_way_bps: float = 10.0,
    borrow_bps: float = 50.0,
) -> pd.Series:
    """Net the hedge return for transaction costs and short-leg borrow.

    - Transaction cost per year = 2 legs x one_way_bps x turnover (both legs rebalance).
    - Borrow = borrow_bps/yr on the short leg (always carried, full NAV on the short side).

    Returns the net hedge series.
    """
    t = turnover.reindex(hedge.index).fillna(turnover.mean() if len(turnover) else 0.0)
    trade_cost = 2.0 * (one_way_bps / 1e4) * t
    borrow = borrow_bps / 1e4
    return (hedge - trade_cost - borrow).rename("hedge_net")


# --------------------------------------------------------------------------- #
# Information coefficient (rank-IC)
# --------------------------------------------------------------------------- #
def information_coefficient(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    min_names: int = 15,
) -> dict:
    """Year-by-year cross-sectional Spearman rank-IC between signal and next-year return.

    Returns ``{"mean_ic", "ic_t", "n_years", "series"}``. A *negative* mean IC means low-ETR
    firms out-earn (the quality framing); positive means high-ETR firms out-earn. The t-stat is a
    plain one-sample t on the annual IC series.
    """
    ics: dict[int, float] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue
        rs = s.rank()
        rn = nxt.rank()
        if rs.std(ddof=1) == 0 or rn.std(ddof=1) == 0:
            continue
        ics[int(y)] = float(np.corrcoef(rs, rn)[0, 1])
    ser = pd.Series(ics, name="ic").sort_index()
    if len(ser) < 2:
        return {"mean_ic": float("nan"), "ic_t": float("nan"), "n_years": len(ser), "series": ser}
    mu = float(ser.mean())
    se = float(ser.std(ddof=1) / np.sqrt(len(ser)))
    return {"mean_ic": mu, "ic_t": (mu / se if se > 0 else float("nan")),
            "n_years": len(ser), "series": ser}


# --------------------------------------------------------------------------- #
# Null distributions
# --------------------------------------------------------------------------- #
def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    min_names: int = 15,
    seed: int = 568,
) -> pd.Series:
    """Random-portfolio excess-vs-market null (long-leg size). One value per (year x draw)."""
    rng = np.random.default_rng(seed)
    excesses: list[float] = []
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue
        n_pick = max(1, int(round(len(s) * q)))
        ret_arr = nxt.to_numpy()
        mkt = ret_arr.mean()
        for _ in range(n_draws):
            pick = rng.choice(len(ret_arr), size=n_pick, replace=False)
            excesses.append(float(ret_arr[pick].mean() - mkt))
    return pd.Series(excesses, name="random_excess")


def placebo_hedge_t(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_perm: int = 1000,
    min_names: int = 15,
    seed: int = 568,
) -> tuple[float, np.ndarray]:
    """Label-shuffle placebo: permute ETR labels within each year.

    Breaks the signal->return link while preserving each year's marginal return distribution.
    Returns ``(p_value, null_tstats)`` where the p-value is the two-sided fraction of permuted
    hedge t-stats whose |t| meets or beats the real |t|. A real edge should die under the shuffle.
    """
    real_t = summary(quintile_returns(signal, fwd_ret, q=q, min_names=min_names)["hedge"])["tstat"]
    rng = np.random.default_rng(seed)
    null_t: list[float] = []
    for _ in range(n_perm):
        shuf = signal.copy()
        for y in shuf.index:
            row = shuf.loc[y].dropna()
            if len(row) < min_names:
                continue
            perm = rng.permutation(row.values)
            shuf.loc[y, row.index] = perm
        h = quintile_returns(shuf, fwd_ret, q=q, min_names=min_names)
        if "hedge" not in h or h["hedge"].dropna().empty:
            continue
        null_t.append(summary(h["hedge"])["tstat"])
    null_arr = np.array([t for t in null_t if np.isfinite(t)])
    if null_arr.size == 0 or not np.isfinite(real_t):
        return float("nan"), null_arr
    p = float((np.abs(null_arr) >= abs(real_t)).mean())
    return p, null_arr


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def summary(annual_returns: pd.Series) -> dict:
    """Headline stats for an annual return series (HAC Newey-West t-stat, Sharpe, hit rate)."""
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {"mean": mu, "vol": std, "sharpe": sr, "tstat": tstat,
            "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": n}


def market_annual(fwd_ret: pd.DataFrame, years: list[int] | None = None) -> pd.Series:
    """Equal-weight annual market return across all tickers in the panel."""
    if years is not None:
        fwd_ret = fwd_ret.reindex(years)
    return fwd_ret.mean(axis=1).rename("market").dropna()


# --------------------------------------------------------------------------- #
# Robustness sweep
# --------------------------------------------------------------------------- #
def window_sweep(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    windows: list[tuple[int, int]],
    q: float = 0.20,
    min_names: int = 15,
) -> pd.DataFrame:
    """Hedge mean and t-stat over several (start_year, end_year) sub-samples.

    Reads the sign of the low-minus-high-ETR hedge in each window. A signal whose sign flips
    across windows is not bankable.
    """
    h_all = quintile_returns(signal, fwd_ret, q=q, min_names=min_names)["hedge"]
    rows = []
    for (a, b) in windows:
        sub = h_all[(h_all.index >= a) & (h_all.index <= b)]
        st = summary(sub)
        rows.append({"window": f"{a}-{b}", "mean%": st["mean"] * 100 if np.isfinite(st["mean"]) else np.nan,
                     "t": st["tstat"], "n_years": st["n"]})
    return pd.DataFrame(rows).set_index("window")


# --------------------------------------------------------------------------- #
# Synthetic positive control wrapper
# --------------------------------------------------------------------------- #
def synthetic_hedge_t(premium: float, seed: int = 568) -> float:
    """HAC t-stat of the Q1-Q5 (low-minus-high ETR) hedge on a synthetic panel with known premium.

    A faithful engine check: ``premium = 0`` should give |t| typically < 2; ``premium < 0``
    (low-ETR firms out-earn) should light the hedge up *positive*; ``premium > 0`` drives it
    negative. NEVER cited to support a real-tape stamp — only proves the machinery works.
    """
    from .data import synthetic_panel
    etr, fwd, _ = synthetic_panel(premium=premium, seed=seed)
    h = quintile_returns(etr, fwd, q=0.20, min_names=15)
    return summary(h["hedge"])["tstat"]


def synthetic_mean_hedge_t(premium: float, n_seeds: int = 25, base_seed: int = 568) -> float:
    """Average the Q1-Q5 hedge t over ``n_seeds`` synthetic worlds (house rule: >= 20 seeds).

    So no single lucky RNG seed can manufacture significance. Returns the mean hedge-t across
    seeds for a planted ``premium``.
    """
    ts = [synthetic_hedge_t(premium, seed=s) for s in range(base_seed, base_seed + n_seeds)]
    return float(np.nanmean(ts))
