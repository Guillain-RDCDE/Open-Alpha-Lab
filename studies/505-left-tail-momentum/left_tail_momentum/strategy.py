"""The Left-Tail-Momentum strategy -- Atilgan, Bali, Demirtas & Gunaydin (2020).

The claim (JFE 2020): stocks with the *worst* left-tail risk -- the lowest 1%/5%
Value-at-Risk, or the worst recent crash day -- keep underperforming. Investors
under-react to extreme negative returns, so the left tail has *momentum*: crashed
stocks keep crashing. A cross-sectional book that goes **long the safe tail** (least
negative VaR) and **short the crashed tail** (most negative VaR) should earn a positive
spread.

Construction (monthly, on a fixed survivor basket):

  1. At each formation date, build a left-tail signal from the trailing ``lookback``
     trading days of each stock's daily returns. Three flavours:
       * ``var5``  -- the 5% empirical quantile (5th-percentile daily return).
       * ``var1``  -- the 1% empirical quantile.
       * ``worst`` -- the single worst daily return in the window.
     All three are *negative* numbers; a *more negative* value = a worse left tail.
  2. Sort stocks by the signal. Split into quintiles (or a median split if the basket
     is small). Long the **best-tail** group (signal nearest zero = "safe"), short the
     **worst-tail** group (most negative = "crashed").
  3. Enter the close ONE trading day after the formation date (one execution lag, no
     same-bar fill), hold for the next calendar month, equal-weight each leg.

Inference: one-sample t on the monthly long-short spread (the house bar is |t| >= 2),
a placebo label-shuffle null (re-randomise the cross-sectional ranking and re-run),
seed-robustness on the synthetic control, and costs = one-way bps x turnover (+ borrow
on the short leg). Gross vs net is labelled.

The basket is **survivorship-biased** (names still trading in 2026); we name it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
DEFAULT_LOOKBACK = 252            # trailing window for the left-tail signal (1 year)
DEFAULT_QUANTILE_GROUPS = 5       # quintile sort when the basket is large enough


# ---------------------------------------------------------------------------
# The left-tail signal
# ---------------------------------------------------------------------------
def left_tail_signal(window_rets: pd.DataFrame, kind: str = "var5") -> pd.Series:
    """Left-tail risk per stock over a window of daily returns.

    ``window_rets`` is a (day x ticker) frame of daily simple returns. Returns a Series
    (indexed by ticker) of the chosen left-tail statistic -- a *negative* number; a more
    negative value means a worse left tail (a deeper recent crash / lower VaR).

      * ``var5``  -- 5th-percentile daily return (95% VaR, sign-as-return).
      * ``var1``  -- 1st-percentile daily return (99% VaR).
      * ``worst`` -- the single worst daily return.
    """
    w = window_rets.dropna(how="all", axis=1)
    if kind == "var5":
        sig = w.quantile(0.05)
    elif kind == "var1":
        sig = w.quantile(0.01)
    elif kind == "worst":
        sig = w.min()
    else:
        raise ValueError(f"unknown signal kind: {kind!r}")
    return sig.dropna()


# ---------------------------------------------------------------------------
# Long-short book construction
# ---------------------------------------------------------------------------
def long_short_book(
    prices: pd.DataFrame,
    kind: str = "var5",
    lookback: int = DEFAULT_LOOKBACK,
    n_groups: int = DEFAULT_QUANTILE_GROUPS,
    min_stocks: int = 20,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
) -> pd.DataFrame:
    """Monthly long-safe / short-crashed left-tail-momentum book.

    For each month-end formation date *t*:
      1. Left-tail signal from the trailing ``lookback`` days up to *t*.
      2. Rank; the **long** leg is the best-tail group (signal nearest 0 -- "safe");
         the **short** leg is the worst-tail group (most negative -- "crashed").
         Quintiles when ``n_groups`` deciles cleanly; else a median split.
      3. Hold the NEXT calendar month, entering one trading day after *t*
         (one execution lag). Each leg is equal-weight, return = compounded daily.
      4. ``ls_ret`` = long_ret - short_ret. Net of costs: one-way ``cost_bps`` x
         turnover on each leg per rebalance, plus borrow on the short leg.

    Returns a DataFrame indexed by formation date with columns:
        ``long_ret``, ``short_ret``, ``ls_ret`` (gross), ``ls_ret_net``,
        ``turnover`` (fraction of names rotated vs prior month, two-sided/2),
        ``n`` (universe size that month).
    """
    rets = prices.pct_change().dropna(how="all")
    if rets.empty:
        return pd.DataFrame()

    # Month-end formation dates that leave a full trailing window behind them.
    month_ends = rets.resample("ME").last().index
    if len(rets) <= lookback:
        return pd.DataFrame()
    first_valid = rets.index[lookback]
    form_dates = [d for d in month_ends if first_valid <= d <= rets.index[-1]]

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    rows: list[dict] = []

    monthly_borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR

    for i, t in enumerate(form_dates[:-1]):
        t_next = form_dates[i + 1]

        window = rets.loc[rets.index <= t].tail(lookback)
        # Require enough non-missing observations per stock.
        valid = window.notna().sum() >= (lookback // 2)
        window = window.loc[:, valid[valid].index]
        sig = left_tail_signal(window, kind=kind)
        if len(sig) < min_stocks:
            continue

        sig_sorted = sig.sort_values()                # most negative first (worst tail)
        n = len(sig_sorted)
        if n >= n_groups * 3:
            grp = max(1, n // n_groups)
            short_names = list(sig_sorted.index[:grp])      # worst tail = short
            long_names = list(sig_sorted.index[-grp:])      # best tail  = long
        else:
            half = n // 2
            short_names = list(sig_sorted.index[:half])
            long_names = list(sig_sorted.index[half:])

        # Holding window: strictly AFTER t (one execution lag) up to and incl. t_next.
        hold = rets.loc[(rets.index > t) & (rets.index <= t_next)]
        if hold.empty:
            continue

        def _compound(cols: list[str]) -> float:
            sub = hold[cols].dropna(how="all")
            if sub.empty:
                return float("nan")
            per_name = (1.0 + sub).prod() - 1.0
            return float(per_name.mean())

        long_ret = _compound(long_names)
        short_ret = _compound(short_names)
        if np.isnan(long_ret) or np.isnan(short_ret):
            continue
        ls_ret = long_ret - short_ret

        # Turnover vs prior month (fraction of each leg rotated), averaged over legs.
        cur_long, cur_short = set(long_names), set(short_names)
        if prev_long or prev_short:
            to_long = len(cur_long - prev_long) / max(1, len(cur_long))
            to_short = len(cur_short - prev_short) / max(1, len(cur_short))
            turnover = 0.5 * (to_long + to_short)
        else:
            turnover = 1.0
        prev_long, prev_short = cur_long, cur_short

        # Cost: one-way bps x turnover on BOTH legs (long + short trade each rebalance).
        cost = 2.0 * cost_bps * 1e-4 * turnover
        ls_ret_net = ls_ret - cost - monthly_borrow

        rows.append(
            {
                "date": t,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": ls_ret,
                "ls_ret_net": ls_ret_net,
                "turnover": turnover,
                "n": int(n),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def one_sample_t(r: pd.Series) -> float:
    """Plain one-sample t-stat on the mean of a monthly return series (the house bar)."""
    x = pd.Series(r).astype(float).dropna().to_numpy()
    n = x.size
    if n < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(n))) if sd > 0 else float("nan")


def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series.

    Same lag rule ``floor(4*(n/100)^(2/9))`` the desk uses elsewhere.
    """
    x = pd.Series(r).astype(float).dropna().to_numpy()
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(monthly_returns: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised stats for a monthly return series.

    Returns annualised mean, vol, Sharpe, plain one-sample t (inference bar), HAC t,
    hit-rate, max drawdown and n.
    """
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hac_t", "hit_rate", "max_drawdown", "n")}
    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1))
    mean_ann = mean_m * periods_per_year
    vol_ann = vol_m * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": one_sample_t(r),
        "hac_t": hac_tstat(r),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    prices: pd.DataFrame,
    kind: str = "var5",
    lookback: int = DEFAULT_LOOKBACK,
    n_groups: int = DEFAULT_QUANTILE_GROUPS,
    min_stocks: int = 20,
    n_shuffles: int = 200,
    seed: int = 505,
) -> dict:
    """Label-shuffle placebo: how often does a *random* cross-sectional sort beat the
    real left-tail sort's mean spread?

    Each shuffle randomly relabels which stocks are "safe" vs "crashed" (i.e. the same
    long-short machinery, but the sort key is replaced by noise). The placebo p-value is
    the fraction of shuffles whose mean long-short return is at least as large (in the
    real-effect direction) as the real one. A real edge => low p.
    """
    real = long_short_book(prices, kind=kind, lookback=lookback,
                           n_groups=n_groups, min_stocks=min_stocks)
    if real.empty:
        return {"real_mean": float("nan"), "p_value": float("nan"), "n_shuffles": 0}
    real_mean = float(real["ls_ret"].mean())

    rets = prices.pct_change().dropna(how="all")
    month_ends = rets.resample("ME").last().index
    first_valid = rets.index[lookback]
    form_dates = [d for d in month_ends if first_valid <= d <= rets.index[-1]]

    # Precompute, ONCE per month, the per-name compounded holding return so the shuffle
    # loop is pure numpy permutation (no pandas slicing inside the inner loop).
    months: list[np.ndarray] = []   # each entry: per-name holding return vector (valid names)
    for i, t in enumerate(form_dates[:-1]):
        t_next = form_dates[i + 1]
        window = rets.loc[rets.index <= t].tail(lookback)
        valid = window.notna().sum() >= (lookback // 2)
        names = list(valid[valid].index)
        if len(names) < min_stocks:
            continue
        hold = rets.loc[(rets.index > t) & (rets.index <= t_next)]
        if hold.empty:
            continue
        comp = ((1.0 + hold[names]).prod() - 1.0).to_numpy(dtype=float)
        comp = comp[~np.isnan(comp)]
        if comp.size >= min_stocks:
            months.append(comp)

    if not months:
        return {"real_mean": real_mean, "p_value": float("nan"), "n_shuffles": 0}

    rng = np.random.default_rng(seed)
    shuffle_means = np.empty(n_shuffles)
    for s in range(n_shuffles):
        diffs = np.empty(len(months))
        for j, comp in enumerate(months):
            perm = rng.permutation(comp)
            n = perm.size
            if n >= n_groups * 3:
                grp = max(1, n // n_groups)
                short_ret = perm[:grp].mean()
                long_ret = perm[-grp:].mean()
            else:
                half = n // 2
                short_ret = perm[:half].mean()
                long_ret = perm[half:].mean()
            diffs[j] = long_ret - short_ret
        shuffle_means[s] = diffs.mean()

    if real_mean >= 0:
        p = float(np.mean(shuffle_means >= real_mean))
    else:
        p = float(np.mean(shuffle_means <= real_mean))
    return {"real_mean": real_mean, "p_value": p, "n_shuffles": int(n_shuffles)}


# ---------------------------------------------------------------------------
# Synthetic positive control (seed-robust)
# ---------------------------------------------------------------------------
def synthetic_control(
    data_module,
    premiums=(0.0, 0.04, 0.08),
    n_seeds: int = 20,
    kind: str = "var5",
    lookback: int = 252,
    base_seed: int = 505,
) -> pd.DataFrame:
    """Faithful-engine check: plant a left-tail-momentum premium and verify recovery.

    For each premium, average the long-short mean and one-sample t over ``n_seeds``
    synthetic panels (Welch-style averaging so a "Real" claim never rests on one lucky
    RNG seed). This is a POWER check on the engine -- NOT evidence about the real tape.

    Returns a DataFrame: premium, mean_ls_pct (ann), mean_t (avg one-sample t), share_t2
    (fraction of seeds with t>=2).
    """
    rows = []
    for prem in premiums:
        means, tstats = [], []
        for k in range(n_seeds):
            sr, _mkt, _truth = data_module.synthetic_panel(
                ltm_premium=prem, seed=base_seed + k
            )
            prices_s = (1.0 + sr).cumprod() * 100.0
            book = long_short_book(prices_s, kind=kind, lookback=lookback, min_stocks=10)
            if book.empty:
                continue
            s = summary(book["ls_ret"])
            means.append(s["mean"])
            tstats.append(s["tstat"])
        if not means:
            rows.append({"premium": prem, "mean_ls_pct": float("nan"),
                         "mean_t": float("nan"), "share_t2": float("nan")})
            continue
        rows.append({
            "premium": prem,
            "mean_ls_pct": float(np.mean(means)) * 100.0,
            "mean_t": float(np.mean(tstats)),
            "share_t2": float(np.mean(np.array(tstats) >= 2.0)),
        })
    return pd.DataFrame(rows)
