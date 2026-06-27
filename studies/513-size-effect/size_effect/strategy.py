"""The Size-Effect (SMB) strategy -- Banz (1981).

Construction (monthly, on a stable survivor basket):
  1. Each month-end, rank the basket by market capitalisation
     (cap_t = cap_now * price_t / price_now -- shares-constant, documented).
  2. Split at the median cap into a SMALL half and a LARGE half.
  3. Long the small half (equal-weight), short the large half (equal-weight).
  4. Execution lag: the signal uses prices through month-end *m*; the position is held
     over month *m+1*. We enter at the next month's first available close -- one lag,
     no same-bar fill, no look-ahead.
  5. Two books:
       - ``smb_dn``  : dollar-neutral small-minus-large (the literal Banz long-short).
       - ``smb_bn``  : beta-neutral -- the large (short) leg is scaled to the small leg's
         beta so the spread carries ~zero market beta.
  6. Costs: one-way bps x turnover (both legs) + an annual borrow charge on the short leg.

Inference: a Newey-West HAC one-sample t-stat on the monthly spread, plus a label-shuffle
placebo (shuffle the small/large labels each month and re-run -> a null distribution of
t-stats; the placebo p is the share of |t_null| >= |t_real|).

Slices the original anomaly is famous for:
  - ``january_split``  : mean spread in January vs the other 11 months (the "January effect").
  - ``decade_split``   : pre-2000 vs post-2000 spread (the post-publication / post-1980 decay).

The basket is **survivorship-biased**; all positive numbers are upper bounds and named.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RF_ANNUAL = 0.02
RF_DAILY = RF_ANNUAL / 252
BETA_WINDOW = 252  # trading days for the beta hedge
MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Market-cap reconstruction
# ---------------------------------------------------------------------------
def reconstruct_caps(prices: pd.DataFrame, caps_now: pd.Series) -> pd.DataFrame:
    """Time-varying market cap = cap_now * price_t / price_now (shares-constant).

    ``price_now`` is each ticker's last valid price. Returns a (date x ticker) cap frame
    aligned to ``prices``. NaN where price is missing.
    """
    common = [c for c in prices.columns if c in caps_now.index and np.isfinite(caps_now.get(c, np.nan))]
    px = prices[common]
    price_now = px.ffill().iloc[-1]
    cap_now = caps_now.reindex(common).astype(float)
    scaled = px.div(price_now, axis=1).mul(cap_now, axis=1)
    return scaled


# ---------------------------------------------------------------------------
# Rolling beta (for the beta-neutral hedge)
# ---------------------------------------------------------------------------
def _leg_beta(leg_monthly: pd.Series, mkt_monthly: pd.Series, window: int = 36) -> pd.Series:
    """Rolling beta of a monthly leg-return series vs the market (min 12 obs)."""
    cov = leg_monthly.rolling(window, min_periods=12).cov(mkt_monthly)
    var = mkt_monthly.rolling(window, min_periods=12).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# The SMB long-short
# ---------------------------------------------------------------------------
def smb_portfolio(
    prices: pd.DataFrame,
    spy: pd.Series,
    caps_now: pd.Series,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """Monthly small-minus-large long-short with one execution lag.

    Returns a monthly DataFrame indexed by the *held* month, with columns:
        ``small_ret``  -- equal-weight return of the small-cap leg over the held month
        ``large_ret``  -- equal-weight return of the large-cap leg over the held month
        ``mkt_ret``    -- SPY return over the held month
        ``smb_dn``     -- dollar-neutral small - large
        ``turnover``   -- fraction of names that changed half vs the prior month (0..2)
        ``n``          -- basket size that month
    A ``smb_bn`` (beta-neutral) column is appended by ``add_beta_neutral``.
    """
    caps = reconstruct_caps(prices, caps_now)
    if caps.empty:
        return pd.DataFrame()

    # Monthly total returns from daily adjusted prices.
    px = prices[caps.columns]
    m_px = px.resample("ME").last()
    m_ret = m_px.pct_change()
    spy_m = spy.resample("ME").last().pct_change()

    # Month-end cap snapshot used to form the NEXT month's positions (one lag).
    cap_m = caps.resample("ME").last()

    months = m_ret.index
    rows = []
    prev_small: set[str] = set()
    prev_large: set[str] = set()

    for i in range(len(months) - 1):
        signal_m = months[i]      # caps known through here
        held_m = months[i + 1]    # position held / return realised here (the lag)

        snap = cap_m.loc[signal_m].dropna()
        snap = snap[snap > 0]
        if len(snap) < min_stocks:
            continue
        med = snap.median()
        small = snap[snap <= med].index  # small caps: at or below median
        large = snap[snap > med].index

        held = m_ret.loc[held_m]
        small_r = float(held[small].dropna().mean())
        large_r = float(held[large].dropna().mean())
        mkt_r = float(spy_m.loc[held_m]) if held_m in spy_m.index else np.nan
        if not np.isfinite(small_r) or not np.isfinite(large_r):
            continue

        cur_small, cur_large = set(small), set(large)
        if prev_small or prev_large:
            n_now = len(cur_small) + len(cur_large)
            changed = len(cur_small ^ prev_small) + len(cur_large ^ prev_large)
            turnover = changed / max(n_now, 1)
        else:
            turnover = 1.0
        prev_small, prev_large = cur_small, cur_large

        rows.append({
            "date": held_m,
            "small_ret": small_r,
            "large_ret": large_r,
            "mkt_ret": mkt_r,
            "smb_dn": small_r - large_r,
            "turnover": float(turnover),
            "n": int(len(snap)),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def add_beta_neutral(book: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Append a beta-neutral spread: small_ret - beta_ratio * large_ret.

    The hedge ratio is the rolling beta of the small leg over the rolling beta of the large
    leg (both vs SPY), shifted one month so it uses only past info. Falls back to 1.0 (=>
    dollar-neutral) until enough history accrues.
    """
    if book.empty:
        return book
    b_small = _leg_beta(book["small_ret"], book["mkt_ret"], window)
    b_large = _leg_beta(book["large_ret"], book["mkt_ret"], window)
    ratio = (b_small / b_large).replace([np.inf, -np.inf], np.nan).shift(1)
    ratio = ratio.clip(0.3, 3.0).fillna(1.0)
    out = book.copy()
    out["beta_ratio"] = ratio
    out["smb_bn"] = out["small_ret"] - ratio * out["large_ret"]
    return out


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def apply_costs(
    spread: pd.Series,
    turnover: pd.Series,
    one_way_bps: float = 10.0,
    borrow_annual_bps: float = 100.0,
) -> pd.Series:
    """Net monthly spread = gross - trading cost - short borrow.

    Trading cost = one_way_bps * turnover (turnover already counts both legs' name churn).
    Borrow = borrow_annual_bps / 12 charged on the (always-on) short leg each month.
    """
    tc = (one_way_bps / 1e4) * turnover.reindex(spread.index).fillna(0.0)
    borrow = (borrow_annual_bps / 1e4) / MONTHS_PER_YEAR
    return spread - tc - borrow


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def hac_tstat(x: pd.Series) -> float:
    """Newey-West HAC one-sample t-stat on the mean of a monthly series."""
    r = pd.Series(x).astype(float).dropna()
    n = len(r)
    if n < 3:
        return np.nan
    arr = r.to_numpy()
    mu = arr.mean()
    e = arr - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else np.nan


def summary(monthly: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised stats for a monthly return series + HAC t-stat (the inference number)."""
    r = pd.Series(monthly).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_ann = float(r.mean()) * periods_per_year
    vol_ann = float(r.std(ddof=1)) * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else np.nan
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann, "vol": vol_ann, "sharpe": sr,
        "tstat": hac_tstat(r), "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd, "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo: label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    prices: pd.DataFrame,
    spy: pd.Series,
    caps_now: pd.Series,
    n_perm: int = 300,
    min_stocks: int = 10,
    seed: int = 513,
) -> tuple[float, float]:
    """Label-shuffle placebo on the dollar-neutral SMB t-stat.

    Each permutation randomly re-assigns the small/large labels each month (preserving the
    half sizes) and recomputes the SMB spread's HAC t. Returns
    ``(t_real, p)`` where ``p`` = share of permutations with |t_null| >= |t_real|.
    """
    book = smb_portfolio(prices, spy, caps_now, min_stocks=min_stocks)
    if book.empty:
        return np.nan, np.nan
    t_real = hac_tstat(book["smb_dn"])

    caps = reconstruct_caps(prices, caps_now)
    px = prices[caps.columns]
    m_ret = px.resample("ME").last().pct_change()
    cap_m = caps.resample("ME").last()
    months = m_ret.index
    rng = np.random.default_rng(seed)

    count = 0
    valid = 0
    for _ in range(n_perm):
        spreads = []
        for i in range(len(months) - 1):
            snap = cap_m.loc[months[i]].dropna()
            snap = snap[snap > 0]
            if len(snap) < min_stocks:
                continue
            names = snap.index.to_numpy().copy()
            rng.shuffle(names)
            half = len(names) // 2
            small = names[:half]
            large = names[half:]
            held = m_ret.loc[months[i + 1]]
            sr = float(held[small].dropna().mean())
            lr = float(held[large].dropna().mean())
            if np.isfinite(sr) and np.isfinite(lr):
                spreads.append(sr - lr)
        if len(spreads) > 5:
            t_null = hac_tstat(pd.Series(spreads))
            if np.isfinite(t_null):
                valid += 1
                if abs(t_null) >= abs(t_real):
                    count += 1
    p = count / valid if valid else np.nan
    return float(t_real), float(p)


# ---------------------------------------------------------------------------
# Slices: January concentration & post-2000 decay
# ---------------------------------------------------------------------------
def january_split(book: pd.DataFrame, col: str = "smb_dn") -> dict:
    """Mean monthly spread in January vs the other months, with a Welch t between them."""
    s = book[col].dropna()
    jan = s[s.index.month == 1]
    rest = s[s.index.month != 1]
    from scipy import stats
    t, p = (np.nan, np.nan)
    if len(jan) > 2 and len(rest) > 2:
        t, p = stats.ttest_ind(jan, rest, equal_var=False)
    return {
        "jan_mean": float(jan.mean()), "jan_n": int(len(jan)),
        "rest_mean": float(rest.mean()), "rest_n": int(len(rest)),
        "welch_t": float(t), "welch_p": float(p),
    }


def decade_split(book: pd.DataFrame, cut: str = "2000-01-01", col: str = "smb_dn") -> dict:
    """Mean spread + HAC t before vs after ``cut`` (the post-publication / post-1980 decay)."""
    s = book[col].dropna()
    cut_ts = pd.Timestamp(cut)
    pre = s[s.index < cut_ts]
    post = s[s.index >= cut_ts]
    return {
        "pre_mean_ann": float(pre.mean()) * MONTHS_PER_YEAR, "pre_t": hac_tstat(pre), "pre_n": int(len(pre)),
        "post_mean_ann": float(post.mean()) * MONTHS_PER_YEAR, "post_t": hac_tstat(post), "post_n": int(len(post)),
    }


# ---------------------------------------------------------------------------
# Synthetic positive control -- faithful-engine check (seed-robust)
# ---------------------------------------------------------------------------
def synthetic_control(
    size_premium: float = 0.06,
    n_seeds: int = 25,
    base_seed: int = 513,
) -> dict:
    """Run the engine on the synthetic panel across many seeds; average the HAC t.

    Proves the SMB sort RECOVERS a planted premium (power check) and that the null
    (size_premium=0) does NOT print significance. Averages over ``n_seeds`` so no single
    lucky RNG draw can carry the claim. Returns dict with mean t under signal and under null.
    """
    from . import data as _data

    def _engine_t(premium, seed):
        rets, mkt, caps, _ = _data.synthetic_panel(size_premium=premium, seed=seed)
        # Build a faux daily price frame from returns (start at 100) for the same pipeline.
        prices = (1.0 + rets).cumprod() * 100.0
        spy = (1.0 + mkt).cumprod() * 100.0
        book = smb_portfolio(prices, spy, caps, min_stocks=8)
        if book.empty:
            return np.nan
        return hac_tstat(book["smb_dn"])

    sig_ts, null_ts = [], []
    for k in range(n_seeds):
        s = base_seed + k
        ts = _engine_t(size_premium, s)
        tn = _engine_t(0.0, s)
        if np.isfinite(ts):
            sig_ts.append(ts)
        if np.isfinite(tn):
            null_ts.append(tn)
    return {
        "size_premium": size_premium,
        "n_seeds": n_seeds,
        "mean_t_signal": float(np.mean(sig_ts)) if sig_ts else np.nan,
        "mean_t_null": float(np.mean(null_ts)) if null_ts else np.nan,
        "frac_signal_t_gt2": float(np.mean([t > 2 for t in sig_ts])) if sig_ts else np.nan,
        "frac_null_t_gt2": float(np.mean([abs(t) > 2 for t in null_ts])) if null_ts else np.nan,
    }
