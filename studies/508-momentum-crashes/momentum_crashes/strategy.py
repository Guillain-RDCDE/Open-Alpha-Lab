"""The Momentum-Crashes engine and its honest controls -- Study 508.

We build the canonical Jegadeesh-Titman (1993) 12-1 cross-sectional momentum book (rank by
trailing 12-month return skipping the most recent month, long the top fraction / short the
bottom, equal-weight, dollar-neutral, monthly hold), then study its *crash dynamics* in the
spirit of Daniel & Moskowitz (2016):

1. **The signal & WML spread** -- monthly winners-minus-losers, GROSS and NET, with a robust
   one-sample *t* (Newey-West HAC), Sharpe, hit-rate, max-drawdown and worst month.
2. **The crash anatomy** -- the worst WML months and the deepest drawdown episode (when the
   left tail bites, and how big it gets on this survivor basket).
3. **Regime conditioning** -- Daniel-Moskowitz's two states: a BEAR indicator (cumulative
   market return over the trailing 2 years < 0) and a PANIC indicator (bear AND the market is
   currently rebounding). The paper's claim is that momentum's worst returns concentrate in the
   panic/rebound state, when the past-loser leg snaps back violently.
4. **The repair -- vol-scaling** -- "dynamic" / constant-volatility momentum: scale each month's
   WML position inversely to the recent realised volatility of the WML spread, targeting a
   constant ex-ante vol. Daniel-Moskowitz show this lifts the Sharpe and tames the crash.
5. **The null** -- a label-shuffle placebo permutation p-value on the WML mean.
6. **The positive control** -- a deterministic synthetic panel with a planted relative-strength
   drift AND a planted loser snap-back the engine must recover (faithful-engine / power check
   ONLY -- never cited for the real-tape stamp).

Execution lag (documented exactly, ONE shift): the signal uses prices up to and including
month-end *m*; we do NOT trade on that close. We enter at the next month and earn the realised
return of month *m+1*. A single forward-only lag -- no same-bar fill, no look-ahead.

Survivorship: the basket is names still trading in 2026. The loser leg's natural shorts -- firms
that trended into delisting -- are absent, so the WML premium here is an upper bound and the
crash an under-statement. Named on the SIGNAL axis. Opt-in guard: pass a delisting-complete
panel to ``long_short`` to lift the bias; we cannot from yfinance, so we flag it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TRADING_DAYS = 252

# 12-1 lookback: 12 months trailing, skip the most recent 1 month (reversal guard).
LOOKBACK_M = 12
SKIP_M = 1

# Daniel-Moskowitz regime windows
BEAR_LOOKBACK_M = 12   # trailing 1-year market return < 0 -> "bear" state (12m resolves the
                       # COVID/2022 episodes on this survivor tape; a 24m window almost never
                       # fires on large-caps that recover fast -- see notebook 02 for the sweep)
PANIC_REBOUND_M = 1    # within a bear, the current month's market return > 0 -> "panic/rebound"

# Vol-scaling target (annualised) for the dynamic-momentum repair
VOL_TARGET_ANN = 0.12
VOL_EST_WINDOW_M = 6   # trailing months of WML returns used to estimate conditional vol


# ---------------------------------------------------------------------------
# Monthly resampling and the 12-1 signal
# ---------------------------------------------------------------------------
def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted-close prices (last observation in each calendar month)."""
    m = prices.resample("ME").last()
    return m.dropna(how="all")


def to_monthly_series(s: pd.Series) -> pd.Series:
    """Month-end of a single price series."""
    return s.resample("ME").last().dropna()


def momentum_signal(
    monthly_prices: pd.DataFrame, lookback: int = LOOKBACK_M, skip: int = SKIP_M
) -> pd.DataFrame:
    """The 12-1 trailing-return signal at each month-end (no look-ahead)."""
    p = monthly_prices
    return p.shift(skip) / p.shift(lookback) - 1.0


# ---------------------------------------------------------------------------
# The winners-minus-losers (WML) long-short book
# ---------------------------------------------------------------------------
def long_short(
    monthly_prices: pd.DataFrame,
    frac: float = 0.2,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = 10,
) -> pd.DataFrame:
    """Monthly winners-minus-losers spread on a fraction sort, gross and net.

    At each month-end *t*: rank by the 12-1 signal, long the top ``frac`` (winners), short the
    bottom ``frac`` (losers), equal-weight, dollar-neutral. Earn the realised return of month
    *t+1* (the single forward execution lag). Costs: one-way turnover x cost_bps x NAV at the
    rebalance (both legs), plus an annual borrow on the short leg pro-rated monthly.

    Returns a DataFrame indexed by the HOLDING month with columns:
        ``win`` / ``los`` -- equal-weight realised leg returns
        ``wml_gross``     -- win - los (dollar-neutral, gross)
        ``turnover``      -- one-way fraction of book turned over vs the prior month
        ``wml_net``       -- wml_gross net of costs and borrow
        ``n_leg``         -- names per leg
    """
    fwd = monthly_prices.pct_change().shift(-1)  # month t -> realised return of month t+1
    sig = momentum_signal(monthly_prices, lookback, skip)

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)
        s = s.loc[common]
        r = r.loc[common]
        n = len(s)
        if n < min_names:
            continue
        k = max(1, int(round(frac * n)))
        ranked = s.sort_values()
        los_names = list(ranked.index[:k])
        win_names = list(ranked.index[-k:])
        if set(los_names) & set(win_names):
            continue

        win_ret = float(r[win_names].mean())
        los_ret = float(r[los_names].mean())
        wml_gross = win_ret - los_ret

        long_set, short_set = set(win_names), set(los_names)
        if prev_long or prev_short:
            long_to = len(long_set ^ prev_long) / (2 * k)
            short_to = len(short_set ^ prev_short) / (2 * k)
            turnover = 0.5 * (long_to + short_to)
        else:
            turnover = 1.0
        prev_long, prev_short = long_set, short_set

        cost = 2.0 * turnover * cost_bps * 1e-4
        borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
        wml_net = wml_gross - cost - borrow

        rows.append(
            {
                "date": t,
                "win": win_ret,
                "los": los_ret,
                "wml_gross": wml_gross,
                "turnover": turnover,
                "wml_net": wml_net,
                "n_leg": k,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").dropna(subset=["wml_gross"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(r: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly return series."""
    x = pd.Series(r).dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def max_drawdown_episode(r: pd.Series) -> dict:
    """Deepest peak-to-trough drawdown of the cumulative WML wealth and its dates."""
    s = pd.Series(r).astype(float).dropna()
    if len(s) < 2:
        return {"max_dd": float("nan"), "peak": None, "trough": None}
    eq = (1.0 + s).cumprod()
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return {"max_dd": float(dd.min()), "peak": peak, "trough": trough}


def summary(r: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised statistics for a monthly return series.

    Returns mean (ann), vol (ann), Sharpe, HAC t-stat, hit-rate, max-drawdown, worst month,
    skewness (the crash signature is a fat *left* tail = negative skew), and n.
    """
    s = pd.Series(r).astype(float).dropna()
    n = len(s)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_dd", "worst", "skew", "n")}
    mean_ann = float(s.mean() * periods_per_year)
    vol_ann = float(s.std(ddof=1) * np.sqrt(periods_per_year))
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    eq = (1.0 + s).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    # Fisher-Pearson skewness (population) -- negative = fat left tail = crash signature.
    z = (s - s.mean())
    denom = (z.pow(2).mean()) ** 1.5
    skew = float((z.pow(3).mean()) / denom) if denom > 0 else float("nan")
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": hac_tstat(s),
        "hit_rate": float((s > 0).mean()),
        "max_dd": dd,
        "worst": float(s.min()),
        "skew": skew,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The crash anatomy
# ---------------------------------------------------------------------------
def crash_table(ls: pd.DataFrame, k: int = 8, col: str = "wml_gross") -> pd.DataFrame:
    """The ``k`` worst WML months (the crash months) with each leg's contribution."""
    if ls.empty:
        return pd.DataFrame()
    worst = ls.sort_values(col).head(k)
    out = worst[[col, "win", "los"]].copy()
    out.columns = ["wml", "win_leg", "los_leg"]
    return out


# ---------------------------------------------------------------------------
# Daniel-Moskowitz regime conditioning
# ---------------------------------------------------------------------------
def market_regimes(
    market_monthly: pd.Series,
    bear_lookback: int = BEAR_LOOKBACK_M,
) -> pd.DataFrame:
    """Bear and panic indicators per holding month (Daniel-Moskowitz 2016 states).

    BEAR_t  = cumulative market return over the trailing ``bear_lookback`` months < 0.
    UP_t    = the market's own realised return *this* holding month > 0.
    PANIC_t = BEAR_t AND UP_t -- a bear-market rebound month, where the past-loser leg snaps
              back and momentum crashes. Returns a (month x {bear, up, panic}) bool frame.
    """
    m = pd.Series(market_monthly).astype(float).dropna()
    mret = m.pct_change()
    cum_2y = m / m.shift(bear_lookback) - 1.0
    bear = (cum_2y < 0.0)
    up = (mret > 0.0)
    panic = bear & up
    out = pd.DataFrame({"bear": bear, "up": up, "panic": panic})
    return out


def regime_split(
    ls: pd.DataFrame,
    market_monthly: pd.Series,
    bear_lookback: int = BEAR_LOOKBACK_M,
    col: str = "wml_gross",
) -> pd.DataFrame:
    """Mean WML return (annualised) conditioned on the market regime.

    Aligns the WML holding-month returns to the bear/up/panic indicators and reports the
    annualised conditional mean and month count in each cell. Daniel-Moskowitz predict the
    PANIC (bear & rebound) cell is where momentum bleeds.
    """
    if ls.empty:
        return pd.DataFrame()
    reg = market_regimes(market_monthly, bear_lookback=bear_lookback)
    # Align on month-end; the WML index already is the holding month-end.
    df = ls[[col]].join(reg, how="inner").dropna(subset=[col])
    if df.empty:
        return pd.DataFrame()

    def _cell(mask: pd.Series) -> tuple[float, int]:
        x = df.loc[mask, col]
        return (float(x.mean() * MONTHS_PER_YEAR) if len(x) else float("nan"), int(len(x)))

    states = {
        "all": pd.Series(True, index=df.index),
        "calm (non-bear)": ~df["bear"],
        "bear (any)": df["bear"],
        "bear & down": df["bear"] & ~df["up"],
        "panic (bear & rebound)": df["panic"],
    }
    rows = []
    for name, mask in states.items():
        mean_ann, n = _cell(mask)
        rows.append({"regime": name, "wml_mean_ann": mean_ann, "n_months": n})
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# The repair -- vol-scaled (dynamic / constant-volatility) momentum
# ---------------------------------------------------------------------------
def vol_scaled(
    ls: pd.DataFrame,
    target_vol_ann: float = VOL_TARGET_ANN,
    est_window: int = VOL_EST_WINDOW_M,
    col: str = "wml_gross",
    cap: float = 3.0,
) -> pd.Series:
    """Constant-volatility "dynamic momentum": scale WML inversely to its recent realised vol.

    Daniel-Moskowitz (2016): position size ``w_t = target / sigma_hat_t`` where ``sigma_hat_t``
    is the annualised volatility of the WML spread estimated from the trailing ``est_window``
    months (known at *t*, so no look-ahead). The scaled return is ``w_t * r_{t+1}``. Leverage
    is capped at ``cap`` to keep the dynamic book honest. Returns the vol-scaled WML series.
    """
    if ls.empty:
        return pd.Series(dtype=float)
    r = ls[col].astype(float)
    # Trailing realised vol (annualised), shifted so weight at t uses only past info.
    sig = r.rolling(est_window, min_periods=max(3, est_window // 2)).std(ddof=1)
    sig_ann = sig * np.sqrt(MONTHS_PER_YEAR)
    w = (target_vol_ann / sig_ann).clip(upper=cap).shift(1)
    scaled = (w * r).dropna()
    scaled.name = "wml_volscaled"
    return scaled


# ---------------------------------------------------------------------------
# The placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_prices: pd.DataFrame,
    frac: float = 0.2,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    n_perm: int = 1000,
    seed: int = 508,
) -> dict:
    """Permutation p-value: does the real WML mean beat a shuffled-label null?

    Keep each month's forward returns but SHUFFLE which stock the momentum signal points at --
    destroying genuine winner/loser persistence while preserving the cross-sectional return
    distribution and leg sizes. Recompute the WML mean ``n_perm`` times; report the one-sided
    share of placebos whose mean >= the real mean. A real edge gives a small p.
    """
    real = long_short(monthly_prices, frac=frac, lookback=lookback, skip=skip)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}
    real_mean = float(real["wml_gross"].mean())

    fwd = monthly_prices.pct_change().shift(-1)
    sig = momentum_signal(monthly_prices, lookback, skip)
    rng = np.random.default_rng(seed)

    months: list[tuple[np.ndarray, np.ndarray, int]] = []
    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)
        if len(common) < 10:
            continue
        sv = s.loc[common].to_numpy()
        rv = r.loc[common].to_numpy()
        k = max(1, int(round(frac * len(common))))
        if 2 * k > len(common):
            continue
        months.append((sv, rv, k))

    if not months:
        return {"real_mean_ann": real_mean * MONTHS_PER_YEAR, "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}

    placebo_means = np.empty(n_perm)
    for b in range(n_perm):
        spread_sum = 0.0
        for sv, rv, k in months:
            perm = rng.permutation(len(sv))
            order = np.argsort(sv[perm])
            los = rv[order[:k]].mean()
            win = rv[order[-k:]].mean()
            spread_sum += win - los
        placebo_means[b] = spread_sum / len(months)

    p = float((placebo_means >= real_mean).mean())
    return {
        "real_mean_ann": real_mean * MONTHS_PER_YEAR,
        "p_value": p,
        "placebo_mean_ann": float(placebo_means.mean() * MONTHS_PER_YEAR),
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(
    strengths: tuple[float, ...] = (0.0, 0.15, 0.30, 0.50),
    n_stocks: int = 40,
    n_days: int = 2600,
    frac: float = 0.2,
    crash_strength: float = 0.0,
    seed: int = 508,
) -> pd.DataFrame:
    """Plant a known relative-strength drift and verify the WML engine recovers it.

    Sweeps ``mom_strength`` on the deterministic synthetic panel; reports the WML mean and
    HAC *t* at each. The engine should score ~0 at strength 0 (null) and rise monotonically as
    the planted drift grows. Faithful-engine / power check ONLY -- never cited for a real stamp.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, _mkt, _truth = _data.synthetic_panel(
            n_stocks=n_stocks, n_days=n_days, mom_strength=strength,
            crash_strength=crash_strength, seed=seed,
        )
        mp = to_monthly(prices)
        ls = long_short(mp, frac=frac)
        if ls.empty:
            rows.append({"mom_strength": strength, "wml_mean_ann": float("nan"),
                         "tstat": float("nan"), "n": 0})
            continue
        s = summary(ls["wml_gross"])
        rows.append({"mom_strength": strength, "wml_mean_ann": s["mean"],
                     "tstat": s["tstat"], "n": s["n"]})
    return pd.DataFrame(rows)
