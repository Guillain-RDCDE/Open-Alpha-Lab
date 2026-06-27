"""The Frog-In-The-Pan (FIP) engine and its honest controls -- Study 510.

Da, Gurun & Warachka (2014), "Frog in the Pan: Continuous Information and Momentum": the market
under-reacts more to information that arrives in many small *continuous* steps than to the same
total move delivered in a few *discrete* jumps. They proxy the continuity of information by an
**information-discreteness (ID)** measure built from the sign-consistency of daily returns over
the formation window, and show momentum is concentrated in the LOW-ID (gradual) names. A frog
dropped in boiling water jumps out (a discrete shock the market prices fast); a frog warmed
slowly never reacts (continuous information the market under-reacts to).

The ID measure (their Eq. 2):

    ID = sign(PRET) * (%neg - %pos)

where PRET is the cumulative formation-period return and %pos / %neg are the fractions of
positive / negative *daily* returns over the same window. A stock that drifted up smoothly has
%pos > %neg, sign(PRET) = +1, so ID < 0 (low, continuous). A stock that jumped up on a few days
amid many small down days has %pos < %neg, so ID > 0 (high, discrete). LOW ID = gradual.

This module measures, honestly:

1. **The signal.** Two winners-minus-losers (WML) books -- one inside the LOW-ID (gradual) half,
   one inside the HIGH-ID (discrete) half -- and their *difference* (the FIP interaction). Each
   carries a robust one-sample HAC *t*, Sharpe, hit-rate, max-drawdown.
2. **The null.** A label-shuffle placebo: shuffle which stock gets which forward return,
   recompute the low-ID WML mean many times, report the share of placebos that beat the real
   mean (a permutation p-value). A real edge survives; a data-mined one does not.
3. **Costs.** One-way bps x NAV x turnover at each monthly rebalance, plus an annual borrow on
   the short (loser) leg pro-rated monthly. Reported gross vs net.
4. **The positive control.** A deterministic synthetic panel where the planted drift is delivered
   smoothly to the low-ID names and lumpily to the high-ID names; the engine must recover a
   bigger low-ID WML than high-ID WML. A faithful-engine / power check ONLY -- never cited for
   the real-tape stamp.

Execution lag (documented, ONE shift): the momentum signal AND the ID measure are computed from
prices up to and including month-end *m*; we do NOT trade on that close. We enter at the close
one trading day later (first session of month *m+1*) and earn the realised return of month
*m+1*. A single forward-only lag -- no same-bar fill, no look-ahead.

Survivorship: the basket is names still trading in 2026. The loser leg's natural short
candidates -- firms that drifted into delisting -- are absent, and a gradual decline into
delisting is precisely a LOW-ID loser, so the FIP slice is the MORE survivorship-inflated of
the two. Named on the SIGNAL axis. Opt-in guard: pass a delisting-complete panel to
``long_short`` to lift the bias; we cannot from yfinance, so we flag it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
TRADING_DAYS = 252

# 12-1 lookback: 12 months trailing, skip the most recent 1 month (reversal guard).
LOOKBACK_M = 12
SKIP_M = 1


# ---------------------------------------------------------------------------
# Monthly resampling, the 12-1 signal, and the information-discreteness measure
# ---------------------------------------------------------------------------
def to_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted-close prices (last observation in each calendar month)."""
    m = prices.resample("ME").last()
    return m.dropna(how="all")


def momentum_signal(
    monthly_prices: pd.DataFrame, lookback: int = LOOKBACK_M, skip: int = SKIP_M
) -> pd.DataFrame:
    """The 12-1 trailing return signal at each month-end (cumulative t-lookback -> t-skip)."""
    p = monthly_prices
    return p.shift(skip) / p.shift(lookback) - 1.0


def information_discreteness(
    prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
) -> pd.DataFrame:
    """Da-Gurun-Warachka information-discreteness, evaluated at each month-end.

    For month-end *t* and the SAME 12-1 formation window used by ``momentum_signal``
    (months t-lookback .. t-skip), using DAILY returns inside that window:

        ID = sign(PRET) * (%neg - %pos)

    where PRET is the cumulative formation return (the momentum signal) and %pos / %neg are the
    fractions of strictly positive / negative daily returns. Low (often negative) ID = gradual,
    continuous information (the Frog-In-The-Pan case). High (positive) ID = discrete, jumpy.

    Returns a (month x ticker) DataFrame of ID values aligned to ``monthly_prices.index``,
    NaN where the formation window has insufficient daily history. No look-ahead: the window
    ends at t-skip month-end, strictly before the t+1 holding month.
    """
    daily_ret = prices.pct_change()
    months = monthly_prices.index
    # Map each month-end to its position so we can slice the formation window by month-ends.
    out = pd.DataFrame(index=months, columns=monthly_prices.columns, dtype=float)

    for i, t in enumerate(months):
        if i < lookback:
            continue
        win_start = months[i - lookback]   # exclusive lower month-end
        win_end = months[i - skip]         # inclusive upper month-end (skip most recent month)
        sub = daily_ret.loc[(daily_ret.index > win_start) & (daily_ret.index <= win_end)]
        if sub.shape[0] < 20:
            continue
        pos = (sub > 0).sum(axis=0)
        neg = (sub < 0).sum(axis=0)
        ndays = (sub.notna()).sum(axis=0).replace(0, np.nan)
        pct_pos = pos / ndays
        pct_neg = neg / ndays
        # PRET sign = sign of the cumulative formation return
        pret = monthly_prices.iloc[i - skip] / monthly_prices.iloc[i - lookback] - 1.0
        sgn = np.sign(pret)
        out.loc[t] = sgn * (pct_neg - pct_pos)
    return out


# ---------------------------------------------------------------------------
# The FIP double-sort: WML inside the low-ID and high-ID halves
# ---------------------------------------------------------------------------
def long_short(
    monthly_prices: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    id_side: str = "low",
    frac: float = 0.3,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = 12,
) -> pd.DataFrame:
    """Monthly winners-minus-losers spread inside one ID half, gross and net.

    Construction at each month-end *t*:
      1. Split the cross-section at its median ID into a LOW-ID (gradual) half and a HIGH-ID
         (discrete) half (``id_side`` selects which half we trade).
      2. Inside that half, rank by the 12-1 momentum signal; long the top ``frac``, short the
         bottom ``frac``, equal-weight, dollar-neutral ($1 long / $1 short).
      3. Earn the realised return of month *t+1* (single forward execution lag -- form on the
         *t* close, hold the *t+1* month; never a same-bar fill).
      4. Costs: turnover x cost_bps x NAV at the rebalance, plus an annual borrow on the short
         leg pro-rated monthly. Net = gross - costs - borrow.

    ``prices`` (daily) is required to compute the ID measure; if omitted, the ID half-split is
    skipped and the book is the plain momentum WML on the whole basket (used only as a baseline).

    Returns a DataFrame indexed by the HOLDING month with columns:
        ``win`` / ``los`` -- equal-weight realised return of each leg
        ``wml_gross``     -- win - los (dollar-neutral, gross)
        ``turnover``      -- one-way fraction of book turned over vs the prior month
        ``wml_net``       -- wml_gross net of costs and borrow
        ``n_leg``         -- names per leg
    """
    fwd = monthly_prices.pct_change().shift(-1)  # month t -> realised return of month t+1
    sig = momentum_signal(monthly_prices, lookback, skip)
    idm = (
        information_discreteness(prices, monthly_prices, lookback, skip)
        if prices is not None
        else None
    )

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)

        if idm is not None:
            idv = idm.loc[t].dropna() if t in idm.index else pd.Series(dtype=float)
            common = common.intersection(idv.index)
            if len(common) < min_names:
                continue
            idv = idv.loc[common]
            med = idv.median()
            if id_side == "low":
                half = idv[idv <= med].index
            else:
                half = idv[idv > med].index
            common = common.intersection(half)

        s = s.loc[common]
        r = r.loc[common]
        n = len(s)
        if n < max(6, int(np.ceil(2 / max(frac, 1e-9)))):
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


def fip_spread(low_book: pd.DataFrame, high_book: pd.DataFrame, col: str = "wml_gross") -> pd.Series:
    """The FIP interaction: low-ID WML minus high-ID WML, month by month (aligned on dates)."""
    if low_book.empty or high_book.empty:
        return pd.Series(dtype=float)
    j = low_book[[col]].join(high_book[[col]], lsuffix="_lo", rsuffix="_hi", how="inner")
    return (j[f"{col}_lo"] - j[f"{col}_hi"]).rename("fip")


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


def summary(r: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised statistics for a monthly return series.

    Returns mean (ann), vol (ann), Sharpe, HAC t-stat, hit-rate, max-drawdown, worst month, n.
    The HAC *t* is the inference-bar number.
    """
    s = pd.Series(r).astype(float).dropna()
    n = len(s)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_dd", "worst", "n")}
    mean_ann = float(s.mean() * periods_per_year)
    vol_ann = float(s.std(ddof=1) * np.sqrt(periods_per_year))
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    eq = (1.0 + s).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": hac_tstat(s),
        "hit_rate": float((s > 0).mean()),
        "max_dd": dd,
        "worst": float(s.min()),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_prices: pd.DataFrame,
    prices: pd.DataFrame,
    id_side: str = "low",
    frac: float = 0.3,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    n_perm: int = 1000,
    seed: int = 510,
) -> dict:
    """Permutation p-value: does the real low-ID WML mean beat a shuffled-label null?

    Inside the chosen ID half each month, we keep the same forward returns but SHUFFLE which
    stock the momentum signal points at -- destroying genuine winner/loser persistence while
    preserving the cross-sectional return distribution, the ID conditioning and the leg sizes.
    We recompute the WML mean ``n_perm`` times and report the one-sided share of placebos whose
    mean >= the real mean. A real edge gives a small p; a data-mined one gives p ~ 0.5.

    Returns ``{"real_mean_ann", "p_value", "placebo_mean_ann", "n_perm"}``.
    """
    real = long_short(monthly_prices, prices=prices, id_side=id_side, frac=frac,
                      lookback=lookback, skip=skip)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}
    real_mean = float(real["wml_gross"].mean())

    fwd = monthly_prices.pct_change().shift(-1)
    sig = momentum_signal(monthly_prices, lookback, skip)
    idm = information_discreteness(prices, monthly_prices, lookback, skip)
    rng = np.random.default_rng(seed)

    months: list[tuple[np.ndarray, np.ndarray, int]] = []
    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        idv = idm.loc[t].dropna() if t in idm.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index).intersection(idv.index)
        if len(common) < 12:
            continue
        idc = idv.loc[common]
        med = idc.median()
        half = idc[idc <= med].index if id_side == "low" else idc[idc > med].index
        common = common.intersection(half)
        if len(common) < 6:
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
    strengths: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60),
    n_stocks: int = 40,
    n_days: int = 2600,
    gradual_frac: float = 0.5,
    frac: float = 0.3,
    seed: int = 510,
) -> pd.DataFrame:
    """Plant a known drift delivered smoothly to low-ID names and lumpily to high-ID names;
    verify the engine recovers a BIGGER low-ID WML than high-ID WML.

    Sweeps ``mom_strength`` on the deterministic synthetic panel where half the trending names
    get the drift smoothly (low-ID) and half get it in jumps (high-ID). The engine should score
    ~0 on both slices at strength 0, and at positive strength the low-ID WML should exceed the
    high-ID WML (the FIP interaction). Faithful-engine / power check ONLY.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, _truth = _data.synthetic_panel(
            n_stocks=n_stocks, n_days=n_days, mom_strength=strength,
            gradual_frac=gradual_frac, seed=seed,
        )
        mp = to_monthly(prices)
        lo = long_short(mp, prices=prices, id_side="low", frac=frac)
        hi = long_short(mp, prices=prices, id_side="high", frac=frac)
        s_lo = summary(lo["wml_gross"]) if not lo.empty else {"mean": float("nan"), "tstat": float("nan"), "n": 0}
        s_hi = summary(hi["wml_gross"]) if not hi.empty else {"mean": float("nan"), "tstat": float("nan"), "n": 0}
        rows.append({
            "mom_strength": strength,
            "low_id_mean_ann": s_lo["mean"], "low_id_t": s_lo["tstat"],
            "high_id_mean_ann": s_hi["mean"], "high_id_t": s_hi["tstat"],
            "fip_gap_ann": (s_lo["mean"] - s_hi["mean"]) if (s_lo["n"] and s_hi["n"]) else float("nan"),
            "n": s_lo["n"],
        })
    return pd.DataFrame(rows)
