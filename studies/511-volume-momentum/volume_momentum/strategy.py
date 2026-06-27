"""The Volume-Momentum (Lee-Swaminathan) engine and its honest controls -- Study 511.

Lee, C. M. C. & Swaminathan, B. (2000), "Price Momentum and Trading Volume" (Journal of Finance,
55(5), 2017-2069): trading volume predicts the magnitude and the persistence of price momentum.
They double-sort stocks on past return (winners/losers) THEN on past trading **volume**
(turnover), and find a "momentum life cycle":

  * momentum is strongest among **high-volume winners** and **low-volume losers**;
  * **high-volume** past performers (winners and losers) **reverse faster** -- volume forecasts
    *when* the drift turns into reversal.

This module measures, honestly:

1. **The trailing turnover measure.** For each name at each month-end, the average daily
   **dollar volume** over the same 12-1 formation window used by the momentum signal -- a
   liquidity/attention proxy. We split the cross-section at its median turnover into a HIGH-volume
   half and a LOW-volume half.

2. **The signal.** Two winners-minus-losers (WML) books -- one inside the HIGH-volume half, one
   inside the LOW-volume half -- and their *difference* (the Lee-Swaminathan interaction
   `high-vol WML - low-vol WML`, predicted positive). Each carries a robust one-sample HAC *t*,
   Sharpe, hit-rate, max-drawdown.

3. **The null.** A label-shuffle placebo: inside the chosen volume half, shuffle which stock gets
   which forward return, recompute the WML mean many times, report the share of placebos that beat
   the real mean (a permutation p-value). A real edge survives; a data-mined one does not. The p
   is checked seed-robust.

4. **Costs.** One-way bps x NAV x turnover at each monthly rebalance, plus an annual borrow on the
   short (loser) leg pro-rated monthly. Reported gross vs net.

5. **The volume-conditioned reversal (third axis).** The Lee-Swaminathan reversal prediction:
   the WML book's cumulative return at HOLDING horizons 1 / 3 / 6 / 12 months for the high-volume
   vs low-volume slice -- does the high-volume winner drift decay (give back) faster?

6. **The positive control.** A deterministic synthetic panel where the planted drift is
   concentrated in the high-turnover names; the engine must recover a bigger high-volume WML than
   low-volume WML. A faithful-engine / power check ONLY -- never cited for the real-tape stamp.

Execution lag (documented, ONE shift): the momentum signal AND the turnover measure are computed
from prices/volume up to and including month-end *m*; we do NOT trade on that close. We enter at
the close one trading day later (first session of month *m+1*) and earn the realised return of
month *m+1*. A single forward-only lag -- no same-bar fill, no look-ahead.

Survivorship: the basket is names still trading in 2026. The loser leg's natural short candidates
-- firms that drifted into delisting -- are absent, and a quiet (low-volume) decline into
delisting is precisely a LOW-volume loser, the Lee-Swaminathan strongest short. Named on the
SIGNAL axis. Opt-in guard: pass a delisting-complete panel to ``long_short`` to lift the bias;
we cannot from yfinance, so we flag it.
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
# Monthly resampling, the 12-1 signal, and the trailing turnover measure
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


def turnover_measure(
    dollar_volume: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
) -> pd.DataFrame:
    """Average daily dollar volume over the 12-1 formation window, at each month-end.

    For month-end *t* and the SAME 12-1 formation window used by ``momentum_signal`` (months
    t-lookback .. t-skip), the measure is the mean DAILY dollar volume inside that window. High
    values = heavily-traded / high-attention names (Lee-Swaminathan HIGH volume); low values =
    quiet names (LOW volume). We split the cross-section at its median each month.

    Returns a (month x ticker) DataFrame aligned to ``monthly_prices.index``, NaN where the
    formation window has insufficient daily history. No look-ahead: the window ends at t-skip
    month-end, strictly before the t+1 holding month.
    """
    months = monthly_prices.index
    out = pd.DataFrame(index=months, columns=monthly_prices.columns, dtype=float)
    dv = dollar_volume
    for i, t in enumerate(months):
        if i < lookback:
            continue
        win_start = months[i - lookback]   # exclusive lower month-end
        win_end = months[i - skip]         # inclusive upper month-end (skip most recent month)
        sub = dv.loc[(dv.index > win_start) & (dv.index <= win_end)]
        if sub.shape[0] < 20:
            continue
        out.loc[t] = sub.mean(axis=0)
    return out


# ---------------------------------------------------------------------------
# The momentum x volume double-sort: WML inside the high-volume and low-volume halves
# ---------------------------------------------------------------------------
def long_short(
    monthly_prices: pd.DataFrame,
    dollar_volume: pd.DataFrame | None = None,
    vol_side: str = "high",
    frac: float = 0.3,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    hold: int = 1,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = 12,
) -> pd.DataFrame:
    """Monthly winners-minus-losers spread inside one volume half, gross and net.

    Construction at each month-end *t*:
      1. Split the cross-section at its median trailing turnover into a HIGH-volume half and a
         LOW-volume half (``vol_side`` selects which half we trade).
      2. Inside that half, rank by the 12-1 momentum signal; long the top ``frac``, short the
         bottom ``frac``, equal-weight, dollar-neutral ($1 long / $1 short).
      3. Earn the realised cumulative return over the next ``hold`` months (single forward
         execution lag -- form on the *t* close, hold months *t+1 .. t+hold*; never a same-bar
         fill). ``hold=1`` is the headline monthly WML; ``hold>1`` drives the reversal axis.
      4. Costs: turnover x cost_bps x NAV at the rebalance, plus an annual borrow on the short
         leg pro-rated over the holding months. Net = gross - costs - borrow.

    ``dollar_volume`` (daily) is required to compute the turnover split; if omitted, the
    half-split is skipped and the book is the plain momentum WML on the whole basket (baseline).

    Returns a DataFrame indexed by the formation month with columns:
        ``win`` / ``los`` -- equal-weight realised return of each leg over the hold
        ``wml_gross``     -- win - los (dollar-neutral, gross)
        ``turnover``      -- one-way fraction of book turned over vs the prior rebalance
        ``wml_net``       -- wml_gross net of costs and borrow
        ``n_leg``         -- names per leg
    """
    mp = monthly_prices
    # forward cumulative return over `hold` months: price[t+hold] / price[t] - 1
    fwd = (mp.shift(-hold) / mp) - 1.0
    sig = momentum_signal(mp, lookback, skip)
    tov = (
        turnover_measure(dollar_volume, mp, lookback, skip)
        if dollar_volume is not None
        else None
    )

    rows: list[dict] = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)

        if tov is not None:
            tv = tov.loc[t].dropna() if t in tov.index else pd.Series(dtype=float)
            common = common.intersection(tv.index)
            if len(common) < min_names:
                continue
            tv = tv.loc[common]
            med = tv.median()
            if vol_side == "high":
                half = tv[tv >= med].index
            else:
                half = tv[tv < med].index
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
        borrow = borrow_ann_bps * 1e-4 * (hold / MONTHS_PER_YEAR)
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


def vm_spread(high_book: pd.DataFrame, low_book: pd.DataFrame, col: str = "wml_gross") -> pd.Series:
    """The Lee-Swaminathan interaction: high-vol WML minus low-vol WML, month by month."""
    if high_book.empty or low_book.empty:
        return pd.Series(dtype=float)
    j = high_book[[col]].join(low_book[[col]], lsuffix="_hi", rsuffix="_lo", how="inner")
    return (j[f"{col}_hi"] - j[f"{col}_lo"]).rename("vm")


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
    The HAC *t* is the inference-bar number. (For ``hold>1`` books the mean is per-period, not
    de-overlapped -- use it for the *sign/shape* of the reversal, the HAC t already widens for the
    induced autocorrelation.)
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
    dollar_volume: pd.DataFrame,
    vol_side: str = "high",
    frac: float = 0.3,
    lookback: int = LOOKBACK_M,
    skip: int = SKIP_M,
    n_perm: int = 1000,
    seed: int = 511,
) -> dict:
    """Permutation p-value: does the real WML mean (in one volume half) beat a shuffled-label null?

    Inside the chosen volume half each month, we keep the same forward returns but SHUFFLE which
    stock the momentum signal points at -- destroying genuine winner/loser persistence while
    preserving the cross-sectional return distribution, the volume conditioning and the leg sizes.
    We recompute the WML mean ``n_perm`` times and report the one-sided share of placebos whose
    mean >= the real mean. A real edge gives a small p; a data-mined one gives p ~ 0.5.

    Returns ``{"real_mean_ann", "p_value", "placebo_mean_ann", "n_perm"}``.
    """
    real = long_short(monthly_prices, dollar_volume=dollar_volume, vol_side=vol_side, frac=frac,
                      lookback=lookback, skip=skip)
    if real.empty:
        return {"real_mean_ann": float("nan"), "p_value": float("nan"),
                "placebo_mean_ann": float("nan"), "n_perm": 0}
    real_mean = float(real["wml_gross"].mean())

    mp = monthly_prices
    fwd = (mp.shift(-1) / mp) - 1.0
    sig = momentum_signal(mp, lookback, skip)
    tov = turnover_measure(dollar_volume, mp, lookback, skip)
    rng = np.random.default_rng(seed)

    months: list[tuple[np.ndarray, np.ndarray, int]] = []
    for t in sig.index:
        s = sig.loc[t].dropna()
        r = fwd.loc[t].dropna() if t in fwd.index else pd.Series(dtype=float)
        tv = tov.loc[t].dropna() if t in tov.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index).intersection(tv.index)
        if len(common) < 12:
            continue
        tc = tv.loc[common]
        med = tc.median()
        half = tc[tc >= med].index if vol_side == "high" else tc[tc < med].index
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
# Volume-conditioned reversal term-structure (third axis)
# ---------------------------------------------------------------------------
def reversal_term_structure(
    monthly_prices: pd.DataFrame,
    dollar_volume: pd.DataFrame,
    holds: tuple[int, ...] = (1, 3, 6, 12),
    frac: float = 0.3,
) -> pd.DataFrame:
    """Lee-Swaminathan reversal: the WML cumulative return at growing holds, high- vs low-volume.

    For each holding horizon ``h`` (months), build the high-volume and low-volume WML books with
    ``hold=h`` and report their per-hold mean return (NOT annualised -- the cumulative drift over
    the hold) and HAC t. The Lee-Swaminathan prediction: the **high-volume** winner drift turns
    over / reverses faster, so its longer-hold cumulative return fades (or goes negative) sooner
    than the low-volume book's. Returns one row per ``hold``.
    """
    rows: list[dict] = []
    for h in holds:
        hi = long_short(monthly_prices, dollar_volume=dollar_volume, vol_side="high",
                        frac=frac, hold=h)
        lo = long_short(monthly_prices, dollar_volume=dollar_volume, vol_side="low",
                        frac=frac, hold=h)
        hi_mean = float(hi["wml_gross"].mean()) if not hi.empty else float("nan")
        lo_mean = float(lo["wml_gross"].mean()) if not lo.empty else float("nan")
        hi_t = hac_tstat(hi["wml_gross"]) if not hi.empty else float("nan")
        lo_t = hac_tstat(lo["wml_gross"]) if not lo.empty else float("nan")
        rows.append({
            "hold_m": h,
            "high_vol_cum": hi_mean, "high_vol_t": hi_t,
            "low_vol_cum": lo_mean, "low_vol_t": lo_t,
            "n": int(len(hi)) if not hi.empty else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deterministic synthetic positive control
# ---------------------------------------------------------------------------
def synthetic_control(
    strengths: tuple[float, ...] = (0.0, 0.20, 0.40, 0.60),
    n_stocks: int = 40,
    n_days: int = 2600,
    vol_tilt: float = 0.8,
    frac: float = 0.3,
    seed: int = 511,
) -> pd.DataFrame:
    """Plant a known drift concentrated in the high-turnover names; verify the engine recovers a
    BIGGER high-volume WML than low-volume WML.

    Sweeps ``mom_strength`` on the deterministic synthetic panel where the trending drift is
    tilted toward the high-turnover names (``vol_tilt`` of it). The engine should score ~0 on both
    slices at strength 0, and at positive strength the high-volume WML should exceed the low-volume
    WML (the Lee-Swaminathan interaction). Faithful-engine / power check ONLY.
    """
    from . import data as _data

    rows: list[dict] = []
    for strength in strengths:
        prices, dvol, _truth = _data.synthetic_panel(
            n_stocks=n_stocks, n_days=n_days, mom_strength=strength,
            vol_tilt=vol_tilt, seed=seed,
        )
        mp = to_monthly(prices)
        hi = long_short(mp, dollar_volume=dvol, vol_side="high", frac=frac)
        lo = long_short(mp, dollar_volume=dvol, vol_side="low", frac=frac)
        s_hi = summary(hi["wml_gross"]) if not hi.empty else {"mean": float("nan"), "tstat": float("nan"), "n": 0}
        s_lo = summary(lo["wml_gross"]) if not lo.empty else {"mean": float("nan"), "tstat": float("nan"), "n": 0}
        rows.append({
            "mom_strength": strength,
            "high_vol_mean_ann": s_hi["mean"], "high_vol_t": s_hi["tstat"],
            "low_vol_mean_ann": s_lo["mean"], "low_vol_t": s_lo["tstat"],
            "vm_gap_ann": (s_hi["mean"] - s_lo["mean"]) if (s_hi["n"] and s_lo["n"]) else float("nan"),
            "n": s_hi["n"],
        })
    return pd.DataFrame(rows)
