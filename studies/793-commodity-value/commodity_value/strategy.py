"""Strategy + inference for Study 793 — Cross-sectional commodity *value*.

The claim (Asness, Moskowitz & Pedersen 2013, "Value and momentum everywhere", the
commodity value leg): **a commodity is "cheap" when its price has fallen over the long
horizon — rank the basket on the ratio of the price ~5 years ago to the current price,
buy the cheap (fallen) third and short the expensive (risen) third, rebalance monthly.**
This is *long-horizon reversal*, the mirror image of the sibling momentum sort — and the
opposite horizon to sibling [792](../../792-commodity-momentum/) (12-1 continuation). It is
also the **commodity value leg on its own**, distinct from the mixed multi-asset
value+momentum combo of [638](../../638-value-momentum-everywhere/).

Construction, stated once:

* **Signal — 5-year value.** At each month-end ``t``, the value score is
  ``log(P_ref) − log(P_t)`` where ``P_ref`` is the average price over the window
  ~4.5–5.5 years before ``t`` (AMP's 5-year reference) and ``P_t`` is the current price.
  A **positive** score means the price fell → the commodity is **cheap** → long.
* **Portfolio.** Rank live assets on the value score; long the top third (cheapest), short
  the bottom third (most expensive), equal-weight each leg (1x long / 1x short = gross 2x
  NAV, dollar-neutral).
* **The ONE execution lag.** Weights formed from data through the close of month ``t`` earn
  the return of month ``t+1`` — a single ``shift(1)``, applied once. No second silent lag.
* **Costs.** One-way cost x traded notional (turnover = sum of ``|Delta w|`` each month);
  the short leg pays borrow at ``borrow_bps_yr`` on its 1x NAV. Gross is labelled gross,
  net is labelled net.

Statistics: the decisive number is the **HAC/Newey-West t** on the monthly L/S return
(monthly factor returns are serially correlated); a one-sample t and an annualised Sharpe
accompany it. A **random-rank placebo** (>= 20 seeds) replaces the value signal with i.i.d.
noise ranks and asks how often that clears the bar; a sub-period contrast splits the sample
and tests the difference. The synthetic positive control plants a known reversal edge and
confirms the sort recovers it while the null (edge=0) does not systematically fire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ASSETS = 6              # a month needs at least this many ranked assets to trade
VAL_LOOKBACK = 60          # months to the value reference price (~5 years, AMP)
VAL_WINDOW = 13            # width of the reference window (~4.5-5.5y ago), centred on LOOKBACK
PLACEBO_SEEDS = 40
TRADING_MONTHS = 12


# --------------------------------------------------------------------------- #
# Robust inference primitives
# --------------------------------------------------------------------------- #
def _nw_lags(n: int) -> int:
    """Newey-West automatic lag length (Bartlett kernel, floor(4*(n/100)^(2/9)))."""
    return int(np.floor(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))


def hac_t(returns, lags: int | None = None) -> dict:
    """Newey-West (HAC) t of a sample mean (Bartlett kernel). The decisive statistic."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 12:
        return {"mean": np.nan, "mean_bps": np.nan, "t": np.nan, "n": n}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = _nw_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "mean_bps": float(mu * 1e4),
            "t": float(mu / se) if se > 0 else np.nan, "n": n}


def one_sample_t(x) -> float:
    """One-sample t of mean(x) vs 0 (i.i.d. SE)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def sharpe(r: pd.Series | np.ndarray, periods: int = TRADING_MONTHS) -> float:
    """Annualised Sharpe of a periodic (monthly) return series."""
    r = pd.Series(r).dropna()
    if len(r) < 12 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def welch_t(a, b) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Signal + portfolio
# --------------------------------------------------------------------------- #
def value_signal(prices: pd.DataFrame, lookback: int = VAL_LOOKBACK,
                 window: int = VAL_WINDOW) -> pd.DataFrame:
    """AMP 5-year commodity value: ``log(P_ref) - log(P_t)``, computed through month t only.

    ``P_ref`` is the average price over a ``window``-month band centred ``lookback`` months
    before ``t`` (AMP's ~4.5-5.5-years-ago reference), so a single stale print does not
    swing the score. A **positive** score = the price fell over the long horizon = the
    commodity is **cheap**. Requires the current price and at least a few prints in the
    reference band, else the cell is NaN and that asset is unranked that month.
    """
    logp = np.log(prices)
    half = window // 2
    lo, hi = lookback + half, lookback - half            # months-ago band (lo=older, hi=newer)
    # average of log price over the reference band [t-lo .. t-hi]
    band = pd.concat([logp.shift(k) for k in range(hi, lo + 1)], axis=1)
    band.columns = pd.MultiIndex.from_product([range(hi, lo + 1), logp.columns])
    ref = band.T.groupby(level=1).mean().T[logp.columns]
    cnt = pd.concat([logp.shift(k).notna() for k in range(hi, lo + 1)], axis=1)
    cnt.columns = pd.MultiIndex.from_product([range(hi, lo + 1), logp.columns])
    n_ref = cnt.T.groupby(level=1).sum().T[logp.columns]
    sig = ref - logp
    return sig.where((n_ref >= max(1, window // 3)) & logp.notna())


def rank_weights(sig: pd.DataFrame, min_assets: int = MIN_ASSETS,
                 frac: float = 1.0 / 3.0) -> pd.DataFrame:
    """Long the top ``frac`` (cheapest = highest value score), short the bottom ``frac``,
    equal-weight each leg (1x long / 1x short, gross 2x NAV, dollar-neutral). NaN row if
    fewer than ``min_assets`` live."""
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    for t in sig.index:
        row = sig.loc[t].dropna()
        n = len(row)
        if n < min_assets:
            w.loc[t] = np.nan
            continue
        k = max(1, int(round(n * frac)))
        order = row.sort_values()
        w.loc[t, order.index[-k:]] = 1.0 / k       # cheapest (long)
        w.loc[t, order.index[:k]] = -1.0 / k       # most expensive (short)
    return w


def portfolio(mret: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """L/S monthly return with the ONE execution lag (weights at month t earn month t+1)
    and the one-way traded notional (sum |Delta w|) per month.

    ``w`` is aligned to ``mret``'s index before shifting so the value signal (computed on
    the price panel) and the P&L (computed on the total-return panel) share a clock.
    """
    w = w.reindex(mret.index)
    wl = w.shift(1)
    live = wl.abs().sum(axis=1) > 0
    ret = (wl * mret).sum(axis=1, min_count=1)[live].dropna()
    traded = (wl.fillna(0.0) - wl.shift(1).fillna(0.0)).abs().sum(axis=1)
    return ret, traded.reindex(ret.index).fillna(0.0)


# --------------------------------------------------------------------------- #
# The headline report
# --------------------------------------------------------------------------- #
def value_report(prices: pd.DataFrame, mret: pd.DataFrame, frac: float = 1.0 / 3.0) -> dict:
    """Build the 5-year value L/S book and its headline statistics (gross)."""
    sig = value_signal(prices)
    w = rank_weights(sig, frac=frac)
    ret, traded = portfolio(mret, w)
    h = hac_t(ret)
    return {
        "ret": ret, "traded": traded,
        "n_months": len(ret),
        "start": str(ret.index.min().date()) if len(ret) else "",
        "end": str(ret.index.max().date()) if len(ret) else "",
        "mean_bps": h["mean_bps"], "hac_t": h["t"], "hac_lags": _nw_lags(len(ret)),
        "one_sample_t": one_sample_t(ret),
        "sharpe": sharpe(ret),
        "ann_pct": float(ret.mean() * TRADING_MONTHS * 100) if len(ret) else float("nan"),
        "avg_turnover": float(traded.mean()) if len(traded) else float("nan"),
        "hit_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
    }


def long_short_legs(prices: pd.DataFrame, mret: pd.DataFrame, frac: float = 1.0 / 3.0) -> dict:
    """Decompose the L/S book into its cheap (long) and expensive (short) legs, each
    measured as an *excess-of-basket* return so the two legs are comparable to each other
    and to the equal-weight basket they are drawn from."""
    sig = value_signal(prices)
    w = rank_weights(sig, frac=frac).reindex(mret.index)
    wl = w.shift(1)
    long_w = wl.clip(lower=0.0)
    short_w = (-wl).clip(lower=0.0)
    live = wl.abs().sum(axis=1) > 0
    ew = mret.mean(axis=1)                               # equal-weight basket return
    long_ret = ((long_w * mret).sum(axis=1, min_count=1) - ew)[live].dropna()
    short_ret = (ew - (short_w * mret).sum(axis=1, min_count=1))[live].dropna()
    return {
        "long_excess_bps": float(long_ret.mean() * 1e4), "long_t": hac_t(long_ret)["t"],
        "short_excess_bps": float(short_ret.mean() * 1e4), "short_t": hac_t(short_ret)["t"],
        "basket_ann_pct": float(ew.reindex(long_ret.index).mean() * TRADING_MONTHS * 100)
        if len(long_ret) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Random-rank placebo (>= 20 seeds by house rule)
# --------------------------------------------------------------------------- #
def random_placebo(prices: pd.DataFrame, mret: pd.DataFrame, n_seeds: int = PLACEBO_SEEDS,
                   frac: float = 1.0 / 3.0, base_seed: int = 793) -> dict:
    """Same construction, random signal: per seed replace the value signal with i.i.d.
    noise ranks (aligned to the real signal's live mask so breadth is held fixed), build
    the same L/S book, record its HAC t and Sharpe. Averaged over ``n_seeds``.

    ``p`` is the share of seeds whose HAC t >= the real HAC t (right tail — the claim is a
    positive value premium)."""
    sig = value_signal(prices)
    real_t = value_report(prices, mret, frac=frac)["hac_t"]
    ts, shs = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        noise = pd.DataFrame(rng.standard_normal(sig.shape),
                             index=sig.index, columns=sig.columns).where(sig.notna())
        w = rank_weights(noise, frac=frac)
        ret, _ = portfolio(mret, w)
        ts.append(hac_t(ret)["t"])
        shs.append(sharpe(ret))
    ts, shs = np.asarray(ts, float), np.asarray(shs, float)
    return {"real_t": real_t, "mean_t": float(np.nanmean(ts)), "sd_t": float(np.nanstd(ts)),
            "mean_sharpe": float(np.nanmean(shs)),
            "frac_t_ge_2": float(np.nanmean(np.abs(ts) >= 2.0)),
            "p_value": float(np.nanmean(ts >= real_t)) if np.isfinite(real_t) else float("nan"),
            "n_seeds": n_seeds, "ts": ts}


# --------------------------------------------------------------------------- #
# Sub-period contrast (a justified split: the effective-sample midpoint)
# --------------------------------------------------------------------------- #
def subperiod_contrast(prices: pd.DataFrame, mret: pd.DataFrame, split: str,
                       frac: float = 1.0 / 3.0) -> dict:
    """L/S HAC t and Sharpe before vs since ``split``, plus a Welch t of the difference in
    the monthly means (is any 'decay' real or just noise?)."""
    ret = value_report(prices, mret, frac=frac)["ret"]
    split_ts = pd.Timestamp(split)
    early = ret[ret.index < split_ts]
    late = ret[ret.index >= split_ts]
    return {
        "n_early": len(early), "n_late": len(late),
        "early_bps": float(early.mean() * 1e4) if len(early) else float("nan"),
        "late_bps": float(late.mean() * 1e4) if len(late) else float("nan"),
        "early_t": hac_t(early)["t"], "late_t": hac_t(late)["t"],
        "early_sharpe": sharpe(early), "late_sharpe": sharpe(late),
        "welch_t_diff": welch_t(late, early),
    }


# --------------------------------------------------------------------------- #
# The costed timer — one-way x NAV, shorts pay borrow
# --------------------------------------------------------------------------- #
def costed_timer(prices: pd.DataFrame, mret: pd.DataFrame, cost_bps: float,
                 borrow_bps_yr: float = 50.0, frac: float = 1.0 / 3.0) -> dict:
    """Net-of-cost L/S book. Each month::

        net = gross - (traded_notional * one-way cost) - (short NAV * borrow / 12)

    ``traded_notional`` is the one-way sum of |Delta w| (turnover x NAV); the short leg
    always carries 1x NAV, so borrow is a flat ``borrow_bps_yr/12`` monthly drag. Returns
    gross/net annualised, net Sharpe and the decisive net HAC t."""
    sig = value_signal(prices)
    w = rank_weights(sig, frac=frac)
    gross, traded = portfolio(mret, w)
    trade_drag = traded * cost_bps * 1e-4
    borrow_drag = borrow_bps_yr * 1e-4 / TRADING_MONTHS          # 1x short NAV, monthly
    net = (gross - trade_drag - borrow_drag).dropna()
    h = hac_t(net)
    return {
        "cost_bps": cost_bps, "borrow_bps_yr": borrow_bps_yr,
        "gross_ann_pct": float(gross.mean() * TRADING_MONTHS * 100) if len(gross) else float("nan"),
        "net_ann_pct": float(net.mean() * TRADING_MONTHS * 100) if len(net) else float("nan"),
        "net_mean_bps": h["mean_bps"], "net_t": h["t"],
        "net_sharpe": sharpe(net), "net_one_sample_t": one_sample_t(net),
        "avg_turnover": float(traded.mean()) if len(traded) else float("nan"), "n": len(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic positive control (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(prices: pd.DataFrame, frac: float = 1.0 / 3.0) -> dict:
    """Run the 5-year value L/S sort on a synthetic **price** panel; report HAC t etc.

    The L/S P&L is measured on the panel's own monthly returns (``pct_change``)."""
    mret = prices.pct_change()
    r = value_report(prices, mret, frac=frac)
    return {"hac_t": r["hac_t"], "sharpe": r["sharpe"],
            "mean_bps": r["mean_bps"], "n_months": r["n_months"]}
