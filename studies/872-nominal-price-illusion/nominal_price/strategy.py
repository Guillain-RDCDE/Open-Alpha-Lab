"""Strategy + inference for Study 872 — Nominal-Price Illusion.

The claim (Kumar 2009; Birru & Wang 2016): the **nominal share price** — the raw
dollar number one share trades at — is a money-illusion characteristic that carries no
value information, yet retail lottery demand clusters in **low-priced** names. If that
demand over-prices cheap-looking stocks, low nominal-price names should show the
lottery look (**higher volatility, more right-skew**) and **lower risk-adjusted
returns**. A book that is **short the cheap (low-priced) names / long the expensive
ones** should therefore earn a positive spread.

We measure the mirror-image spread ``lo − hi`` = (low-priced book) − (high-priced
book) so the *sign* reads straight off the claim:

* claim TRUE  → cheap names under-earn → ``lo − hi < 0``;
* wrong sign  → cheap names out-earn   → ``lo − hi > 0`` (⇒ **None**);
* flat        → nominal price prices nothing → ``lo − hi ≈ 0`` (the money-illusion
  null: a pure characteristic, no return content).

Because the raw-return spread can be dominated by the extra *risk* the cheap book
carries, we also report each book's annualised **volatility**, return **skew**, and
**Sharpe** — the over-priced-lottery hypothesis is fundamentally about *risk-adjusted*
underperformance (same or lower return for much more risk).

This is distinct from four siblings — see ``docs/references.md``:

* [11-vanishing-penny](../../11-vanishing-penny/) — literal **penny stocks** and their
  disappearance, not a cross-sectional price-level sort on liquid names;
* [365-lottery-max-effect](../../365-lottery-max-effect/) — the single **MAX** daily
  return, a realized tail statistic, not the *price level* itself;
* [250-reverse-split](../../250-reverse-split/) — the **event** of a reverse split
  (a discrete price-level reset), not a continuous cross-sectional price sort;
* [93-round-numbers](../../93-round-numbers/) — price **round-number** magnetism
  ($100 barriers), a within-name level effect, not a cross-name cheap-vs-dear sort.

Method:

* **Close-to-close returns.** Per-name daily simple-return panel from adjusted Close.
* **Price-level signal.** The (adjusted) Close *is* the signal — the nominal price
  proxy (exact at the as-of date, back-adjusted into the past; see ``data.py``).
* **Point-in-time sort.** On each day ``t`` rank the cross-section by the price known at
  the close of ``t-1`` (one ``shift``) and hold day ``t``. Long the bottom ``frac`` (low
  price), short the top ``frac`` (high price); equal weight.
* **Inference.** Newey-West (HAC) *t* on the daily long-short spread; a one-sample *t*
  and a pooled Welch *t* (cheap book vs dear book) cross-check; a permutation placebo
  breaks the signal->outcome link; a costed timer charges the round-trip friction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + signal
# --------------------------------------------------------------------------- #
def close_returns(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily simple close-to-close returns (index=date, columns=ticker)."""
    closes = pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()
    return closes.pct_change()


def close_prices(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The price-level panel (index=date, columns=ticker) — the nominal-price proxy."""
    return pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()


def _skew(x: np.ndarray) -> float:
    """Sample skewness (bias-uncorrected, population moment)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    m = x.mean()
    s = x.std(ddof=0)
    if s <= 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 3))


# --------------------------------------------------------------------------- #
# The cross-sectional sort -> long-low-price / short-high-price spread
# --------------------------------------------------------------------------- #
def price_spreads(
    panel: dict[str, pd.DataFrame],
    frac: float = 0.3,
    min_names: int = 10,
) -> pd.DataFrame:
    """Daily equal-weight low-price-minus-high-price fractile spread.

    On each day ``t`` names are ranked by the **price level** known at the close of
    ``t-1`` (one ``shift``). ``lo`` = mean forward day-``t`` return of the bottom
    ``frac`` (cheapest, the long); ``hi`` = mean of the top ``frac`` (most expensive,
    the short). ``spread = lo - hi``. Days with fewer than ``min_names`` ranked names
    are dropped.
    """
    prices = close_prices(panel)
    ret = prices.pct_change()
    ret = ret.reindex(columns=prices.columns)
    sig = prices.shift(1)                       # price known at close t-1
    S = sig.to_numpy(dtype=float)
    R = ret.to_numpy(dtype=float)
    idx = prices.index
    out_spread, out_lo, out_hi, out_n, out_t = [], [], [], [], []
    for i in range(len(idx)):
        row = S[i]
        valid = np.where(~np.isnan(row))[0]
        n = len(valid)
        if n < min_names:
            continue
        k = max(1, int(np.floor(n * frac)))
        order = valid[np.argsort(row[valid], kind="stable")]
        low = order[:k]        # cheapest  -> long
        high = order[-k:]      # priciest  -> short
        rr = R[i]
        lo = float(np.nanmean(rr[low]))
        hi = float(np.nanmean(rr[high]))
        if np.isnan(lo) or np.isnan(hi):
            continue
        out_spread.append(lo - hi); out_lo.append(lo); out_hi.append(hi)
        out_n.append(n); out_t.append(idx[i])
    return pd.DataFrame(
        {"spread": out_spread, "lo": out_lo, "hi": out_hi, "n": out_n}, index=out_t
    ).sort_index()


# --------------------------------------------------------------------------- #
# Inference primitives (verbatim from the desk's canonical set, study 803)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline stats — spread + the risk-adjusted lottery read
# --------------------------------------------------------------------------- #
def _ann_vol(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    return float(x.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(x) > 1 else float("nan")


def _sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def price_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    lo = spreads["lo"].to_numpy(dtype=float)
    hi = spreads["hi"].to_numpy(dtype=float)
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "lo_bps": float(np.nanmean(lo) * 1e4),
        "hi_bps": float(np.nanmean(hi) * 1e4),
        "welch_t": welch_t(lo, hi),
        # the risk-adjusted lottery read: do cheap names carry more risk for less reward?
        "lo_vol": _ann_vol(lo),
        "hi_vol": _ann_vol(hi),
        "lo_skew": _skew(lo),
        "hi_skew": _skew(hi),
        "lo_sharpe": _sharpe(lo),
        "hi_sharpe": _sharpe(hi),
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the sort?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    panel: dict[str, pd.DataFrame],
    frac: float = 0.3,
    min_names: int = 10,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 872,
) -> dict:
    """Keep the price sort but read each day's forward return from a **column-permuted**
    panel (signal->outcome link broken, each day's cross-sectional distribution
    preserved). Two-sided: report the share of permuted worlds at least as extreme (in
    ``|mean|``) as the observed spread."""
    prices = close_prices(panel)
    ret = prices.pct_change()
    cols = list(prices.columns)
    ncol = len(cols)
    sig = prices.shift(1)
    obs = float(price_spreads(panel, frac, min_names)["spread"].mean())

    ret_mat = ret.to_numpy(dtype=float)
    pos_of = {c: i for i, c in enumerate(cols)}
    rows_idx, lows, highs = [], [], []
    row_lookup = {t: r for r, t in enumerate(ret.index)}
    for t in prices.index:
        s = sig.loc[t].dropna()
        if len(s) < min_names:
            continue
        k = max(1, int(np.floor(len(s) * frac)))
        order = s.sort_values()
        rows_idx.append(row_lookup[t])
        lows.append(np.array([pos_of[c] for c in order.index[:k]]))
        highs.append(np.array([pos_of[c] for c in order.index[-k:]]))
    rows_idx = np.asarray(rows_idx)

    means = []
    if len(rows_idx):
        M = ret_mat[rows_idx]
        kl = max(len(a) for a in lows)
        kh = max(len(a) for a in highs)

        def _pad(books, kmax):
            P = np.zeros((len(books), kmax), dtype=int)
            V = np.zeros((len(books), kmax), dtype=bool)
            for j, a in enumerate(books):
                P[j, :len(a)] = a
                V[j, :len(a)] = True
            return P, V

        LOW, LOWv = _pad(lows, kl)
        HIGH, HIGHv = _pad(highs, kh)
        rows_ar = np.arange(len(rows_idx))[:, None]

        def _masked_mean(pos, valid, perm):
            vals = M[rows_ar, perm[pos]]
            vals = np.where(valid, vals, np.nan)
            return np.nanmean(vals, axis=1)

        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                lo_v = _masked_mean(LOW, LOWv, perm)
                hi_v = _masked_mean(HIGH, HIGHv, perm)
                means.append(np.nanmean(lo_v - hi_v))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((np.abs(means) >= abs(obs)).mean()) if len(means) else float("nan"),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-cheap / short-dear book.

    The price-level sort turns over slowly (a name's price rank is highly persistent),
    but to stay comparable to the desk's other cross-sectional timers we charge a
    conservative 2 sides × one-way cost × NAV per day on the long-short book, plus
    borrow on the short leg. Note the short leg is the **expensive** names here — the
    cheap-name lottery segment is the *long*, which flips the usual borrow story.
    """
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * cost_bps / 1e4
    borrow_daily = (borrow_bps_yr / 1e4) / 365.0
    net = sp - round_trip_cost - borrow_daily
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    sharpe = net_mean / sd * np.sqrt(TRADING_DAYS) if sd and sd > 0 else float("nan")
    return {
        "n_days": n,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_day": (round_trip_cost + borrow_daily) * 1e4,
        "ann_net_pct": net_mean * TRADING_DAYS * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], frac: float = 0.3) -> dict:
    """Run the headline price-sort stats on a synthetic panel."""
    sp = price_spreads(panel, frac)
    ts = price_stats(sp)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"],
            "lo_vol": ts["lo_vol"], "hi_vol": ts["hi_vol"],
            "lo_skew": ts["lo_skew"], "hi_skew": ts["hi_skew"]}
