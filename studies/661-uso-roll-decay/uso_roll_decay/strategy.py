"""Strategy + inference for Study 661 — USO-Roll-Decay.

The claim: **"USO tracks oil"** — but USO doesn't hold crude, it holds (mostly front-month)
**WTI futures**, and when the curve is in contango each monthly roll forces the fund to sell
the cheaper expiring contract and buy the pricier next-month one — a structural "sell low, buy
high" that bleeds NAV relative to the spot/front price everyone quotes as "oil". Over enough
contango, the folklore says, USO becomes a mechanical loser even when oil itself goes nowhere.

Measurements:

* **Cumulative divergence** — total return and CAGR of USO vs CL=F since USO's 2006-04-10
  inception, the plainest possible test of "does the ETF track the commodity".
* **Daily roll-drag** — the daily log-return gap (USO − CL=F), a naive one-sample *t*, a
  Newey-West (HAC) cross-check at several lags, a Wilson-bounded down-day hit rate, and a
  circular block-bootstrap CI/placebo for the mean gap (autocorrelation-robust, distribution-
  free companion to the *t*-stats).
* **Contango-stress regimes** — the same daily gap split into the two hardcoded 2009/2020
  super-contango windows vs the rest of the tape: is the drag a smooth daily grind, or
  concentrated in a few crisis episodes?
* **The April-2020 negative-oil episode** — a named, non-parametric case study: what actually
  happened to USO the day the front WTI contract settled at -$37.63.
* **The trade (third axis)** — a constant-notional, monthly-rebalanced **long CL=F / short
  USO** book, charged one-way costs at each rebalance plus an annualized borrow fee on the
  short leg, gross and net, with an ex-crisis vs crisis-only decomposition.

The decisive number is the daily-gap Welch/HAC *t* on the REAL tape; the honest question is
whether the "long spot / short USO" trade the folklore implies is actually a repeatable,
deployable edge, or a backtest that only worked because it happened to straddle two
once-a-decade storage crises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Gap frame
# --------------------------------------------------------------------------- #
def gap_frame(uso: pd.DataFrame, clf: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of USO and CL=F, aligned, plus the gap (USO − CL=F).

    ``np.log`` of the one non-positive CL=F print (2020-04-20, -$37.63) is NaN, so the two
    adjacent daily returns (into and out of that day) drop out automatically via ``dropna`` —
    no manual row surgery, so no adjacency artefact on the days around it.
    """
    idx = uso.index.intersection(clf.index)
    u = uso.loc[idx, "Close"].sort_index()
    c = clf.loc[idx, "Close"].sort_index()
    with np.errstate(invalid="ignore"):
        r_uso = np.log(u).diff()
        r_clf = np.log(c).diff()
    df = pd.DataFrame({"uso_close": u, "clf_close": c, "r_uso": r_uso, "r_clf": r_clf})
    df["gap"] = df["r_uso"] - df["r_clf"]
    return df.dropna(subset=["gap"])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    """Naive (i.i.d.) one-sample t that mean(x) != 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def newey_west_mean_t(x: np.ndarray, lags: int = 5) -> dict:
    """HAC (Newey-West, Bartlett kernel) t of the sample mean of a single series ``x``.

    Unlike the FOMC-day dummy regression, there is no "other group" here — every day carries a
    gap — so this is a one-sample long-run-variance estimator, not a two-group split.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    m = x.mean()
    xm = x - m
    s = float((xm ** 2).sum()) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        s += 2.0 * w * float((xm[lag:] * xm[:-lag]).sum()) / n
    se = np.sqrt(s / n)
    return {"mean": float(m), "se": float(se), "t": float(m / se) if se > 0 else float("nan"),
            "n": n}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def block_bootstrap_mean_ci(x: np.ndarray, block: int = 21, n_boot: int = 5000,
                            seed: int = 661, alpha: float = 0.05) -> dict:
    """Circular block-bootstrap CI + p(mean >= 0) for the mean of ``x``.

    Blocks of ``block`` consecutive days (~one trading month, the roll-cycle frequency)
    preserve the gap series' serial correlation that an i.i.d. resample would destroy.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[i] = x[idx].ravel()[:n].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "p_ge0": float((means >= 0).mean()),
            "n_boot": n_boot, "block": block}


# --------------------------------------------------------------------------- #
# The headline — cumulative divergence and the daily roll-drag
# --------------------------------------------------------------------------- #
def cumulative_stats(df: pd.DataFrame) -> dict:
    """Total return and CAGR of USO vs CL=F over the full sample."""
    u0, u1 = df["uso_close"].iloc[0], df["uso_close"].iloc[-1]
    c0, c1 = df["clf_close"].iloc[0], df["clf_close"].iloc[-1]
    years = (df.index[-1] - df.index[0]).days / 365.25
    tr_uso = u1 / u0 - 1.0
    tr_clf = c1 / c0 - 1.0
    cagr_uso = (u1 / u0) ** (1.0 / years) - 1.0
    cagr_clf = (c1 / c0) ** (1.0 / years) - 1.0
    return {"years": years, "start": df.index[0], "end": df.index[-1],
            "total_return_uso": float(tr_uso), "total_return_clf": float(tr_clf),
            "cagr_uso": float(cagr_uso), "cagr_clf": float(cagr_clf),
            "cagr_gap": float(cagr_uso - cagr_clf)}


def headline_drag_stats(df: pd.DataFrame, nw_lags: tuple[int, ...] = (5, 21, 63)) -> dict:
    """Daily gap (USO − CL=F): mean/ann, naive t, NW t at several lags, Wilson hit rate."""
    g = df["gap"].values
    k_down = int((g < 0).sum())
    lo, hi = wilson_interval(k_down, len(g))
    out = {
        "n": len(g), "mean": float(g.mean()), "ann_pct": float(g.mean() * TRADING_DAYS * 100),
        "naive_t": one_sample_t(g),
        "hit_down": k_down, "hit_rate": k_down / len(g), "hit_lo": lo, "hit_hi": hi,
    }
    for lags in nw_lags:
        out[f"nw_t_{lags}"] = newey_west_mean_t(g, lags=lags)["t"]
    return out


# --------------------------------------------------------------------------- #
# Contango-stress regimes
# --------------------------------------------------------------------------- #
def regime_stats(df: pd.DataFrame, stress: pd.Series) -> dict:
    """Daily gap inside the hardcoded contango-stress windows vs the rest of the tape."""
    g = df["gap"]
    stress = stress.reindex(g.index).fillna(False)
    a, b = g[stress].values, g[~stress].values
    total_log = float(g.sum())
    stress_log = float(g[stress].sum())
    return {
        "n_stress": int(stress.sum()), "n_rest": int((~stress).sum()),
        "stress_ann_pct": float(a.mean() * TRADING_DAYS * 100),
        "rest_ann_pct": float(b.mean() * TRADING_DAYS * 100),
        "welch_t_stress_vs_rest": welch_t(a, b),
        "one_t_stress": one_sample_t(a), "one_t_rest": one_sample_t(b),
        "cum_log_total": total_log, "cum_log_stress": stress_log,
        "stress_share_of_cum": float(stress_log / total_log) if total_log != 0 else float("nan"),
    }


def per_window_stats(df: pd.DataFrame, windows) -> list[dict]:
    """One row per hardcoded contango-stress window: n, mean ann%, one-sample t."""
    g = df["gap"]
    rows = []
    for lo, hi, label in windows:
        x = g[(g.index >= lo) & (g.index <= hi)].values
        rows.append({"lo": lo, "hi": hi, "label": label, "n": len(x),
                     "ann_pct": float(x.mean() * TRADING_DAYS * 100), "t": one_sample_t(x)})
    return rows


# --------------------------------------------------------------------------- #
# The April-2020 negative-oil case study
# --------------------------------------------------------------------------- #
def negative_oil_case(uso: pd.DataFrame, clf: pd.DataFrame,
                      lo: str = "2020-04-16", hi: str = "2020-04-24") -> pd.DataFrame:
    """USO vs CL=F, simple (not log) returns, around the -$37.63 negative-oil settlement.

    Simple returns stay well-defined across a sign flip; log returns do not, which is exactly
    why this window is treated as a separate case study instead of folded into the continuous
    gap series.
    """
    u = uso.loc[lo:hi, "Close"]
    c = clf.loc[lo:hi, "Close"]
    out = pd.DataFrame({"uso_close": u, "clf_close": c})
    out["uso_ret_pct"] = u.pct_change() * 100
    out["clf_ret_pct"] = c.pct_change() * 100
    return out


# --------------------------------------------------------------------------- #
# Third axis — the "long spot / short USO" carry-capture book
# --------------------------------------------------------------------------- #
def carry_book(df: pd.DataFrame) -> pd.Series:
    """Daily return of a constant-notional long-CL=F / short-USO book (gross, no costs)."""
    return -df["gap"]


def net_of_costs(book: pd.Series, borrow_annual: float = 0.0075, cost_bps: float = 5.0
                 ) -> pd.Series:
    """Charge the book its real frictions.

    * **Borrow** on the short USO leg — an assumed ``borrow_annual`` stock-loan fee (USO is a
      large, liquid, easy-to-borrow ETF; this is a conservative retail-platform assumption, not
      a crisis-borrow spike), accrued as ``borrow_annual / 252`` every day the short is held.
    * **Rebalancing turnover** — the book is rebalanced to constant notional on the first
      trading day of each calendar month; that rebalance trades **both legs**, so it pays
      ``2 x cost_bps`` one-way x NAV on rebalance days only (not every day).
    """
    idx = book.index
    months = idx.to_period("M")
    is_rebal = pd.Series(months != pd.Series(months, index=idx).shift(1).values, index=idx)
    borrow_drag = borrow_annual / TRADING_DAYS
    txn_drag = is_rebal.astype(float) * (2.0 * cost_bps / 1e4)
    return book - borrow_drag - txn_drag


def perf(r: pd.Series) -> dict:
    """Annualised return/vol/Sharpe, compounded max drawdown, worst day."""
    r = pd.Series(r).dropna()
    mu = r.mean() * TRADING_DAYS
    sd = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = mu / sd if sd > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"ann_ret": float(mu), "ann_vol": float(sd), "sharpe": float(sharpe),
            "max_dd": dd, "worst_day": float(r.min()), "n": int(len(r))}


def sign_shuffle_placebo(x: np.ndarray, n_draws: int = 5_000, seed: int = 661) -> dict:
    """Sign-shuffle placebo null: p = P[shuffled mean >= observed mean]."""
    x = np.asarray(x, dtype=float)
    obs = float(x.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=len(x))
        means[i] = float((signs * x).mean())
    return {"obs_mean": obs, "placebo_mean": float(means.mean()),
            "p_value": float((means >= obs).mean())}


def carry_capture_summary(df: pd.DataFrame, stress: pd.Series,
                          borrow_annual: float = 0.0075,
                          cost_bps_sweep: tuple[float, ...] = (5.0, 10.0),
                          nw_lags: int = 21) -> dict:
    """Gross/net performance of the carry book, HAC t, placebo p, and the ex-crisis split."""
    book = carry_book(df)
    stress = stress.reindex(book.index).fillna(False)
    out = {"gross": perf(book),
          "hac_t": newey_west_mean_t(book.values, lags=nw_lags)["t"],
          "placebo_p": sign_shuffle_placebo(book.values)["p_value"]}
    for cb in cost_bps_sweep:
        net = net_of_costs(book, borrow_annual=borrow_annual, cost_bps=cb)
        out[f"net_{cb:g}"] = perf(net)
        if cb == cost_bps_sweep[0]:
            out["net5_ex_stress"] = perf(net[~stress])
            out["net5_stress_only"] = perf(net[stress])
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(spot_r: np.ndarray, fund_r: np.ndarray, nw_lags: int = 5) -> dict:
    """Run the headline naive-t / NW-t split on a synthetic (spot, fund) pair."""
    gap = np.asarray(fund_r) - np.asarray(spot_r)
    return {"naive_t": one_sample_t(gap), "nw_t": newey_west_mean_t(gap, lags=nw_lags)["t"],
           "ann_pct": float(gap.mean() * TRADING_DAYS * 100)}
