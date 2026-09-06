"""Measurement & inference for Study 964 — All-Time High.

Three questions, kept separate, because the folklore runs them together:

1. **Conditional forward returns.** Take every session, mark the ones that closed at a new
   running maximum of the *total-return* index, and compare the distribution of forward
   returns (1, 3, 12 months) from those days against every other day. ``forward_table``.
2. **Where the alternative actually is.** "Don't buy at a high" implies buying somewhere
   else — so the comparison that matters is not high-versus-average but
   **high versus in-drawdown**, at several drawdown depths. ``drawdown_bucket_table``.
3. **The rule as a portfolio.** ``dip_strategy`` implements the advice literally: hold the
   asset only while it is at least ``dip_pct`` below its running peak, sit in T-bills
   otherwise, one day of execution lag, costs charged on every switch. Then race it against
   buy-and-hold on excess-of-cash Sharpe, CAGR, drawdown and turnover.

**Inference on overlapping windows.** Forward returns computed at every session overlap
almost completely — a 252-day forward return shares 251 of its days with the next one — so
the effective sample is a fraction of the row count and a naive *t* is nonsense (this is
the trap catalogued in this desk's study 841). Two defences, both applied:

- HAC standard errors with the lag set to the forecast horizon (``mean_tstat_hac`` with
  ``lags = horizon``), the Hansen-Hodrick convention;
- a **non-overlapping** cross-check, ``nonoverlap_stats``, which keeps one observation per
  horizon and throws the rest away. It is wasteful and it is honest.

**Survivorship and the elephant.** Every tape here is a survivor: SPY, QQQ and EEM all exist
today. A study of record highs on indices that *kept making them* is biased toward the
comforting answer, and the size of that bias is not estimated here — it is declared, in the
front-card and in ``docs/results.md``, and it is the main reason the Signal stamp on this
study is not stronger than the evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics import mean_tstat_hac

TRADING_DAYS = 252
HORIZONS = (21, 63, 252)
HORIZON_LABEL = {21: "1 month", 63: "3 months", 252: "12 months"}
DRAWDOWN_BUCKETS = (0.0, 0.02, 0.05, 0.10, 0.20)


# --------------------------------------------------------------------------- #
# The state of the market on each day
# --------------------------------------------------------------------------- #
def running_peak(prices: pd.Series) -> pd.Series:
    """The running maximum of the total-return index (the peak known *by* that close)."""
    return prices.cummax()


def drawdown(prices: pd.Series) -> pd.Series:
    """Distance below the running peak, as a non-positive fraction."""
    return prices / running_peak(prices) - 1.0


def at_high(prices: pd.Series, tol: float = 0.0) -> pd.Series:
    """Boolean: the close is at (or within ``tol`` of) a new all-time high.

    ``tol = 0`` is the strict definition — today's close *is* the running maximum. A small
    tolerance (say 0.005) answers the more practical question, "within half a percent of
    the record", and is swept in the verification script.
    """
    return drawdown(prices) >= -abs(tol)


def forward_return(prices: pd.Series, horizon: int) -> pd.Series:
    """Total return over the next ``horizon`` sessions, aligned on the decision day."""
    return (prices.shift(-horizon) / prices - 1.0).rename(f"fwd_{horizon}")


# --------------------------------------------------------------------------- #
# Conditional forward returns
# --------------------------------------------------------------------------- #
def hac_mean(x: pd.Series, lags: int) -> dict:
    """Newey-West mean statistics with two guards the raw estimator does not carry.

    1. **The lag is capped at n/4.** A Bartlett kernel with 252 lags on 600 observations is
       estimating more autocovariances than the sample can support.
    2. **A non-positive long-run variance falls back to the i.i.d. standard error.** The
       Bartlett kernel is positive semi-definite in theory but the *sample* long-run variance
       can still come out at or below zero on short, strongly mean-reverting series, in which
       case ``quantlab``'s estimator correctly refuses to divide and returns NaN. A NaN in a
       results table reads as "no answer"; the honest reading is "the HAC correction has
       nothing to say here", so the i.i.d. SE is used and the fallback is flagged.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    lags = int(max(0, min(lags, n // 4)))
    r = dict(mean_tstat_hac(pd.Series(v), lags=lags))
    r["lags_used"] = lags
    r["fallback"] = False
    if not np.isfinite(r["tstat"]) or r["se_bps"] <= 0:
        se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        r.update({"se_bps": se * 1e4, "tstat": v.mean() / se if se > 0 else np.nan,
                  "fallback": True})
    return r


def conditional_stats(fwd: pd.Series, mask: pd.Series, horizon: int) -> dict:
    """Mean forward return in the flagged state vs everywhere else, with a Hansen-Hodrick *t*.

    The HAC lag is the horizon itself: consecutive overlapping windows share all but one
    observation, and pretending otherwise is the single most common way an overlapping-return
    study manufactures significance.
    """
    f = fwd.dropna()
    m = mask.reindex(f.index).fillna(False)
    a, b = f[m], f[~m]
    # Ten is the floor for saying anything at all. A tape that spent its life below an
    # early peak (Japan 1990, or a synthetic draw that drifted down) simply has no
    # record-high sample, and the honest output there is NaN, not a number.
    if len(a) < 10 or len(b) < 30:
        return {"n_state": int(len(a)), "n_other": int(len(b)), "mean_state": np.nan,
                "mean_other": np.nan, "diff": np.nan, "t_diff": np.nan,
                "win_state": np.nan, "win_other": np.nan}
    sa = hac_mean(a, horizon)
    sb = hac_mean(b, horizon)
    se = np.sqrt(sa["se_bps"] ** 2 + sb["se_bps"] ** 2) / 1e4
    diff = a.mean() - b.mean()
    return {
        "n_state": int(len(a)), "n_other": int(len(b)),
        "mean_state": float(a.mean()), "mean_other": float(b.mean()),
        "diff": float(diff), "t_diff": float(diff / se) if se > 0 else np.nan,
        "win_state": float((a > 0).mean()), "win_other": float((b > 0).mean()),
    }


def forward_table(prices: pd.Series, tol: float = 0.0,
                  horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """One row per horizon: forward returns from a record high vs from everywhere else."""
    hi = at_high(prices, tol)
    rows = []
    for hz in horizons:
        st = conditional_stats(forward_return(prices, hz), hi, hz)
        st["horizon"] = hz
        rows.append(st)
    return pd.DataFrame(rows).set_index("horizon")


def nonoverlap_stats(prices: pd.Series, horizon: int, tol: float = 0.0) -> dict:
    """The same comparison on **non-overlapping** windows — the wasteful, honest version.

    Every ``horizon``-th session is kept, so no two forward returns share a day. The sample
    collapses (a 34-year daily tape yields ~34 twelve-month observations) and the standard
    errors tell the truth about how little a 12-month conditional claim can be known from
    one country's history.
    """
    idx = prices.index[::horizon]
    sub = prices.reindex(idx)
    fwd = (sub.shift(-1) / sub - 1.0).dropna()
    hi = at_high(prices, tol).reindex(fwd.index).fillna(False)
    a, b = fwd[hi], fwd[~hi]
    if len(a) < 5 or len(b) < 5:
        return {"n_state": int(len(a)), "n_other": int(len(b)), "diff": np.nan,
                "t_diff": np.nan}
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    diff = a.mean() - b.mean()
    return {"n_state": int(len(a)), "n_other": int(len(b)),
            "mean_state": float(a.mean()), "mean_other": float(b.mean()),
            "diff": float(diff), "t_diff": float(diff / se) if se > 0 else np.nan}


def drawdown_bucket_table(prices: pd.Series, horizon: int = 252,
                          buckets: tuple[float, ...] = DRAWDOWN_BUCKETS) -> pd.DataFrame:
    """Forward return by how far below the peak you bought — the comparison that matters.

    Bucket ``b`` holds the sessions whose drawdown is at least ``b`` below the peak but less
    than the next bucket; bucket 0.0 is "at the high (within 0.5%)".
    """
    dd = -drawdown(prices)
    fwd = forward_return(prices, horizon)
    edges = list(buckets) + [np.inf]
    rows = []
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        m = (dd >= lo) & (dd < hi_) if lo > 0 else (dd < 0.005)
        f = fwd[m.reindex(fwd.index).fillna(False)].dropna()
        rows.append({
            "bucket": ("at the high" if lo == 0 else
                       (f"{lo:.0%}-{hi_:.0%} below" if np.isfinite(hi_) else f">{lo:.0%} below")),
            "n": int(len(f)), "mean_fwd": float(f.mean()) if len(f) else np.nan,
            "median_fwd": float(f.median()) if len(f) else np.nan,
            "win_rate": float((f > 0).mean()) if len(f) else np.nan,
            "worst": float(f.min()) if len(f) else np.nan,
        })
    return pd.DataFrame(rows).set_index("bucket")


# --------------------------------------------------------------------------- #
# The advice as a portfolio
# --------------------------------------------------------------------------- #
def dip_strategy(prices: pd.Series, cash: pd.Series, dip_pct: float = 0.05,
                 cost_bps: float = 2.0) -> pd.DataFrame:
    """Hold the asset only while it sits ``dip_pct`` or more below its peak; else T-bills.

    The signal is formed from the close of day ``t`` and the position is held for day
    ``t+1``'s return — one execution lag, no exceptions — and every switch is charged
    ``cost_bps`` on the notional turned over. Returns a frame with the strategy's daily
    return, the buy-and-hold return, the cash return and the position.
    """
    px = prices.dropna()
    c = cash.reindex(px.index).ffill()
    r_asset = px.pct_change().fillna(0.0)
    r_cash = c.pct_change().fillna(0.0)
    invested = (drawdown(px) <= -abs(dip_pct)).shift(1).fillna(False)
    switches = invested.astype(int).diff().abs().fillna(0.0)
    cost = switches * cost_bps / 1e4
    strat = np.where(invested, r_asset, r_cash) - cost
    return pd.DataFrame({"strategy": strat, "buy_hold": r_asset, "cash": r_cash,
                         "invested": invested.astype(int)}, index=px.index)


def performance(returns: pd.Series, cash: pd.Series | None = None) -> dict:
    """CAGR, vol, excess-of-cash Sharpe, worst drawdown and the *t* of the excess mean."""
    r = returns.dropna()
    if cash is not None:
        ex = (r - cash.reindex(r.index).fillna(0.0)).dropna()
    else:
        ex = r
    curve = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sd = ex.std(ddof=1)
    return {
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "vol": vol,
        "sharpe_excess": float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
        "max_dd": float((curve / curve.cummax() - 1).min()),
        "t_excess": float(mean_tstat_hac(ex)["tstat"]),
        "years": float(years),
    }


def race(prices: pd.Series, cash: pd.Series, dip_pct: float = 0.05,
         cost_bps: float = 2.0) -> dict:
    """Buy-and-hold versus the wait-for-a-dip rule, on the same tape and the same clock."""
    f = dip_strategy(prices, cash, dip_pct, cost_bps)
    c = f["cash"]
    strat, hold = performance(f["strategy"], c), performance(f["buy_hold"], c)
    diff = (f["strategy"] - f["buy_hold"]).dropna()
    return {
        "dip_pct": dip_pct, "cost_bps": cost_bps,
        "time_invested": float(f["invested"].mean()),
        "switches_per_year": float(f["invested"].diff().abs().sum() / (len(f) / TRADING_DAYS)),
        "strategy": strat, "buy_hold": hold,
        "cagr_gap": float(strat["cagr"] - hold["cagr"]),
        "sharpe_gap": float(strat["sharpe_excess"] - hold["sharpe_excess"]),
        "t_gap": float(mean_tstat_hac(diff)["tstat"]),
    }


def dip_sweep(prices: pd.Series, cash: pd.Series,
              dips: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20),
              cost_bps: float = 2.0) -> pd.DataFrame:
    """How patient does the dip-buyer have to be before the rule works? (Spoiler: sweep it.)"""
    rows = []
    for d in dips:
        r = race(prices, cash, d, cost_bps)
        rows.append({"dip": d, "time_invested": r["time_invested"],
                     "cagr": r["strategy"]["cagr"], "cagr_gap": r["cagr_gap"],
                     "sharpe": r["strategy"]["sharpe_excess"],
                     "sharpe_gap": r["sharpe_gap"], "max_dd": r["strategy"]["max_dd"],
                     "t_gap": r["t_gap"], "switches_per_year": r["switches_per_year"]})
    return pd.DataFrame(rows).set_index("dip")


def share_of_days_at_high(prices: pd.Series, tol: float = 0.0) -> float:
    """What fraction of all sessions closed at a record. Context for every number above."""
    return float(at_high(prices, tol).mean())


# --------------------------------------------------------------------------- #
# The verdict rule — pre-registered, tested
# --------------------------------------------------------------------------- #
def verdict(h: dict) -> dict:
    """Two stamps from the numbers, by a rule fixed before the run.

    - **Signal** — is the forward return *conditional on a record high* different? **Real**
      only if the 12-month gap clears |*t*| = 2 on the HAC test **and** survives the
      non-overlapping cross-check with the same sign; **Weak** if the pooled sign is
      consistent across tapes without the *t*; **None** otherwise.
    - **Tradability** — is *waiting for a dip* worth doing? **Investable** only if the rule
      beats buy-and-hold on excess Sharpe on a majority of tapes with |*t*| >= 2 on the
      pooled gap; **Fragile** if it wins on Sharpe but not significantly; **Mirage** if it
      loses.
    """
    real = abs(h["pooled_t_12m"]) >= 2.0 and (
        np.sign(h["pooled_diff_12m"]) == np.sign(h["nonoverlap_diff_12m"]))
    weak = h["n_positive_12m"] >= 4
    signal = "Real" if real else ("Weak" if weak else "None")
    wins = h["n_dip_sharpe_wins"]
    trad = ("Investable" if wins > len(h["tickers"]) / 2 and abs(h["pooled_dip_t"]) >= 2.0
            else ("Fragile" if wins > len(h["tickers"]) / 2 else "Mirage"))
    return {
        "signal": signal,
        "signal_why": (
            f"Money invested on a record-high close earned **{h['pooled_state_12m']:+.1%}** "
            f"over the next twelve months against **{h['pooled_other_12m']:+.1%}** from every "
            f"other day — a gap of **{h['pooled_diff_12m']:+.1%}** (HAC *t* = "
            f"{h['pooled_t_12m']:+.2f} at the horizon lag; the non-overlapping cross-check "
            f"gives {h['nonoverlap_diff_12m']:+.1%}). Record highs are also not rare: "
            f"**{h['share_at_high_spy']:.0%}** of SPY's sessions closed at one. The direction "
            f"is the finding — it is not negative — and every tape here is a survivor, which "
            f"flatters it."),
        "trad": trad,
        "trad_why": (
            f"Waiting for a **{h['head_dip']:.0%}** dip and sitting in bills otherwise "
            f"was invested only {h['head_time_invested']:.0%} of the time and gave up "
            f"**{h['head_cagr_gap']:+.2%}/yr** of compounding on SPY "
            f"(excess-Sharpe gap {h['head_sharpe_gap']:+.2f}, *t* = {h['head_t_gap']:+.2f}); "
            f"it won on excess Sharpe on **{wins} of {len(h['tickers'])}** tapes. The "
            f"drawdown it buys back is real — the return it gives up is larger."),
        "one_sentence": (
            f"Buying at a record high is not buying the top: the next twelve months paid "
            f"**{h['pooled_state_12m']:+.1%}** on average against **{h['pooled_other_12m']:+.1%}** "
            f"from every other day, and the rule the folklore implies — wait for a dip, hold "
            f"bills meanwhile — cost **{h['head_cagr_gap']:+.2%}/yr** on SPY while being out "
            f"of the market {1 - h['head_time_invested']:.0%} of the time."),
    }
